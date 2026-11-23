"""Bounded amplification, independent stage composition and portable circuits."""
from dataclasses import asdict
import contextlib
import io
import json
import math
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from qiskit import QuantumCircuit, qasm2, qasm3, qpy
from qiskit.quantum_info import Operator, Statevector

from amplify import optimize_database
from direct_search.amplification import amplification_metadata, best_amplified_expected_t
from direct_search.axial import z_rotation_metadata
from direct_search.circuits import circuit_from_template, clifford_group, matrix_key, t_count
from direct_search.dsa import SearchConfig, analyze_w, encode_matrix
from direct_search.export import export_database
from direct_search.main import main as search_main, parser as search_parser
from direct_search.workflow import export_variants, variant_paths
from direct_search.tests.test_direct_search import known_entry, execute_forced_path
from reduction.clucr import canonicalize, reduce_entries
from reduction.common import decode
from reduction.dataset import restore_member
from reduction.pipeline import analyze_database, export_entry, require_amplified_source


def low_probability_entry(swap_success=False):
    """Native two-CZ example with p=5/16, improved by one amplification round."""
    template = {'anc_g0': ['x'], 'C_a0': ['h', 't', 'h', 't'], 'anc_g1': ['s', 'h'],
                'anc_g2': ['h', 's'], 'C_a1': ['h', 't', 'h', 't'], 'anc_g3': ['h', 's', 'h'],
                'anc_g4': [], 'C_a2': ['h', 't', 'h', 't'], 'anc_g5': ['h', 's', 'h'],
                'dat_g0': ['h', 's'], 'C_d0': [], 'dat_g1': ['h', 's', 'h']}
    if swap_success:
        template['anc_g5'].append('x')
    analysis = analyze_w(Operator(circuit_from_template(template)).data)
    witness = z_rotation_metadata(analysis.unitary)
    return dict(template_ops=template, template_labels={}, fig3_t_partition=[2, 2, 2, 0],
                unitary_key=matrix_key(analysis.unitary), unitary_exact_key=matrix_key(analysis.unitary),
                unitary_matrix=encode_matrix(analysis.unitary), raw_t_count=6,
                success_probability=analysis.probability, expected_t_count=6/analysis.probability,
                success_outcomes=analysis.success_outcomes, failure_outcomes=analysis.failure_labels,
                failure_outcome_ops=analysis.failure_ops, branch_probabilities=analysis.branch_probabilities,
                zero_probability_outcomes=analysis.zero_outcomes, deduplication_method='exact',
                axial=witness['axial'], axial_angle=witness['angle'], z_rotation=witness)


class AmplificationWorkflowTests(unittest.TestCase):
    def test_default_verified_search_has_no_clifford_parser_option(self):
        from direct_search.dsa import prepare_search
        self.assertEqual(SearchConfig().clifford_sets, 'verified')
        _, _, stats = prepare_search(SearchConfig(max_raw_t=0))
        self.assertTrue(stats['position_coverage']['all_slots_cover_clifford_group'])
        self.assertNotIn('--clifford-sets', search_parser().format_help())
        self.assertEqual(search_parser().parse_args([]).amp_max_j, 10)

    def test_integrated_cli_exports_both_variants_and_reduction_uses_amplified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = low_probability_entry()
            database = {source['unitary_key']: source}
            stats = {'config': asdict(SearchConfig(workers=1))}
            before = json.dumps(database, sort_keys=True)
            with patch('direct_search.main.build_database', return_value=(database, stats)) as search, \
                 contextlib.redirect_stdout(io.StringIO()):
                code = search_main(['--max-raw-t', '6', '--workers', '1', '--output-dir', str(root/'direct'),
                                    '--amplified-output-dir', str(root/'amp'), '--export-attempt-qasm2'])
            self.assertEqual(code, 0)
            self.assertEqual(search.call_args.args[0].clifford_sets, 'verified')
            self.assertEqual(search.call_args.args[0].amp_max_j, 0)
            self.assertEqual(json.dumps(database, sort_keys=True), before)
            direct = json.loads((root/'direct/database.json').read_text())
            amp = json.loads((root/'amp/database.json').read_text())
            self.assertEqual(direct['entries'][0]['amplification_rounds'], 0)
            amplified = amp['entries'][0]
            self.assertEqual(amplified['amplification_rounds'], 1)
            self.assertLess(amplified['effective_t_count'], amplified['expected_t_count'])
            self.assertEqual(amplified['entry_id'], direct['entries'][0]['entry_id'])
            self.assertEqual(amplified['unitary_matrix'], direct['entries'][0]['unitary_matrix'])
            qasm3.load(str(root/'amp'/amplified['export_files']['amplified_rus_loop_qasm3']))
            with self.assertRaisesRegex(ValueError, 'amplified'):
                analyze_database(root/'direct', {'clucr': root/'wrong'}, workers=1)
            with self.assertRaisesRegex(ValueError, 'effective'):
                analyze_database(root/'amp', {'clucr': root/'wrong'}, workers=1, cost='direct')
            result = analyze_database(root/'amp', {'clucr': root/'reduced'}, workers=1)
            self.assertEqual(result['clucr']['selection_cost'], 'effective')
            self.assertEqual(result['clucr']['output_entries'], 1)
            amplified['effective_t_count'] = amplified['expected_t_count']
            with self.assertRaisesRegex(ValueError, 'effective_t_count'):
                require_amplified_source(amp, 1e-10)

    def test_pair_paths_and_existing_amplified_output_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct, amplified = variant_paths(root/'out_record_paper')
            self.assertEqual(amplified, root/'out_amplified_paper')
            for output in (direct, direct/'nested', root):
                with self.assertRaises(ValueError):
                    variant_paths(direct, output)
            amplified.mkdir()
            (amplified/'database.json').write_text('{}')
            with self.assertRaises(FileExistsError):
                export_variants({}, {'config': asdict(SearchConfig())}, direct)
            self.assertFalse(direct.exists())

    def test_pipeline_routes_integrated_outputs_and_rejects_unamplified_reuse(self):
        from run_all import main as pipeline_main
        output = io.StringIO()
        with patch.object(sys, 'argv', ['run_all.py', '--dry-run']), contextlib.redirect_stdout(output):
            pipeline_main()
        commands = [shlex.split(line) for line in output.getvalue().splitlines()]
        self.assertEqual(commands[0][2], 'direct_search')
        self.assertNotIn('amplify', [command[2] for command in commands])
        amplified = commands[0][commands[0].index('--amplified-output-dir')+1]
        reduced = next(command for command in commands if command[2] == 'reduction')
        self.assertEqual(reduced[reduced.index('--input-dir')+1], amplified)
        self.assertNotIn('--clifford-sets', commands[0])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'database.json').write_text(json.dumps(dict(algorithm='clucr', stats={
                'export_ok': True, 'selection_cost': 'direct'})))
            with patch.object(sys, 'argv', ['run_all.py', '--skip-reduction', '--reduction-dir', str(root)]), \
                 patch('run_all.subprocess.run') as run, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    pipeline_main()
                run.assert_not_called()

    def test_optimum_matches_full_enumeration_with_both_caps(self):
        for raw in (0, 2, 15):
            for probability in (.0001, .001, .01, .1, .25, 1/3, .75, 1.):
                for maximum in (0, 1, 10, 100):
                    for cap in (None, raw, 3*raw, 17*raw):
                        with self.subTest(raw=raw, p=probability, j=maximum, cap=cap):
                            theta = math.asin(math.sqrt(probability))
                            values = [(raw/probability, 0, probability)]
                            for j in range(1, maximum+1):
                                if cap is not None and (2*j+1)*raw > cap:
                                    continue
                                p = math.sin((2*j+1)*theta)**2
                                if p:
                                    values.append(((2*j+1)*raw/p, j, p))
                            expected = min(values, key=lambda v: (v[0], v[1]))
                            actual = best_amplified_expected_t(raw, probability, maximum, cap)
                            self.assertAlmostEqual(actual[0], expected[0])
                            self.assertEqual(actual[1], expected[1])

    def test_invalid_bounds_rejected(self):
        for values in ((2, .1, -1, None), (2, .1, 1.5, None), (2, .1, 10, 1),
                       (-1, .1, 10, None), (2, float('nan'), 10, None)):
            with self.assertRaises(ValueError):
                best_amplified_expected_t(*values)

    def test_independent_stage_is_portable_and_does_not_modify_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = known_entry()
            low = low_probability_entry()
            export_database({e['unitary_key']: e for e in (original, low)},
                            {'config': asdict(SearchConfig(workers=1))}, root/'source', export_qasm2=True)
            before = (root/'source/database.json').read_bytes()
            optimize_database(root/'source', root/'amp', max_j=10, workers=2, chunk_size=1)
            payload = json.loads((root/'amp/database.json').read_text())
            entry = next(e for e in payload['entries'] if e['raw_t_count'] == 2)
            amplified = next(e for e in payload['entries'] if e['raw_t_count'] == 6)
            self.assertEqual(amplified['amplification_rounds'], 1)
            self.assertLess(amplified['effective_t_count'], amplified['expected_t_count'])
            self.assertEqual(payload['stats']['amplification']['workers_used'], 2)
            qasm3.load(str(root/'amp'/amplified['export_files']['amplified_rus_loop_qasm3']))
            self.assertEqual((root/'source/database.json').read_bytes(), before)
            self.assertEqual(entry['amplification_rounds'], 0)
            self.assertEqual(entry['amplification_search']['max_j'], 10)
            self.assertTrue(payload['stats']['export_ok'])
            (root/'source').rename(root/'moved_source')
            files = entry['export_files']
            qasm2.load(str(root/'amp'/files['attempt_qasm2']))
            qasm3.load(str(root/'amp'/files['rus_loop_qasm3']))
            with self.assertRaises(FileExistsError):
                optimize_database(root/'moved_source', root/'amp')
            with self.assertRaises(ValueError):
                optimize_database(root/'moved_source', root/'cap_too_small', max_raw_t=1)

    def test_effective_and_direct_select_different_class_winners(self):
        source = known_entry()
        canonical = canonicalize(decode(source['unitary_matrix']))
        cheap_direct = dict(source, source_entry_id='direct', canonical=canonical,
                            expected_t_count=12., effective_t_count=12.)
        cheap_amplified = dict(source, source_entry_id='amp', canonical=canonical,
                               expected_t_count=20., effective_t_count=8.)
        self.assertEqual(reduce_entries([cheap_amplified, cheap_direct], cost='direct')[0][0]['source_entry_id'], 'direct')
        winner, members = reduce_entries([cheap_amplified, cheap_direct])
        self.assertEqual(winner[0]['source_entry_id'], 'amp')
        self.assertEqual(len(members), 2)

    def test_reduced_amplified_qasm_and_qpy_execute_recovery(self):
        for swap in (False, True):
            entry = known_entry(swap)
            entry.update(amplification_metadata(2, .75, 0))
            # j=2 has p=3/4 again: this checks the executable export, not selection.
            entry.update(entry_id='amplified_example', amplification_rounds=2,
                         amplified_raw_t_count=10, effective_t_count=10/.75,
                         transformation=canonicalize(decode(entry['unitary_matrix'])))
            with tempfile.TemporaryDirectory() as tmp:
                result, _ = export_entry(entry, Path(tmp), 1e-10)
                files = result['export_files']
                W = qasm2.load(str(Path(tmp)/files['amplified_w_qasm2']))
                self.assertEqual(t_count(W), 10)
                loop = qasm3.load(str(Path(tmp)/files['amplified_target_rus_loop_qasm3']))
                success = int(entry['success_outcomes'][0])
                psi = np.array([1, 1j])/np.sqrt(2)
                state, _ = execute_forced_path(loop, psi, [1-success, 1-success, success])
                target = decode(entry['transformation']['unitary_matrix']) @ psi
                self.assertTrue(Statevector(state.data[2*success:2*success+2]).equiv(Statevector(target)))
                with (Path(tmp)/files['qpy']).open('rb') as handle:
                    circuits = qpy.load(handle)
                index = result['qpy_indices']['amplified_w']
                self.assertTrue(Operator(circuits[index]).equiv(Operator(W)))

    def test_restore_member_uses_amplified_winner_cost_and_recovery(self):
        for swap in (False, True):
            with self.subTest(swap_success=swap), tempfile.TemporaryDirectory() as tmp:
                entry = low_probability_entry(swap)
                entry.update(amplification_metadata(entry['raw_t_count'], entry['success_probability'], 10))
                self.assertEqual(entry['amplification_rounds'], 1)
                self.assertLess(entry['effective_t_count'], entry['expected_t_count'])
                entry.update(entry_id='winner', class_id='class', source_entry_id='source',
                             transformation=canonicalize(decode(entry['unitary_matrix'])))
                decorated = (clifford_group()[1].matrix @ decode(entry['unitary_matrix'])
                             @ clifford_group()[4].matrix)
                member = dict(class_id='class', transformation=canonicalize(decorated))
                exported, _ = export_entry(entry, Path(tmp), 1e-10)
                restored = restore_member(tmp, exported, member)

                # Recover the actual attempt from the loaded dynamic loop so
                # the check detects a valid-U circuit with the wrong T/p cost.
                loop = next(inst for inst in restored.data if inst.operation.name == 'while_loop')
                body = loop.operation.blocks[0]
                wires = [restored.find_bit(qubit).index for qubit in loop.qubits]
                attempt = QuantumCircuit(2)
                for inst in body.data:
                    if inst.operation.name == 'measure':
                        break
                    if inst.operation.name != 'reset':
                        attempt.append(inst.operation, [wires[body.find_bit(q).index] for q in inst.qubits])
                analysis = analyze_w(Operator(attempt).data)
                self.assertTrue(analysis.accepted)
                self.assertEqual(t_count(attempt), entry['amplified_raw_t_count'])
                self.assertAlmostEqual(analysis.probability, entry['amplified_success_probability'])
                self.assertAlmostEqual(t_count(attempt) / analysis.probability, entry['effective_t_count'])

                success = int(entry['success_outcomes'][0])
                psi = np.array([1, 1j]) / np.sqrt(2)
                for failures in (0, 2):
                    state, _ = execute_forced_path(restored, psi, [1-success] * (failures+1) + [success])
                    actual = Statevector(state.data[2*success:2*success+2])
                    self.assertTrue(actual.equiv(Statevector(decorated @ psi), atol=1e-10, rtol=0))


if __name__ == '__main__':
    unittest.main()
