"""Independent physics and determinism checks for the expanded search."""
import math
import tempfile
import unittest
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.quantum_info import Operator, Statevector, random_unitary

from direct_search.amplification import amplify_circuit, amplification_metadata, best_amplified_expected_t
from direct_search.circuits import (SLOTS, allowed_sets, circuit_from_ops, circuit_from_template,
                                clifford_group, equivalent, matrix_key, t_count)
from direct_search.dsa import SearchConfig, analyze_w, build_database, decode_matrix, prepare_search
from direct_search.export import validate_entry
from direct_search.hardware import hardware_profile
from direct_search.examples import paper_circuits
from direct_search.research import v3_cost_estimate
from direct_search.parallel import analyze_batch, candidate_matrices
from direct_search.axial import z_rotation_metadata
from direct_search.validate_qasm import reference_circuit
from test_direct_search import known_entry, execute_forced_path


class AxialTests(unittest.TestCase):
    def test_double_coset_detects_all_pauli_axes_with_constructive_witness(self):
        for angle in (0, 1e-7, .219, math.pi/4, math.pi/2, 2.1, -2.1, math.pi):
            U = np.diag([1, np.exp(1j*angle)])
            direct = z_rotation_metadata(U)
            self.assertEqual(direct["clifford_ops"], [])
            self.assertAlmostEqual(direct["angle"], angle)
            for C in clifford_group():
                V = np.exp(.917j) * C.matrix @ U @ C.matrix.conj().T
                witness = z_rotation_metadata(V)
                self.assertIsNotNone(witness["angle"])
                D = Operator(circuit_from_ops(witness["clifford_ops"])).data
                R = Operator(circuit_from_ops(witness["right_ops"])).data
                diagonal = D @ V @ R
                self.assertTrue(equivalent(diagonal, np.diag([1, np.exp(1j*witness["angle"])])))

    def test_independent_left_right_cliffords_are_axial(self):
        U = np.diag([1, np.exp(.321j)])
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        for L in clifford_group():
            for R in clifford_group():
                self.assertIsNotNone(z_rotation_metadata(L.matrix @ U @ R.matrix)["angle"])
        self.assertIsNone(z_rotation_metadata(random_unitary(2, seed=41).data)["angle"])

    def test_classification_rejects_invalid_input(self):
        for U in (np.eye(4), np.zeros((2, 2)), np.full((2, 2), np.nan)):
            with self.assertRaises(ValueError):
                z_rotation_metadata(U)

    def test_export_validation_rejects_stale_reduction_and_wrong_classification(self):
        for field, value in (("canonical", {}), ("deduplication_method", "clifford"),
                             ("axial", False), ("axial_angle", .3398),
                             ("unitary_key", "wrong")):
            entry = known_entry()
            entry[field] = value
            with self.assertRaises(ValueError):
                validate_entry(entry)
        with self.assertRaisesRegex(ValueError, "Unknown exported circuit kind"):
            reference_circuit(known_entry(), "canonical_rus_loop_qasm3")

    def test_paper_position_sets_are_literal_matrix_products(self):
        H = np.array([[1, 1], [1, -1]])/np.sqrt(2)
        S = np.diag([1, 1j])
        X = np.array([[0, 1], [1, 0]])
        paper = allowed_sets("paper")
        expected = {"anc_g0": [np.eye(2), X, S@H, S@H@X, H@S@H, H@S@H@X],
                    "anc_g1": [H, H@S, H@S@H], "anc_g2": [np.eye(2), S@H, H@S@H]}
        for slot, matrices in expected.items():
            for sequence, matrix in zip(paper[slot], matrices):
                self.assertTrue(equivalent(sequence.matrix, matrix))



class AmplificationTests(unittest.TestCase):
    def test_paper_numerical_example_and_optimum_bound(self):
        cost, j, p = best_amplified_expected_t(15, .1, 10)
        self.assertEqual(j, 1)
        self.assertAlmostEqual(p, .676)
        self.assertAlmostEqual(cost, 45/.676)
        for probability in (.01, .1, .25, 1/3, .75, 1):
            cost, j, p = best_amplified_expected_t(9, probability, 10)
            theta = math.asin(math.sqrt(probability))
            brute = min((2*k+1)*9 / max(math.sin((2*k+1)*theta)**2, 1e-300) for k in range(11))
            self.assertAlmostEqual(cost, brute)
        self.assertEqual(best_amplified_expected_t(2,.1,0), (20,0,.1))

    def test_oblivious_amplification_preserves_arbitrary_input_and_recovery(self):
        U = random_unitary(2, seed=801).data
        R = Operator(circuit_from_ops(["h", "s"])).data
        for p in (.1, .25, .625):
            W = np.block([[np.sqrt(p)*U, -np.sqrt(1-p)*U],
                          [np.sqrt(1-p)*R, np.sqrt(p)*R]])
            for success in ("0", "1"):
                circuit = QuantumCircuit(2)
                circuit.unitary(W, [0,1])
                if success == "1":
                    circuit.x(1)
                for j in (1,2,3):
                    amp = Operator(amplify_circuit(circuit, j, success)).data
                    expected_p = math.sin((2*j+1)*math.asin(math.sqrt(p)))**2
                    for outcome, target, probability in ((int(success), U, expected_p),
                                                         (1-int(success), R, 1-expected_p)):
                        block = amp[2*outcome:2*outcome+2,:2]
                        np.testing.assert_allclose(block.conj().T@block, probability*np.eye(2),atol=1e-12)
                        if probability > 1e-10:
                            self.assertTrue(equivalent(block/np.sqrt(probability), target))

    def test_native_amplified_loop_retries_and_validation_rejects_bad_metadata(self):
        for swap in (False, True):
            entry = known_entry(swap)
            # j=2 returns this p=3/4 circuit to p=3/4 with five W/W† copies.
            entry.update(amplification_metadata(2,.75,0))
            entry.update(amplification_rounds=2, amplified_raw_t_count=10, effective_t_count=10/.75)
            self.assertTrue(validate_entry(entry)["amplified_operator_and_recovery"])
            loop = qasm3.loads(qasm3.dumps(reference_circuit(entry,"amplified_rus_loop_qasm3")))
            psi=np.array([1,1j])/np.sqrt(2)
            success=int(entry["success_outcomes"][0])
            result,_=execute_forced_path(loop,psi,[1-success]*3+[success])
            self.assertTrue(Statevector(result.data[2*success:2*success+2]).equiv(
                Statevector(decode_matrix(entry["unitary_matrix"])@psi)))
            entry["amplified_success_probability"] = .1
            with self.assertRaisesRegex(ValueError,"Amplified probability"):
                validate_entry(entry)


class ParallelTests(unittest.TestCase):
    def test_direct_winners_match_unpruned_scalar_enumeration(self):
        config = SearchConfig(max_raw_t=2, job=(0, 0, 0, 2), workers=1,
                              chunk_size=307, batch_size=103)
        jobs, segments, _ = prepare_search(config)
        minima = {}
        accepted = 0
        for items in product(*(segments[i, t] for i, t in enumerate(jobs[0]))):
            template = {k: v for item in items for k, v in item.ops.items()}
            analysis = analyze_w(Operator(circuit_from_template(template)).data, config)
            if analysis.accepted:
                accepted += 1
                key = matrix_key(analysis.unitary)
                cost = sum(jobs[0]) / analysis.probability
                minima[key] = min(minima.get(key, float("inf")), cost)
        actual, stats = build_database(config)
        self.assertEqual(stats["accepted_candidates"], accepted)
        self.assertEqual(set(actual), set(minima))
        self.assertGreater(len(actual), 1)
        for key, cost in minima.items():
            self.assertAlmostEqual(actual[key]["expected_t_count"], cost, places=10)
            self.assertEqual(actual[key]["unitary_exact_key"], key)
            self.assertNotIn("canonical", actual[key])
        self.assertTrue(any(e["axial"] for e in actual.values()))

    def test_amplification_cannot_change_direct_winner_across_partitions(self):
        # Inject valid RUS branch operators to isolate database selection:
        # A: T=2,p=.1 -> direct 20, amplified ~8.88.
        # B: T=10,p=.75 -> direct 13.33; B must win even when A amplifies better.
        U = np.diag([1, np.exp(.321j)])
        jobs = [(2, 0, 0, 0), (10, 0, 0, 0)]
        segments = {(i, t): [SimpleNamespace(sequence=SimpleNamespace(matrix=np.eye(2)*t),
                                              labels={}, ops={})]
                    for job in jobs for i, t in enumerate(job)}

        def matrices(arrays, indices):
            p = .1 if arrays[0][0, 0, 0] == 2 else .75
            return np.array([np.block([[np.sqrt(p)*U, -np.sqrt(1-p)*U],
                                       [np.sqrt(1-p)*np.eye(2), np.sqrt(p)*np.eye(2)]])])

        self.assertLess(amplification_metadata(2, .1, 10)["effective_t_count"], 10/.75)
        for j in (0, 10):
            config = SearchConfig(max_raw_t=10, workers=1, chunk_size=1, batch_size=1, amp_max_j=j)
            stats = {"config": asdict(config), "planned_candidates": 2, "planned_partitions": 2}
            with patch("direct_search.parallel.prepare_search", return_value=(jobs, segments, stats)), \
                 patch("direct_search.parallel.candidate_matrices", side_effect=matrices):
                db, _ = build_database(config)
            self.assertEqual(len(db), 1)
            winner = next(iter(db.values()))
            self.assertEqual(winner["raw_t_count"], 10)
            self.assertAlmostEqual(winner["expected_t_count"], 10/.75)

    def test_worker_sizing_obeys_affinity_and_cgroup_limits(self):
        def read(path):
            path = str(path)
            fixtures = {"/proc/self/cgroup": "0::/test", "/proc/meminfo": "MemTotal: 8388608 kB\nMemAvailable: 4194304 kB",
                        "/sys/fs/cgroup/test/cpu.max": "200000 100000",
                        "/sys/fs/cgroup/test/memory.max": str(1024**3),
                        "/sys/fs/cgroup/test/memory.current": str(256*1024**2)}
            if path.endswith("/core_id"):
                return str(int(path.split("/cpu")[-1].split("/")[0]))
            if path.endswith("/physical_package_id"):
                return "0"
            return fixtures.get(path, "")
        with patch("direct_search.hardware._read", side_effect=read), \
             patch("direct_search.hardware.os.sched_getaffinity", return_value={0,1,2,3}), \
             patch("direct_search.hardware.os.cpu_count", return_value=32):
            profile=hardware_profile()
        self.assertEqual(profile["physical_cores_available"],4)
        self.assertEqual(profile["cpu_quota"],2)
        self.assertEqual(profile["recommended_workers"],2)
        self.assertEqual(profile["memory_available_bytes"],768*1024**2)

    def test_batched_matrix_and_analysis_match_qiskit(self):
        config=SearchConfig(max_raw_t=6,job=(1,2,1,2),workers=1)
        jobs,segments,_=prepare_search(config)
        pools=[segments[i,t] for i,t in enumerate(jobs[0])]
        rng=np.random.default_rng(982)
        indices=[rng.integers(len(pool),size=200) for pool in pools]
        matrices=candidate_matrices([np.array([s.sequence.matrix for s in pool]) for pool in pools],indices)
        reasons,unitaries,probabilities=analyze_batch(matrices,config)
        for i,position in enumerate(zip(*indices)):
            template={k:v for pool,index in zip(pools,position) for k,v in pool[index].ops.items()}
            reference=Operator(circuit_from_template(template)).data
            np.testing.assert_allclose(matrices[i],reference,atol=1e-12)
            analysis=analyze_w(reference,config)
            self.assertEqual(reasons[i],analysis.reason)
            if analysis.accepted:
                self.assertTrue(equivalent(unitaries[i],analysis.unitary))
                self.assertAlmostEqual(probabilities[i],analysis.probability)

    def test_process_count_chunking_and_backend_preserve_results(self):
        config=SearchConfig(max_raw_t=2,job=(0,0,0,2),workers=1,max_examined=2501,
                            chunk_size=1000,batch_size=173,dedup_mode="exact")
        expected,stats=build_database(config)
        for variant in (replace(config,workers=2,chunk_size=613,batch_size=101),
                        replace(config,backend="qiskit",batch_size=313)):
            actual,other=build_database(variant)
            self.assertEqual(set(actual),set(expected))
            for field in ("total_examined","accepted_candidates","rejections","completed_partitions","search_complete"):
                self.assertEqual(stats[field],other[field],field)
            for key,entry in actual.items():
                self.assertEqual(entry["candidate_ordinal"],expected[key]["candidate_ordinal"])
                self.assertEqual(entry["template_ops"],expected[key]["template_ops"])
                self.assertAlmostEqual(entry["effective_t_count"],expected[key]["effective_t_count"])
                validate_entry(entry)

    def test_parallel_caps_on_and_across_partition_boundaries(self):
        config=SearchConfig(max_raw_t=1,workers=2,chunk_size=1000)
        jobs,segments,_=prepare_search(config)
        boundary=math.prod(len(segments[i,t]) for i,t in enumerate(jobs[0]))
        for limit in (0,7,boundary,boundary+3):
            _,stats=build_database(replace(config,max_examined=limit))
            self.assertEqual(stats["total_examined"],limit)
            self.assertEqual(stats["accepted_candidates"]+sum(stats["rejections"].values()),limit)
            self.assertEqual(stats["completed_partitions"],int(limit>=boundary))


class PaperApplicationTests(unittest.TestCase):
    def test_fig8_fig9_and_section_6_1_costs(self):
        for name,circuit,target,probability in paper_circuits():
            analysis=analyze_w(Operator(circuit).data)
            self.assertTrue(analysis.accepted,name)
            self.assertTrue(equivalent(analysis.unitary,target),name)
            self.assertAlmostEqual(analysis.probability,probability)
            self.assertEqual(t_count(circuit),2 if name=="fig8" else 4)
            self.assertEqual(analysis.failure_ops,{"1":[]})
        self.assertAlmostEqual(v3_cost_estimate(1e-6)["estimated_expected_t_count"],
                               15.78*math.log(1e6,5))
        self.assertGreater(v3_cost_estimate(1e-6,6.4)["estimated_expected_t_count"],
                           v3_cost_estimate(1e-6)["estimated_expected_t_count"])


if __name__ == "__main__":
    unittest.main()
