"""Independent Clifford, branch-physics, parser and process-determinism checks."""
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from qiskit import qasm2, qasm3, qpy
from qiskit.quantum_info import Operator, Statevector, random_unitary

from direct_search.circuits import circuit_from_ops, circuit_from_template, clifford_group
from direct_search.amplification import amplification_metadata
from direct_search.tests.test_direct_search import known_entry, execute_forced_path
from reduction.common import decode, encode, phase_error, verify_transformation, wrapper_circuit
from reduction.cuc_dagger import classify_axial
from reduction.clucr import canonicalize, reduce_entries
from reduction.pipeline import analyze_database, axial_entry, export_database, export_entry, validate_source
from reduction.dataset import restore_member
from reduction.audit import equivalent_class_pairs


def fixture(root, count=4, qasm2_only=False):
    root.mkdir(parents=True)
    entries = []
    for i in range(count):
        entry = known_entry(bool(i % 2))
        entry.update(amplification_metadata(entry["raw_t_count"], entry["success_probability"], 10))
        entry.update(entry_id=f"source_{i}", candidate_ordinal=i)
        name = f"source_{i}.qasm"
        if qasm2_only:
            circuit = circuit_from_template(entry["template_ops"], measure=True)
            writer, kind = qasm2.dump, "attempt_qasm2"
        else:
            circuit = circuit_from_template(entry["template_ops"])
            writer, kind = qasm3.dump, "w_qasm3"
        with (root / name).open("w") as handle:
            writer(circuit, handle)
        entry["export_files"] = {kind: name}
        entries.append(entry)
    payload = dict(schema_version=3, algorithm="direct_search", qubit_order={"data": 0, "ancilla": 1}, entries=entries,
                   stats=dict(dataset_variant="amplified", amplification=dict(max_j=10, max_raw_t=None)))
    (root / "database.json").write_text(json.dumps(payload))
    return entries


def member(unitary, name, cost, ordinal=0):
    return dict(unitary_matrix=encode(unitary), canonical=canonicalize(unitary), source_entry_id=name,
                expected_t_count=cost, effective_t_count=cost, raw_t_count=2, candidate_ordinal=ordinal,
                axial=classify_axial(unitary)["axial"])


class AxialTests(unittest.TestCase):
    def test_pauli_axes_phases_and_unfolded_angles(self):
        for angle in (0, 1e-7, .219, 2.1, -2.1, math.pi):
            diagonal = np.diag([1, np.exp(1j * angle)])
            for clifford in clifford_group():
                unitary = np.exp(.817j) * clifford.matrix @ diagonal @ clifford.matrix.conj().T
                result = classify_axial(unitary)
                self.assertTrue(result["axial"])
                c = Operator(circuit_from_ops(result["clifford_ops"])).data
                np.testing.assert_allclose(c @ unitary @ Operator(circuit_from_ops(result["right_ops"])).data, decode(result["diagonal_unitary_matrix"]), atol=1e-12)
        self.assertEqual(classify_axial(diagonal)["clifford_ops"], [])

    def test_left_right_diagonalizable_is_axial(self):
        h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        diagonal = np.diag([1, np.exp(.321j)])
        for unitary in (h @ diagonal, diagonal @ h):
            self.assertTrue(classify_axial(unitary)["axial"])
            self.assertLess(phase_error(decode(canonicalize(unitary)["unitary_matrix"]),
                                       decode(canonicalize(diagonal)["unitary_matrix"])), 1e-10)

    def test_invalid_matrices_and_tolerances(self):
        for function in (classify_axial, canonicalize):
            for matrix in (np.eye(4), np.zeros((2, 2)), np.full((2, 2), np.nan), np.eye(2) * 2):
                with self.assertRaises(ValueError):
                    function(matrix)
            for tol in (0, -1, np.nan, np.inf, .1):
                with self.assertRaises(ValueError):
                    function(np.eye(2), tol)


class CliffordReductionTests(unittest.TestCase):
    def test_independent_audit_detects_unmerged_orbits(self):
        unitary = random_unitary(2, seed=321).data
        decorated = clifford_group()[1].matrix @ unitary @ clifford_group()[4].matrix
        entries = [dict(class_id=name, unitary_matrix=encode(matrix)) for name, matrix in
                   (("a", unitary), ("b", decorated), ("c", random_unitary(2, seed=432).data))]
        self.assertEqual(equivalent_class_pairs(entries), [("a", "b")])

    def test_all_576_pairs_and_inverse_relation(self):
        for unitary in (random_unitary(2, seed=41).data, np.diag([1, np.exp(-.219j)]), np.array([[0, 1], [1, 0]])):
            target = decode(canonicalize(unitary)["unitary_matrix"])
            for left in clifford_group():
                for right in clifford_group():
                    decorated = np.exp(.719j) * left.matrix @ unitary @ right.matrix
                    result = canonicalize(decorated)
                    self.assertLess(phase_error(target, decode(result["unitary_matrix"])), 1e-10)
                    verify_transformation(dict(unitary_matrix=encode(decorated), transformation=result), 1e-10)

    def test_representative_is_lexicographic_not_axial_folding(self):
        unitary = random_unitary(2, seed=92).data
        values = []
        for left in clifford_group():
            for right in clifford_group():
                v = left.matrix @ unitary @ right.matrix
                v /= v[0, 0] / abs(v[0, 0])
                values.append(tuple(np.round(np.stack((v.real, v.imag), axis=-1).ravel(), 10)))
        result = decode(canonicalize(unitary)["unitary_matrix"])
        self.assertEqual(tuple(np.round(np.stack((result.real, result.imag), axis=-1).ravel(), 10)), min(values))
        axial = canonicalize(np.diag([1, np.exp(.219j)]))
        self.assertGreater(abs(decode(axial["unitary_matrix"])[0, 1]), .01)

    def test_first_row_phase_pivot_in_second_column(self):
        unitary = np.exp(.719j) * np.array([[0, 1], [np.exp(.321j), 0]])
        pivot = unitary[0, 1]
        normalized = unitary * pivot.conjugate() / np.sqrt(abs(pivot)**2)
        self.assertAlmostEqual(normalized[0, 1], 1.)
        orbit = []
        for left in clifford_group():
            for right in clifford_group():
                v = left.matrix @ normalized @ right.matrix
                u = next(value for value in v[0] if abs(value) > 1e-10)
                v = v * u.conjugate() / np.sqrt(abs(u)**2)
                orbit.append(tuple(np.round(np.stack((v.real, v.imag), axis=-1).ravel(), 10)))
        witness = canonicalize(unitary)
        result = decode(witness["unitary_matrix"])
        self.assertEqual(tuple(np.round(np.stack((result.real, result.imag), axis=-1).ravel(), 10)), min(orbit))
        verify_transformation(dict(unitary_matrix=encode(unitary), transformation=witness), 1e-10)

    def test_cost_ranking_membership_and_rounding_boundaries(self):
        unitary = np.diag([1, np.exp(.219j)])
        h = clifford_group()[1].matrix
        a, b, c = member(unitary, "a", 8), member(h @ unitary, "b", 5), member(unitary, "c", 5, 1)
        a["effective_t_count"] = .5  # Effective cost now decides the winner.
        a["canonical"]["key"] = "different_rounded_key"
        unrelated = member(np.diag([1, np.exp(.229j)]), "other", 4)
        winners, mapping = reduce_entries([c, unrelated, a, b])
        self.assertEqual(len(winners), 2)
        self.assertEqual(len(mapping), 4)
        self.assertEqual(winners[0]["source_entry_id"], "a")
        self.assertEqual(winners[0]["member_count"], 3)
        for row in mapping:
            verify_transformation(row, 1e-10)
        self.assertEqual(reduce_entries([b, a, unrelated, c])[0], winners)


class ExportPhysicsTests(unittest.TestCase):
    def test_dynamic_retries_and_source_restoration_for_both_success_outcomes(self):
        for swap in (False, True):
            entry = known_entry(swap)
            entry["transformation"] = canonicalize(decode(entry["unitary_matrix"]))
            success, failure = int(entry["success_outcomes"][0]), int(next(iter(entry["failure_outcome_ops"])))
            for restore in (False, True):
                circuit = wrapper_circuit(entry, restore=restore)
                imported = qasm3.loads(qasm3.dumps(circuit))
                desired = decode(entry["unitary_matrix"] if restore else entry["transformation"]["unitary_matrix"])
                for psi in ([1, 0], [0, 1], np.array([1, 1j]) / np.sqrt(2)):
                    for failures in (0, 1, 3):
                        state, _ = execute_forced_path(imported, psi, [failure] * (failures + 1) + [success])
                        actual = Statevector(state.data[2 * success:2 * success + 2])
                        self.assertTrue(actual.equiv(Statevector(desired @ psi), atol=1e-10, rtol=0))

    def test_corrupt_source_rejected_and_qasm2_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            entry = fixture(root, 1, qasm2_only=True)[0]
            self.assertEqual(validate_source(entry, root, 1e-10)["parser"], "qiskit.qasm2.load")
            entry["failure_outcome_ops"] = {"1": []}
            with self.assertRaisesRegex(ValueError, "recovery"):
                validate_source(entry, root, 1e-10)
            entry["success_probability"] = float("nan")
            with self.assertRaisesRegex(ValueError, "probability"):
                validate_source(entry, root, 1e-10)

    def test_restore_a_different_member_using_the_winning_circuit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = known_entry()
            entry.update(entry_id="winner", source_entry_id="original", class_id="class")
            entry["transformation"] = canonicalize(decode(entry["unitary_matrix"]))
            exported, _ = export_entry(entry, root, 1e-10)
            decorated = clifford_group()[1].matrix @ decode(entry["unitary_matrix"]) @ clifford_group()[4].matrix
            member = dict(class_id="class", transformation=canonicalize(decorated))
            restored = restore_member(root, exported, member)
            psi = np.array([1, 1j]) / np.sqrt(2)
            state, _ = execute_forced_path(restored, psi, [1, 1, 1, 0])
            self.assertTrue(Statevector(state.data[:2]).equiv(Statevector(decorated @ psi), atol=1e-10, rtol=0))

    def test_qpy_and_qasm_are_independently_loadable(self):
        with tempfile.TemporaryDirectory() as temporary:
            entry = known_entry()
            entry.update(entry_id="test", source_entry_id="source")
            entry["axial_witness"] = classify_axial(decode(entry["unitary_matrix"]))
            entry = axial_entry(entry)
            entry, rows = export_entry(entry, Path(temporary), 1e-10)
            self.assertEqual(len(rows), 5)
            with (Path(temporary) / entry["export_files"]["qpy"]).open("rb") as handle:
                circuits = qpy.load(handle)
            reference = circuits[entry["qpy_indices"]["target_unitary_reference"]]
            self.assertTrue(Operator(reference).equiv(Operator(decode(entry["transformation"]["unitary_matrix"]))))
            attempt = qasm2.load(str(Path(temporary) / entry["export_files"]["attempt_qasm2"]))
            self.assertEqual(attempt.count_ops()["measure"], 1)


class PipelineTests(unittest.TestCase):
    def test_serial_parallel_two_stage_equivalence_and_reuse_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root / "source")
            summaries = []
            for workers in (1, 2):
                outputs = {mode: root / str(workers) / mode for mode in ("cuc_dagger", "clucr")}
                stats = analyze_database(root / "source", outputs, workers=workers, chunk_size=1)
                self.assertEqual(stats["cuc_dagger"]["output_entries"], 1)
                self.assertEqual(stats["clucr"]["output_entries"], 1)
                self.assertFalse((outputs["cuc_dagger"] / "database.json").exists())
                for output in outputs.values():
                    result = export_database(output, workers=workers, chunk_size=1)
                    self.assertEqual(result["validation_failures"], 0)
                    with self.assertRaises(FileExistsError):
                        export_database(output, workers=workers)
                summaries.append({mode: json.loads((output / "database.json").read_text())["entries"]
                                  for mode, output in outputs.items()})
            self.assertEqual(summaries[0], summaries[1])
            with self.assertRaises(FileExistsError):
                analyze_database(root / "source", outputs)


if __name__ == "__main__":
    unittest.main()
