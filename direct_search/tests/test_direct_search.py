"""Physics, gate ordering, search accounting, and executable export regressions."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qasm2, qasm3
from qiskit.quantum_info import Clifford, Operator, Statevector

from direct_search.circuits import (
    NAMED, SLOTS, allowed_sets, candidate_circuit, canonical_words, circuit_from_ops,
    circuit_from_template, clifford_group, core_pool, core_word_to_ops, equivalent,
    matrix_key, rus_loop_circuit, segment_pool, t_count, validate_template_form,
)
from direct_search.dsa import (
    SearchConfig, analyze_w, build_database, encode_matrix, partitions, unitary_identity,
)
from direct_search.export import export_database, validate_entry


def known_template(swap_success=False):
    # Original project's T=2 example: U=diag(1,(-1+2i sqrt(2))/3), p=3/4.
    return {
        "anc_g0": [], "C_a0": [], "anc_g1": ["h", "s", "s", "s"],
        "anc_g2": [], "C_a1": [], "anc_g3": ["h"],
        "anc_g4": ["s", "s", "s"], "C_a2": [],
        "anc_g5": ["h", "x"] if swap_success else ["h"],
        "dat_g0": ["s", "h"], "C_d0": ["t", "h", "t", "h"],
        "dat_g1": ["h", "s", "h"],
    }


def known_entry(swap_success=False):
    template = known_template(swap_success)
    U = np.diag([1, (-1 + 2j * np.sqrt(2)) / 3])
    success, failure = ("1", "0") if swap_success else ("0", "1")
    return {
        "unitary_key": matrix_key(U), "unitary_exact_key": matrix_key(U),
        "deduplication_method": "exact", "unitary_matrix": encode_matrix(U),
        "raw_t_count": 2, "success_probability": 0.75, "expected_t_count": 8 / 3,
        "success_outcomes": [success], "failure_outcomes": {failure: "X"},
        "failure_outcome_ops": {failure: ["x"]},
        "branch_probabilities": {success: 0.75, failure: 0.25},
        "zero_probability_outcomes": [], "template_ops": template,
        "template_labels": {}, "fig3_t_partition": [0, 0, 0, 2],
        "axial": True, "axial_angle": math.acos(-1 / 3),
    }


def execute_forced_path(circuit, psi, measurement_outcomes):
    """Interpret Qiskit control-flow blocks, forcing a specified measurement path.

    Used to verify the exported loop's actual control logic, qubit mapping and
    recovery gates. No backend or stochastic sampling is needed.
    """
    state = Statevector(np.r_[np.asarray(psi, dtype=complex), [0, 0]])
    outcomes = iter(measurement_outcomes)
    classical = [0] * circuit.num_clbits
    measurements = 0

    def execute(block, qmap, cmap):
        nonlocal state, measurements
        for instruction in block.data:
            op = instruction.operation
            wires = [qmap[block.find_bit(q).index] for q in instruction.qubits]
            bits = [cmap[block.find_bit(c).index] for c in instruction.clbits]
            if op.name in {"while_loop", "if_else"}:
                bit, value = op.condition
                index = cmap[block.find_bit(bit).index]
                if op.name == "if_else":
                    if classical[index] == value:
                        execute(op.blocks[0], wires, bits)
                else:
                    iterations = 0
                    while classical[index] == value:
                        iterations += 1
                        if iterations > 5:
                            raise AssertionError("RUS loop failed to terminate")
                        execute(op.blocks[0], wires, bits)
            elif op.name == "measure":
                value = next(outcomes)
                measurements += 1
                indices = np.arange(4)
                vector = state.data.copy()
                vector[((indices >> wires[0]) & 1) != value] = 0
                norm = np.linalg.norm(vector)
                if norm < 1e-10:
                    raise AssertionError("Forced a zero-probability measurement")
                state = Statevector(vector / norm)
                classical[bits[0]] = value
            elif op.name == "reset":
                # These protocols only reset a previously measured ancilla.
                p = state.probabilities([wires[0]])
                if max(p) < 1 - 1e-10:
                    raise AssertionError("Unexpected reset of an entangled qubit")
                if p[1] > 0.5:
                    flip = QuantumCircuit(1)
                    flip.x(0)
                    state = state.evolve(flip, wires)
            else:
                state = state.evolve(Operator(op), wires)
    execute(circuit, list(range(2)), list(range(circuit.num_clbits)))
    return state, measurements


class CircuitTests(unittest.TestCase):
    def test_bs12_definition_and_gate_cost(self):
        pools = core_pool(8)
        self.assertEqual([len(pools[t]) for t in range(9)], [1, 1, 1, 1, 1, 2, 4, 8, 16])
        for t, pool in pools.items():
            self.assertEqual(len({matrix_key(s.matrix) for s in pool}), len(pool))
            for word in canonical_words(t):
                self.assertTrue(all(token == "TH" for token in word[:4]))
            for seq in pool:
                self.assertEqual(t_count(seq.circuit), t)
        self.assertIn(("TH",) * 4 + ("SHTH",), list(canonical_words(5)))

    def test_cliffords_and_original_slot_choices(self):
        cliffords = clifford_group()
        self.assertEqual(len({Clifford(c.circuit).tableau.tobytes() for c in cliffords}), 24)
        self.assertEqual(len(allowed_sets()["anc_g1"]), 24)
        self.assertEqual(len(allowed_sets()["anc_g4"]), 24)
        self.assertEqual(len(allowed_sets("fig4")["anc_g1"]), 3)
        self.assertEqual(len(allowed_sets("fig4")["anc_g4"]), 3)

    def test_cached_candidate_matches_native_gate_order(self):
        allowed = allowed_sets()
        cores = core_pool(1)
        pools = [segment_pool(slots, cores[1], allowed) for slots in SLOTS]
        rng = np.random.default_rng(731)
        for _ in range(12):
            items = [pool[rng.integers(len(pool))] for pool in pools]
            template = {key: value for item in items for key, value in item.ops.items()}
            cached = Operator(candidate_circuit(*items)).data
            native = circuit_from_template(template)
            np.testing.assert_allclose(cached, Operator(native).data, atol=1e-12, rtol=0)
            self.assertEqual(t_count(native), 4)
            # Independent tensor expression used in project_fixed_v2.
            a0, a1, a2, d0 = [item.sequence.matrix for item in items]
            cz = np.diag([1, 1, 1, -1])
            reference = np.kron(a2, np.eye(2)) @ cz @ np.kron(a1, d0) @ cz @ np.kron(a0, np.eye(2))
            np.testing.assert_allclose(cached, reference, atol=1e-12, rtol=0)

    def test_known_analytic_unitary_and_recovery_both_outcomes(self):
        for swap in (False, True):
            entry = known_entry(swap)
            result = analyze_w(Operator(circuit_from_template(entry["template_ops"])).data)
            self.assertTrue(result.accepted)
            self.assertAlmostEqual(result.probability, 0.75)
            self.assertEqual(result.success_outcomes, entry["success_outcomes"])
            U = np.diag([1, (-1 + 2j * np.sqrt(2)) / 3])
            self.assertTrue(equivalent(U, result.unitary))
            self.assertTrue(validate_entry(entry)["statevector_success_and_recovery"])

    def test_branch_rejections_and_deterministic_options(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        self.assertEqual(analyze_w(Operator(qc).data).reason, "state_dependent_probability")
        self.assertFalse(analyze_w(np.ones((4, 4))).accepted)
        self.assertFalse(analyze_w(np.eye(4)).accepted)
        permissive = SearchConfig(require_recovery_branch=False, strict_nondeterministic=False)
        for both in (False, True):
            qc = QuantumCircuit(2)
            qc.t(0)
            if both:
                qc.h(1)
            W = Operator(qc).data
            self.assertFalse(analyze_w(W).accepted)
            self.assertTrue(analyze_w(W, permissive).accepted)
        qc.cz(1, 0)
        self.assertEqual(analyze_w(Operator(qc).data, permissive).reason, "different_success_unitaries")

    def test_native_and_exported_loop_recovers_before_retry(self):
        psi = np.array([np.sqrt(0.3), 1j * np.sqrt(0.7)])
        U = np.diag([1, (-1 + 2j * np.sqrt(2)) / 3])
        for swap in (False, True):
            entry = known_entry(swap)
            loop = rus_loop_circuit(entry["template_ops"], entry["success_outcomes"], entry["failure_outcome_ops"])
            success = int(entry["success_outcomes"][0])
            failure = 1 - success
            # Bootstrap, two failures (each corrected), then success.
            state, measured = execute_forced_path(loop, psi, [failure, failure, failure, success])
            actual = state.data[success * 2:success * 2 + 2]
            self.assertTrue(Statevector(actual).equiv(Statevector(U @ psi)))
            self.assertEqual(measured, 4)
            source = qasm3.dumps(loop)
            self.assertIn("while", source)
            self.assertIn("if", source)
            self.assertIn("reset", source)

    def test_identity_uses_only_physical_u_up_to_global_phase(self):
        U = np.diag([1, np.exp(0.321j)])
        key, angle = unitary_identity(U)
        self.assertAlmostEqual(angle, .321)
        self.assertEqual(key, unitary_identity(np.exp(.731j) * U)[0])
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        for V in (H @ U, U @ H, H @ U @ H, U.conj().T,
                  np.diag([1, 1j]) @ U):
            self.assertNotEqual(key, unitary_identity(V)[0])
        with self.assertRaisesRegex(ValueError, "exact"):
            unitary_identity(U, "clifford")


class PaperFormTests(unittest.TestCase):
    # Independent matrix definitions from BS12 Sec. II (not Qiskit op lists).
    H = 1j * np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    T = np.diag([np.exp(-1j * np.pi / 8), np.exp(1j * np.pi / 8)])
    S = T @ T
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.diag([1, -1])

    def test_appendix_e_adjoint_identities_fix_syllable_direction(self):
        th = Operator(circuit_from_ops(core_word_to_ops(("TH",)))).data
        shth = Operator(circuit_from_ops(core_word_to_ops(("SHTH",)))).data
        for U, expected in (
            (th, (self.Z, (self.X - self.Y) / np.sqrt(2), (self.X + self.Y) / np.sqrt(2))),
            (shth, (self.Y, (-self.X + self.Z) / np.sqrt(2), (self.X + self.Z) / np.sqrt(2))),
        ):
            for pauli, transformed in zip((self.X, self.Y, self.Z), expected):
                np.testing.assert_allclose(U @ pauli @ U.conj().T, transformed, atol=1e-12, rtol=0)

    def test_entire_mixed_core_word_is_in_matrix_product_order(self):
        word = ("TH",) * 4 + ("SHTH", "TH")
        expected = np.linalg.matrix_power(self.T @ self.H, 4) @ (self.S @ self.H @ self.T @ self.H) @ (self.T @ self.H)
        actual = Operator(circuit_from_ops(core_word_to_ops(word))).data
        self.assertTrue(equivalent(actual, expected))
        # In execution order this word starts with the rightmost TH, not the prefix.
        self.assertEqual(core_word_to_ops(word)[:6], ("h", "t", "h", "t", "h", "s"))

    def test_appendix_a_group_matches_independent_h_s_enumeration(self):
        # Compare the explicit paper table with an independent generator traversal.
        queue = deque([np.eye(2, dtype=complex)])
        seen = set()
        while queue:
            matrix = queue.popleft()
            key = matrix_key(matrix)
            if key in seen:
                continue
            seen.add(key)
            queue.extend((self.H @ matrix, self.S @ matrix))
        group = clifford_group()
        self.assertEqual([g.label for g in group], [f"G{i}" for i in range(24)])
        self.assertEqual({matrix_key(g.matrix) for g in group}, seen)
        self.assertEqual(len(seen), 24)
        for a in group:
            self.assertIn(matrix_key(a.matrix.conj().T), seen)
            for b in group:
                self.assertIn(matrix_key(a.matrix @ b.matrix), seen)

    def test_appendix_b_all_clifford_t_commutation_relations(self):
        # Each row encodes Gi T = prefix T Gj, copied from Appendix B.
        rows = (
            ("I", 0), ("H", 0), ("I", 12), ("I", 3), ("I", 4), ("I", 5),
            ("H", 3), ("H", 12), ("HSH", 2), ("HSH", 4), ("I", 11),
            ("I", 2), ("I", 10), ("H", 4), ("H", 5), ("H", 11),
            ("HSH", 10), ("HSH", 5), ("HSH", 0), ("HSH", 12),
            ("H", 2), ("H", 10), ("HSH", 3), ("HSH", 11),
        )
        prefixes = {"I": np.eye(2), "H": self.H, "HSH": self.H @ self.S @ self.H}
        group = clifford_group()
        for i, (prefix, j) in enumerate(rows):
            with self.subTest(clifford=i):
                self.assertTrue(equivalent(group[i].matrix @ self.T,
                                           prefixes[prefix] @ self.T @ group[j].matrix))

    def test_named_cliffords_are_matrix_products(self):
        for name, target in (("SH", self.S @ self.H), ("HS", self.H @ self.S),
                             ("SHX", self.S @ self.H @ self.X),
                             ("XHS", self.X @ self.H @ self.S)):
            self.assertTrue(equivalent(Operator(circuit_from_ops(NAMED[name])).data, target))

    def test_position_exceptions_keep_existing_physical_operations(self):
        allowed = allowed_sets("fig4")
        self.assertEqual([s.ops for s in allowed["anc_g0"]],
                         [(), ("x",), ("s", "h"), ("s", "h", "x"),
                          ("h", "s", "h"), ("h", "s", "h", "x")])
        self.assertEqual([s.ops for s in allowed["anc_g1"]],
                         [("h",), ("h", "s"), ("h", "s", "h")])
        self.assertEqual([s.ops for s in allowed["anc_g2"]],
                         [(), ("s", "h"), ("h", "s", "h")])

    def test_segment_is_g_out_times_core_times_g_in(self):
        allowed = allowed_sets()
        core = core_pool(5)[5][1]
        # Explicitly use all 24 Cliffords at both ends, bypassing RUS exceptions.
        allowed["anc_g0"] = [clifford_group()[8]]  # G8 = SH
        allowed["anc_g1"] = [clifford_group()[13]]  # G13 = HS
        segment = segment_pool(SLOTS[0], [core], allowed)[0]
        C = np.linalg.matrix_power(self.T @ self.H, 4) @ self.S @ self.H @ self.T @ self.H
        expected = (self.H @ self.S) @ C @ (self.S @ self.H)
        self.assertTrue(equivalent(segment.sequence.matrix, expected))

    def test_template_validation_rejects_reversed_core_and_bad_labels(self):
        allowed, cores = allowed_sets(), core_pool(5)
        parts = (5, 0, 0, 0)
        items = [segment_pool(slots, [cores[t][-1]], allowed)[0] for slots, t in zip(SLOTS, parts)]
        labels = {k: v for item in items for k, v in item.labels.items()}
        ops = {k: v for item in items for k, v in item.ops.items()}
        validate_template_form(labels, ops, parts)
        wrong_ops = dict(ops, C_a0=list(reversed(ops["C_a0"])))
        with self.assertRaisesRegex(ValueError, "execution order"):
            validate_template_form(labels, wrong_ops, parts)
        with self.assertRaisesRegex(ValueError, "canonical grammar"):
            validate_template_form(dict(labels, C_a0="C[SHTH|TH|TH|TH|TH]"), ops, parts)


class SearchTests(unittest.TestCase):
    def test_input_validation(self):
        for kwargs in ({"max_raw_t": -1}, {"max_examined": -1},
                       {"job": (1, 2, 3, 4)}, {"job": (-1, 0, 0, 1)}, {"tol": float("nan")}, {"dedup_mode": "clifford"}):
            with self.assertRaises(ValueError):
                SearchConfig(**kwargs)
        self.assertEqual(len(list(partitions(3))), math.comb(7, 4) - 1)

    def test_candidate_cap_is_exact_and_marked_incomplete(self):
        for limit in (0, 7):
            _, stats = build_database(SearchConfig(max_raw_t=1, max_examined=limit))
            self.assertEqual(stats["total_examined"], limit)
            self.assertEqual(sum(stats["rejections"].values()) + stats["accepted_candidates"], limit)
            self.assertFalse(stats["search_complete"])

    def test_empty_complete_search(self):
        db, stats = build_database(SearchConfig(max_raw_t=0))
        self.assertEqual(db, {})
        self.assertTrue(stats["search_complete"])
        with tempfile.TemporaryDirectory() as directory:
            summary = export_database(db, stats, directory)
            self.assertEqual(summary["validated_entries"], 0)
            self.assertTrue((Path(directory) / "summary.csv").exists())

    def test_search_finds_and_validates_real_rus_entries(self):
        config = SearchConfig(max_raw_t=2, job=(0, 0, 0, 2), clifford_sets="fig4",
                              max_examined=1000, dedup_mode="exact")
        db, stats = build_database(config)
        self.assertGreater(len(db), 0)
        self.assertEqual(stats["total_examined"], 1000)
        for entry in db.values():
            self.assertAlmostEqual(entry["success_probability"], 0.75)
            self.assertAlmostEqual(entry["expected_t_count"], 8 / 3)
            validate_entry(entry)

    def test_exports_and_qasm2_operator_roundtrip(self):
        entry = known_entry()
        stats = {"config": asdict(SearchConfig()), "search_complete": True}
        with tempfile.TemporaryDirectory() as directory:
            export_database({entry["unitary_key"]: entry}, stats, directory, export_qasm2=True)
            payload = json.loads((Path(directory) / "database.json").read_text())
            self.assertEqual(len(payload["entries"]), 1)
            self.assertEqual(payload["schema_version"], 3)
            self.assertNotIn("canonical", payload["entries"][0])
            files = payload["entries"][0]["export_files"]
            self.assertEqual(set(files), {"json", "w_qasm3", "attempt_qasm3", "rus_loop_qasm3", "attempt_qasm2"})
            qc = qasm2.loads((Path(directory) / files["attempt_qasm2"]).read_text())
            native = circuit_from_template(entry["template_ops"])
            self.assertTrue(Operator(qc.remove_final_measurements(inplace=False)).equiv(Operator(native)))
            self.assertEqual(t_count(qc), 2)
            for relative in files.values():
                self.assertTrue((Path(directory) / relative).is_file())
            with self.assertRaises(FileExistsError):
                export_database({}, stats, directory)


if __name__ == "__main__":
    unittest.main()
