"""Independent group geometry, branch physics, lookup completeness and CLI data."""
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from qiskit import qasm2, qasm3, qpy
from qiskit.quantum_info import Operator, Statevector, random_unitary, partial_trace
from direct_search.circuits import circuit_from_ops, clifford_group, clifford_matrices
from direct_search.amplification import amplification_metadata
from direct_search.tests.test_direct_search import known_entry, execute_forced_path
from direct_search.export import export_database as export_direct
from reduction.clucr import canonicalize
from reduction.common import decode, encode, phase_error
from composition.geometry import canonicalize_g2, distance, fold_transform, rotation
from composition.build import Config, Pool, build_database, fill_gaps, gap_metrics, node, product_entry, seed_entry
from composition.export import export_database, recipe_circuit
from composition.lookup import Lookup, hybrid_estimate, lookup_database


def source(swap=False):
    entry = known_entry(swap)
    entry.update(entry_id="source1" if swap else "source0")
    entry.update(amplification_metadata(entry["raw_t_count"], entry["success_probability"], 0))
    return entry


def fake_payload(entries, mode):
    return dict(entries=entries, stats=dict(config=dict(mode=mode)))


class GeometryTests(unittest.TestCase):
    def test_g2_equals_full_and_is_orbit_invariant(self):
        rng = np.random.default_rng(84)
        for seed in range(12):
            U = random_unitary(2, seed=seed).data
            expected = decode(canonicalize(U)["unitary_matrix"])
            for _ in range(6):
                l, r = rng.integers(0, 24, 2)
                V = clifford_group()[l].matrix @ U @ clifford_group()[r].matrix
                witness = canonicalize_g2(np.exp(.47j)*V)
                actual = decode(witness["unitary_matrix"])
                self.assertLess(phase_error(actual, expected), 1e-10)
                left = Operator(circuit_from_ops(witness["left_ops"])).data
                right = Operator(circuit_from_ops(witness["right_ops"])).data
                self.assertLess(phase_error(left @ V @ right, actual), 1e-10)

    def test_axial_boundary_folding_constructive(self):
        for theta in list(np.linspace(-20, 20, 50)) + [0, math.pi/4, math.pi/2, math.pi, 1e-11]:
            a, l, r = fold_transform(theta)
            self.assertTrue(0 <= a <= math.pi/4 + 1e-14)
            actual = Operator(circuit_from_ops(l)).data @ rotation(theta) @ Operator(circuit_from_ops(r)).data
            self.assertLess(phase_error(actual, rotation(a)), 1e-12)
        self.assertAlmostEqual(distance(rotation(0), rotation(.2)), math.sqrt(1-math.cos(.1)))
        self.assertLess(distance(rotation(0), rotation(1e-12)), 1e-12)

    def test_verified_pool_rounding_boundaries_and_cost(self):
        pool = Pool(1e-10)
        for angle, cost in ((.30000000004, 5), (.30000000006, 3), (.30001, 1)):
            pool.insert(node(rotation(angle), [], 1, 2, .5, cost, angle))
        self.assertEqual(len(pool.entries), 2)
        self.assertEqual(sorted(e["expected_t_count"] for e in pool.entries), [1, 3])


class PhysicsTests(unittest.TestCase):
    def test_sum_difference_and_general_middle_clifford_retry_paths(self):
        components = {e["entry_id"]: e for e in (source(), source(True))}
        psi = np.array([1, 1j])/math.sqrt(2)
        for mode in ("axial", "all"):
            config = Config(mode=mode)
            a, b = [seed_entry(e, config) for e in components.values()]
            for middle in ((1, -1) if mode == "axial" else (0, 5, 13, 23)):
                e = product_entry(a, b, middle, config)
                qc = qasm3.loads(qasm3.dumps(recipe_circuit(e, components)))
                # Product A C B executes B first. Both RUS protocols bootstrap,
                # fail twice with recovery, and then succeed independently.
                state, _ = execute_forced_path(qc, psi, [0,0,0,1, 1,1,1,0])
                self.assertTrue(Statevector(state.data[:2]).equiv(Statevector(decode(e["unitary_matrix"]) @ psi), atol=1e-10))
                self.assertAlmostEqual(e["expected_t_count"], 16/3)
                self.assertEqual(e["class_depth"], 2)
                self.assertNotAlmostEqual(e["raw_t_count"]/e["success_probability"], e["expected_t_count"])

    def test_cost_depth_caps(self):
        config = Config(max_class=1)
        a = seed_entry(source(), config)
        self.assertIsNone(product_entry(a, a, 1, config))
        self.assertIsNone(product_entry(a, a, 1, Config(max_expected_t=5)))


class LookupTests(unittest.TestCase):
    def test_axial_chooses_cheapest_in_ball_not_nearest(self):
        entries = []
        for i, (angle, cost) in enumerate(((.29, 1), (.3, 5), (.5, .1))):
            e = node(rotation(angle), [], 1, 2, .5, cost, angle)
            e["entry_id"] = str(i)
            entries.append(e)
        index = Lookup(fake_payload(entries, "axial"))
        result = index.query(rotation(.3+3*math.pi/2), .01)
        self.assertEqual(result["matched_entry_id"], "0")
        self.assertLessEqual(result["distance"], .01)
        self.assertIsNone(index.query(rotation(.7), 1e-5))

    def test_kdtree_matches_full_orbit_bruteforce_including_seams(self):
        entries = []
        for i in range(12):
            U = decode(canonicalize_g2(random_unitary(2, seed=i).data)["unitary_matrix"])
            e = node(U, [], 1, 2, .5, 12-i)
            e["entry_id"] = str(i)
            entries.append(e)
        index = Lookup(fake_payload(entries, "all"))
        group = clifford_matrices()
        for seed in range(12, 20):
            target = random_unitary(2, seed=seed).data
            orbit = (group[:,None] @ target @ group[None,:]).reshape(-1,2,2)
            epsilon = .06
            eligible = [e for e in entries if min(distance(v, decode(e["unitary_matrix"])) for v in orbit) <= epsilon]
            result = index.query(target, epsilon)
            if eligible:
                winner = min(eligible, key=lambda e: e["expected_t_count"])
                self.assertEqual(result["matched_entry_id"], winner["entry_id"])
                self.assertLessEqual(result["distance"], epsilon+1e-14)
            else:
                self.assertIsNone(result)
        # Exact query with a nontrivial left/right frame and a global phase.
        target = np.exp(.413j)*group[8] @ decode(entries[5]["unitary_matrix"]) @ group[16]
        result = index.query(target, 1e-12)
        self.assertEqual(result["matched_entry_id"], "5")
        self.assertLess(result["distance"], 1e-12)

    def test_gap_fill_obeys_caps_and_reduces_covering_radius(self):
        config = Config(max_class=4, max_pool=0, gap_fill_iterations=2)
        pool = Pool(config.tol)
        pool.insert(node(rotation(0), [], 0, 0, 1, 0, 0))
        pool.insert(node(rotation(.22), [], 1, 2, .75, 8/3, .22))
        before = gap_metrics(pool.entries)["covering_radius"]
        added = fill_gaps(pool, config)
        self.assertGreater(len(added), 0)
        self.assertLess(gap_metrics(pool.entries)["covering_radius"], before)
        self.assertTrue(all(e["class_depth"] <= 4 and e["expected_t_count"] <= 30 for e in added))


class DatasetTests(unittest.TestCase):
    def test_serial_parallel_export_and_lookup_are_self_contained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = source()
            stats = dict(config=dict(tol=1e-10, workers=1), search_complete=True)
            export_direct({entry["unitary_key"]: entry}, stats, root / "direct", export_qasm2=True)
            results = []
            for mode in ("axial", "all"):
                for workers in (1, 2):
                    output = root / f"{mode}{workers}"
                    config = Config(mode=mode, max_class=3, max_pool=30, workers=workers, chunk_size=1)
                    build_database(root / "direct", output, config)
                    summary = export_database(output, workers, 1)
                    self.assertTrue(summary["export_ok"])
                    payload = json.loads((output / "database.json").read_text())
                    results.append(payload["entries"])
                    for e in payload["entries"]:
                        qasm3.load(str(output/e["export_files"]["target_rus_loop_qasm3"]))
                        with (output/e["export_files"]["qpy"]).open("rb") as f:
                            circuits = qpy.load(f)
                        self.assertLess(phase_error(Operator(circuits[1]).data, decode(e["unitary_matrix"])), 1e-10)
                    for component in payload["components"].values():
                        qasm2.load(str(output/component["export_files"]["attempt_qasm2"]))
                    chosen = next(e for e in payload["entries"] if e["class_depth"] == 2)
                    result = lookup_database(output, root / f"lookup_{mode}{workers}", decode(chosen["unitary_matrix"]), 1e-9, workers)
                    self.assertIsNotNone(result)
                    with self.assertRaises(FileExistsError):
                        build_database(root / "direct", output, config)
                self.assertEqual(results[-1], results[-2])


class GearboxTests(unittest.TestCase):
    def test_independent_kraus_and_cost_recurrence(self):
        from composition.gearbox import gearbox_angle, gearbox_cost, gearbox_probability
        for a in (.01, .2, .6):
            qc = __import__("qiskit").QuantumCircuit(2)
            qc.rx(a, 1)
            qc.cz(1, 0)
            qc.sdg(1)
            qc.rx(-a, 1)
            W = Operator(qc).data
            p = gearbox_probability(a)
            np.testing.assert_allclose(W[:2,:2].conj().T @ W[:2,:2], p*np.eye(2), atol=1e-12)
            self.assertLess(phase_error(W[:2,:2]/math.sqrt(p), rotation(gearbox_angle(a))), 1e-12)
            correction = np.diag([1, 1j])
            self.assertLess(phase_error(correction @ W[2:,:2]/math.sqrt(1-p), np.eye(2)), 1e-10)
            angle, cost, probabilities = gearbox_cost(a, 4., 2)
            self.assertAlmostEqual(cost, 16./np.prod(probabilities))
            self.assertAlmostEqual(angle, gearbox_angle(gearbox_angle(a)))

    def test_nested_native_circuit_entangled_subroutines_and_readback(self):
        from composition.gearbox import gearbox_circuit, gearbox_cost
        entry = source()
        components = {entry["entry_id"]: entry}
        seed = seed_entry(entry, Config())
        rng = np.random.default_rng(112)
        psi = np.array([1, 1j])/math.sqrt(2)
        for depth in (1, 2):
            circuit = gearbox_circuit(seed, components, depth)
            circuit = qasm3.loads(qasm3.dumps(circuit))
            angle, _, _ = gearbox_cost(seed["axial_angle"], seed["expected_t_count"], depth)
            wanted = Statevector(rotation(angle) @ psi).to_operator().data
            for shot in range(6):
                state = Statevector(np.r_[psi, np.zeros(2**circuit.num_qubits-2)])
                classical = [0]*circuit.num_clbits
                def execute(block, qmap, cmap):
                    nonlocal state
                    for instruction in block.data:
                        op = instruction.operation
                        wires = [qmap[block.find_bit(q).index] for q in instruction.qubits]
                        bits = [cmap[block.find_bit(c).index] for c in instruction.clbits]
                        if op.name in {"while_loop", "if_else"}:
                            bit, value = op.condition
                            ci = cmap[block.find_bit(bit).index]
                            if op.name == "if_else":
                                if classical[ci] == value:
                                    execute(op.blocks[0], wires, bits)
                            else:
                                iterations = 0
                                while classical[ci] == value:
                                    iterations += 1
                                    self.assertLess(iterations, 100)
                                    execute(op.blocks[0], wires, bits)
                        elif op.name == "measure":
                            probabilities = state.probabilities([wires[0]])
                            value = int(rng.random() >= probabilities[0])
                            vector = state.data.copy()
                            vector[((np.arange(len(vector)) >> wires[0]) & 1) != value] = 0
                            state = Statevector(vector/np.linalg.norm(vector))
                            classical[bits[0]] = value
                        elif op.name == "reset":
                            probabilities = state.probabilities([wires[0]])
                            self.assertGreater(max(probabilities), 1-1e-10)
                            if probabilities[1] > .5:
                                state = state.evolve(Operator(np.array([[0,1],[1,0]])), wires)
                        else:
                            state = state.evolve(Operator(op), wires)
                execute(circuit, list(range(circuit.num_qubits)), list(range(circuit.num_clbits)))
                actual = partial_trace(state, list(range(1,circuit.num_qubits))).data
                np.testing.assert_allclose(actual, wanted, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
