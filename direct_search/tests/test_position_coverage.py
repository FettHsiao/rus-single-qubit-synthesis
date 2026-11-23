"""Independent checks for Fig. 4's state, measurement and CZ absorption claims."""
import unittest

import numpy as np

from direct_search.circuits import allowed_sets, clifford_group, matrix_key, position_coverage
from direct_search.dsa import SearchConfig, prepare_search


class PositionCoverageTests(unittest.TestCase):
    def test_start_states_cover_all_six_stabilizer_states(self):
        group = clifford_group()
        zero = np.array([1, 0])

        def projectors(pool):
            return {tuple(np.round(np.outer(g.matrix @ zero, (g.matrix @ zero).conj()).ravel(), 10))
                    for g in pool}

        expected = projectors(group)
        self.assertEqual(len(expected), 6)
        self.assertEqual(projectors(allowed_sets("verified")["anc_g0"]), expected)
        self.assertEqual(len(projectors(allowed_sets("paper")["anc_g0"])), 4)
        self.assertEqual(len(projectors(allowed_sets("fig4")["anc_g0"])), 5)

    def test_measurement_bases_cover_xyz_up_to_outcome_relabeling(self):
        z = np.diag([1, -1])
        xyz = (np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), z)
        bases = [g.matrix.conj().T @ z @ g.matrix for g in allowed_sets("verified")["anc_g5"]]
        for pauli in xyz:
            self.assertTrue(any(np.allclose(base, pauli) or np.allclose(base, -pauli) for base in bases))

    def test_each_coset_covers_all_24_cliffords(self):
        coverage = position_coverage(allowed_sets("verified"))
        self.assertTrue(coverage["all_slots_cover_clifford_group"])
        self.assertTrue(all(slot["covered_cliffords"] == 24 for slot in coverage["slots"].values()))
        for mode in ("paper", "fig4", "project"):
            self.assertFalse(position_coverage(allowed_sets(mode))["all_slots_cover_clifford_group"])

    def test_cz_moves_every_pair_of_monomials_to_local_cliffords(self):
        z = np.diag([1, -1])
        group = clifford_group()
        monomials = [g.matrix for g in group if
                    np.allclose(g.matrix @ z @ g.matrix.conj().T, z)
                    or np.allclose(g.matrix @ z @ g.matrix.conj().T, -z)]
        self.assertEqual(len(monomials), 8)
        local = {matrix_key(np.kron(a, b)) for a in monomials for b in monomials}
        cz = np.diag([1, 1, 1, -1])
        for a in monomials:
            for b in monomials:
                self.assertIn(matrix_key(cz @ np.kron(a, b) @ cz), local)
        # Moved gates need an unrestricted neighboring ancilla Clifford.
        sets = allowed_sets("verified")
        for slot in ("anc_g1", "anc_g4"):
            self.assertEqual(len({matrix_key(g.matrix) for g in sets[slot]}), 24)

    def test_preflight_records_real_sets_and_coverage(self):
        _, _, stats = prepare_search(SearchConfig(max_raw_t=0, clifford_sets="verified"))
        self.assertEqual(stats["position_policy"], "verified_clifford_cosets")
        self.assertTrue(stats["position_coverage"]["all_slots_cover_clifford_group"])
        self.assertEqual(stats["clifford_position_sets"]["anc_g0"][3], {"label": "HX", "ops": ["x", "h"]})


if __name__ == "__main__":
    unittest.main()
