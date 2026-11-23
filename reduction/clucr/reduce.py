"""The 24 x 24 orbit and lexicographic representative after Eq. (26).

Complex entries are ordered by (Re u00, Im u00, Re u01, Im u01, ...).
Coordinates within tol of the minimum tie, with G0...G23 breaking final ties.
Every orbit member is phase normalized, making the class phase independent.
There is deliberately no section-6.2.1 axial-angle folding here.
"""
import numpy as np
from scipy.spatial import cKDTree

from direct_search.circuits import (
    clifford_group, clifford_matrices, matrix_key, phase_normalize_batch,
)
from reduction.common import checked_unitary, decode, encode, inverse_ops, phase_error


def canonicalize(unitary, tol=1e-10):
    unitary = checked_unitary(unitary, tol)
    # Paper convention: u is the first nonzero entry of the FIRST ROW.
    # Unitarity guarantees that this row has a nonzero pivot.
    u = unitary[0, np.flatnonzero(np.abs(unitary[0]) > tol)[0]]
    unitary = unitary * (u.conjugate() / abs(u))
    group = clifford_matrices()
    # Re-normalize each LUR as well: Cliffords can introduce another global
    # phase. For a unitary the first flattened nonzero entry is in row zero.
    orbit = phase_normalize_batch((group[:, None] @ unitary @ group[None, :]).reshape(576, 2, 2), tol)
    values = np.stack((orbit.real, orbit.imag), axis=-1).reshape(576, 8)
    candidates = np.arange(576)
    for coordinate in range(8):
        column = values[candidates, coordinate]
        candidates = candidates[column <= column.min() + tol]
    index = int(candidates[0])
    left_index, right_index = divmod(index, 24)
    left, right = clifford_group()[left_index], clifford_group()[right_index]
    target = orbit[index]
    # The key is a label only. Grouping below verifies matrices, including
    # neighbors across decimal rounding boundaries, rather than trusting keys.
    return dict(method="left_right_clifford", key=matrix_key(target),
                unitary_matrix=encode(target), left_label=left.label, right_label=right.label,
                left_ops=list(left.ops), right_ops=list(right.ops),
                left_inverse_ops=inverse_ops(left.ops), right_inverse_ops=inverse_ops(right.ops),
                orbit_size=576,
                phase_normalization="conj(u)/sqrt(abs(u)**2), first nonzero entry in first row",
                representative_selection="lexicographic minimum of phase-normalized 24x24 LUR orbit",
                lexicographic_order=["Re u00", "Im u00", "Re u01", "Im u01",
                                     "Re u10", "Im u10", "Re u11", "Im u11"],
                relation="representative = left @ physical_success_unitary @ right (up to global phase)",
                inverse_relation="physical_success_unitary = left.conj().T @ representative @ right.conj().T (up to global phase)")


def entry_rank(entry, cost="effective"):
    field = "effective_t_count" if cost == "effective" else "expected_t_count"
    return (entry[field], entry["raw_t_count"],
            entry.get("candidate_ordinal", 0), entry["source_entry_id"])


def reduce_entries(entries, tol=1e-10, *, cost="effective"):
    """Keep minimum selected expected T per orbit, retaining every membership.

    A spatial index catches matching matrices across rounded-key boundaries.
    Each proposed merge is checked against its winner, avoiding transitive
    tolerance chains. Classification of the physical U is kept separately.
    """
    if cost not in {"direct", "effective"}:
        raise ValueError("cost must be direct or effective")
    if not entries:
        return [], []
    matrices = np.array([decode(e["canonical"]["unitary_matrix"]) for e in entries])
    vectors = np.stack((matrices.real, matrices.imag), axis=-1).reshape(-1, 8)
    tree = cKDTree(vectors)
    assigned = {}
    winners, membership = [], []
    for index in sorted(range(len(entries)), key=lambda i: entry_rank(entries[i], cost)):
        if index in assigned:
            continue
        winner = dict(entries[index])
        class_id = f"clifford_{len(winners):05d}"
        neighbors = tree.query_ball_point(vectors[index], 4 * tol)
        members = [i for i in sorted(neighbors) if i not in assigned
                   and phase_error(matrices[i], matrices[index]) <= tol]
        winner.update(class_id=class_id, member_count=len(members),
                      transformation=winner["canonical"], entry_id=class_id,
                      source_deduplication_method=winner.get("deduplication_method", "exact"),
                      deduplication_method="left_right_clifford")
        for i in members:
            assigned[i] = class_id
            source = entries[i]
            membership.append(dict(source_entry_id=source["source_entry_id"], class_id=class_id,
                                   selected_source_entry_id=winner["source_entry_id"],
                                   source_axial=source["axial"],
                                   expected_t_count=source["expected_t_count"],
                                   effective_t_count=source.get("effective_t_count", source["expected_t_count"]),
                                   raw_t_count=source["raw_t_count"],
                                   candidate_ordinal=source.get("candidate_ordinal", 0),
                                   unitary_matrix=source["unitary_matrix"],
                                   transformation=dict(source["canonical"],
                                       unitary_matrix=winner["canonical"]["unitary_matrix"])))
        winners.append(winner)
    if len(assigned) != len(entries):
        raise RuntimeError("Incomplete Clifford class membership")
    return winners, sorted(membership, key=lambda e: e["source_entry_id"])
