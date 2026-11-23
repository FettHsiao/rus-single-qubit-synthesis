"""Axial membership of a Clifford double coset: L U R is diagonal."""
from __future__ import annotations
import numpy as np
from .circuits import TOL, clifford_group, clifford_matrices


def z_rotation_metadata(U, tol=TOL):
    """Find independent L/R in G0...G23 order; keep physical U unchanged.

    Angles are unfolded. ``clifford_*`` remain aliases for the left witness
    for old JSON readers; they no longer imply a conjugate right witness.
    """
    U = np.asarray(U, dtype=complex)
    if (not np.isfinite(tol) or not 0 < tol < 1e-3 or U.shape != (2, 2)
            or not np.all(np.isfinite(U))
            or not np.allclose(U.conj().T @ U, np.eye(2), rtol=0, atol=tol)):
        raise ValueError("Axial classification requires a finite 2x2 unitary and valid tolerance")
    group = clifford_matrices()
    orbit = (group[:, None] @ U @ group[None, :]).reshape(576, 2, 2)
    errors = np.maximum(np.abs(orbit[:, 0, 1]), np.abs(orbit[:, 1, 0]))
    hits = np.flatnonzero(errors <= tol)
    result = dict(method="left_right_clifford", axial=bool(len(hits)),
                  relation="diagonal_unitary = left @ physical_success_unitary @ right",
                  clifford_label=None, clifford_ops=None,
                  left_label=None, right_label=None, left_ops=None, right_ops=None,
                  diagonal_unitary_matrix=None, angle=None,
                  minimum_offdiagonal_error=float(errors.min()))
    if len(hits):
        index = int(hits[0])
        li, ri = divmod(index, 24)
        left, right = clifford_group()[li], clifford_group()[ri]
        V = orbit[index]
        result.update(clifford_label=left.label, clifford_ops=list(left.ops),
                      left_label=left.label, right_label=right.label,
                      left_ops=list(left.ops), right_ops=list(right.ops),
                      diagonal_unitary_matrix=np.stack((V.real, V.imag), axis=-1).tolist(),
                      angle=float(np.angle(V[1, 1] / V[0, 0])))
    return result
