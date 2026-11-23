"""Clifford double-coset geometry, sections 6.2.1 and 6.2.3."""
from functools import lru_cache
import math
import numpy as np
from qiskit.quantum_info import Operator
from direct_search.circuits import circuit_from_ops, clifford_group, clifford_matrices, matrix_key, phase_normalize_batch
from reduction.common import checked_unitary, encode, phase_error
from reduction.clucr import canonicalize

PERIOD = math.pi / 2


def rotation(angle):
    return np.diag([1., np.exp(1j * angle)])


def distance(U, V):
    """Eq. 27, evaluated stably as a phase-aligned Frobenius distance."""
    overlap = np.vdot(V, U)
    phase = overlap / abs(overlap) if abs(overlap) else 1.
    return float(np.linalg.norm(U - phase * V) / 2)


def fold_angle(angle):
    """Return a in [0,pi/4], sign s, integer n: a=s*angle+n*pi/2."""
    n = -math.floor(angle / PERIOD + .5)
    value = angle + n * PERIOD
    sign = -1 if value < 0 else 1
    return abs(value), sign, sign * n


def fold_transform(angle):
    value, sign, n = fold_angle(angle)
    # Rz(value) = S^n X^b Rz(angle) X^b, up to phase.
    return value, (["x"] if sign < 0 else []) + ["s"] * (n % 4), (["x"] if sign < 0 else [])


def clifford_ops(matrix):
    group = clifford_matrices()
    overlaps = np.einsum("aij,ij->a", group.conj(), matrix)
    phases = np.divide(overlaps, np.abs(overlaps), out=np.ones_like(overlaps), where=np.abs(overlaps)>0)
    errors = np.max(np.abs(matrix-group*phases[:, None, None]), axis=(1,2))
    index = int(np.argmin(errors))
    if errors[index] > 1e-8:
        raise ValueError("Expected a single-qubit Clifford")
    return list(clifford_group()[index].ops)


@lru_cache(None)
def g2_data():
    # Eq. 35 words are matrix products, reversed for Qiskit time order.
    words = ((), ("h",), ("x",), ("h", "x"), ("s", "h"),
             ("s", "h", "x"), ("h", "s", "h"), ("h", "s", "h", "x"))
    matrices = np.array([Operator(circuit_from_ops(w)).data for w in words])
    # G = G1 G2, hence also G = G2† G1. Validate rather than assume coverage.
    phases = np.array([np.diag([1, 1j ** k]) for k in range(4)])
    cover = (phases[:, None] @ matrices[None, :]).reshape(-1, 2, 2)
    for c in clifford_group():
        if min(phase_error(c.matrix, x) for x in cover) > 1e-10:
            raise ValueError("Eq. 35 G1/G2 does not cover the Clifford group")
    return matrices, phases


def canonicalize_g2(U, tol=1e-10):
    """64 G2 pairs; eliminate outer G1 phases analytically (Eqs. 35–37).

    Pick the same matrix lexicographic minimum as the 576-pair reference.
    Four scalar phase choices per nonzero entry avoid constructing 576 matrices.
    Right G2 inverses follow G=G2† G1 and handle matrix-order conventions.
    """
    U = checked_unitary(U, tol)
    g2, phases = g2_data()
    right = g2.conj().swapaxes(-1, -2)
    orbit = phase_normalize_batch((g2[:, None] @ U @ right[None, :]).reshape(64, 2, 2), tol)
    # Singular Euler charts (diagonal or anti-diagonal) are small and uncommon;
    # the full reference gives unambiguous boundary/tolerance behavior.
    if np.any(np.abs(orbit[:, 0, 0]) <= tol) or np.any(np.abs(orbit[:, 0, 1]) <= tol):
        return canonicalize(U, tol)
    choices = 1j ** np.arange(4)
    def best_phase(values):
        values = values[:, None] * choices
        valid = values.real <= values.real.min(axis=1)[:, None] + tol
        imag = np.where(valid, values.imag, np.inf)
        valid &= imag <= imag.min(axis=1)[:, None] + tol
        return np.argmax(valid, axis=1)
    rp = best_phase(orbit[:, 0, 1])
    lp = best_phase(orbit[:, 1, 0])
    orbit = phases[lp] @ orbit @ phases[rp]
    values = np.stack((orbit.real, orbit.imag), axis=-1).reshape(64, 8)
    selected = np.arange(64)
    for coordinate in range(8):
        column = values[selected, coordinate]
        selected = selected[column <= column.min() + tol]
    index = int(selected[0])
    li, ri = divmod(index, 8)
    left_ops = clifford_ops(phases[lp[index]] @ g2[li])
    right_ops = clifford_ops(right[ri] @ phases[rp[index]])
    target = orbit[index]
    return dict(method="left_right_clifford_g2", key=matrix_key(target),
                unitary_matrix=encode(target), left_ops=left_ops, right_ops=right_ops,
                orbit_size=64, relation="representative = left @ physical_success_unitary @ right (up to global phase)")


def quaternion(U):
    """SU(2) unit quaternion; ± represent the same projective unitary."""
    V = np.asarray(U) / np.sqrt(np.linalg.det(U))
    return np.array([V[0, 0].real, V[0, 0].imag, V[0, 1].real, V[0, 1].imag])
