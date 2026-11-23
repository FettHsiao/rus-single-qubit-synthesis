"""Shared numerical conventions and native Qiskit wrapper construction."""
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from direct_search.circuits import circuit_from_ops, rus_loop_circuit


def encode(matrix):
    return np.stack((np.asarray(matrix).real, np.asarray(matrix).imag), axis=-1).tolist()


def decode(values):
    array = np.asarray(values, dtype=float)
    if array.shape != (2, 2, 2):
        raise ValueError("Expected a 2x2 matrix encoded as [real, imaginary] pairs")
    return array[..., 0] + 1j * array[..., 1]


def checked_unitary(matrix, tol=1e-10):
    if not math.isfinite(tol) or not 0 < tol < 1e-3:
        raise ValueError("tol must be finite and in (0, 1e-3)")
    matrix = np.asarray(matrix, dtype=complex)
    if (matrix.shape != (2, 2) or not np.isfinite(matrix).all()
            or not np.allclose(matrix.conj().T @ matrix, np.eye(2), atol=tol, rtol=0)):
        raise ValueError("Expected a finite 2x2 unitary")
    return matrix


def inverse_ops(ops):
    inverses = dict(h="h", s="sdg", sdg="s", x="x", y="y", z="z", i="i")
    return [inverses[op] for op in reversed(ops)]


def phase_error(left, right):
    """Entrywise error after optimal Frobenius global-phase alignment."""
    overlap = np.vdot(right, left)
    phase = overlap / abs(overlap) if abs(overlap) else 1.0
    return float(np.max(np.abs(left - phase * right)))


def wrapper_circuit(entry, *, restore=False, amplified=False):
    """Implement V=LUR, or restore the selected source U=L†VR†.

    Matrix products act right to left. R is executed once before all retries,
    L once after success; the physical attempt and failure recovery stay paired.
    """
    left = circuit_from_ops(entry["transformation"]["left_ops"])
    right = circuit_from_ops(entry["transformation"]["right_ops"])
    from direct_search.amplification import amplified_entry_circuit
    attempt = amplified_entry_circuit(entry) if amplified else None
    physical = rus_loop_circuit(entry["template_ops"], entry["success_outcomes"],
                                entry["failure_outcome_ops"], attempt=attempt)
    target = QuantumCircuit(2, 1, name="target_rus_loop")
    target.compose(right, [0], inplace=True)
    target.compose(physical, inplace=True)
    target.compose(left, [0], inplace=True)
    if not restore:
        return target
    restored = QuantumCircuit(2, 1, name="restored_source_rus_loop")
    restored.compose(right.inverse(), [0], inplace=True)
    restored.compose(target, inplace=True)
    restored.compose(left.inverse(), [0], inplace=True)
    return restored


def verify_transformation(entry, tol):
    transform = entry["transformation"]
    left = Operator(circuit_from_ops(transform["left_ops"])).data
    right = Operator(circuit_from_ops(transform["right_ops"])).data
    unitary = checked_unitary(decode(entry["unitary_matrix"]), tol)
    target = checked_unitary(decode(transform["unitary_matrix"]), tol)
    error = phase_error(left @ unitary @ right, target)
    inverse_error = phase_error(left.conj().T @ target @ right.conj().T, unitary)
    if max(error, inverse_error) > tol:
        raise ValueError("Clifford wrappers do not implement the stored target and inverse")
    return dict(forward_error=error, inverse_error=inverse_error)
