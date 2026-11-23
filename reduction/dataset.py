"""Convenience readers; all stored circuits also load through Qiskit directly."""
import json
from pathlib import Path

from qiskit import QuantumCircuit, qasm2, qasm3, qpy

from direct_search.circuits import circuit_from_ops


def load_database(directory):
    return json.loads((Path(directory) / "database.json").read_text(encoding="utf-8"))


def load_circuit(directory, entry, kind="target_rus_loop_qasm3"):
    """Load a QASM file, or a named circuit in the QPY bundle."""
    directory = Path(directory)
    if kind in entry["qpy_indices"]:
        with (directory / entry["export_files"]["qpy"]).open("rb") as handle:
            return qpy.load(handle)[entry["qpy_indices"][kind]]
    path = str(directory / entry["export_files"][kind])
    if kind.endswith("_qasm2"):
        return qasm2.load(path)
    if kind.endswith("_qasm3"):
        return qasm3.load(path)
    raise ValueError(f"Unknown circuit kind: {kind}")


def restore_member(directory, winner, member):
    """Implement any original class member with the winning RUS circuit's cost.

    A member's V=L U R implies U=L† V R†, with R† executed first.
    The member and winner can have different physical success unitaries.
    Use the amplified representative loop when the winner selected j > 0.
    """
    if member["class_id"] != winner["class_id"]:
        raise ValueError("Membership and selected winner belong to different classes")
    kind = ("amplified_target_rus_loop_qasm3" if winner.get("amplification_rounds", 0)
            else "target_rus_loop_qasm3")
    target = load_circuit(directory, winner, kind=kind)
    restored = QuantumCircuit(2, 1, name="restored_member_rus_loop")
    transformation = member["transformation"]
    restored.compose(circuit_from_ops(transformation["right_ops"]).inverse(), [0], inplace=True)
    restored.compose(target, inplace=True)
    restored.compose(circuit_from_ops(transformation["left_ops"]).inverse(), [0], inplace=True)
    return restored
