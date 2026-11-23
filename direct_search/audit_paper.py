"""Read-only audit of an existing direct-search database against paper conventions.

Run from the workspace root with the project's Python environment:
  python -m direct_search.audit_paper INPUT_DIR --output-dir NEW_DIR

Clifford orbit counts are diagnostics, not a claim that section 5 used a
particular equivalence relation. No database entries or search rules are changed.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from functools import lru_cache
from itertools import permutations, product
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .circuits import allowed_sets

PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
IDENTITY = np.eye(2, dtype=complex)
PRIMITIVES = {
    "h": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
    "s": np.diag([1, 1j]), "sdg": np.diag([1, -1j]),
    "t": np.diag([1, np.exp(1j * np.pi / 4)]),
    "tdg": np.diag([1, np.exp(-1j * np.pi / 4)]),
    "x": PAULI[0], "y": PAULI[1], "z": PAULI[2], "i": IDENTITY,
}


def bloch_rotations(unitaries):
    """R_ab = Tr(P_a U P_b U†)/2, independent of global phase."""
    return np.einsum("aij,njk,bkl,nil->nab", PAULI, unitaries, PAULI,
                     unitaries.conj()).real / 2


def clifford_rotations():
    """Independently generate the 24 orientation-preserving signed permutations."""
    result = []
    for perm in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            rotation = np.eye(3)[list(perm)] * np.array(signs)[:, None]
            if np.linalg.det(rotation) > 0:
                result.append(rotation)
    return np.array(result)


def orbit_groups(rotations, mode, tol):
    """Group only observed entries; check actual distances, not rounded hashes."""
    if mode not in {"exact", "conjugate", "double"}:
        raise ValueError(mode)
    group = clifford_rotations()
    tree = cKDTree(rotations.reshape(-1, 9))
    seen, groups = set(), []
    for index, rotation in enumerate(rotations):
        if index in seen:
            continue
        if mode == "exact":
            orbit = rotation[None]
        elif mode == "conjugate":
            orbit = group @ rotation @ group.transpose(0, 2, 1)
        else:
            orbit = (group[:, None] @ rotation @ group[None]).reshape(-1, 3, 3)
        hits = tree.query_ball_point(orbit.reshape(-1, 9), tol, p=np.inf)
        members = sorted({j for row in hits for j in row})
        if index not in members or seen.intersection(members):
            raise ValueError("Ambiguous tolerance clustering; inspect near-boundary matrices")
        seen.update(members)
        groups.append(members)
    return groups


@lru_cache(maxsize=None)
def operation_matrix(ops):
    result = IDENTITY.copy()
    for gate in ops:
        result = PRIMITIVES[gate] @ result
    return result


def phase_distance(actual, target):
    overlap = np.vdot(target, actual)
    phase = overlap / abs(overlap) if abs(overlap) else 1
    return float(np.max(np.abs(actual - phase * target)))


def validate_physics(entries, unitaries, tol):
    """Reconstruct W from primitive matrices without the search acceptance code."""
    maximum = Counter()
    failures = []
    cz = np.diag([1, 1, 1, -1])
    for entry, target in zip(entries, unitaries):
        ops = entry["template_ops"]
        segments = []
        for slots in (("anc_g0", "C_a0", "anc_g1"), ("anc_g2", "C_a1", "anc_g3"),
                      ("anc_g4", "C_a2", "anc_g5"), ("dat_g0", "C_d0", "dat_g1")):
            segments.append(operation_matrix(tuple(gate for slot in slots for gate in ops[slot])))
        a0, a1, a2, data = segments
        w = np.kron(a2, IDENTITY) @ cz @ np.kron(a1, data) @ cz @ np.kron(a0, IDENTITY)
        errors = {"w_unitarity": float(np.max(abs(w.conj().T @ w - np.eye(4))))}
        raw_t = sum(gate in {"t", "tdg"} for seq in ops.values() for gate in seq)
        errors["raw_t_count"] = abs(raw_t - entry["raw_t_count"])
        success_p = 0.0
        for outcome in ("0", "1"):
            block = w[2 * int(outcome):2 * int(outcome) + 2, :2]
            gram = block.conj().T @ block
            probability = float(np.trace(gram).real / 2)
            errors[f"gram_{outcome}"] = float(np.max(abs(gram - probability * IDENTITY)))
            errors[f"stored_probability_{outcome}"] = abs(probability - entry["branch_probabilities"][outcome])
            if probability <= tol:
                errors["zero_branch"] = 1.0
                continue
            unitary = block / np.sqrt(probability)
            if outcome in entry["success_outcomes"]:
                success_p += probability
                errors["success_unitary"] = phase_distance(unitary, target)
            else:
                recovery = operation_matrix(tuple(entry["failure_outcome_ops"][outcome]))
                errors["failure_recovery"] = phase_distance(recovery.conj().T @ unitary, IDENTITY)
                r = bloch_rotations(recovery[None])[0]
                errors["recovery_clifford"] = float(np.max(abs(r - np.round(r))))
        errors["success_probability"] = abs(success_p - entry["success_probability"])
        expected = raw_t / success_p if success_p else float("inf")
        errors["expected_t_relative"] = abs(expected - entry["expected_t_count"]) / max(1, expected)
        for name, value in errors.items():
            maximum[name] = max(maximum[name], value)
        if any(value > tol for value in errors.values()):
            failures.append({"entry_id": entry["entry_id"], "errors": errors})
    return {"checked": len(entries), "failed": len(failures),
            "max_errors": dict(maximum), "failures": failures}


def position_audit(mode):
    result = {}
    for slot, sequences in allowed_sets(mode).items():
        measurement = slot in {"anc_g1", "anc_g3", "anc_g5", "dat_g1"}
        rows = []
        for sequence in sequences:
            matrix = sequence.matrix
            observable = (matrix.conj().T @ PAULI[2] @ matrix if measurement
                          else matrix @ PAULI[2] @ matrix.conj().T)
            vector = np.rint(np.einsum("aij,ji->a", PAULI, observable).real / 2).astype(int)
            rows.append({"label": sequence.label, "bloch_vector": vector.tolist()})
        result[slot] = {
            "relation": "g† Z g" if measurement else "g Z g†",
            "rows": rows,
            "distinct_signed_vectors": len({tuple(row["bloch_vector"]) for row in rows}),
            "distinct_unsigned_axes": len({tuple(abs(v) for v in row["bloch_vector"]) for row in rows}),
        }
    return result


def audit(input_dir, output_dir, tolerances=(1e-10, 1e-9)):
    # Refuse to overwrite either search data or an earlier audit.
    output_dir.mkdir(parents=True, exist_ok=False)
    document = json.loads((input_dir / "database.json").read_text())
    entries = document["entries"]
    values = np.asarray([entry["unitary_matrix"] for entry in entries]).reshape(len(entries), 2, 2, 2)
    unitaries = values[..., 0] + 1j * values[..., 1]
    rotations = bloch_rotations(unitaries)
    report = {
        "input": str(input_dir.resolve()), "entries": len(entries),
        "axial_classification": "left_right_clifford (legacy strict_axial field names retained)",
        "scope": "Diagnostics only; Clifford orbit counts do not assert section-5 database semantics.",
        "physics": validate_physics(entries, unitaries, min(tolerances)),
        "position_sets": {mode: position_audit(mode) for mode in ("paper", "fig4")},
        "orbit_counts": {},
    }
    for tol in tolerances:
        from .axial import z_rotation_metadata
        axial = np.array([z_rotation_metadata(U, tol)["axial"] for U in unitaries])
        convertible = np.any(abs(abs(rotations) - 1) <= tol, axis=(1, 2))
        counts = {}
        for mode in ("exact", "conjugate", "double"):
            groups = orbit_groups(rotations, mode, tol)
            counts[mode] = {
                "classes": len(groups),
                "classes_with_observed_strict_axial_member": sum(bool(axial[group].any()) for group in groups),
                "classes_convertible_to_axial_by_two_cliffords": sum(bool(convertible[group].any()) for group in groups),
                "multiplicity_histogram": dict(sorted(Counter(map(len, groups)).items())),
            }
            if mode == "double" and tol == tolerances[0]:
                with (output_dir / "clifford_orbits.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["class_id", "members", "entry_id", "best_observed_entry_id",
                                     "strict_axial", "convertible_to_axial", "raw_t", "probability", "expected_t"])
                    for class_id, group in enumerate(groups):
                        best = min(group, key=lambda i: (entries[i]["expected_t_count"], entries[i]["raw_t_count"]))
                        for i in group:
                            entry = entries[i]
                            writer.writerow([class_id, len(group), entry["entry_id"], entries[best]["entry_id"],
                                             bool(axial[i]), bool(convertible[i]), entry["raw_t_count"],
                                             entry["success_probability"], entry["expected_t_count"]])
        counts["strict_axial_entries"] = int(axial.sum())
        counts["stored_axial_mismatches"] = int(np.sum(axial != [entry["axial"] for entry in entries]))
        counts["convertible_to_axial_entries"] = int(convertible.sum())
        report["orbit_counts"][str(tol)] = counts
        print(f"tol={tol}: " + ", ".join(f"{mode}={counts[mode]['classes']}"
              for mode in ("exact", "conjugate", "double")), flush=True)
    report["raw_t_2"] = {
        "entries": sum(entry["raw_t_count"] == 2 for entry in entries),
        "strict_axial": sum(entry["raw_t_count"] == 2 and entry["axial"] for entry in entries),
    }
    (output_dir / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Physics: {report['physics']['checked']} checked; {report['physics']['failed']} failed", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    audit(arguments.input_dir, arguments.output_dir)
