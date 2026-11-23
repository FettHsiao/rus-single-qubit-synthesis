"""Minimum-cost lookup with exact Fowler-distance filtering (Eq. 27)."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from qiskit.quantum_info import Operator
from direct_search.axial import z_rotation_metadata
from direct_search.circuits import circuit_from_ops, clifford_group, clifford_matrices, matrix_key
from reduction.common import checked_unitary, decode, encode, inverse_ops
from reduction.pipeline import write_json
from .build import rank, wrap
from .export import export_database
from .geometry import distance, fold_transform, quaternion, rotation


class Lookup:
    def __init__(self, payload):
        self.payload = payload
        self.entries = payload["entries"]
        self.mode = payload["stats"]["config"]["mode"]
        self.matrices = np.array([decode(e["unitary_matrix"]) for e in self.entries])
        if self.mode == "axial":
            self.order = np.argsort([e["axial_angle"] for e in self.entries])
            self.angles = np.array([self.entries[i]["axial_angle"] for i in self.order])
        else:
            self.quaternions = np.array([quaternion(U) for U in self.matrices])
            self.tree = cKDTree(self.quaternions)

    def query(self, target, epsilon):
        target = checked_unitary(target)
        if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be finite and in [0,1]")
        matches = []
        if self.mode == "axial":
            witness = z_rotation_metadata(target)
            if not witness["axial"]:
                raise ValueError("Axial database queries require a target in an axial double coset")
            angle, left, right = fold_transform(witness["angle"])
            # Lfold Lw target Rw Rfold is the folded target.
            left = witness["left_ops"] + left
            right = right + witness["right_ops"]
            radius = 4*math.asin(min(1., epsilon/math.sqrt(2)))
            start, end = np.searchsorted(self.angles, [angle-radius-1e-14, angle+radius+1e-14], side="left")
            end = int(np.searchsorted(self.angles, angle+radius+1e-14, side="right"))
            for index in self.order[int(start):end]:
                error = distance(rotation(angle), self.matrices[index])
                if error <= epsilon + 1e-14:
                    matches.append((int(index), error, left, right))
        else:
            # Canonical coordinates have seams. Query all target Clifford images,
            # and both quaternion signs, to avoid losing valid neighbors there.
            group = clifford_matrices()
            orbit = (group[:, None] @ target @ group[None, :]).reshape(576, 2, 2)
            vectors = np.array([quaternion(U) for U in orbit])
            radius = math.sqrt(2)*epsilon + 2e-14
            hits = self.tree.query_ball_point(np.r_[vectors, -vectors], radius)
            best = {}
            for j, candidates in enumerate(hits):
                oi = j % 576
                for index in candidates:
                    error = distance(orbit[oi], self.matrices[index])
                    if error <= epsilon+1e-14 and (index not in best or (error, oi) < best[index]):
                        best[index] = error, oi
            for index, (error, oi) in best.items():
                li, ri = divmod(oi, 24)
                matches.append((index, error, list(clifford_group()[li].ops), list(clifford_group()[ri].ops)))
        if not matches:
            return None
        i, error, left, right = min(matches, key=lambda m: (self.entries[m[0]]["expected_t_count"], m[1], rank(self.entries[m[0]])))
        entry = dict(self.entries[i])
        entry["recipe"] = wrap(entry["recipe"], inverse_ops(left), inverse_ops(right))
        L, R = Operator(circuit_from_ops(left)).data, Operator(circuit_from_ops(right)).data
        actual = L.conj().T @ self.matrices[i] @ R.conj().T
        entry.update(unitary_matrix=encode(actual), unitary_key=matrix_key(actual))
        witness = z_rotation_metadata(actual)
        entry.update(axial=witness["axial"], axial_angle=witness["angle"], axial_witness=witness)
        return dict(entry=entry, distance=distance(actual, target), epsilon=epsilon,
                    candidates_within_epsilon=len({m[0] for m in matches}),
                    target_unitary_matrix=encode(target), matched_entry_id=self.entries[i]["entry_id"],
                    lookup_method="binary_search" if self.mode == "axial" else "quaternion_kdtree_all_target_clifford_images")


def lookup_database(input_dir, output_dir, target, epsilon, workers=0):
    root, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Lookup output must be a new or empty directory")
    payload = json.loads((root / "database.json").read_text())
    if payload.get("algorithm") != "composition" or not payload["stats"].get("export_ok"):
        raise ValueError("Expected a completed composition database")
    result = Lookup(payload).query(target, epsilon)
    from output_paths import register_output
    register_output(output, "lookup")
    if result is None:
        write_json(output / "lookup.json", dict(found=False, epsilon=epsilon, target_unitary_matrix=encode(target)))
        return None
    selected = {k: v for k, v in result["entry"].items() if k not in {"export_files", "qpy_indices", "validation"}}
    selected["entry_id"] = "approximation_0000000"
    stats = dict(payload["stats"], output_entries=1, export_ok=False, lookup={k: v for k, v in result.items() if k != "entry"})
    # The selected unitary need not be folded; use all-mode lookup if this small
    # exported dataset is itself used as a database later.
    stats["config"] = dict(stats["config"], mode="all")
    write_json(output / "analysis.json", dict(payload, entries=[selected], stats=stats))
    export_database(output, workers)
    write_json(output / "lookup.json", dict(found=True, **{k: v for k, v in result.items() if k != "entry"}))
    return result


def hybrid_estimate(payload, theta, epsilon, coarse_epsilon=1e-6):
    """Section 6.2.2 resource estimate, not a synthesized gearbox circuit.

    Eqs. 31–34 are scaling estimates in the source paper. Return a constructive
    coarse lookup plus the residual/mantissa parameters and estimated costs;
    never export a continuous rotation as a counted Clifford+T gearbox.
    """
    if not all(math.isfinite(x) for x in (theta, epsilon, coarse_epsilon)) or not 0 < epsilon < coarse_epsilon <= 1:
        raise ValueError("Require 0 < epsilon < coarse_epsilon <= 1 and finite theta")
    index = Lookup(payload)
    coarse = index.query(rotation(theta), coarse_epsilon)
    if coarse is None:
        return dict(available=False, reason="No coarse approximation within coarse_epsilon", circuit_synthesized=False)
    actual = decode(coarse["entry"]["unitary_matrix"])
    actual_angle = float(np.angle(actual[1, 1]/actual[0, 0]))
    residual = float(np.angle(np.exp(1j*(theta-actual_angle))))
    result = dict(available=True, circuit_synthesized=False, method="paper_section_6.2.2_resource_estimate",
                  coarse_entry_id=coarse["matched_entry_id"], coarse_distance=coarse["distance"],
                  coarse_expected_t=coarse["entry"]["expected_t_count"], residual_angle=residual,
                  epsilon=epsilon, coarse_epsilon=coarse_epsilon)
    if distance(actual, rotation(theta)) <= epsilon:
        return dict(result, gearbox_needed=False, estimated_expected_t=coarse["entry"]["expected_t_count"])
    # residual = sign * a * 10^(-gamma), with .1 <= a < 1.
    gamma = -math.floor(math.log10(abs(residual)))-1
    mantissa = abs(residual)*10**gamma
    mantissa_epsilon = min(1., epsilon*10**gamma)
    lookup = index.query(rotation(mantissa), mantissa_epsilon)
    result.update(gearbox_needed=True, gamma=gamma, mantissa=mantissa,
                  mantissa_epsilon=mantissa_epsilon, mantissa_available=lookup is not None,
                  eq32_estimate=2.52*math.log2(10**(-gamma)/epsilon)+1.14*math.log2(10**gamma)+5.14,
                  eq34_estimate=2.52*math.log2(1/epsilon)-.12*math.log2(1/coarse_epsilon)-2.97)
    if lookup is not None:
        result.update(mantissa_entry_id=lookup["matched_entry_id"], mantissa_distance=lookup["distance"],
                      estimated_expected_t=coarse["entry"]["expected_t_count"]+2*lookup["entry"]["expected_t_count"]+1.14*math.log2(10**gamma)+12.2)
    else:
        result["reason"] = "Mantissa requires a larger database or recursive hybrid approximation"
    return result
