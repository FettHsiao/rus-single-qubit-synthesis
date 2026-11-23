"""Independently audit coverage, cost winners and class uniqueness in SO(3)."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from direct_search.circuits import clifford_matrices
from .common import decode, phase_error, verify_transformation
from .pipeline import BASE, fingerprint, write_json


def bloch_rotations(matrices):
    """R_ab = Tr(Pauli_a U Pauli_b U†)/2, eliminating global phase independently."""
    pauli = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    return np.einsum("aij,...jk,bkl,...il->...ab", pauli, matrices, pauli, matrices.conj()).real / 2


def equivalent_class_pairs(entries, tol=1e-10):
    """Search every Clifford pair in SO(3), then confirm hits using 2x2 unitaries."""
    unitaries = np.array([decode(entry["unitary_matrix"]) for entry in entries])
    group = clifford_matrices()
    rotations, cliffords = bloch_rotations(unitaries), bloch_rotations(group)
    tree = cKDTree(rotations.reshape(-1, 9))
    duplicates = set()
    for index, rotation in enumerate(rotations):
        orbit = (cliffords[:, None] @ rotation @ cliffords[None, :]).reshape(576, 9)
        neighbors = tree.query_ball_point(orbit, 12 * tol)
        for pair, hits in enumerate(neighbors):
            for other in hits:
                if other <= index:
                    continue
                left, right = divmod(pair, 24)
                if phase_error(group[left] @ unitaries[index] @ group[right], unitaries[other]) <= tol:
                    duplicates.add((entries[index]["class_id"], entries[other]["class_id"]))
    return sorted(duplicates)


def audit(directory):
    directory = Path(directory).resolve()
    payload = json.loads((directory / "database.json").read_text())
    if payload["algorithm"] != "clucr":
        raise ValueError("This audit requires a clucr dataset")
    source = Path(payload["stats"]["source_database"])
    if fingerprint(source) != payload["stats"]["source_database_sha256"]:
        raise ValueError("Source database has changed since analysis")
    sources = json.loads(source.read_text())["entries"]
    source_ids = {entry["entry_id"] for entry in sources}
    members = [json.loads(line) for line in (directory / "membership.jsonl").open()]
    if len({member["source_entry_id"] for member in members}) != len(members):
        raise ValueError("Duplicate source membership")
    if payload["stats"]["full_input"] and {m["source_entry_id"] for m in members} != source_ids:
        raise ValueError("Membership does not cover the complete source database")
    tol = payload["stats"]["tolerance"]
    grouped = defaultdict(list)
    for member in members:
        verify_transformation(member, tol)
        grouped[member["class_id"]].append(member)
    for winner in payload["entries"]:
        group = grouped[winner["class_id"]]
        cost_field = "effective_t_count" if payload["stats"].get("selection_cost") == "effective" else "expected_t_count"
        rank = lambda entry: (entry[cost_field], entry["raw_t_count"],
                              entry["candidate_ordinal"], entry["source_entry_id"])
        if min(group, key=rank)["source_entry_id"] != winner["source_entry_id"]:
            raise ValueError(f"Selected circuit is not the minimum {cost_field} member")
        target = decode(winner["transformation"]["unitary_matrix"])
        if any(phase_error(target, decode(m["transformation"]["unitary_matrix"])) > tol for m in group):
            raise ValueError("Membership target differs from its winner")
    duplicates = equivalent_class_pairs(payload["entries"], tol)
    result = dict(ok=not duplicates, source_entries=len(sources), membership_entries=len(members),
                  representative_entries=len(payload["entries"]), minimum_cost_winners_verified=len(payload["entries"]),
                  membership_wrappers_verified=len(members), equivalent_cross_class_pairs=duplicates,
                  classes_containing_axial_members=sum(any(m["source_axial"] for m in group) for group in grouped.values()),
                  classes_with_mixed_source_classification=sum(len({m["source_axial"] for m in group}) > 1 for group in grouped.values()),
                  independence="SO(3) Clifford orbits; no canonical-key or lexicographic reuse",
                  tolerance=tol, source_database_sha256=fingerprint(source),
                  membership_sha256=fingerprint(directory / "membership.jsonl"))
    write_json(directory / "audit_summary.json", result)
    if duplicates:
        raise ValueError(f"Equivalent classes were not merged: {duplicates}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=BASE / "out_record_paper")
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), indent=2))
