"""Re-optimize a direct dataset within bounded oblivious amplification rounds."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path

from direct_search.amplification import amplification_metadata
from direct_search.axial import z_rotation_metadata
from direct_search.export import export_database
from direct_search.hardware import resolve_workers
from reduction.common import decode
from reduction.pipeline import chunks, fingerprint, parallel_chunks, report, validate_source
from output_paths import register_output


def optimize_chunk(task):
    """Validate physical QASM, classify LUR and find the bounded optimal j."""
    entries, root, max_j, max_raw_t, tol = task
    result = []
    for original in entries:
        entry = dict(original)
        validation = validate_source(entry, root, tol)
        witness = z_rotation_metadata(decode(entry["unitary_matrix"]), tol)
        entry.update(axial=witness["axial"], axial_angle=witness["angle"], z_rotation=witness,
                     source_entry_id=entry["entry_id"], source_validation=validation,
                     amplification_search=dict(max_j=max_j, max_raw_t=max_raw_t,
                                               includes_unamplified=True))
        entry.update(amplification_metadata(entry["raw_t_count"], entry["success_probability"],
                                           max_j, max_raw_t))
        for key in ("export_files", "validation", "qpy_indices"):
            entry.pop(key, None)
        result.append(entry)
    return result, os.getpid()


def optimize_database(input_dir, output_dir, *, max_j=10, max_raw_t=None, workers=0, chunk_size=16):
    """Write a portable schema-3 dataset, preserving direct costs and circuits."""
    if (not isinstance(max_j, int) or max_j < 0 or workers < 0 or chunk_size < 1
            or (max_raw_t is not None and (not isinstance(max_raw_t, int) or max_raw_t < 0))):
        raise ValueError("Require nonnegative integer j/T/worker limits and positive chunk size")
    source, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if source.is_dir():
        source /= "database.json"
    if output == source.parent or output.is_relative_to(source.parent) or source.parent.is_relative_to(output):
        raise ValueError("Input/output dataset directories must not overlap")
    if (output / "database.json").exists():
        raise FileExistsError(f"Completed output already exists: {output}")
    payload = json.loads(source.read_text())
    if (payload.get("algorithm") != "direct_search" or payload.get("schema_version") != 3
            or payload.get("qubit_order") != {"data": 0, "ancilla": 1}):
        raise ValueError("Amplification requires an unreduced schema-3 direct_search dataset")
    if not payload.get("stats", {}).get("export_ok") or not payload.get("entries"):
        raise ValueError("Input must be nonempty and have passed Qiskit export validation")
    entries = payload["entries"]
    for field in ("entry_id", "unitary_exact_key"):
        if len({e[field] for e in entries}) != len(entries):
            raise ValueError(f"Duplicate {field} in input dataset")
    if max_raw_t is not None and any(e["raw_t_count"] > max_raw_t for e in entries):
        raise ValueError("--max-raw-t is smaller than an input attempt; no feasible j for that entry")
    stats = deepcopy(payload["stats"])
    tol = stats["config"]["tol"]
    used = resolve_workers(workers, math.ceil(len(entries) / chunk_size))
    register_output(output, "amplification")
    report("amplification", entries=len(entries), max_j=max_j, max_raw_t=max_raw_t, workers=used)
    optimized, pids = [], set()
    tasks = ((part, source.parent, max_j, max_raw_t, tol) for part in chunks(entries, chunk_size))
    for part, pid in parallel_chunks(optimize_chunk, tasks, used):
        optimized.extend(part)
        pids.add(pid)
    stats["config"].update(amp_max_j=max_j, workers=workers)
    stats["dataset_variant"] = "amplified" if max_j > 0 else "unamplified"
    stats["amplification"] = dict(source_database=str(source), source_sha256=fingerprint(source),
                                  max_j=max_j, max_raw_t=max_raw_t, workers_used=used,
                                  worker_pids=sorted(pids), optimized_entries=len(optimized),
                                  amplified_entries=sum(e["amplification_rounds"] > 0 for e in optimized),
                                  objective="minimum expected T among j=0..max_j within attempt T cap")
    stats["axial_classification"] = "left_right_clifford"
    summary = export_database({e["unitary_exact_key"]: e for e in optimized}, stats, output,
                              export_qasm=True, export_qasm2=True)
    if not summary["export_ok"]:
        raise ValueError(f"Amplification export/readback failed; see {output}/qasm_validation_errors.csv")
    report("amplification_complete", entries=len(optimized), output=str(output))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Direct dataset directory or database.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="New amplified dataset directory")
    parser.add_argument("--max-j", "--amp-max-j", type=int, default=10, help="Inclusive maximum rounds; 0 disables amplification")
    parser.add_argument("--max-raw-t", type=int, help="Optional per-amplified-attempt native T cap (must admit every input)")
    parser.add_argument("--workers", type=int, default=0, help="0: all available CPUs within affinity/quota/memory limits")
    parser.add_argument("--chunk-size", type=int, default=16, help="Entries per optimization process task")
    args = parser.parse_args(argv)
    try:
        optimize_database(**vars(args))
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
