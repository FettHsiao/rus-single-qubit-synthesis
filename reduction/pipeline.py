"""Bounded process workers for parsing/analysis and Qiskit export/readback."""
from __future__ import annotations

import argparse
from collections import Counter
from execution import chunks, fingerprint, parallel_chunks, report, write_json
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import qiskit
from qiskit import QuantumCircuit, qasm2, qasm3, qpy
from qiskit.quantum_info import Operator

from direct_search.circuits import (
    TEMPLATE_CONVENTION, circuit_from_ops, circuit_from_template, matrix_key,
    rus_loop_circuit, t_count, validate_template_form,
)
from direct_search.validate_qasm import compare_circuits
from direct_search.hardware import resolve_workers
from .common import checked_unitary, decode, phase_error, verify_transformation, wrapper_circuit
from direct_search.axial import z_rotation_metadata as classify_axial
from .clucr import canonicalize, reduce_entries

BASE = Path(__file__).resolve().parent
MODES = ("cuc_dagger", "clucr")


def source_file(root, relative):
    """Resolve moved database exports against their current parent directory."""
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Export path must be relative to the input database: {relative}")
    return path


def validate_source(entry, root, tol):
    """Parse actual source QASM; independently check the full success/recovery maps."""
    root = Path(root).resolve()
    unitary = checked_unitary(decode(entry["unitary_matrix"]), tol)
    if entry.get("deduplication_method") != "exact" or "canonical" in entry:
        raise ValueError("Expected an unreduced direct-search entry")
    if any(entry[key] != matrix_key(unitary) for key in ("unitary_key", "unitary_exact_key")):
        raise ValueError("Source identity key differs from its physical unitary")
    probability = entry["success_probability"]
    if not math.isfinite(probability) or not 0 < probability <= 1:
        raise ValueError("Invalid stored success probability")
    if (set(entry["branch_probabilities"]) != {"0", "1"}
            or any(not math.isfinite(p) or not 0 <= p <= 1 for p in entry["branch_probabilities"].values())):
        raise ValueError("Invalid stored branch probabilities")
    files = entry["export_files"]
    if "w_qasm3" in files:
        path = source_file(root, files["w_qasm3"])
        circuit = qasm3.load(str(path))
        parser = "qiskit.qasm3.load"
    else:
        path = source_file(root, files["attempt_qasm2"])
        attempt = qasm2.load(str(path))
        expected = circuit_from_template(entry["template_ops"], measure=True)
        compare_circuits(expected, attempt, tol)
        circuit = attempt.remove_final_measurements(inplace=False)
        parser = "qiskit.qasm2.load"
    if circuit.num_qubits != 2 or circuit.num_clbits:
        raise ValueError("The parsed W must be a two-qubit unitary circuit")
    W = Operator(circuit).data
    expected_w = Operator(circuit_from_template(entry["template_ops"])).data
    if phase_error(W, expected_w) > tol:
        raise ValueError("Source QASM differs from template_ops")
    if (not np.allclose(W.conj().T @ W, np.eye(4), atol=tol, rtol=0)
            or t_count(circuit) != entry["raw_t_count"]):
        raise ValueError("Invalid W or inconsistent native T count")
    if entry.get("template_convention") == TEMPLATE_CONVENTION:
        validate_template_form(entry["template_labels"], entry["template_ops"], entry["fig3_t_partition"])
    success = set(entry["success_outcomes"])
    failure = set(entry["failure_outcome_ops"])
    zero = set(entry["zero_probability_outcomes"])
    if (success & failure or success & zero or failure & zero
            or success | failure | zero != {"0", "1"} or not success):
        raise ValueError("Invalid measurement outcome partition")
    total_success, max_error = 0.0, 0.0
    for outcome in (0, 1):
        label = str(outcome)
        block = W[2 * outcome:2 * outcome + 2, :2]
        gram = block.conj().T @ block
        probability = float(np.trace(gram).real / 2)
        if (abs(probability - entry["branch_probabilities"][label]) > tol
                or not np.allclose(gram, probability * np.eye(2), atol=tol, rtol=0)):
            raise ValueError("Source branch probability is inconsistent or depends on the input state")
        if label in zero:
            if probability > tol:
                raise ValueError("Nonzero branch labeled zero")
            continue
        if probability <= tol:
            raise ValueError("Zero branch labeled success/recovery")
        branch = block / math.sqrt(probability)
        if label in success:
            error = phase_error(branch, unitary)
            total_success += probability
        else:
            recovery = Operator(circuit_from_ops(entry["failure_outcome_ops"][label])).data
            error = phase_error(recovery.conj().T @ branch, np.eye(2))
        max_error = max(max_error, error)
        if error > tol:
            raise ValueError("Source success unitary or inverse-Clifford recovery is inconsistent")
    expected_cost = entry["raw_t_count"] / total_success
    if (abs(total_success - entry["success_probability"]) > tol
            or not math.isfinite(entry["expected_t_count"])
            or not math.isclose(entry["expected_t_count"], expected_cost, rel_tol=tol, abs_tol=tol)):
        raise ValueError("Stored success probability or direct expected T count is inconsistent")
    return dict(parser=parser, source_qasm=str(path.relative_to(root)),
                source_qasm_sha256=fingerprint(path), branch_operator_error=max_error,
                native_t_count=True, success_and_recovery=True, template_operator=True)


def analyze_chunk(task):
    entries, root, modes, tol = task
    output = []
    for source in entries:
        try:
            validation = validate_source(source, root, tol)
            entry = {key: value for key, value in source.items()
                     if key not in {"validation", "export_files", "canonical", "transformation"}}
            entry["source_entry_id"] = source["entry_id"]
            entry["source_axial"] = source.get("axial")
            entry["source_validation"] = validation
            witness = classify_axial(decode(entry["unitary_matrix"]), tol)
            entry.update(axial=witness["axial"], axial_angle=witness["angle"], axial_witness=witness, z_rotation=witness)
            if "amplification_rounds" in entry:
                from direct_search.export import validate_entry
                validate_entry(entry, tol)
            entry["canonical"] = canonicalize(decode(entry["unitary_matrix"]), tol)
            output.append(entry)
        except Exception as exc:
            raise ValueError(f"Analysis failed for {source.get('entry_id')}: {exc}") from exc
    return output, os.getpid()


def axial_entry(source):
    entry = {key: value for key, value in source.items() if key != "canonical"}
    witness = entry["axial_witness"]
    left = witness["clifford_ops"] if entry["axial"] else []
    right = witness["right_ops"] if entry["axial"] else []
    target = witness["diagonal_unitary_matrix"] if entry["axial"] else entry["unitary_matrix"]
    entry["transformation"] = dict(method="left_right_clifford" if entry["axial"] else "identity",
                                   left_ops=left, right_ops=right, unitary_matrix=target,
                                   relation="target = left @ physical_success_unitary @ right" if entry["axial"]
                                   else "target = physical_success_unitary (no axial witness)")
    return entry


def require_amplified_source(payload, tol):
    """Reject unoptimized inputs and stale or nonoptimal amplification costs."""
    from direct_search.amplification import amplification_metadata
    stats = payload.get("stats", {})
    amplification = stats.get("amplification") or {}
    max_j = amplification.get("max_j", 0)
    if (not isinstance(max_j, int) or max_j < 1
            or stats.get("dataset_variant") == "unamplified"):
        raise ValueError("Reduction requires the amplified direct-search dataset (max_j > 0)")
    max_raw_t = amplification.get("max_raw_t")
    for entry in payload["entries"]:
        expected = amplification_metadata(entry["raw_t_count"], entry["success_probability"], max_j, max_raw_t)
        for field in ("amplification_rounds", "amplified_raw_t_count", "amplified_success_probability", "effective_t_count"):
            actual = entry.get(field)
            if actual is None or not math.isclose(actual, expected[field], rel_tol=tol, abs_tol=tol):
                raise ValueError(f"Invalid optimized {field} for {entry.get('entry_id')}")
    return amplification


def analyze_database(input_path, outputs, *, workers=0, chunk_size=32, tol=1e-10, limit=None, cost="effective"):
    started = time.monotonic()
    input_path = Path(input_path).resolve()
    if input_path.is_dir():
        input_path /= "database.json"
    for output in outputs.values():
        output = Path(output)
        if any((output / name).exists() for name in ("analysis.json", "database.json")):
            raise FileExistsError(f"Analysis output already exists: {output}; use --stage export or a new output directory")
        if output.resolve() == input_path.parent:
            raise ValueError("Output must differ from the source dataset")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 3 or payload.get("algorithm") != "direct_search":
        raise ValueError("Expected the schema-3 direct_search database, before Clifford reduction")
    if payload.get("qubit_order") != {"data": 0, "ancilla": 1}:
        raise ValueError("Unsupported source qubit order")
    sources = payload["entries"][:limit]
    if cost != "effective":
        raise ValueError("Reduction always selects winners by amplified effective T cost")
    require_amplified_source(payload, tol)
    if not outputs or any(mode not in MODES for mode in outputs):
        raise ValueError("Unsupported reduction mode")
    if not sources or len({entry["entry_id"] for entry in sources}) != len(sources):
        raise ValueError("The input must have nonempty, uniquely identified entries")
    actual_workers = resolve_workers(workers, math.ceil(len(sources) / chunk_size))
    report("analyze", entries=len(sources), workers=actual_workers, modes=list(outputs))
    entries, pids, last = [], set(), time.monotonic()
    tasks = ((chunk, input_path.parent, tuple(outputs), tol) for chunk in chunks(sources, chunk_size))
    for result, pid in parallel_chunks(analyze_chunk, tasks, actual_workers):
        entries.extend(result)
        pids.add(pid)
        if time.monotonic() - last >= 10:
            report("analyze", completed=len(entries), total=len(sources))
            last = time.monotonic()
    common = dict(source_search_complete=payload.get("stats", {}).get("search_complete"), axial_classification="left_right_clifford", source_database=str(input_path), source_database_sha256=fingerprint(input_path),
                  total_source_entries=len(payload["entries"]), input_entries=len(entries),
                  full_input=limit is None or limit >= len(payload["entries"]),
                  tolerance=tol, workers_used=actual_workers, worker_pids=sorted(pids),
                  chunk_size=chunk_size, multiprocessing_start_method="spawn", worker_blas_threads=1,
                  source_qasm_parsed=len(entries), source_qasm_validation_failures=0,
                  source_axial_mismatches=sum(e["source_axial"] != e["axial"] for e in entries),
                  qiskit_version=qiskit.__version__,
                  qasm3_importer_version=importlib.metadata.version("qiskit-qasm3-import"),
                  composition_performed=False, angle_folding_performed=False,
                  source_amplification=payload.get("stats", {}).get("amplification"))
    from output_paths import register_output
    result_summaries = {}
    for mode, output in outputs.items():
        output = Path(output)
        register_output(output, "reduction")
        selected, membership = reduce_entries(entries, tol, cost=cost)
        # Legacy mode names also perform the full independent-L/R reduction.
        for member in membership:
            verify_transformation(member, tol)
        with (output / "membership.jsonl").open("w", encoding="utf-8") as handle:
            for member in membership:
                handle.write(json.dumps(member, ensure_ascii=False, allow_nan=False) + "\n")
        for entry in selected:
            verify_transformation(entry, tol)
        stats = dict(common, mode="clucr", requested_mode=mode, selection_cost=cost,
                     equivalence_relation="L U R with independent single-qubit Cliffords",
                     representative_selection="first-row phase normalization then lexicographic minimum over 576 LUR pairs",
                     ranking=f"{cost} expected T, raw T, candidate ordinal, source id", output_entries=len(selected),
                     axial_entries=sum(e["axial"] for e in selected),
                     non_axial_entries=sum(not e["axial"] for e in selected),
                     removed_entries=len(entries) - len(selected), membership_entries=len(membership),
                     analysis_seconds=time.monotonic() - started)
        analysis = dict(schema_version=1, algorithm="clucr", qubit_order={"data": 0, "ancilla": 1},
                        stats=stats, entries=selected)
        write_json(output / "analysis.json", analysis)
        write_json(output / "analysis_summary.json", stats)
        result_summaries[mode] = stats
        report("analysis_complete", mode=mode, output_entries=len(selected), removed_entries=stats["removed_entries"])
    return result_summaries


def write_and_check(circuit, path, version, tol):
    writer, reader = {"qasm2": (qasm2.dump, qasm2.load), "qasm3": (qasm3.dump, qasm3.load)}[version]
    with path.open("w", encoding="utf-8") as handle:
        writer(circuit, handle)
    loaded = reader(str(path))
    compare_circuits(circuit, loaded, tol)


def export_entry(source, output, tol):
    entry = dict(source)
    group = "axial" if entry["axial"] else "non_axial"
    stem = entry["entry_id"]
    if Path(stem).name != stem or stem in {".", ".."}:
        raise ValueError("Invalid entry identifier")
    folder = output / group
    folder.mkdir(exist_ok=True)
    checks = verify_transformation(entry, tol)
    W = circuit_from_template(entry["template_ops"])
    W.name = "physical_w"
    attempt = circuit_from_template(entry["template_ops"], measure=True)
    attempt.name = "physical_attempt"
    physical = rus_loop_circuit(entry["template_ops"], entry["success_outcomes"], entry["failure_outcome_ops"])
    physical.name = "physical_rus_loop"
    target = wrapper_circuit(entry)
    target_matrix = decode(entry["transformation"]["unitary_matrix"])
    reference = QuantumCircuit(1, name="target_unitary_reference")
    reference.unitary(target_matrix, [0])
    specs = [("w_qasm2", "qasm2", W), ("attempt_qasm2", "qasm2", attempt),
             ("physical_rus_loop_qasm3", "qasm3", physical), ("target_rus_loop_qasm3", "qasm3", target)]
    bundle = [W, attempt, physical, target, reference]
    if entry.get("amplification_rounds", 0):
        from direct_search.amplification import amplified_entry_circuit
        from direct_search.export import validate_entry
        validation_source = {k: v for k, v in entry.items() if k != "canonical"}
        validation_source["deduplication_method"] = "exact"
        validate_entry(validation_source, tol)
        amp_w = amplified_entry_circuit(entry)
        amp_w.name = "amplified_w"
        amp_attempt = QuantumCircuit(2, 1, name="amplified_attempt")
        amp_attempt.compose(amp_w, inplace=True)
        amp_attempt.measure(1, 0)
        amp_physical = rus_loop_circuit(entry["template_ops"], entry["success_outcomes"],
                                       entry["failure_outcome_ops"], attempt=amp_w)
        amp_physical.name = "amplified_physical_rus_loop"
        amp_target = wrapper_circuit(entry, amplified=True)
        amp_target.name = "amplified_target_rus_loop"
        specs.extend([("amplified_w_qasm2", "qasm2", amp_w),
                      ("amplified_attempt_qasm2", "qasm2", amp_attempt),
                      ("amplified_physical_rus_loop_qasm3", "qasm3", amp_physical),
                      ("amplified_target_rus_loop_qasm3", "qasm3", amp_target)])
        bundle.extend([amp_w, amp_attempt, amp_physical, amp_target])
    files, rows = {}, []
    for kind, version, circuit in specs:
        relative = f"{group}/{stem}_{kind}.qasm"
        write_and_check(circuit, output / relative, version, tol)
        files[kind] = relative
        rows.append(dict(entry_id=stem, kind=kind, file=relative,
                         read_function=f"qiskit.{version}.load", status="passed", circuits_checked=1))
    relative = f"{group}/{stem}_circuits.qpy"
    with (output / relative).open("wb") as handle:
        qpy.dump(bundle, handle)
    with (output / relative).open("rb") as handle:
        loaded = qpy.load(handle)
    if len(loaded) != len(bundle):
        raise ValueError("QPY circuit count differs after readback")
    for before, after in zip(bundle, loaded):
        compare_circuits(before, after, tol)
    files["qpy"] = relative
    files["json"] = f"{group}/{stem}.json"
    rows.append(dict(entry_id=stem, kind="qpy", file=relative, read_function="qiskit.qpy.load",
                     status="passed", circuits_checked=len(bundle)))
    entry.update(export_group=group, export_files=files,
                 qpy_indices={circuit.name: index for index, circuit in enumerate(bundle)},
                 validation=dict(checks, qasm_readback=True, qpy_readback=True,
                                 control_flow_and_operators=True))
    write_json(output / files["json"], entry)
    return entry, rows


def export_chunk(task):
    entries, output, tol = task
    exported, rows = [], []
    for entry in entries:
        try:
            result, checks = export_entry(entry, output, tol)
            exported.append(result)
            rows.extend(checks)
        except Exception as exc:
            raise ValueError(f"Export/readback failed for {entry['entry_id']}: {exc}") from exc
    return exported, rows, os.getpid()


def export_database(output, *, workers=0, chunk_size=32):
    """Export a completed analysis checkpoint; successful outputs are never overwritten."""
    started = time.monotonic()
    output = Path(output).resolve()
    if (output / "database.json").exists():
        raise FileExistsError(f"Completed dataset already exists: {output}")
    payload = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("algorithm") != "clucr":
        raise ValueError("Unsupported analysis checkpoint")
    if (payload["stats"].get("selection_cost") != "effective"
            or (payload["stats"].get("source_amplification") or {}).get("max_j", 0) < 1):
        raise ValueError("Export requires LUR analysis of an amplified dataset using effective cost")
    tol = payload["stats"]["tolerance"]
    sources = payload["entries"]
    if not sources:
        raise ValueError("Analysis checkpoint contains no entries")
    actual_workers = resolve_workers(workers, math.ceil(len(sources) / chunk_size))
    rows, entries, pids, last = [], [], set(), time.monotonic()
    report("export", mode=payload["algorithm"], entries=len(sources), workers=actual_workers)
    tasks = ((chunk, output, tol) for chunk in chunks(sources, chunk_size))
    try:
        for result, checks, pid in parallel_chunks(export_chunk, tasks, actual_workers):
            entries.extend(result)
            rows.extend(checks)
            pids.add(pid)
            if time.monotonic() - last >= 10:
                report("export", mode=payload["algorithm"], completed=len(entries), total=len(sources))
                last = time.monotonic()
    except Exception as exc:
        write_json(output / "export_error.json", dict(export_ok=False, error_type=type(exc).__name__, error=str(exc)))
        raise
    fields = ["entry_id", "kind", "file", "read_function", "status", "circuits_checked"]
    with (output / "validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(payload["stats"], export_ok=True, export_workers_used=actual_workers,
                   export_worker_pids=sorted(pids), export_seconds=time.monotonic() - started,
                   files_readback_verified=len(rows), circuits_readback_verified=sum(r["circuits_checked"] for r in rows),
                   validation_failures=0, formats=dict(Counter(r["kind"] for r in rows)),
                   analysis_sha256=fingerprint(output / "analysis.json"))
    fields = ["entry_id", "source_entry_id", "axial", "axial_angle", "raw_t_count", "success_probability",
              "expected_t_count", "class_id", "member_count"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)
    write_json(output / "summary.json", summary)
    write_json(output / "database.json", dict(payload, stats=summary, entries=entries))
    (output / "export_error.json").unlink(missing_ok=True)
    report("export_complete", mode=payload["algorithm"], entries=len(entries), verified_files=len(rows))
    return summary


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def run_cli(default_mode="clucr"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=(*MODES, "both"), default=default_mode)
    parser.add_argument("--stage", choices=("analyze", "export", "all"), default="all")
    parser.add_argument("--input", "--input-dir", type=Path, default=BASE.parent / "direct_search/out_amplified_paper")
    parser.add_argument("--output-root", type=Path, default=BASE,
                        help="Outputs are ROOT/out_record_paper (both: ROOT/MODE/out_record_paper)")
    parser.add_argument("--output", "--output-dir", type=Path, help="Exact output directory (single mode only)")
    parser.add_argument("--workers", type=int, default=0, help="0: all available CPUs within affinity/quota/memory limits")
    parser.add_argument("--chunk-size", type=positive_int, default=32)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--limit", type=positive_int, help="Explicit partial-input run for development")
    parser.add_argument("--cost", choices=("effective",), default="effective",
                        help="Always select class winners by optimized amplified T/p")
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers must be >= 0")
    if args.output and args.mode == "both":
        parser.error("--output requires a single --mode; use --output-root for both")
    if not math.isfinite(args.tol) or not 0 < args.tol < 1e-3:
        parser.error("--tol must be finite and in (0, 1e-3)")
    if args.stage == "export" and args.limit is not None:
        parser.error("--limit applies only to the analysis stage")
    modes = MODES if args.mode == "both" else (args.mode,)
    outputs = {mode: (args.output or (args.output_root / mode if args.mode == "both" else args.output_root)
                     / "out_record_paper").resolve() for mode in modes}
    if args.stage in {"analyze", "all"}:
        analyze_database(args.input, outputs, workers=args.workers, chunk_size=args.chunk_size,
                         tol=args.tol, limit=args.limit, cost=args.cost)
    if args.stage in {"export", "all"}:
        for output in outputs.values():
            export_database(output, workers=args.workers, chunk_size=args.chunk_size)


if __name__ == "__main__":
    run_cli()
