"""Validate direct entries and export circuits through Qiskit's QASM writers."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import qiskit
from qiskit.quantum_info import Operator, Statevector

from .circuits import (
    ANC, TEMPLATE_CONVENTION, TOL, circuit_from_ops, circuit_from_template, equivalent,
    matrix_key, t_count, validate_template_form,
)
from .dsa import SearchConfig, analyze_w, decode_matrix
from .amplification import amplification_metadata, amplified_entry_circuit
from .axial import z_rotation_metadata
from .validate_qasm import (
    dump_and_validate, record_error, reference_circuit, validation_record,
    write_validation_reports,
)


def validate_entry(entry, tol=TOL):
    """Check native gates, branches and failure recovery on independent inputs."""
    if "canonical" in entry or entry.get("deduplication_method", "exact") != "exact":
        raise ValueError("Legacy reduced entries require a fresh direct search")
    bs12_template = entry.get("template_convention") == TEMPLATE_CONVENTION
    if bs12_template:
        validate_template_form(entry["template_labels"], entry["template_ops"], entry["fig3_t_partition"])
    circuit = circuit_from_template(entry["template_ops"])
    W = Operator(circuit).data
    if circuit.count_ops().get("cz", 0) != 2:
        raise ValueError("A Fig. 3 attempt must contain exactly two CZ gates")
    if t_count(circuit) != entry["raw_t_count"]:
        raise ValueError("Stored T count differs from the native Qiskit circuit")
    permissive = SearchConfig(require_recovery_branch=False, strict_nondeterministic=False, tol=tol)
    result = analyze_w(W, permissive)
    U = decode_matrix(entry["unitary_matrix"])
    if not result.accepted or not equivalent(result.unitary, U, tol):
        raise ValueError("Reconstructed W has a different success unitary")
    if abs(result.probability - entry["success_probability"]) > tol:
        raise ValueError("Reconstructed W has a different success probability")
    if result.success_outcomes != entry["success_outcomes"]:
        raise ValueError("Reconstructed W has different success outcomes")
    if set(result.failure_ops) != set(entry["failure_outcome_ops"]):
        raise ValueError("Stored recovery outcomes differ from reconstructed W")
    if abs(entry["expected_t_count"] - entry["raw_t_count"] / result.probability) > tol:
        raise ValueError("Expected T count must equal raw T count / success probability")
    # Input states span the Bloch axes, including complex relative phases.
    probes = ([1, 0], [0, 1], np.array([1, 1]) / np.sqrt(2),
              np.array([1, 1j]) / np.sqrt(2))
    for psi in probes:
        psi = np.asarray(psi, dtype=complex)
        state = Statevector(np.r_[psi, [0, 0]]).evolve(circuit)
        for outcome in (0, 1):
            label = str(outcome)
            branch = state.data[outcome * 2:outcome * 2 + 2]
            p = float(np.vdot(branch, branch).real)
            if abs(p - entry["branch_probabilities"][label]) > tol:
                raise ValueError("Branch probability depends on the data input")
            if p <= tol:
                continue
            branch /= np.sqrt(p)
            if label in entry["success_outcomes"]:
                expected = U @ psi
            else:
                correction = circuit_from_ops(entry["failure_outcome_ops"][label]).inverse()
                branch = Statevector(branch).evolve(correction).data
                expected = psi
            if not Statevector(branch).equiv(Statevector(expected), rtol=0, atol=tol):
                raise ValueError("Success action or inverse-Clifford recovery failed")
    for field in ("unitary_key", "unitary_exact_key"):
        if entry[field] != matrix_key(U):
            raise ValueError("Unitary key must identify the physical success U")
    axial = z_rotation_metadata(U, tol)
    if entry["axial"] != (axial["angle"] is not None):
        raise ValueError("Axial classification differs from L U R (independent Cliffords)")
    if axial["angle"] is None:
        if entry["axial_angle"] is not None:
            raise ValueError("Non-axial U must not have an axial angle")
    elif (entry["axial_angle"] is None or not np.isfinite(entry["axial_angle"])
          or abs(entry["axial_angle"] - axial["angle"]) > tol):
        raise ValueError("Axial angle differs from the unfolded L U R (independent Cliffords) angle")
    if "z_rotation" in entry:
        for field in ("method", "relation", "clifford_label", "clifford_ops", "left_ops", "right_ops"):
            if entry["z_rotation"][field] != axial[field]:
                raise ValueError("Z rotation witness differs from L U R (independent Cliffords)")
        angle = entry["z_rotation"]["angle"]
        if ((angle is None) != (axial["angle"] is None)
                or (angle is not None and (not np.isfinite(angle) or abs(angle - axial["angle"]) > tol))):
            raise ValueError("Z rotation witness angle differs from L U R (independent Cliffords)")
    if "amplification_rounds" in entry:
        amplified = amplified_entry_circuit(entry)
        if t_count(amplified) != entry["amplified_raw_t_count"]:
            raise ValueError("Amplified native T count differs from (2j+1)T")
        amp_w = Operator(amplified).data
        result_amp = analyze_w(amp_w, permissive)
        if not result_amp.accepted or not equivalent(result_amp.unitary, U, tol):
            raise ValueError("Amplification changed the success unitary")
        p = result_amp.probability
        if abs(p - entry["amplified_success_probability"]) > tol:
            raise ValueError("Amplified probability differs from sin^2((2j+1)theta)")
        if abs(entry["effective_t_count"] - t_count(amplified) / p) > tol * max(1, entry["effective_t_count"]):
            raise ValueError("Effective T cost differs from the executable amplified circuit")
        for outcome, ops in result_amp.failure_ops.items():
            if outcome not in entry["failure_outcome_ops"] or not equivalent(
                    Operator(circuit_from_ops(ops)).data,
                    Operator(circuit_from_ops(entry["failure_outcome_ops"][outcome])).data, tol):
                raise ValueError("Amplification changed the failure recovery")
    return {"qiskit_operator": True, "native_gate_t_count": True,
            "statevector_success_and_recovery": True, "input_probes": len(probes),
            "bs12_template": bs12_template, "exact_unitary_and_axial_classification": True,
            "amplified_operator_and_recovery": "amplification_rounds" in entry}


def _export_one(task):
    index, source, output_dir, tol, export_qasm, export_qasm2 = task
    qasm_rows, entry_failures = [], 0
    entry = dict(source)
    group = "axial" if entry["axial"] else "non_axial"
    stem = f"rus_{index:05d}_T{entry['raw_t_count']}_p{entry['success_probability']:.10f}"
    folder = output_dir / group
    folder.mkdir(exist_ok=True)
    files = {"json": f"{group}/{stem}.json"}
    entry.update(entry_id=stem, export_group=group)
    try:
        if "z_rotation" not in entry:
            entry["z_rotation"] = z_rotation_metadata(decode_matrix(entry["unitary_matrix"]), tol)
        if "amplification_rounds" not in entry:
            entry.update(amplification_metadata(entry["raw_t_count"], entry["success_probability"], 0))
        entry["validation"] = validate_entry(entry, tol)
    except Exception as exc:
        entry_failures += 1
        qasm_rows.append(record_error(validation_record(folder / f"{stem}.json", "entry", entry=entry),
                                      "entry_validation", exc))
        return None, qasm_rows, entry_failures, os.getpid()
    formats = []
    if export_qasm:
        formats.extend((kind + "_qasm3", "qasm3", f"{stem}_{kind}.qasm")
                       for kind in ("w", "attempt", "rus_loop"))
        if entry["amplification_rounds"]:
            formats.extend((kind + "_qasm3", "qasm3", f"{stem}_{kind}.qasm")
                           for kind in ("amplified_w", "amplified_rus_loop"))
    if export_qasm2:
        formats.append(("attempt_qasm2", "qasm2", f"{stem}_attempt_qasm2.qasm"))
    entry["validation"]["qasm_files"] = {}
    for kind, version, filename in formats:
        path = folder / filename
        files[kind] = f"{group}/{filename}"
        try:
            circuit = reference_circuit(entry, kind)
        except Exception as exc:
            result = record_error(validation_record(path, version, kind, entry), "build_reference", exc)
        else:
            result = dump_and_validate(circuit, path, version, kind, entry, tol)
        qasm_rows.append(result)
        entry["validation"]["qasm_files"][kind] = {
            key: result[key] for key in ("status", "stage", "read_function", "verification", "error_type", "error_message")
        }
    entry.update({"entry_id": stem, "export_group": group, "export_files": files})
    (folder / f"{stem}.json").write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    return entry, qasm_rows, entry_failures, os.getpid()


def export_database(database, stats, output_dir, export_qasm=True, export_qasm2=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Refuse reuse of a completed output directory so obsolete circuits cannot
    # silently remain mixed with results of another run.
    if (output_dir / "database.json").exists():
        raise FileExistsError(f"Output already contains a database: {output_dir}; choose a new directory")
    entries = sorted(database.values(), key=lambda e: (
        not e["axial"], e["axial_angle"] if e["axial"] else 0,
        e["expected_t_count"], e["unitary_key"]))
    records, rows, qasm_rows = [], [], []
    entry_failures = 0
    from execution import parallel_chunks
    from .hardware import resolve_workers
    workers = resolve_workers(stats.get("config", {}).get("workers", 0), len(entries))
    pids = set()
    tasks = ((i, source, output_dir, stats["config"]["tol"], export_qasm, export_qasm2) for i, source in enumerate(entries))
    for entry, checks, failures, pid in parallel_chunks(_export_one, tasks, workers):
        qasm_rows.extend(checks)
        entry_failures += failures
        pids.add(pid)
        if entry is not None:
            records.append(entry)
            rows.append({key: entry[key] for key in (
                "entry_id", "export_group", "raw_t_count", "success_probability",
                "expected_t_count", "effective_t_count", "amplification_rounds",
                "amplified_raw_t_count", "amplified_success_probability", "axial_angle")})
    qasm_summary = write_validation_reports(qasm_rows, output_dir)
    summary = dict(stats, export_workers_used=workers, export_worker_pids=sorted(pids), qiskit_version=qiskit.__version__, validated_entries=len(records),
                   invalid_entries=entry_failures, qasm_validation=qasm_summary,
                   export_ok=qasm_summary["ok"],
                   axial_entries=sum(e["axial"] for e in records),
                   non_axial_entries=sum(not e["axial"] for e in records))
    from .research import dataset_statistics
    summary["research_statistics"] = dataset_statistics(records)
    payload = {"schema_version": 3, "algorithm": "direct_search",
               "qubit_order": {"data": 0, "ancilla": ANC},
               "stats": summary, "entries": records}
    (output_dir / "database.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fields = ("entry_id", "export_group", "raw_t_count", "success_probability",
              "expected_t_count", "effective_t_count", "amplification_rounds",
              "amplified_raw_t_count", "amplified_success_probability", "axial_angle")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return summary
