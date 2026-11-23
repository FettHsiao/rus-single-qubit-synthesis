"""Native Qiskit circuits with local recovery/retries and verified QASM/QPY."""
from __future__ import annotations
import csv
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Operator
from direct_search.axial import z_rotation_metadata
from direct_search.amplification import amplification_metadata, amplified_entry_circuit
from direct_search.circuits import circuit_from_ops, rus_loop_circuit
from direct_search.export import validate_entry
from direct_search.hardware import resolve_workers
from direct_search.validate_qasm import compare_circuits
from reduction.common import decode, phase_error
from reduction.pipeline import chunks, parallel_chunks, report, validate_source, write_and_check, write_json


def validate_source_chunk(task):
    entries, root, config = task
    result = []
    for original in entries:
        # A reduction winner still has its unchanged physical template and U.
        entry = {k: v for k, v in original.items() if k not in {"canonical", "transformation"}}
        entry["deduplication_method"] = "exact"
        witness = z_rotation_metadata(decode(entry["unitary_matrix"]), config.tol)
        entry.update(z_rotation=witness, axial=witness["axial"], axial_angle=witness["angle"])
        validate_source(entry, root, config.tol)
        validate_entry(entry, config.tol)
        if config.cost == "direct":
            entry.update(amplification_metadata(entry["raw_t_count"], entry["success_probability"], 0))
        entry.pop("export_files", None)
        entry.pop("qpy_indices", None)
        result.append(entry)
    return result


def validate_sources(entries, root, config, workers):
    used = min(workers, math.ceil(len(entries)/config.chunk_size))
    report("composition_validate_sources", entries=len(entries), workers=used)
    result = []
    tasks = ((part, root, config) for part in chunks(entries, config.chunk_size))
    for part in parallel_chunks(validate_source_chunk, tasks, used):
        result.extend(part)
    return result


def component_loop(source):
    return rus_loop_circuit(source["template_ops"], source["success_outcomes"],
                            source["failure_outcome_ops"], attempt=amplified_entry_circuit(source))


def recipe_circuit(entry, components, tol=1e-10):
    """Time-ordered native operations, with an independent retry loop per RUS."""
    circuit = QuantumCircuit(2, 1, name="target_rus_loop")
    unitary = np.eye(2, dtype=complex)
    cost, raw, probability, depth = 0., 0, 1., 0
    for token in entry["recipe"]:
        if token["kind"] == "clifford":
            gate = circuit_from_ops(token["ops"])
            circuit.compose(gate, [0], inplace=True)
            matrix = Operator(gate).data
        elif token["kind"] == "rus":
            source = components[token["source_entry_id"]]
            circuit.compose(component_loop(source), inplace=True)
            matrix = decode(source["unitary_matrix"])
            cost += source["effective_t_count"]
            raw += source["amplified_raw_t_count"]
            probability *= source["amplified_success_probability"]
            depth += 1
        else:
            raise ValueError("Unknown composition recipe token")
        unitary = matrix @ unitary
    error = phase_error(unitary, decode(entry["unitary_matrix"]))
    if error > tol:
        raise ValueError(f"Recipe does not implement stored unitary: error={error}")
    if (not math.isclose(cost, entry["expected_t_count"], rel_tol=tol, abs_tol=tol)
            or raw != entry["raw_t_count"] or depth != entry["class_depth"]
            or abs(probability-entry["success_probability"]) > tol):
        raise ValueError("Recipe costs, class depth or branch probabilities disagree")
    return circuit


def export_entry(entry, components, output, tol):
    entry = dict(entry)
    group = "axial" if entry["axial"] else "non_axial"
    folder = output / group
    folder.mkdir(exist_ok=True)
    stem = entry["entry_id"]
    circuit = recipe_circuit(entry, components, tol)
    reference = QuantumCircuit(1, name="target_unitary_reference")
    reference.unitary(decode(entry["unitary_matrix"]), [0])
    files = dict(target_rus_loop_qasm3=f"{group}/{stem}_target_rus_loop_qasm3.qasm",
                 qpy=f"{group}/{stem}_circuits.qpy", json=f"{group}/{stem}.json")
    write_and_check(circuit, output / files["target_rus_loop_qasm3"], "qasm3", tol)
    with (output / files["qpy"]).open("wb") as handle:
        qpy.dump([circuit, reference], handle)
    with (output / files["qpy"]).open("rb") as handle:
        loaded = qpy.load(handle)
    if len(loaded) != 2:
        raise ValueError("Invalid QPY bundle")
    for before, after in zip([circuit, reference], loaded):
        compare_circuits(before, after, tol)
    entry.update(export_files=files, export_group=group,
                 qpy_indices=dict(target_rus_loop=0, target_unitary_reference=1),
                 validation=dict(recipe_operator=True, additive_expected_t=True, local_recovery=True,
                                 qasm_readback=True, qpy_readback=True))
    write_json(output / files["json"], entry)
    return entry


def export_chunk(task):
    entries, components, output, tol = task
    return [export_entry(e, components, output, tol) for e in entries], os.getpid()


def export_component_chunk(task):
    entries, output, tol = task
    result = {}
    folder = output / "components"
    folder.mkdir(exist_ok=True)
    for i, source in entries:
        entry = dict(source)
        W = amplified_entry_circuit(entry)
        attempt = QuantumCircuit(2, 1)
        attempt.compose(W, inplace=True)
        attempt.measure(1, 0)
        files = {}
        for name, circuit in (("w_qasm2", W), ("attempt_qasm2", attempt)):
            files[name] = f"components/component_{i:05d}_{name}.qasm"
            write_and_check(circuit, output / files[name], "qasm2", tol)
        entry["export_files"] = files
        entry["exported_attempt_raw_t_count"] = entry["amplified_raw_t_count"]
        entry["exported_attempt_success_probability"] = entry["amplified_success_probability"]
        result[entry["entry_id"]] = entry
    return result


def export_database(output_dir, workers=0, chunk_size=32):
    output = Path(output_dir).resolve()
    if (output / "database.json").exists():
        raise FileExistsError("Completed composition dataset already exists")
    payload = json.loads((output / "analysis.json").read_text())
    entries, components = payload["entries"], payload["components"]
    tol = payload["stats"]["config"]["tol"]
    used = resolve_workers(workers, math.ceil(len(entries)/chunk_size))
    report("composition_export", entries=len(entries), workers=used)
    started, exported, pids = time.monotonic(), [], set()
    try:
        ids = sorted({t["source_entry_id"] for e in entries for t in e["recipe"] if t["kind"] == "rus"})
        tasks = ((part, output, tol) for part in chunks(list(enumerate(components[i] for i in ids)), chunk_size))
        exported_components = {}
        for part in parallel_chunks(export_component_chunk, tasks, resolve_workers(workers, max(1, math.ceil(len(ids)/chunk_size)))):
            exported_components.update(part)
        def tasks():
            for part in chunks(entries, chunk_size):
                ids = {t["source_entry_id"] for e in part for t in e["recipe"] if t["kind"] == "rus"}
                yield part, {i: components[i] for i in ids}, output, tol
        for part, pid in parallel_chunks(export_chunk, tasks(), used):
            exported.extend(part)
            pids.add(pid)
    except Exception as exc:
        write_json(output / "export_error.json", dict(error=str(exc), export_ok=False))
        raise
    stats = dict(payload["stats"], export_ok=True, export_workers_used=used, export_worker_pids=sorted(pids),
                 validation_failures=0, files_readback_verified=2*len(exported)+2*len(exported_components),
                 circuits_readback_verified=3*len(exported)+2*len(exported_components),
                 export_seconds=time.monotonic()-started)
    fields = ["entry_id", "composition_class", "axial", "axial_angle", "raw_t_count", "success_probability", "expected_t_count", "effective_t_count"]
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(exported)
    with (output / "validation.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["entry_id", "kind", "file", "read_function", "status"])
        for e in [*exported, *exported_components.values()]:
            for kind, path in e["export_files"].items():
                if kind != "json":
                    function = "qpy" if kind == "qpy" else ("qasm2" if kind.endswith("qasm2") else "qasm3")
                    writer.writerow([e["entry_id"], kind, path, f"qiskit.{function}.load", "passed"])
    write_json(output / "summary.json", stats)
    write_json(output / "database.json", dict(payload, entries=exported, components=exported_components, stats=stats))
    (output / "export_error.json").unlink(missing_ok=True)
    report("composition_export_complete", entries=len(exported), output=str(output))
    return stats
