"""Executable nested one-input WK13 gearbox and section-6.2.2 hybrid lookup.

A gearbox maps a Z angle a to g(a)=2 atan(tan(a/2)^2). Its successful
Kraus operator is cos(a/2)^2 I - i sin(a/2)^2 Z; its failure is S† up
to phase and scale, corrected by S. Every child RUS retries locally,
so measured subroutines can act coherently on an entangled input.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit, qpy
from direct_search.hardware import resolve_workers
from direct_search.validate_qasm import compare_circuits
from reduction.common import decode, encode
from reduction.pipeline import chunks, parallel_chunks, report, write_and_check, write_json
from .export import recipe_circuit
from .geometry import distance, rotation
from .lookup import Lookup


def gearbox_angle(angle):
    return 2*math.atan(math.tan(angle/2)**2)


def gearbox_probability(angle):
    return math.cos(angle/2)**4 + math.sin(angle/2)**4


def gearbox_cost(angle, cost, depth):
    probabilities = []
    for _ in range(depth):
        p = gearbox_probability(angle)
        probabilities.append(p)
        cost = 2*cost/p
        angle = gearbox_angle(angle)
    return angle, cost, probabilities


def append_mapped(source, target, qmap, cmap):
    """Copy nested control flow using explicit bit maps at every block level.

    Avoid Qiskit 2.4 compose remapping a nested condition into its parent's
    classical register while leaving that block's own bits unchanged.
    """
    target.global_phase += source.global_phase
    for instruction in source.data:
        op = instruction.operation
        qs = [qmap[source.find_bit(q).index] for q in instruction.qubits]
        cs = [cmap[source.find_bit(c).index] for c in instruction.clbits]
        if op.name in {"while_loop", "if_else"}:
            bit, value = op.condition
            condition = (cmap[source.find_bit(bit).index], value)
            blocks = []
            for block in op.blocks:
                mapped = QuantumCircuit(target.qubits, target.clbits)
                append_mapped(block, mapped, qs, cs)
                blocks.append(mapped)
            if op.name == "while_loop":
                target.while_loop(condition, blocks[0], target.qubits, target.clbits)
            else:
                target.if_else(condition, blocks[0], blocks[1] if len(blocks)>1 else None,
                               target.qubits, target.clbits)
        else:
            target.append(op, qs, cs)


def gearbox_circuit(seed, components, depth):
    """Implement Rz(g^depth(seed.angle)) using native gates and nested retries.

    Each gearbox level needs a fresh control ancilla and measurement bit;
    the bottom source-RUS subroutine uses one reusable scratch ancilla/bit.
    No arbitrary-angle primitive is used in the executable circuit.
    """
    if not isinstance(depth, int) or depth < 0:
        raise ValueError("Gearbox depth must be a nonnegative integer")
    base = recipe_circuit(seed, components)
    angle = seed["axial_angle"]
    from reduction.common import phase_error
    if phase_error(decode(seed["unitary_matrix"]), rotation(angle)) > 1e-10:
        raise ValueError("Gearbox seed must be a diagonal axial representative")

    def build(level):
        if not level:
            return base
        child = build(level-1)
        qc = QuantumCircuit(level+2, level+1, name=f"gearbox_{level}")
        # q0=data, q1=outer control, q2...=child's controls/scratch.
        qc.reset(1)
        qc.x(1)
        qc.measure(1, 0)  # bootstrap failure flag, deterministic
        # Build the body explicitly. Qiskit 2.4's compose inside an active
        # builder scope cannot remap previously unused nested classical bits.
        body = QuantumCircuit(level+2, level+1)
        child_q, child_c = body.qubits[1:], body.clbits[1:]
        body.reset(1)
        body.h(1)
        append_mapped(child, body, child_q, child_c)
        body.h(1)
        body.cz(1, 0)
        body.sdg(1)  # controlled(-i Z)
        # Inverse Rx(child_angle) = H X Rz(child_angle) X H.
        body.h(1)
        body.x(1)
        append_mapped(child, body, child_q, child_c)
        body.x(1)
        body.h(1)
        body.measure(1, 0)
        with body.if_test((body.clbits[0], 1)):
            body.s(0)  # inverse of the failure S†
        qc.while_loop((qc.clbits[0], 1), body, qc.qubits, qc.clbits)
        return qc
    return build(depth)


def candidate_chunk(task):
    entries, theta, coarse_angle, epsilon, max_depth = task
    best = None
    for entry in entries:
        angle, cost = entry["axial_angle"], entry["expected_t_count"]
        probabilities = []
        for depth in range(max_depth+1):
            for sign in (1, -1):
                error = distance(rotation(coarse_angle+sign*angle), rotation(theta))
                if error <= epsilon+1e-14:
                    order = (round(cost, 10), error, depth, entry["entry_id"], sign)
                    if best is None or order < best[0]:
                        best = (order, entry, depth, sign, angle, cost, list(probabilities))
            p = gearbox_probability(angle)
            probabilities.append(p)
            cost = 2*cost/p
            angle = gearbox_angle(angle)
    return best, os.getpid()


def hybrid_database(input_dir, output_dir, theta, epsilon, coarse_epsilon=1e-6,
                    max_depth=6, workers=0, chunk_size=128):
    if (not all(math.isfinite(x) for x in (theta, epsilon, coarse_epsilon))
            or not 0 < epsilon <= coarse_epsilon <= 1 or max_depth < 0 or chunk_size < 1):
        raise ValueError("Require finite angle, 0 < epsilon <= coarse_epsilon <= 1, depth >= 0, chunk size > 0")
    root, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Hybrid output must be new or empty")
    payload = json.loads((root / "database.json").read_text())
    if payload.get("algorithm") != "composition" or payload["stats"]["config"]["mode"] != "axial" or not payload["stats"].get("export_ok"):
        raise ValueError("Hybrid synthesis requires a completed axial composition dataset")
    from output_paths import register_output
    register_output(output, "hybrid_gearbox")
    coarse = Lookup(payload).query(rotation(theta), coarse_epsilon)
    if coarse is None:
        write_json(output / "lookup.json", dict(found=False, reason="No coarse approximation within coarse_epsilon"))
        return None
    coarse_entry = coarse["entry"]
    U = decode(coarse_entry["unitary_matrix"])
    coarse_angle = float(np.angle(U[1,1]/U[0,0]))
    entries, components = payload["entries"], payload["components"]
    used = resolve_workers(workers, math.ceil(len(entries)/chunk_size))
    report("hybrid_search", seeds=len(entries), max_gearbox_depth=max_depth, workers=used)
    tasks = ((part, theta, coarse_angle, epsilon, max_depth) for part in chunks(entries, chunk_size))
    winner, pids = None, set()
    for result, pid in parallel_chunks(candidate_chunk, tasks, used):
        pids.add(pid)
        if result is not None and (winner is None or result[0] < winner[0]):
            winner = result
    if winner is None:
        write_json(output / "lookup.json", dict(found=False, reason="No seed/depth meets epsilon; enlarge the dataset or max-gearbox-depth",
                   coarse_distance=coarse["distance"], epsilon=epsilon, max_gearbox_depth=max_depth))
        return None
    _, seed, depth, sign, correction_angle, correction_cost, probabilities = winner
    correction = gearbox_circuit(seed, components, depth)
    circuit = QuantumCircuit(correction.num_qubits, correction.num_clbits, name="target_rus_loop")
    coarse_circuit = recipe_circuit(coarse_entry, components)
    append_mapped(coarse_circuit, circuit, [circuit.qubits[0], circuit.qubits[-1]], [circuit.clbits[-1]])
    if sign < 0:
        circuit.x(0)
    append_mapped(correction, circuit, circuit.qubits, circuit.clbits)
    if sign < 0:
        circuit.x(0)
    actual = rotation(coarse_angle+sign*correction_angle)
    error = distance(actual, rotation(theta))
    if error > epsilon+1e-14:
        raise ValueError("Hybrid circuit misses requested epsilon")
    entry = dict(entry_id="hybrid_0000000", unitary_matrix=encode(actual), axial=True,
                 axial_angle=float(np.angle(actual[1,1]/actual[0,0])), expected_t_count=coarse_entry["expected_t_count"]+correction_cost,
                 effective_t_count=coarse_entry["expected_t_count"]+correction_cost,
                 raw_t_count=coarse_entry["raw_t_count"]+2**depth*seed["raw_t_count"],
                 success_probability=probabilities[-1] if probabilities else 1.,
                 success_probability_meaning="outer gearbox attempt success; all component RUS retry locally",
                 raw_t_count_meaning="syntactic leaf T count of one pass through nested loop bodies",
                 cost_model="E_gearbox=2 E_child/(cos(a/2)^4+sin(a/2)^4); total=E_coarse+E_gearbox",
                 gearbox_depth=depth, gearbox_sign=sign, gearbox_probabilities=probabilities,
                 coarse_entry=coarse_entry, seed_entry=seed, distance=error, epsilon=epsilon,
                 target_unitary_matrix=encode(rotation(theta)), circuit_synthesized=True)
    reference = QuantumCircuit(1, name="target_unitary_reference")
    reference.unitary(actual, [0])
    folder = output / "axial"
    folder.mkdir()
    files = dict(target_rus_loop_qasm3="axial/hybrid_0000000_target_rus_loop_qasm3.qasm",
                 qpy="axial/hybrid_0000000_circuits.qpy", json="axial/hybrid_0000000.json")
    write_and_check(circuit, output/files["target_rus_loop_qasm3"], "qasm3", 1e-10)
    with (output/files["qpy"]).open("wb") as handle:
        qpy.dump([circuit, reference], handle)
    with (output/files["qpy"]).open("rb") as handle:
        loaded = qpy.load(handle)
    if len(loaded) != 2:
        raise ValueError("Invalid hybrid QPY bundle")
    for a, b in zip([circuit, reference], loaded):
        compare_circuits(a, b, 1e-10)
    entry.update(export_files=files, qpy_indices=dict(target_rus_loop=0, target_unitary_reference=1),
                 validation=dict(qasm_readback=True, qpy_readback=True, analytic_gearbox_kraus=True))
    stats = dict(export_ok=True, output_entries=1, workers_used=used, worker_pids=sorted(pids),
                 max_gearbox_depth=max_depth, coarse_epsilon=coarse_epsilon, epsilon=epsilon,
                 method="section_6.2.2_hybrid_with_nested_WK13_one_input_gearboxes",
                 optimization_scope="minimum correction cost for the selected minimum-cost coarse approximation",
                 circuit_synthesized=True, num_qubits=circuit.num_qubits, num_clbits=circuit.num_clbits,
                 validation_failures=0, files_readback_verified=2, circuits_readback_verified=3)
    # Recipes are self-contained; source export paths are not needed in this file.
    components = {k: {f:v for f,v in e.items() if f != "export_files"} for k,e in components.items()}
    write_json(output/files["json"], entry)
    write_json(output/"summary.json", stats)
    write_json(output/"database.json", dict(schema_version=1, algorithm="hybrid_gearbox", entries=[entry], components=components,
                                          qubit_order=dict(data=0, gearbox_ancillas=list(range(1, depth+1)), rus_scratch=depth+1), stats=stats))
    write_json(output/"lookup.json", dict(found=True, distance=error, epsilon=epsilon, expected_t_count=entry["expected_t_count"], circuit_synthesized=True))
    import csv
    with (output/"summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        fields = ["entry_id", "axial", "axial_angle", "raw_t_count", "success_probability", "expected_t_count", "effective_t_count", "distance", "gearbox_depth"]
        writer.writerow(fields)
        writer.writerow([entry[k] for k in fields])
    return entry
