"""Bounded, deterministic process search over mixed-radix candidate ranges.

Workers receive segment matrices once. Only locally best entries cross process
boundaries; the coordinator merges them in enumeration order and owns output.
"""
from __future__ import annotations

import math
import multiprocessing
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from execution import single_thread_blas
from itertools import chain, islice

import numpy as np
from qiskit.quantum_info import Operator

from .amplification import amplification_metadata
from .axial import z_rotation_metadata
from .circuits import (TEMPLATE_CONVENTION, circuit_from_template, matrix_key,
                       clifford_matrices, phase_normalize_batch)
from .dsa import analyze_w, encode_matrix, prepare_search
from .hardware import hardware_profile

_POOLS = _CONFIG = None
_IDENTITIES = {}


def candidate_matrices(arrays, indices):
    """Batched kron(A2,I) CZ kron(A1,D0) CZ kron(A0,I)."""
    a0, a1, a2, d0 = [pool[index] for pool, index in zip(arrays, indices)]
    identity = np.broadcast_to(np.eye(2), a0.shape)

    def kron(a, b):
        return np.einsum("nab,ncd->nacbd", a, b).reshape(-1, 4, 4)

    cz = np.array([1, 1, 1, -1])
    return (kron(a2, identity) * cz) @ (kron(a1, d0) * cz) @ kron(a0, identity)


def analyze_batch(matrices, config):
    """Vectorized counterpart to analyze_w, including rejection accounting."""
    tol = config.tol
    W = np.asarray(matrices)
    n = len(W)
    blocks = W[:, :, :2].reshape(n, 2, 2, 2)
    gram = blocks.conj().swapaxes(-1, -2) @ blocks
    p = np.trace(gram, axis1=-2, axis2=-1).real / 2
    zero = p <= tol
    U = phase_normalize_batch(blocks / np.sqrt(np.maximum(p, tol))[..., None, None], tol)
    # Normalize the cached Cliffords with the same phase convention.
    group = phase_normalize_batch(clifford_matrices(), tol)
    distance = np.max(np.abs(U[:, :, None] - group[None, None]), axis=(-1, -2))
    matches = distance <= tol
    cliff = np.where(np.any(matches, axis=-1), np.argmax(matches, axis=-1), -1)
    noncliff = (~zero) & (cliff < 0)
    successes = noncliff.sum(axis=1)
    probabilities = np.minimum(1.0, np.sum(np.where(noncliff, p, 0), axis=1))
    reasons = np.full(n, "", dtype=object)

    def reject(mask, reason):
        reasons[(reasons == "") & mask] = reason

    reject(~np.all(np.isfinite(W), axis=(1, 2)), "invalid_matrix")
    reject(np.max(np.abs(W.conj().swapaxes(-1, -2) @ W - np.eye(4)), axis=(1, 2)) > tol,
           "not_unitary")
    for outcome in (0, 1):
        reject((~zero[:, outcome]) & (np.max(np.abs(gram[:, outcome] - p[:, outcome, None, None] * np.eye(2)),
                                                   axis=(1, 2)) > tol), "state_dependent_probability")
        reject((~zero[:, outcome]) & (np.max(np.abs(U[:, outcome].conj().swapaxes(-1, -2) @ U[:, outcome]
                                                   - np.eye(2)), axis=(1, 2)) > tol), "branch_not_unitary")
    reject(successes == 0, "no_nonclifford_success")
    reject((successes == 2) & (np.max(np.abs(U[:, 0] - U[:, 1]), axis=(1, 2)) > tol),
           "different_success_unitaries")
    reject(np.abs(p.sum(axis=1) - 1) > tol, "probabilities_not_normalized")
    if config.require_recovery_branch:
        reject(~np.any((~zero) & (cliff >= 0), axis=1), "no_recovery_branch")
    if config.strict_nondeterministic:
        reject(np.any(zero, axis=1) | (probabilities >= 1 - tol), "deterministic_or_zero_branch")
    reference = U[np.arange(n), np.argmax(noncliff, axis=1)]
    return reasons, reference, probabilities


def entry_rank(entry):
    # A total ordering makes chunk size and completion order irrelevant. Round
    # only the comparison cost; all reported probabilities remain unrounded.
    return (round(entry["expected_t_count"], 10), entry["raw_t_count"], entry["candidate_ordinal"])


def _initialize(pools, config):
    global _POOLS, _CONFIG, _IDENTITIES
    _POOLS, _CONFIG, _IDENTITIES = pools, config, {}
    clifford_matrices()


def _search_chunk(task):
    job, start, stop, offset = task
    config = _CONFIG
    pools = [_POOLS[i, t] for i, t in enumerate(job)]
    arrays = [pool[0] for pool in pools]
    sizes = tuple(len(array) for array in arrays)
    db, rejects = {}, Counter()
    accepted = 0
    for begin in range(start, stop, config.batch_size):
        end = min(stop, begin + config.batch_size)
        indices = np.unravel_index(np.arange(begin, end), sizes)
        if config.backend == "numpy":
            matrices = candidate_matrices(arrays, indices)
            reasons, unitaries, probabilities = analyze_batch(matrices, config)
        else:
            matrices = np.array([Operator(circuit_from_template({k: v for pool, index in zip(pools, position)
                                  for k, v in pool[2][index].items()})).data for position in zip(*indices)])
            analyses = [analyze_w(W, config) for W in matrices]
            reasons = np.array([a.reason for a in analyses], dtype=object)
            unitaries = np.array([a.unitary if a.accepted else np.eye(2) for a in analyses])
            probabilities = np.array([a.probability for a in analyses])
        rejects.update(reason for reason in reasons if reason)
        hits = np.flatnonzero(reasons == "")
        accepted += len(hits)
        if not len(hits):
            continue
        # Only phase-identical U values compete, using the direct cost T / p.
        values = np.round(np.stack((unitaries[hits].real, unitaries[hits].imag), axis=-1).reshape(len(hits), 8), 10)
        _, classes = np.unique(values, axis=0, return_inverse=True)
        costs = sum(job) / probabilities[hits]
        ordered = np.lexsort((hits, np.round(costs, 10), classes))
        unique = ordered[np.r_[True, np.diff(classes[ordered]) != 0]]
        for h in hits[unique]:
            U = unitaries[h]
            key = matrix_key(U)
            axial = _IDENTITIES.get(key)
            if axial is None:
                axial = z_rotation_metadata(U, config.tol)
                if len(_IDENTITIES) >= 65536:
                    _IDENTITIES.clear()
                _IDENTITIES[key] = axial
            ordinal = offset + begin + int(h)
            rank = (round(sum(job) / probabilities[h], 10), sum(job), ordinal)
            if key in db and entry_rank(db[key]) <= rank:
                continue
            # Independent scalar analysis for every prospective retained entry.
            analysis = analyze_w(matrices[h], config)
            if not analysis.accepted:
                raise RuntimeError("Batch and scalar RUS acceptance disagree")
            position = [index[h] for index in indices]
            db[key] = {
                "template_convention": TEMPLATE_CONVENTION,
                "unitary_key": key, "unitary_exact_key": key,
                "deduplication_method": "exact",
                "unitary_matrix": encode_matrix(U), "z_rotation": axial,
                "raw_t_count": sum(job), "success_probability": analysis.probability,
                "expected_t_count": sum(job) / analysis.probability,
                "candidate_ordinal": ordinal,
                "success_outcomes": analysis.success_outcomes,
                "failure_outcomes": analysis.failure_labels,
                "failure_outcome_ops": analysis.failure_ops,
                "branch_probabilities": analysis.branch_probabilities,
                "zero_probability_outcomes": analysis.zero_outcomes,
                "template_labels": {k: v for pool, index in zip(pools, position) for k, v in pool[1][index].items()},
                "template_ops": {k: v for pool, index in zip(pools, position) for k, v in pool[2][index].items()},
                "fig3_t_partition": list(job),
                "axial": axial["angle"] is not None, "axial_angle": axial["angle"],
            }
    return stop - start, accepted, rejects, db, job


def search_tasks(jobs, segments, config):
    offset = 0
    for job in jobs:
        size = math.prod(len(segments[i, t]) for i, t in enumerate(job))
        allowed = size if config.max_examined is None else max(0, min(size, config.max_examined - offset))
        for start in range(0, allowed, config.chunk_size):
            yield job, start, min(allowed, start + config.chunk_size), offset
        offset += size
        if config.max_examined is not None and offset >= config.max_examined:
            break




def run_search(config, progress=None):
    started = time.monotonic()
    jobs, segments, stats = prepare_search(config)
    hardware = hardware_profile()
    tasks = iter(search_tasks(jobs, segments, config))
    # Do not start more processes than there are chunks to execute.
    requested = config.workers or hardware["recommended_workers"]
    initial = list(islice(tasks, requested))
    workers = min(requested, len(initial))
    tasks = iter(chain(initial, tasks))
    pools = {key: (np.array([s.sequence.matrix for s in pool]),
                   [s.labels for s in pool], [s.ops for s in pool]) for key, pool in segments.items()}
    stats.update(hardware=hardware, workers_used=workers, multiprocessing_start_method="spawn",
                 auto_worker_policy="all_available_cpus_with_affinity_quota_memory_limits",
                 worker_blas_threads=1 if workers > 1 else None,
                 chunk_size=config.chunk_size, batch_size=config.batch_size,
                 deduplication="same_unitary_up_to_global_phase",
                 axial_classification="L U R is diagonal for independent Cliffords L and R",
                 ranking="expected_t_count,raw_t_count,candidate_ordinal")
    if progress:
        progress({"phase": "prepared", **stats})
    db, rejects = {}, Counter()
    examined = accepted = completed_chunks = 0
    per_job = Counter()
    last_report = time.monotonic()

    def merge(result):
        nonlocal examined, accepted, completed_chunks, last_report
        count, successes, failures, entries, job = result
        examined += count
        accepted += successes
        completed_chunks += 1
        per_job[job] += count
        rejects.update(failures)
        for key, entry in entries.items():
            if key not in db or entry_rank(entry) < entry_rank(db[key]):
                db[key] = entry
        if progress and time.monotonic() - last_report >= 10:
            progress({"phase": "search", "examined": examined, "accepted": accepted,
                      "unique_unitaries": len(db), "job": job,
                      "planned_candidates": stats["planned_candidates"]})
            last_report = time.monotonic()

    if workers <= 1:
        _initialize(pools, config)
        for task in tasks:
            merge(_search_chunk(task))
    else:
        with single_thread_blas(), ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize, initargs=(pools, config)) as executor:
            pending = deque()
            for task in islice(tasks, 2 * workers):
                pending.append(executor.submit(_search_chunk, task))
            while pending:
                merge(pending.popleft().result())
                task = next(tasks, None)
                if task is not None:
                    pending.append(executor.submit(_search_chunk, task))
    # Optional section-3.3 estimates apply after direct winners have been chosen.
    for entry in db.values():
        entry.update(amplification_metadata(entry["raw_t_count"], entry["success_probability"], config.amp_max_j))
    elapsed = time.monotonic() - started
    completed = sum(per_job[job] == math.prod(len(segments[i, t]) for i, t in enumerate(job)) for job in jobs)
    stats.update(total_examined=examined, accepted_candidates=accepted, unique_unitaries=len(db),
                 rejections=dict(rejects), completed_partitions=completed, completed_chunks=completed_chunks,
                 search_complete=examined == stats["planned_candidates"], elapsed_seconds=elapsed,
                 candidates_per_second=examined / elapsed if elapsed else 0)
    return db, stats
