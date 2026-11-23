"""Section 4 direct search and independent Qiskit branch validation."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from .circuits import (
    SLOTS, TEMPLATE_CONVENTION, TOL, allowed_sets, clifford_group, core_pool,
    equivalent, matrix_key, position_coverage, remove_global_phase, segment_pool,
)


@dataclass(frozen=True)
class SearchConfig:
    max_raw_t: int = 6
    clifford_sets: str = "verified"
    dedup_mode: str = "exact"
    require_recovery_branch: bool = True
    strict_nondeterministic: bool = True
    tol: float = TOL
    max_examined: int | None = None
    job: tuple[int, int, int, int] | None = None
    amp_max_j: int = 0
    workers: int = 0
    chunk_size: int = 8192
    batch_size: int = 2048
    backend: str = "numpy"

    def __post_init__(self):
        if self.workers < 0 or self.chunk_size <= 0 or self.batch_size <= 0 or self.amp_max_j < 0:
            raise ValueError("workers and amp_max_j must be >= 0; chunk_size and batch_size must be > 0")
        if self.backend not in {"numpy", "qiskit"}:
            raise ValueError("backend must be numpy or qiskit")
        if self.max_raw_t < 0:
            raise ValueError("max_raw_t must be nonnegative")
        if self.max_examined is not None and self.max_examined < 0:
            raise ValueError("max_examined must be nonnegative")
        if not math.isfinite(self.tol) or not 0 < self.tol < 1e-3:
            raise ValueError("tol must be finite and between 0 and 1e-3")
        if self.clifford_sets not in {"project", "fig4", "paper", "verified"}:
            raise ValueError("clifford_sets must be project, fig4, paper or verified")
        if self.dedup_mode != "exact":
            raise ValueError("dedup_mode must be exact; only the same U may be deduplicated")
        if self.job is not None and (
            len(self.job) != 4 or any(t < 0 for t in self.job)
            or sum(self.job) > self.max_raw_t
        ):
            raise ValueError("job must contain four nonnegative counts summing to <= max_raw_t")


@dataclass
class Analysis:
    accepted: bool
    reason: str = ""
    unitary: np.ndarray | None = None
    probability: float = 0.0
    success_outcomes: list[str] = field(default_factory=list)
    failure_ops: dict[str, list[str]] = field(default_factory=dict)
    failure_labels: dict[str, str] = field(default_factory=dict)
    branch_probabilities: dict[str, float] = field(default_factory=dict)
    zero_outcomes: list[str] = field(default_factory=list)


def analyze_w(W, config: SearchConfig | None = None) -> Analysis:
    config = config or SearchConfig()
    tol = config.tol
    W = np.asarray(W, dtype=complex)
    if W.shape != (4, 4) or not np.all(np.isfinite(W)):
        return Analysis(False, "invalid_matrix")
    if not np.allclose(W.conj().T @ W, np.eye(4), rtol=0, atol=tol):
        return Analysis(False, "not_unitary")
    result = Analysis(False)
    successes = []
    for outcome in (0, 1):
        label = str(outcome)
        # Qiskit orders basis vectors as |ancilla,data>; input ancilla is |0>.
        block = W[2 * outcome:2 * outcome + 2, :2]
        gram = block.conj().T @ block
        p = float(np.trace(gram).real / 2)
        result.branch_probabilities[label] = p
        if p <= tol:
            result.zero_outcomes.append(label)
            continue
        if not np.allclose(gram, p * np.eye(2), rtol=0, atol=tol):
            return Analysis(False, "state_dependent_probability")
        U = remove_global_phase(block / math.sqrt(p), tol)
        if not np.allclose(U.conj().T @ U, np.eye(2), rtol=0, atol=tol):
            return Analysis(False, "branch_not_unitary")
        recovery = next((c for c in clifford_group()
                         if equivalent(U, c.matrix, tol)), None)
        if recovery is None:
            successes.append((label, U, p))
        else:
            result.failure_labels[label] = recovery.label
            result.failure_ops[label] = list(recovery.ops)
    if not successes:
        return Analysis(False, "no_nonclifford_success")
    if any(not equivalent(U, successes[0][1], tol) for _, U, _ in successes[1:]):
        return Analysis(False, "different_success_unitaries")
    if abs(sum(result.branch_probabilities.values()) - 1) > tol:
        return Analysis(False, "probabilities_not_normalized")
    result.probability = min(1.0, sum(p for _, _, p in successes))
    if config.require_recovery_branch and not result.failure_ops:
        return Analysis(False, "no_recovery_branch")
    if config.strict_nondeterministic and (
        result.zero_outcomes or result.probability >= 1 - tol
    ):
        return Analysis(False, "deterministic_or_zero_branch")
    result.accepted = True
    result.unitary = successes[0][1]
    result.success_outcomes = [label for label, _, _ in successes]
    return result


def partitions(max_raw_t):
    # Keep the original project's partition traversal order.
    for a0 in range(max_raw_t + 1):
        for a1 in range(max_raw_t - a0 + 1):
            for a2 in range(max_raw_t - a0 - a1 + 1):
                for d0 in range(max_raw_t - a0 - a1 - a2 + 1):
                    if a0 + a1 + a2 + d0:
                        yield a0, a1, a2, d0


def unitary_identity(U, mode="exact", tol=TOL):
    """Key only the physical U up to global phase; classify by independent L U R separately."""
    from .axial import z_rotation_metadata
    if mode != "exact":
        raise ValueError("mode must be exact")
    axial = z_rotation_metadata(U, tol)
    return matrix_key(U), axial["angle"]


def encode_matrix(matrix):
    return [[[float(z.real), float(z.imag)] for z in row] for row in matrix]


def decode_matrix(value):
    values = np.asarray(value, dtype=float)
    return values[..., 0] + 1j * values[..., 1]


def prepare_search(config):
    jobs = [config.job] if config.job is not None else list(partitions(config.max_raw_t))
    allowed = allowed_sets(config.clifford_sets)
    # A single partition need not allocate unrelated high-T cores or segments.
    needed_t = max((max(job) for job in jobs), default=0)
    cores = core_pool(needed_t)
    segments = {}
    for slot_index, slots in enumerate(SLOTS):
        for t in sorted({job[slot_index] for job in jobs}):
            segments[slot_index, t] = segment_pool(slots, cores[t], allowed)
    total = sum(math.prod(len(segments[i, t]) for i, t in enumerate(job)) for job in jobs)
    stats = {
        "config": asdict(config), "planned_candidates": total,
        "planned_partitions": len(jobs),
        "core_counts": {str(t): len(pool) for t, pool in cores.items()},
        "clifford_set_sizes": {slot: len(pool) for slot, pool in allowed.items()},
        "clifford_position_sets": {slot: [{"label": seq.label, "ops": list(seq.ops)} for seq in pool]
                                   for slot, pool in allowed.items()},
        "position_coverage": position_coverage(allowed),
        "segment_counts": {f"{i}:{t}": len(pool) for (i, t), pool in segments.items()},
        "segment_deduplication": "global_phase_at_fixed_t_count",
        "matrix_backend": config.backend,
        "segment_matrix_backend": "qiskit.quantum_info.Operator(QuantumCircuit)",
        "template_convention": TEMPLATE_CONVENTION,
        "clifford_labels": "1206.3223v1 Appendix A: G0...G23",
        "position_policy": {"paper": "paper_matrix_products", "verified": "verified_clifford_cosets"}.get(
            config.clifford_sets, "inherited_physical_wrappers"),
    }
    return jobs, segments, stats


def build_database(config: SearchConfig | None = None,
                   progress: Callable[[dict], None] | None = None):
    from .parallel import run_search
    return run_search(config or SearchConfig(), progress)
