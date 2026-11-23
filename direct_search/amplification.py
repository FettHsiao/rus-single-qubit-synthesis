"""Executable oblivious amplitude amplification, RUS section 3.3."""
from __future__ import annotations

import math

from .circuits import ANC, circuit_from_template


def best_amplified_expected_t(raw_t_count, p_success, max_j=10, max_raw_t=None):
    """Minimize (2j+1)T / sin²((2j+1)asin(sqrt(p))) including j=0.

    max_raw_t optionally caps native T gates in one amplified attempt. Equal
    costs prefer fewer rounds. The lower bound (2j+1)T permits exact pruning.
    """
    if (not isinstance(raw_t_count, int) or raw_t_count < 0
            or not isinstance(max_j, int) or max_j < 0
            or not math.isfinite(p_success) or not 0 < p_success <= 1):
        raise ValueError("Require T >= 0, j >= 0 and 0 < probability <= 1")
    if max_raw_t is not None:
        if not isinstance(max_raw_t, int) or max_raw_t < raw_t_count:
            raise ValueError("max_raw_t must be an integer >= the original attempt T count")
        if raw_t_count:
            max_j = min(max_j, (max_raw_t // raw_t_count - 1) // 2)
    best = (raw_t_count / p_success, 0, p_success)
    if p_success >= 1 / 3 or raw_t_count == 0:
        return best
    theta = math.asin(math.sqrt(p_success))
    for j in range(1, max_j + 1):
        # Even probability one cannot improve the current best above this cost.
        if (2 * j + 1) * raw_t_count >= best[0]:
            break
        p = math.sin((2 * j + 1) * theta) ** 2
        if p > 0 and (2 * j + 1) * raw_t_count / p < best[0]:
            best = ((2 * j + 1) * raw_t_count / p, j, p)
    return best


def amplification_metadata(raw_t, probability, max_j=10, max_raw_t=None):
    """Return the selected rounds, probability, expected cost and variances."""
    cost, rounds, p = best_amplified_expected_t(raw_t, probability, max_j, max_raw_t)
    attempt_t = (2 * rounds + 1) * raw_t
    return {
        "effective_t_count": cost,
        "amplification_rounds": rounds,
        "amplified_success_probability": p,
        "amplified_raw_t_count": attempt_t,
        "expected_attempts": 1 / probability,
        "attempt_variance": (1 - probability) / probability ** 2,
        "t_count_variance": raw_t ** 2 * (1 - probability) / probability ** 2,
        "amplified_expected_attempts": 1 / p,
        "amplified_t_count_variance": attempt_t ** 2 * (1 - p) / p ** 2,
    }


def amplify_circuit(attempt, rounds, success_outcome="0"):
    """Return (-W S_input W† S_success)^j W for a single ancilla.

    Both rank-two reflections are Z on the ancilla, up to a global sign.
    Thus either success label is supported without changing the measured label.
    """
    if (rounds < 0 or int(rounds) != rounds or success_outcome not in {"0", "1"}
            or attempt.num_qubits != 2 or attempt.num_clbits):
        raise ValueError("Expected a two-qubit unitary attempt, integer j >= 0 and success 0/1")
    qc = attempt.copy()
    inverse = attempt.inverse()
    for _ in range(rounds):
        qc.z(ANC)  # S_success
        qc.compose(inverse, inplace=True)
        qc.z(ANC)  # S_input, initial ancilla |0>
        qc.compose(attempt, inplace=True)
    if success_outcome == "0":
        qc.global_phase += rounds * math.pi
    qc.name = f"W_amp_{rounds}"
    return qc


def amplified_entry_circuit(entry):
    return amplify_circuit(circuit_from_template(entry["template_ops"]),
                           entry.get("amplification_rounds", 0), entry["success_outcomes"][0])
