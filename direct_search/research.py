"""Section-5 dataset statistics and section-6.1 resource estimates.

These summarize the generated search, not the paper's cluster run or KMM
comparison data. Section 6.2 composition and approximate lookup are excluded.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def dataset_statistics(entries):
    entries = list(entries)

    def distribution(field):
        values = np.array([e.get(field, e["expected_t_count"]) for e in entries], dtype=float)
        if not len(values):
            return {"count": 0, "min": None, "median": None, "max": None}
        return {"count": len(values), "min": float(values.min()), "median": float(np.median(values)),
                "max": float(values.max())}

    bins = np.arange(0, 105, 5)
    before = np.array([e["expected_t_count"] for e in entries])
    after = np.array([e.get("effective_t_count", e["expected_t_count"]) for e in entries])
    return {
        "amplified_entries": sum(e.get("amplification_rounds", 0) > 0 for e in entries),
        "raw_t_count": distribution("raw_t_count"),
        "success_probability": distribution("success_probability"),
        "expected_t_count": distribution("expected_t_count"),
        "effective_t_count": distribution("effective_t_count"),
        "expected_t_histogram": {"edges": bins.tolist(),
            "before": np.histogram(before, bins)[0].tolist(), "after": np.histogram(after, bins)[0].tolist(),
            "before_over_100": int(np.sum(before > 100)), "after_over_100": int(np.sum(after > 100))},
    }


def v3_cost_estimate(epsilon, expected_v3_t=5.26):
    """Section 6.1 Eq. (20), a typical asymptotic estimate, not synthesis."""
    if not math.isfinite(epsilon) or not 0 < epsilon < 1 or not math.isfinite(expected_v3_t) or expected_v3_t <= 0:
        raise ValueError("Require 0 < epsilon < 1 and positive finite V3 cost")
    gates = -3 * math.log(epsilon) / math.log(5)
    return {"epsilon": epsilon, "typical_v3_count": gates,
            "estimated_expected_t_count": expected_v3_t * gates,
            "estimate_only": True, "source": "1311.1074v2 section 6.1, equation (20)"}


def plot_database(entries, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    entries = list(entries)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].hist([e["raw_t_count"] for e in entries], bins=np.arange(0.5, max([e["raw_t_count"] for e in entries], default=1) + 1.5))
    axes[0].set_xlabel("Raw T count")
    axes[1].hist([e["expected_t_count"] for e in entries], bins=np.arange(0, 105, 5), alpha=.6, label="Before")
    axes[1].hist([e.get("effective_t_count", e["expected_t_count"]) for e in entries], bins=np.arange(0, 105, 5), alpha=.6, label="After amplification")
    axes[1].set_xlabel("Expected T count (0–100)")
    axes[1].legend()
    axes[2].hist([e["success_probability"] for e in entries], bins=np.linspace(0, 1, 21))
    axes[2].set_xlabel("Success probability")
    for ax in axes:
        ax.set_ylabel("Retained circuits")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(destination / f"direct_search_statistics.{suffix}", dpi=180)
    plt.close(fig)
