"""Reproducible same-input speed measurements, with semantic equality checks."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .dsa import SearchConfig, build_database
from .hardware import hardware_profile
from .main import parse_job


def benchmark(config, worker_counts=(1, 8, 24), include_qiskit=False):
    variants = ([replace(config, workers=1, backend="qiskit")] if include_qiskit else [])
    variants += [replace(config, workers=n, backend="numpy") for n in worker_counts]
    baseline = baseline_stats = None
    runs = []
    for variant in variants:
        db, stats = build_database(variant)
        if baseline is None:
            baseline, baseline_stats = db, stats
        if db.keys() != baseline.keys():
            raise AssertionError("Benchmark configurations produced different unitaries")
        for field in ("total_examined", "accepted_candidates", "rejections", "search_complete"):
            if stats[field] != baseline_stats[field]:
                raise AssertionError(f"Benchmark configurations disagree on {field}")
        for key, entry in db.items():
            reference = baseline[key]
            if (entry["template_ops"] != reference["template_ops"]
                    or abs(entry["expected_t_count"] - reference["expected_t_count"]) > 1e-8):
                raise AssertionError("Benchmark configurations selected different circuits")
        runs.append({"backend": variant.backend, "workers_requested": variant.workers,
                     **{k: stats[k] for k in ("workers_used", "elapsed_seconds", "candidates_per_second",
                                             "total_examined", "accepted_candidates", "unique_unitaries", "search_complete")},
                     "same_results": True})
    for run in runs:
        run["speedup_vs_first_run"] = runs[0]["elapsed_seconds"] / run["elapsed_seconds"]
    return {"hardware": hardware_profile(), "config": baseline_stats["config"],
            "timing_scope": "pool preparation + worker startup + direct search + exact deduplication; excludes export",
            "repetitions": 1, "runs": runs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-raw-t", type=int, default=6)
    parser.add_argument("--max-examined", type=int)
    parser.add_argument("--job", type=parse_job)
    parser.add_argument("--workers", default="1,8,24")
    parser.add_argument("--include-qiskit", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = SearchConfig(max_raw_t=args.max_raw_t, max_examined=args.max_examined,
                          job=args.job)
    result = benchmark(config, [int(n) for n in args.workers.split(",")], args.include_qiskit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["runs"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
