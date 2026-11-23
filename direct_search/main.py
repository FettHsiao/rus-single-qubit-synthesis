#!/usr/bin/env python3
"""Verified direct search with paired unamplified/amplified datasets."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from direct_search.dsa import SearchConfig, build_database, prepare_search
from direct_search.workflow import export_variants, variant_paths


def parse_job(value):
    try:
        counts = tuple(int(piece.strip()) for piece in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("job must be four integers: tA0,tA1,tA2,tD0") from exc
    if len(counts) != 4 or any(t < 0 for t in counts):
        raise argparse.ArgumentTypeError("job must be four nonnegative integers: tA0,tA1,tA2,tD0")
    return counts


def parser():
    cli = argparse.ArgumentParser(description="Generate Fig. 3 RUS circuits using Qiskit direct search.")
    cli.add_argument("--max-raw-t", "--max-t-count", type=int, default=6)
    cli.add_argument("--job", type=parse_job, help="Single T partition: tA0,tA1,tA2,tD0")
    cli.add_argument("--dedup-mode", choices=("exact",), default="exact",
                     help="Keep minimum direct expected T for the same U up to global phase")
    cli.add_argument("--require-recovery-branch", action=argparse.BooleanOptionalAction, default=True)
    cli.add_argument("--strict-nondeterministic", action=argparse.BooleanOptionalAction, default=True)
    cli.add_argument("--tol", type=float, default=1e-10)
    cli.add_argument("--amp-max-j", type=int, default=10,
                     help="Positive amplification search bound for the second dataset, including j=0")
    cli.add_argument("--amp-max-raw-t", type=int, help="Optional T cap per amplified attempt")
    cli.add_argument("--workers", type=int, default=0, help="0: all affinity/quota CPUs, bounded by memory; 1: serial")
    cli.add_argument("--chunk-size", type=int, default=8192, help="Candidates per process task")
    cli.add_argument("--batch-size", type=int, default=2048, help="Matrices per NumPy batch")
    cli.add_argument("--backend", choices=("numpy", "qiskit"), default="numpy",
                     help="Batched Qiskit-derived matrices or per-candidate Qiskit reference")
    cli.add_argument("--plots", action="store_true", help="Write section-5 histograms (requires matplotlib)")
    cli.add_argument("--max-examined", type=int, help="Cap evaluated candidates; output records incompleteness")
    cli.add_argument("--estimate-only", action="store_true", help="Build pools and report search size without evaluating candidates")
    cli.add_argument("--output-dir", type=Path, default=Path("out_record_paper"),
                     help="Relative paths are resolved under direct_search/")
    cli.add_argument("--amplified-output-dir", type=Path,
                     help="Second dataset; defaults to out_amplified_paper, or OUTPUT_amplified for a custom output")
    cli.add_argument("--no-qasm", action="store_true", help="Export validated JSON/CSV only")
    cli.add_argument("--export-attempt-qasm2", action="store_true")
    cli.add_argument("--quiet", action="store_true", help="Suppress progress on stderr")
    return cli


def report(progress):
    if progress["phase"] == "prepared":
        print(f"Prepared {progress['planned_partitions']} partitions; "
              f"{progress['planned_candidates']:,} candidates; "
              f"workers={progress.get('workers_used', 0)}", file=sys.stderr, flush=True)
    else:
        print(f"Examined {progress['examined']:,}/{progress['planned_candidates']:,}; "
              f"accepted {progress['accepted']:,}; unique {progress['unique_unitaries']}; "
              f"partition {progress['job']}", file=sys.stderr, flush=True)


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        config = SearchConfig(**{key: getattr(args, key) for key in asdict(SearchConfig())
                                 if key not in {"clifford_sets", "amp_max_j"}})
        args.output_dir, args.amplified_output_dir = variant_paths(args.output_dir, args.amplified_output_dir)
        if args.amp_max_j < 1:
            raise ValueError("--amp-max-j must be positive; the unamplified dataset always uses j=0")
        if args.amp_max_raw_t is not None and args.amp_max_raw_t < config.max_raw_t:
            raise ValueError("--amp-max-raw-t must admit the search's --max-raw-t")
    except ValueError as exc:
        cli.error(str(exc))
    if args.estimate_only:
        from direct_search.hardware import hardware_profile
        _, _, stats = prepare_search(config)
        stats["hardware"] = hardware_profile()
        print(json.dumps(stats, indent=2))
        return 0
    for output in (args.output_dir, args.amplified_output_dir):
        if (output / "database.json").exists():
            cli.error(f"{output} already contains a database; choose new output directories")
    if args.plots:
        try:
            import matplotlib
        except ImportError:
            cli.error("--plots requires matplotlib: python -m pip install matplotlib")
    database, stats = build_database(config, progress=None if args.quiet else report)
    try:
        summaries = export_variants(database, stats, args.output_dir, args.amplified_output_dir,
                                    max_j=args.amp_max_j, max_raw_t=args.amp_max_raw_t,
                                    export_qasm=not args.no_qasm,
                                    export_qasm2=args.export_attempt_qasm2, plots=args.plots)
    except (ValueError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
