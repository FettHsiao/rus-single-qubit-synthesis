#!/usr/bin/env python3
"""Run direct search → amplification → double-coset reduction → Fig. 5 → section 6.2."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

BASE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--direct-dir", type=Path, default=BASE / "direct_search/out_record_paper")
    p.add_argument("--amp-dir", type=Path, default=BASE / "direct_search/out_amplified_paper")
    p.add_argument("--reduction-dir", type=Path, default=BASE / "reduction/out_record_paper")
    p.add_argument("--fig-dir", type=Path, default=BASE / "draw/fig5")
    p.add_argument("--axial-dir", type=Path, default=BASE / "composition/out_axial")
    p.add_argument("--general-dir", type=Path, default=BASE / "composition/out_general")
    p.add_argument("--lookup-dir", type=Path, default=BASE / "composition/lookup_z")
    p.add_argument("--hybrid-dir", type=Path, default=BASE / "composition/hybrid_z")
    p.add_argument("--hybrid-estimate-dir", type=Path, default=BASE / "composition/hybrid_estimate_z")
    p.add_argument("--target-z", type=float, help="Optional target angle; also run axial lookup")
    p.add_argument("--epsilon", type=float, default=1e-6, help="Optional lookup tolerance")
    p.add_argument("--hybrid-epsilon", type=float, help="Also synthesize and estimate gearbox correction; requires --target-z")
    p.add_argument("--max-gearbox-depth", type=int, default=6)
    p.add_argument("--max-raw-t", type=int, default=15)
    p.add_argument("--amp-max-j", type=int, default=10)
    p.add_argument("--amp-max-raw-t", type=int, help="Optional native T cap per amplified attempt")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--axial-max-class", type=int, default=6)
    p.add_argument("--general-max-class", type=int, default=3)
    p.add_argument("--max-pool", "--maxpool", type=int, default=100000)
    p.add_argument("--max-expected-t", type=float, default=30.)
    p.add_argument("--base-limit", type=int, default=0)
    p.add_argument("--gap-fill-iterations", type=int, default=100)
    p.add_argument("--gap-fill-pool", type=int, default=1000)
    p.add_argument("--composition-mode", choices=("both", "axial", "all"), default="both")
    p.add_argument("--cost", choices=("effective",), default="effective")
    p.add_argument("--canonicalizer", choices=("g2", "full"), default="g2")
    p.add_argument("--search-chunk-size", type=int, default=8192)
    p.add_argument("--amp-chunk-size", type=int, default=16)
    p.add_argument("--reduction-chunk-size", type=int, default=16)
    p.add_argument("--composition-chunk-size", type=int, default=256)
    p.add_argument("--export-chunk-size", type=int, default=16)
    p.add_argument("--fig-bin-width", type=float, default=2.)
    p.add_argument("--max-examined", type=int, help="Smoke test only; resulting database records an incomplete search")
    p.add_argument("--job", help="Optional one-partition smoke test: tA0,tA1,tA2,tD0")
    p.add_argument("--skip-direct", action="store_true", help="Use existing direct results; optimize only if amp-dir is missing")
    p.add_argument("--skip-amplification", action="store_true", help="Require an existing amp-dir; also requires --skip-direct")
    p.add_argument("--skip-reduction", action="store_true", help="Use an existing completed --reduction-dir")
    p.add_argument("--dry-run", action="store_true", help="Print the exact commands without running them")
    a = p.parse_args()
    if a.amp_max_j < 1:
        p.error("--amp-max-j must be positive; reduction requires optimized amplification")
    if a.skip_amplification and not a.skip_direct:
        p.error("--skip-amplification requires --skip-direct because direct search exports both variants")
    if any(getattr(a, name) < 0 for name in ("workers", "amp_max_j", "max_pool", "base_limit", "gap_fill_iterations", "gap_fill_pool", "max_gearbox_depth")):
        p.error("Worker, amplification, pool and gearbox limits must be nonnegative")
    if any(getattr(a, name) < 1 for name in ("max_raw_t", "axial_max_class", "general_max_class", "search_chunk_size", "amp_chunk_size", "reduction_chunk_size", "composition_chunk_size", "export_chunk_size")):
        p.error("T, class and chunk limits must be positive")
    if a.amp_max_raw_t is not None and a.amp_max_raw_t < 0:
        p.error("amp-max-raw-t must be nonnegative")
    if a.max_examined is not None and a.max_examined < 0:
        p.error("max-examined must be nonnegative")
    if not math.isfinite(a.fig_bin_width) or a.fig_bin_width <= 0:
        p.error("fig-bin-width must be finite and positive")
    if not math.isfinite(a.max_expected_t) or a.max_expected_t <= 0:
        p.error("max-expected-t must be finite and positive")
    if a.target_z is not None and (not math.isfinite(a.target_z) or not math.isfinite(a.epsilon) or not 0 < a.epsilon <= 1):
        p.error("Target must be finite and epsilon in (0,1]")
    if a.hybrid_epsilon is not None and (not math.isfinite(a.hybrid_epsilon) or not 0 < a.hybrid_epsilon < a.epsilon):
        p.error("Require 0 < hybrid-epsilon < epsilon")
    if a.hybrid_epsilon is not None and a.target_z is None:
        p.error("--hybrid-epsilon requires --target-z")
    if a.target_z is not None and a.composition_mode == "all":
        p.error("Target/hybrid pipeline requires axial or both composition modes")
    for name in ("direct_dir", "amp_dir", "reduction_dir", "fig_dir", "axial_dir", "general_dir", "lookup_dir", "hybrid_dir", "hybrid_estimate_dir"):
        setattr(a, name, getattr(a, name).resolve())
    dirs = [a.direct_dir, a.amp_dir, a.reduction_dir, a.fig_dir, a.axial_dir, a.general_dir, a.lookup_dir, a.hybrid_dir, a.hybrid_estimate_dir]
    if len(set(dirs)) != len(dirs) or any(x != y and x.is_relative_to(y) for x in dirs for y in dirs):
        p.error("Pipeline output directories must be distinct and must not contain each other")
    commands = []
    if not a.skip_direct:
        direct = ["direct_search", "--max-raw-t", a.max_raw_t,
                  "--dedup-mode", "exact", "--require-recovery-branch", "--strict-nondeterministic",
                  "--amp-max-j", a.amp_max_j, "--workers", a.workers, "--chunk-size", a.search_chunk_size,
                  "--export-attempt-qasm2", "--output-dir", a.direct_dir,
                  "--amplified-output-dir", a.amp_dir]
        if a.amp_max_raw_t is not None:
            direct += ["--amp-max-raw-t", a.amp_max_raw_t]
        if a.max_examined is not None:
            direct += ["--max-examined", a.max_examined]
        if a.job:
            direct += ["--job", a.job]
        commands.append(direct)
    if a.skip_direct and not a.skip_amplification and not (a.amp_dir / "database.json").exists():
        amplification = ["amplify", "--input-dir", a.direct_dir, "--output-dir", a.amp_dir,
                         "--max-j", a.amp_max_j, "--workers", a.workers, "--chunk-size", a.amp_chunk_size]
        if a.amp_max_raw_t is not None:
            amplification += ["--max-raw-t", a.amp_max_raw_t]
        commands.append(amplification)
    if a.skip_direct and (a.skip_amplification or (a.amp_dir / "database.json").exists()) and not a.dry_run:
        from reduction.pipeline import require_amplified_source
        require_amplified_source(json.loads((a.amp_dir / "database.json").read_text()), 1e-10)
    if a.skip_reduction and not a.dry_run:
        reduced = json.loads((a.reduction_dir / "database.json").read_text())
        stats = reduced.get("stats", {})
        if (reduced.get("algorithm") != "clucr" or not stats.get("export_ok")
                or stats.get("selection_cost") != "effective"
                or (stats.get("source_amplification") or {}).get("max_j", 0) < 1):
            p.error("--skip-reduction requires a completed LUR dataset selected using amplified effective cost")
    if not a.skip_reduction:
        commands.append(["reduction", "--input-dir", a.amp_dir, "--output-dir", a.reduction_dir, "--cost", a.cost,
                         "--workers", a.workers, "--chunk-size", a.reduction_chunk_size])
    commands.append(["reduction.plot_fig5", "--input-dir", a.reduction_dir, "--output-dir", a.fig_dir,
                     "--bin-width", a.fig_bin_width])
    modes = ["axial", "all"] if a.composition_mode == "both" else [a.composition_mode]
    for mode in modes:
        command = ["composition", "build", "--input-dir", a.reduction_dir,
                   "--output-dir", a.axial_dir if mode == "axial" else a.general_dir,
                   "--mode", mode, "--max-class", a.axial_max_class if mode == "axial" else a.general_max_class,
                   "--max-pool", a.max_pool, "--max-expected-t", a.max_expected_t,
                   "--base-limit", a.base_limit, "--workers", a.workers, "--cost", a.cost,
                   "--canonicalizer", a.canonicalizer, "--chunk-size", a.composition_chunk_size,
                   "--export-chunk-size", a.export_chunk_size]
        if mode == "axial":
            command += ["--gap-fill-iterations", a.gap_fill_iterations, "--gap-fill-pool", a.gap_fill_pool]
        commands.append(command)
    if a.target_z is not None:
        commands.append(["composition", "lookup", "--input-dir", a.axial_dir, "--output-dir", a.lookup_dir,
                         "--target-z", a.target_z, "--epsilon", a.epsilon, "--workers", a.workers])
        if a.hybrid_epsilon is not None:
            commands.append(["composition", "hybrid", "--input-dir", a.axial_dir, "--output-dir", a.hybrid_dir,
                             "--target-z", a.target_z, "--epsilon", a.hybrid_epsilon, "--coarse-epsilon", a.epsilon,
                             "--max-gearbox-depth", a.max_gearbox_depth, "--workers", a.workers])
            commands.append(["composition", "hybrid-estimate", "--input-dir", a.axial_dir, "--output-dir", a.hybrid_estimate_dir,
                             "--target-z", a.target_z, "--epsilon", a.hybrid_epsilon, "--coarse-epsilon", a.epsilon])
    env = dict(os.environ)
    env.update({name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    env.setdefault("MPLCONFIGDIR", "/tmp/rus-matplotlib")
    if not a.dry_run:
        for command in commands:
            flags = ["--output-dir"] + (["--amplified-output-dir"] if command[0] == "direct_search" else [])
            for flag in flags:
                output = Path(command[command.index(flag) + 1])
                if any((output / name).exists() for name in ("database.json", "analysis.json", "lookup.json", "hybrid_estimate.json")):
                    p.error(f"Output already exists: {output}; skip completed stages or choose a new directory")
    for command in commands:
        command = [sys.executable, "-m", *map(str, command)]
        print(shlex.join(command), flush=True)
        if not a.dry_run:
            subprocess.run(command, cwd=BASE, env=env, check=True)


if __name__ == "__main__":
    main()
