"""CLI for section 6.2 class-k expansion, approximation, and gearbox estimates."""
import argparse
import json
from pathlib import Path
from .build import Config, build_database
from .export import export_database
from .geometry import rotation
from .lookup import hybrid_estimate, lookup_database
from reduction.common import decode
from reduction.pipeline import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Expand class-k RUS circuits")
    build.add_argument("--input-dir", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--stage", choices=("all", "analyze", "export"), default="all")
    build.add_argument("--mode", choices=("axial", "all"), default="axial", help="all includes axial and non-axial with all 24 middle Cliffords")
    build.add_argument("--max-class", "--classes", type=int, default=3)
    build.add_argument("--max-pool", "--maxpool", type=int, default=10000, help="Hard pool cap; 0 = unlimited. Positive cap can make search non-exhaustive")
    build.add_argument("--max-expected-t", "--t0", type=float, default=30.)
    build.add_argument("--base-limit", type=int, default=0, help="Lowest-cost class-1 seed limit; 0 = all")
    build.add_argument("--workers", type=int, default=0)
    build.add_argument("--chunk-size", type=int, default=256)
    build.add_argument("--export-chunk-size", type=int, default=16)
    build.add_argument("--canonicalizer", choices=("g2", "full"), default="g2")
    build.add_argument("--cost", choices=("effective", "direct"), default="effective")
    build.add_argument("--gap-fill-iterations", type=int, default=0)
    build.add_argument("--gap-fill-pool", type=int, default=1000, help="0 = all retained entries")
    build.add_argument("--tol", type=float, default=1e-10)
    lookup = commands.add_parser("lookup", help="Lowest expected T among every match within epsilon")
    lookup.add_argument("--input-dir", type=Path, required=True)
    lookup.add_argument("--output-dir", type=Path, required=True)
    target = lookup.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-z", type=float, help="Z angle in radians")
    target.add_argument("--target-json", type=Path, help="2x2 matrix encoded as [real,imag], or an object with unitary_matrix")
    lookup.add_argument("--epsilon", type=float, required=True)
    lookup.add_argument("--workers", type=int, default=0)
    hybrid = commands.add_parser("hybrid-estimate", help="Section 6.2.2 gearbox cost estimates; no gearbox synthesis")
    hybrid.add_argument("--input-dir", type=Path, required=True)
    hybrid.add_argument("--output-dir", type=Path, required=True)
    hybrid.add_argument("--target-z", type=float, required=True)
    hybrid.add_argument("--epsilon", type=float, required=True)
    hybrid.add_argument("--coarse-epsilon", type=float, default=1e-6)
    actual_hybrid = commands.add_parser("hybrid", help="Executable coarse RUS plus nested WK13 gearbox correction")
    actual_hybrid.add_argument("--input-dir", type=Path, required=True)
    actual_hybrid.add_argument("--output-dir", type=Path, required=True)
    actual_hybrid.add_argument("--target-z", type=float, required=True)
    actual_hybrid.add_argument("--epsilon", type=float, required=True)
    actual_hybrid.add_argument("--coarse-epsilon", type=float, default=1e-6)
    actual_hybrid.add_argument("--max-gearbox-depth", type=int, default=6)
    actual_hybrid.add_argument("--workers", type=int, default=0)
    actual_hybrid.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args(argv)
    if args.command == "build":
        try:
            config = Config(**{k: getattr(args, k) for k in Config.__dataclass_fields__})
            if args.export_chunk_size <= 0:
                raise ValueError("export-chunk-size must be positive")
        except ValueError as exc:
            parser.error(str(exc))
        if args.stage in {"all", "analyze"}:
            build_database(args.input_dir, args.output_dir, config)
        if args.stage in {"all", "export"}:
            export_database(args.output_dir, args.workers, args.export_chunk_size)
    elif args.command == "lookup":
        if args.target_json:
            value = json.loads(args.target_json.read_text())
            matrix = decode(value["unitary_matrix"] if isinstance(value, dict) else value)
        else:
            matrix = rotation(args.target_z)
        result = lookup_database(args.input_dir, args.output_dir, matrix, args.epsilon, args.workers)
        print(json.dumps(dict(found=result is not None, **({k: v for k, v in result.items() if k != "entry"} if result else {})), indent=2))
        return 0 if result else 2
    elif args.command == "hybrid":
        from .gearbox import hybrid_database
        result = hybrid_database(args.input_dir, args.output_dir, args.target_z, args.epsilon,
                                 args.coarse_epsilon, args.max_gearbox_depth, args.workers, args.chunk_size)
        print(json.dumps(dict(found=result is not None, distance=result["distance"] if result else None,
                              expected_t_count=result["expected_t_count"] if result else None), indent=2))
        return 0 if result else 2
    else:
        payload = json.loads((args.input_dir / "database.json").read_text())
        result = hybrid_estimate(payload, args.target_z, args.epsilon, args.coarse_epsilon)
        from output_paths import register_output
        register_output(args.output_dir, "hybrid_estimate")
        write_json(args.output_dir / "hybrid_estimate.json", result)
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
