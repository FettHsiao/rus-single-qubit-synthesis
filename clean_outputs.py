#!/usr/bin/env python3
"""Remove generated datasets; --all also discovers historical outputs and caches."""
import argparse
import json
import os
from pathlib import Path
import shutil
from output_paths import BASE, MARKER, REGISTRY, registry_lock


def discover_outputs(base=BASE):
    """Find local output roots without descending into datasets or symlinks."""
    base = Path(base).resolve()
    signatures = {MARKER, "database.json", "analysis.json", "plot_summary.json",
                  "lookup.json", "hybrid_estimate.json", "audit.json"}
    found = set()
    for root, dirs, files in os.walk(base, followlinks=False):
        path = Path(root)
        dirs[:] = [d for d in dirs if d not in {".git", ".agents", ".codex"}
                   and not (path / d).is_symlink()]
        if "pyvenv.cfg" in files:
            dirs[:] = []
            continue
        if path != base and (path.name == "__pycache__" or path.name == "out"
                             or path.name.startswith("out_") or signatures.intersection(files)):
            found.add(path)
            dirs[:] = []
    found.update(p for p in base.glob("VALIDATION*.json") if p.is_file() and not p.is_symlink())
    return found


def validate_output(path, registered, discovered, base=BASE):
    """Allow discovered local artifacts or marked registered datasets, never code."""
    base = Path(base).resolve()
    if path.is_symlink():
        raise ValueError(f"Refusing output symlink: {path}")
    path = path.resolve()
    if path == base or base.is_relative_to(path):
        raise ValueError(f"Refusing source directory: {path}")
    local = path in discovered and path.is_relative_to(base)
    if str(path) not in registered and not local:
        raise ValueError(f"Not a registered or discovered output: {path}")
    if not path.exists():
        return path
    if path.is_file():
        if local and path.parent == base and path.match("VALIDATION*.json"):
            return path
        raise ValueError(f"Not a generated validation report: {path}")
    if any(path.rglob("*.py")):
        raise ValueError(f"Refusing directory containing Python source: {path}")
    if not local:
        marker = path / MARKER
        if not marker.is_file() or json.loads(marker.read_text()).get("workspace") != str(base):
            raise ValueError(f"Refusing unverified output: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="Actually remove the listed output directories")
    parser.add_argument("--all", action="store_true", help="Include local historical outputs, validation reports and Python caches")
    parser.add_argument("--dirs", nargs="+", type=Path, help="Only these registered dataset directories (default: all)")
    args = parser.parse_args()
    if args.all and args.dirs:
        parser.error("Choose --all or --dirs, not both")
    with registry_lock():
        registered = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
        discovered = discover_outputs(BASE) if args.all else set()
        selected = set(args.dirs) if args.dirs else {Path(name) for name in registered}
        if args.all:
            # --all is scoped to this project, even if old registry entries point outside it.
            selected = {p for p in selected if p.resolve().is_relative_to(BASE)} | discovered
        try:
            valid = sorted({validate_output(p, registered, discovered, BASE) for p in selected})
        except ValueError as exc:
            parser.error(str(exc))
        for path in valid:
            print(("DELETE " if args.yes else "WOULD DELETE ")+str(path))
        if args.yes:
            for path in sorted(valid, key=lambda p: len(p.parts), reverse=True):
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                registered.pop(str(path), None)
            registered = {name: value for name, value in registered.items() if Path(name).exists()}
            REGISTRY.write_text(json.dumps(registered, indent=2)+"\n")
        elif valid:
            print("Add --yes to remove these generated outputs.")
        else:
            print("No outputs to remove.")


if __name__ == "__main__":
    main()
