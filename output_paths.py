"""Track generated output roots so cleanup never relies on broad wildcards."""
import json
import fcntl
from contextlib import contextmanager
from pathlib import Path

BASE = Path(__file__).resolve().parent
REGISTRY = BASE / ".output_registry.json"
MARKER = ".rus-output.json"


@contextmanager
def registry_lock():
    with (BASE / ".output_registry.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def register_output(directory, stage):
    path = Path(directory).resolve()
    protected = {BASE, *(BASE / n for n in ("direct_search", "reduction", "composition", "draw", "docs"))}
    if path in protected or any(p.is_relative_to(path) for p in protected):
        raise ValueError("Output must be a dataset subdirectory, not a source-code directory")
    if path.exists() and any(path.rglob("*.py")):
        raise ValueError("Output directory contains Python source code")
    path.mkdir(parents=True, exist_ok=True)
    marker = dict(owner="rus_generator", workspace=str(BASE), stage=stage)
    (path / MARKER).write_text(json.dumps(marker, indent=2)+"\n")
    with registry_lock():
        paths = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
        paths = {name: value for name, value in paths.items() if Path(name).exists()}
        paths[str(path)] = marker
        temporary = REGISTRY.with_suffix(".tmp")
        temporary.write_text(json.dumps(paths, indent=2)+"\n")
        temporary.replace(REGISTRY)
