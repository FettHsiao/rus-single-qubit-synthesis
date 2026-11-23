"""Shared bounded multiprocessing, progress and atomic JSON utilities."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import multiprocessing
import os
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fingerprint(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def report(phase, **fields):
    print(json.dumps(dict(phase=phase, **fields), ensure_ascii=False), flush=True)


@contextmanager
def single_thread_blas():
    names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update({name: "1" for name in names})
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def parallel_chunks(function, tasks, workers):
    """At most 2*workers queued tasks; merge in input order, not completion order."""
    tasks = iter(tasks)
    if workers == 1:
        yield from map(function, tasks)
        return
    with single_thread_blas(), ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        pending = deque()
        for _ in range(2 * workers):
            task = next(tasks, None)
            if task is not None:
                pending.append(executor.submit(function, task))
        while pending:
            yield pending.popleft().result()
            task = next(tasks, None)
            if task is not None:
                pending.append(executor.submit(function, task))


def chunks(entries, size):
    for begin in range(0, len(entries), size):
        yield entries[begin:begin + size]
