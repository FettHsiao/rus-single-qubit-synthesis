"""Conservative local worker sizing, respecting affinity and cgroup limits."""
from __future__ import annotations

import math
import os
from pathlib import Path


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def hardware_profile():
    logical = os.cpu_count() or 1
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(logical))
    cores = {}
    for cpu in cpus:
        root = f"/sys/devices/system/cpu/cpu{cpu}/topology"
        core = _read(root + "/core_id")
        if core:
            key = (_read(root + "/physical_package_id"), core)
            cores[key] = cores.get(key, 0) + 1
    physical = len(cores) or len(cpus)
    mem = dict(line.split(":", 1) for line in _read("/proc/meminfo").splitlines() if ":" in line)
    available = int(mem.get("MemAvailable", "0 kB").split()[0]) * 1024
    total = int(mem.get("MemTotal", "0 kB").split()[0]) * 1024
    quota = None
    # Inspect the current cgroup and its parents (v2); fall back to v1 roots.
    roots = {Path("/sys/fs/cgroup")}
    for line in _read("/proc/self/cgroup").splitlines():
        _, controllers, path = line.split(":", 2)
        if not controllers:
            current = Path("/sys/fs/cgroup") / path.lstrip("/")
            roots.update([current, *[p for p in current.parents if str(p).startswith("/sys/fs/cgroup")]])
    for root in roots:
        fields = _read(root / "cpu.max").split()
        if len(fields) == 2 and fields[0] != "max":
            limit = max(1, math.ceil(int(fields[0]) / int(fields[1])))
            quota = min(quota or limit, limit)
        maximum, used = _read(root / "memory.max"), _read(root / "memory.current")
        if maximum.isdigit() and used.isdigit():
            remaining = max(0, int(maximum) - int(used))
            available = min(available, remaining) if available else remaining
    q, period = (_read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
                 _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
    if q and period and int(q) > 0:
        quota = min(quota or logical, max(1, math.ceil(int(q) / int(period))))
    maximum = _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    used = _read("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if maximum.isdigit() and used.isdigit():
        remaining = max(0, int(maximum) - int(used))
        available = min(available, remaining) if available else remaining
    workers = min(len(cpus), quota or logical)
    smt_cores = sum(count > 1 for count in cores.values())
    hybrid = bool(smt_cores and any(count == 1 for count in cores.values()))
    if available:
        workers = min(workers, max(1, available // (256 * 1024 ** 2)))
    model = next((line.split(":", 1)[1].strip() for line in _read("/proc/cpuinfo").splitlines()
                  if line.startswith("model name")), "unknown")
    return {"cpu_model": model, "logical_cpus": logical, "affinity_cpus": cpus,
            "physical_cores_available": physical, "cpu_quota": quota,
            "mixed_smt_topology": hybrid, "smt_cores_available": smt_cores,
            "memory_total_bytes": total, "memory_available_bytes": available,
            "recommended_workers": max(1, workers)}


def resolve_workers(requested=0, tasks=None):
    """Use all quota/affinity CPUs, bounded by memory and useful tasks."""
    if requested < 0:
        raise ValueError("workers must be >= 0")
    count = requested or hardware_profile()["recommended_workers"]
    return max(1, min(count, tasks)) if tasks is not None else count
