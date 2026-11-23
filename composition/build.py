"""Cost-bounded class-k expansion with deterministic, bounded process tasks."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from direct_search.axial import z_rotation_metadata
from direct_search.circuits import clifford_group, matrix_key
from direct_search.hardware import resolve_workers
from reduction.common import decode, encode, phase_error
from reduction.clucr import canonicalize
from execution import fingerprint, parallel_chunks, report, write_json
from .geometry import PERIOD, canonicalize_g2, fold_transform, rotation


@dataclass
class Config:
    mode: str = "axial"
    max_class: int = 3
    max_pool: int = 10000
    max_expected_t: float = 30.
    base_limit: int = 0
    workers: int = 0
    chunk_size: int = 256
    canonicalizer: str = "g2"
    cost: str = "effective"
    gap_fill_iterations: int = 0
    gap_fill_pool: int = 1000
    tol: float = 1e-10

    def __post_init__(self):
        if self.mode not in {"axial", "all"} or self.cost not in {"direct", "effective"}:
            raise ValueError("Invalid mode/cost")
        if self.canonicalizer not in {"g2", "full"}:
            raise ValueError("Invalid canonicalizer")
        if (self.max_class < 1 or self.max_pool < 0 or self.base_limit < 0 or self.workers < 0
                or self.chunk_size < 1 or self.gap_fill_iterations < 0 or self.gap_fill_pool < 0):
            raise ValueError("Invalid class/pool/worker/chunk/gap limit")
        if not math.isfinite(self.max_expected_t) or self.max_expected_t <= 0:
            raise ValueError("max_expected_t must be finite and positive")
        if not math.isfinite(self.tol) or not 0 < self.tol < 1e-3:
            raise ValueError("tol must be finite and in (0,1e-3)")
        if self.mode != "axial" and self.gap_fill_iterations:
            raise ValueError("Gap filling applies only to --mode axial")


def gate(ops):
    return [{"kind": "clifford", "ops": list(ops)}] if ops else []


def wrap(recipe, left, right):
    return gate(right) + recipe + gate(left)


def rank(e):
    return (e["expected_t_count"], e["class_depth"], e["raw_t_count"],
            json.dumps(e["recipe"], sort_keys=True, separators=(",", ":")))


def node(matrix, recipe, depth, raw, probability, cost, angle=None):
    return dict(unitary_matrix=encode(matrix), unitary_key=matrix_key(matrix), recipe=recipe,
                class_depth=depth, raw_t_count=raw, success_probability=probability,
                expected_t_count=cost, effective_t_count=cost, axial_angle=angle)


class Pool:
    """Tolerance-aware spatial buckets with explicit matrix verification.

    Neighbor buckets catch rounding boundaries. Only three coordinates index
    candidates; all matrix entries are checked before merging. Updating the
    anchor with the winner prevents tolerance chains. Bulk trimming does not
    rebuild a KD-tree after every candidate batch.
    """
    def __init__(self, tol):
        self.tol, self.entries, self.matrices, self.buckets = tol, [], [], {}

    def bucket(self, matrix):
        # Useful for both diagonal axial matrices and general representatives.
        features = (abs(matrix[0, 0]), matrix[0, 1].real, matrix[1, 1].imag)
        return tuple(math.floor(float(v)/(8*self.tol)) for v in features)

    def insert(self, entry):
        from itertools import product
        matrix = decode(entry["unitary_matrix"])
        key = self.bucket(matrix)
        indices = []
        for shift in product((-1, 0, 1), repeat=3):
            indices.extend(self.buckets.get(tuple(k+s for k, s in zip(key, shift)), ()))
        for i in sorted(indices):
            if phase_error(matrix, self.matrices[i]) <= self.tol:
                if rank(entry) < rank(self.entries[i]):
                    previous = self.bucket(self.matrices[i])
                    if previous != key:
                        self.buckets[previous].remove(i)
                        self.buckets.setdefault(key, []).append(i)
                    self.entries[i], self.matrices[i] = entry, matrix
                    return True
                return False
        self.buckets.setdefault(key, []).append(len(self.entries))
        self.entries.append(entry)
        self.matrices.append(matrix)
        return True

    def trim(self, maximum):
        if not maximum or len(self.entries) <= maximum:
            return 0
        removed = len(self.entries)-maximum
        keep = sorted(self.entries, key=rank)[:maximum]
        self.__init__(self.tol)
        for e in keep:
            matrix = decode(e["unitary_matrix"])
            self.buckets.setdefault(self.bucket(matrix), []).append(len(self.entries))
            self.entries.append(e)
            self.matrices.append(matrix)
        return removed


def canonical(matrix, config):
    return (canonicalize_g2 if config.canonicalizer == "g2" else canonicalize)(matrix, config.tol)


def seed_entry(source, config):
    matrix = decode(source["unitary_matrix"])
    recipe = [{"kind": "rus", "source_entry_id": source["entry_id"]}]
    angle = None
    if config.mode == "axial":
        witness = z_rotation_metadata(matrix, config.tol)
        if not witness["axial"]:
            return None
        recipe = wrap(recipe, witness["left_ops"], witness["right_ops"])
        angle, left, right = fold_transform(witness["angle"])
        recipe, matrix = wrap(recipe, left, right), rotation(angle)
    else:
        witness = canonical(matrix, config)
        recipe = wrap(recipe, witness["left_ops"], witness["right_ops"])
        matrix = decode(witness["unitary_matrix"])
    amplified = config.cost == "effective"
    cost = source["effective_t_count"] if amplified else source["expected_t_count"]
    raw = source["amplified_raw_t_count"] if amplified else source["raw_t_count"]
    p = source["amplified_success_probability"] if amplified else source["success_probability"]
    return node(matrix, recipe, 1, raw, p, cost, angle)


def product_entry(a, b, middle, config):
    cost = a["expected_t_count"] + b["expected_t_count"]
    depth = a["class_depth"] + b["class_depth"]
    if cost > config.max_expected_t + config.tol or depth > config.max_class:
        return None
    angle = None
    if config.mode == "axial":
        sign = middle
        b_recipe = b["recipe"] if sign == 1 else wrap(b["recipe"], ["x"], ["x"])
        angle, left, right = fold_transform(a["axial_angle"] + sign*b["axial_angle"])
        recipe = wrap(b_recipe + a["recipe"], left, right)
        matrix = rotation(angle)
    else:
        clifford = clifford_group()[middle]
        matrix = decode(a["unitary_matrix"]) @ clifford.matrix @ decode(b["unitary_matrix"])
        witness = canonical(matrix, config)
        recipe = wrap(b["recipe"] + gate(clifford.ops) + a["recipe"], witness["left_ops"], witness["right_ops"])
        matrix = decode(witness["unitary_matrix"])
    return node(matrix, recipe, depth, a["raw_t_count"]+b["raw_t_count"],
                a["success_probability"]*b["success_probability"], cost, angle)


def expand_chunk(task):
    pairs, config = task
    pool, examined, pruned = Pool(config.tol), 0, 0
    for a, b in pairs:
        for middle in ((1, -1) if config.mode == "axial" else range(24)):
            examined += 1
            entry = product_entry(a, b, middle, config)
            if entry is None:
                pruned += 1
            else:
                pool.insert(entry)
    return sorted(pool.entries, key=rank), examined, pruned, os.getpid()


def pair_chunks(frontier, base, size):
    batch = []
    for a in frontier:
        for b in base:
            batch.append((a, b))
            if len(batch) == size:
                yield batch
                batch = []
    if batch:
        yield batch


def gap_metrics(entries):
    angles = sorted(set([0., PERIOD/2] + [e["axial_angle"] for e in entries]))
    widths = np.diff(angles)
    index = int(np.argmax(widths))
    gap = float(widths[index])
    # Include endpoint π/4 even when no entry exists there: separately account
    # for the distance from the last actual angle to that boundary.
    actual = sorted(set([0.] + [e["axial_angle"] for e in entries]))
    worst_angle = max(float(np.max(np.diff(actual)))/2 if len(actual)>1 else 0., PERIOD/2-actual[-1])
    return dict(largest_gap_radians=gap, largest_neighbor_distance=2**.5*math.sin(gap/4),
                covering_radius=2**.5*math.sin(worst_angle/4),
                largest_gap=[angles[index], angles[index+1]])


def fill_gaps(pool, config):
    """Target the largest interval, using binary search for complementary angles.

    Search both sum and difference branches of folded angles, then retain the
    point nearest the midpoint. Added circuits obey class and expected-T caps.
    """
    added = []
    for _ in range(config.gap_fill_iterations):
        entries = sorted(pool.entries, key=rank)
        if config.gap_fill_pool:
            entries = entries[:config.gap_fill_pool]
        entries = sorted(entries, key=lambda e: e["axial_angle"])
        angles = np.array([e["axial_angle"] for e in entries])
        lo, hi = gap_metrics(pool.entries)["largest_gap"]
        midpoint, candidates = (lo+hi)/2, []
        for a in entries:
            for sign in (1, -1):
                for n in range(-2, 3):
                    for orientation in (1, -1):
                        target_b = sign*(orientation*midpoint+n*PERIOD-a["axial_angle"])
                        i = int(np.searchsorted(angles, target_b))
                        for j in (i-1, i):
                            if 0 <= j < len(entries):
                                result = product_entry(a, entries[j], sign, config)
                                if result is not None and lo+config.tol < result["axial_angle"] < hi-config.tol:
                                    candidates.append(result)
        if not candidates:
            break
        candidate = min(candidates, key=lambda e: (abs(e["axial_angle"]-midpoint), rank(e)))
        if pool.insert(candidate):
            added.append(candidate)
        else:
            break
    return added


def build_database(input_dir, output_dir, config):
    """Write an analysis checkpoint; export is a separate restartable stage."""
    from .export import validate_sources
    root, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    path = root / "database.json" if root.is_dir() else root
    root = path.parent
    if output == root or any((output / f).exists() for f in ("analysis.json", "database.json")):
        raise FileExistsError("Use a new output directory; analysis/export checkpoints already exist")
    payload = json.loads(path.read_text())
    if payload.get("algorithm") not in {"direct_search", "clucr"} or not payload.get("entries"):
        raise ValueError("Expected a nonempty direct_search or clucr dataset")
    if not payload.get("stats", {}).get("export_ok", False):
        raise ValueError("Input dataset must have completed Qiskit export/readback validation")
    from output_paths import register_output
    register_output(output, "composition")
    started = time.monotonic()
    workers = resolve_workers(config.workers)
    sources = validate_sources(payload["entries"], root, config, workers)
    seeds = Pool(config.tol)
    for source in sources:
        entry = seed_entry(source, config)
        if entry is not None and entry["expected_t_count"] <= config.max_expected_t + config.tol:
            seeds.insert(entry)
    base = sorted(seeds.entries, key=rank)
    base_available = len(base)
    if config.base_limit:
        base = base[:config.base_limit]
    if not base:
        raise ValueError("No seed RUS entries meet the selected mode/cost cap")
    total = Pool(config.tol)
    # Free Cliffords are legitimate class-0 approximants and dominate paid identities.
    identity = node(np.eye(2), [], 0, 0, 1., 0., 0. if config.mode == "axial" else None)
    if config.mode == "all":
        witness = canonical(np.eye(2), config)
        identity.update(unitary_matrix=witness["unitary_matrix"], recipe=wrap([], witness["left_ops"], witness["right_ops"]))
        identity["unitary_key"] = matrix_key(decode(identity["unitary_matrix"]))
    total.insert(identity)
    for e in base:
        total.insert(e)
    truncated = total.trim(config.max_pool)
    frontier = [e for e in total.entries if e["class_depth"] == 1]
    levels, pids = [], set()
    for depth in range(2, config.max_class+1):
        if not frontier:
            break
        current = Pool(config.tol)
        count = math.ceil(len(frontier)*len(base)/config.chunk_size)
        used = min(workers, count)
        report("composition", class_depth=depth, frontier=len(frontier), base=len(base), workers=used, tasks=count)
        tasks = ((pairs, config) for pairs in pair_chunks(frontier, base, config.chunk_size))
        examined = pruned = completed = 0
        last_report = time.monotonic()
        for results, seen, cut, pid in parallel_chunks(expand_chunk, tasks, used):
            examined += seen
            pruned += cut
            completed += 1
            pids.add(pid)
            for e in results:
                current.insert(e)
            # Streaming beam pruning keeps peak pool memory bounded.
            truncated += current.trim(config.max_pool)
            if time.monotonic()-last_report >= 10:
                report("composition_progress", class_depth=depth, tasks_done=completed, tasks=count, candidates=examined)
                last_report = time.monotonic()
        # Earlier levels have no greater depth: cheaper existing representatives
        # dominate these candidates in both cost and remaining class budget.
        frontier = [e for e in sorted(current.entries, key=rank) if total.insert(e)]
        truncated += total.trim(config.max_pool)
        retained = {id(e) for e in total.entries}
        frontier = [e for e in frontier if id(e) in retained]
        levels.append(dict(class_depth=depth, examined=examined, cost_or_depth_pruned=pruned,
                           improved_or_new=len(frontier), pool=len(total.entries), workers=used))
    before_gap = gap_metrics(total.entries) if config.mode == "axial" else None
    # max_pool remains a hard bound. Reserve space explicitly by choosing a larger
    # pool if gap filling is wanted after a truncated expansion.
    room = max(0, config.max_pool-len(total.entries)) if config.max_pool else config.gap_fill_iterations
    gap_config = Config(**dict(asdict(config), gap_fill_iterations=min(config.gap_fill_iterations, room)))
    added = fill_gaps(total, gap_config) if config.mode == "axial" else []
    entries = sorted(total.entries, key=rank)
    for i, e in enumerate(entries):
        e["entry_id"] = f"composed_{i:07d}"
        witness = z_rotation_metadata(decode(e["unitary_matrix"]), config.tol)
        e.update(axial=witness["axial"], axial_witness=witness,
                 axial_angle=e["axial_angle"] if config.mode == "axial" else witness["angle"],
                 deduplication_method="left_right_clifford", composition_class=e["class_depth"],
                 cost_model="sum of component expected T; each component retries locally",
                 success_probability_meaning="product of component single-attempt probabilities; not used as the total-loop cost denominator")
    stats = dict(config=asdict(config), source_database=str(path), source_sha256=fingerprint(path),
                 source_search_complete=payload.get("stats", {}).get("search_complete", payload.get("stats", {}).get("source_search_complete")),
                 seed_classes_available=base_available, seed_classes_used=len(base), output_entries=len(entries),
                 workers_requested=config.workers, workers_available=workers, worker_pids=sorted(pids),
                 levels=levels, pool_pruning_events=truncated,
                 exhaustive_within_class_and_cost_caps=not truncated and base_available == len(base),
                 gap_fill_added=len(added), gap_fill_capacity_limited=room < config.gap_fill_iterations,
                 gap_before=before_gap, gap_after=gap_metrics(entries) if config.mode == "axial" else None,
                 middle_cliffords=24 if config.mode == "all" else "angle sum and difference",
                 build_seconds=time.monotonic()-started, export_ok=False)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = dict(schema_version=1, algorithm="composition", qubit_order={"data": 0, "ancilla": 1},
                      stats=stats, components={s["entry_id"]: s for s in sources}, entries=entries)
    write_json(output / "analysis.json", checkpoint)
    write_json(output / "analysis_summary.json", stats)
    report("composition_analysis_complete", entries=len(entries), output=str(output), truncated=bool(truncated))
    return checkpoint
