"""Export both cost variants from one verified exact-search result."""
from copy import deepcopy
from pathlib import Path

from execution import fingerprint
from output_paths import register_output
from .amplification import amplification_metadata
from .export import export_database

BASE = Path(__file__).resolve().parent


def variant_paths(output_dir, amplified_output_dir=None):
    """Resolve two distinct sibling datasets, with direct-search-relative paths."""
    direct = Path(output_dir)
    direct = (direct if direct.is_absolute() else BASE / direct).resolve()
    if amplified_output_dir is None:
        name = "out_amplified_paper" if direct.name == "out_record_paper" else direct.name + "_amplified"
        amplified = direct.with_name(name)
    else:
        amplified = Path(amplified_output_dir)
        amplified = (amplified if amplified.is_absolute() else BASE / amplified).resolve()
    if direct.is_relative_to(amplified) or amplified.is_relative_to(direct):
        raise ValueError("Unamplified/amplified output directories must not overlap")
    return direct, amplified


def export_variants(database, stats, output_dir, amplified_output_dir=None, *, max_j=10,
                    max_raw_t=None, export_qasm=True, export_qasm2=False, plots=False):
    """Preserve direct T/p winners and export j=0 and optimized j=0..max_j copies."""
    direct, amplified = variant_paths(output_dir, amplified_output_dir)
    if not isinstance(max_j, int) or max_j < 1:
        raise ValueError("Paired export requires a positive amplification search bound")
    if max_raw_t is not None and (not isinstance(max_raw_t, int) or max_raw_t < 0
                                 or any(e["raw_t_count"] > max_raw_t for e in database.values())):
        raise ValueError("Amplified attempt T cap must admit every original attempt")
    for output in (direct, amplified):
        if (output / "database.json").exists():
            raise FileExistsError(f"Completed output already exists: {output}")
    summaries = {}
    for variant, output, bound in (("unamplified", direct, 0), ("amplified", amplified, max_j)):
        variant_stats = deepcopy(stats)
        variant_stats.pop("amplification", None)
        variant_stats["config"]["amp_max_j"] = bound
        variant_stats["dataset_variant"] = variant
        entries = {}
        for key, original in database.items():
            entry = dict(original)
            entry.update(amplification_metadata(entry["raw_t_count"], entry["success_probability"], bound, max_raw_t))
            entry["amplification_search"] = dict(max_j=bound, max_raw_t=max_raw_t,
                                                includes_unamplified=True)
            entries[key] = entry
        if bound:
            variant_stats["amplification"] = dict(
                source_database=str(direct / "database.json"),
                source_sha256=fingerprint(direct / "database.json"),
                max_j=bound, max_raw_t=max_raw_t, optimized_entries=len(entries),
                amplified_entries=sum(e["amplification_rounds"] > 0 for e in entries.values()),
                objective="minimum expected T among j=0..max_j within attempt T cap",
                integrated_with_direct_search=True)
        register_output(output, "direct_search")
        summary = export_database(entries, variant_stats, output, export_qasm=export_qasm,
                                  export_qasm2=export_qasm2)
        if not summary["export_ok"] or summary["invalid_entries"]:
            raise ValueError(f"Export/readback validation failed: {output}")
        if plots:
            from .research import plot_database
            plot_database(entries.values(), output)
        summaries[variant] = dict(summary, output_dir=str(output))
    return summaries
