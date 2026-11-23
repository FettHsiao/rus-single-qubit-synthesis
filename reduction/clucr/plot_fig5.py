"""Plot all reduced RUS entries in the style of 1311.1074v2 Fig. 5."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rus-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np


def histogram_panel(ax, edges, before, after, unchanged, *, legend=True):
    width = np.diff(edges)
    ax.bar(edges[:-1], after, width=width, align="edge", color="#7767cf",
           edgecolor="#352a75", linewidth=.45, alpha=.70)
    ax.bar(edges[:-1], before, width=width, align="edge", color="#f1ab66",
           edgecolor="#8d5329", linewidth=.45, alpha=.42)
    ax.bar(edges[:-1], unchanged, width=width, align="edge", facecolor="none",
           edgecolor="#3d3446", linewidth=.35, hatch="////")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, max(1, before.max(), after.max()) * 1.13)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if legend:
        ax.legend(handles=[Patch(facecolor="#7767cf", alpha=.70, label="Amplified"),
                           Patch(facecolor="#f1ab66", alpha=.60, label="Unamplified"),
                           Patch(facecolor="white", edgecolor="#3d3446", hatch="////", label="Unaffected circuits")],
                  loc="upper right", frameon=False, fontsize=9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument("--database", "--input-dir", type=Path, default=base.parent / "out_record_paper/database.json")
    parser.add_argument("--output", "--output-dir", type=Path, default=base.parent.parent / "draw/fig5")
    parser.add_argument("--bin-width", type=float, default=2.)
    args = parser.parse_args()
    if not np.isfinite(args.bin_width) or args.bin_width <= 0:
        parser.error("--bin-width must be finite and positive")
    if args.database.is_dir():
        args.database /= "database.json"
    source = args.database.read_bytes()
    database = json.loads(source)
    if database.get("algorithm") != "clucr":
        raise ValueError("Expected a completed clucr database")
    entries = database["entries"]
    if not entries or len({e["entry_id"] for e in entries}) != len(entries):
        raise ValueError("Expected nonempty, uniquely identified RUS entries")
    raw = np.array([e["raw_t_count"] for e in entries], dtype=float)
    p = np.array([e["success_probability"] for e in entries])
    before = np.array([e["expected_t_count"] for e in entries])
    after = np.array([e["effective_t_count"] for e in entries])
    rounds = np.array([e["amplification_rounds"] for e in entries], dtype=float)
    if (not all(np.isfinite(a).all() for a in (raw, p, before, after, rounds))
            or np.any(raw != np.floor(raw)) or np.any(raw < 0)
            or np.any((p <= 0) | (p > 1)) or np.any(rounds != np.floor(rounds)) or np.any(rounds < 0)):
        raise ValueError("Invalid circuit statistics")
    np.testing.assert_allclose(before, raw / p, rtol=1e-10, atol=1e-10)
    amplified_p = np.sin((2 * rounds + 1) * np.arcsin(np.sqrt(p))) ** 2
    np.testing.assert_allclose(after, (2 * rounds + 1) * raw / amplified_p, rtol=1e-10, atol=1e-10)
    if np.any(after > before + 1e-9):
        raise ValueError("Stored amplification increases expected T cost")
    unaffected = rounds == 0
    raw_values = np.arange(int(raw.min()), int(raw.max()) + 1)
    probability_edges = np.linspace(0, 1, 11)
    probability_bin = np.minimum(np.searchsorted(probability_edges, p, side="right") - 1, 9)
    stacked = np.array([[np.count_nonzero((raw == t) & (probability_bin == i))
                         for t in raw_values] for i in range(10)])
    assert stacked.sum() == len(entries)
    bin_count = int(np.ceil(max(100, before.max(), after.max()) / args.bin_width))
    full_edges = np.arange(bin_count + 1) * args.bin_width
    upper = full_edges[-1]
    paper_edges = np.unique(np.r_[np.arange(0, 100, args.bin_width), 100.])
    from output_paths import register_output
    register_output(args.output, "fig5")
    histograms = {}
    for name, edges in (("all", full_edges), ("paper_range", paper_edges)):
        histograms[name] = [np.histogram(a, edges)[0] for a in (before, after, before[unaffected])]
        with (args.output / f"expected_t_{name}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["left_edge", "right_edge", "unamplified", "amplified", "unaffected"])
            writer.writerows(zip(edges[:-1], edges[1:], *histograms[name]))
    assert histograms["all"][0].sum() == histograms["all"][1].sum() == len(entries)
    with (args.output / "raw_t_success_probability.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["raw_t_count"] + [f"p_{i/10:.1f}_to_{(i+1)/10:.1f}" for i in range(10)])
        writer.writerows(zip(raw_values, *stacked))
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11,
                         "axes.labelsize": 12, "axes.linewidth": .7, "hatch.linewidth": .4,
                         "xtick.direction": "in", "ytick.direction": "in",
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    colors = ["#e5262b", "#f06428", "#ee9737", "#d0b941", "#a7bf4a",
              "#77b565", "#54aaa1", "#4c94bb", "#4468c4", "#4933ad"]
    output_files = []
    for mode, edges in (("all", full_edges), ("paper_range", paper_edges)):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 5.8))
        fig.subplots_adjust(left=.075, right=.98, bottom=.20, top=.85, wspace=.23)
        fig.suptitle(f"RUS circuit statistics after Clifford reduction (N = {len(entries):,})", y=.965, fontsize=16)
        fig.text(.5, .91, "All axial and non-axial winners; one circuit per Clifford equivalence class", ha="center", fontsize=10, color="#555555")
        bottom = np.zeros(len(raw_values))
        bars = []
        for i, counts in enumerate(stacked):
            bar = ax.bar(raw_values, counts, bottom=bottom, width=.9, color=colors[i],
                         edgecolor="#242424", linewidth=.38)
            bars.append(bar[0])
            bottom += counts
        ax.set(xlabel="Raw T count", ylabel="Number of circuits", xlim=(raw_values[0]-.7, raw_values[-1]+.7))
        ax.set_xticks(raw_values)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        labels = [f"{i/10:.1f} ≤ p < {(i+1)/10:.1f}" for i in range(10)]
        labels[-1] = "0.9 ≤ p ≤ 1.0"
        ax.legend(bars[::-1], labels[::-1], title="Pr[success]", loc="upper left", frameon=False, fontsize=8.8, title_fontsize=10)
        histogram_panel(bx, edges, *histograms[mode], legend=mode == "paper_range")
        bx.set(xlabel="Expected T count", ylabel="Number of circuits")
        if mode == "all":
            inset = bx.inset_axes([.33, .31, .64, .61])
            histogram_panel(inset, paper_edges, *histograms["paper_range"], legend=True)
            inset.set_title("Detail: expected T ≤ 100", fontsize=9)
            inset.tick_params(labelsize=8)
            note = f"All {len(entries):,} circuits included in both histograms. Full range: 0–{upper:g}; inset: 0–100."
        else:
            bx.set_xticks(np.arange(0, 101, 10))
            note = (f"Paper Fig. 5 range (≤100): {np.count_nonzero(before > 100)} unamplified and "
                    f"{np.count_nonzero(after > 100)} amplified values lie outside this panel.")
        ax.text(.5, -.15, "(a)", transform=ax.transAxes, ha="center")
        bx.text(.5, -.15, "(b)", transform=bx.transAxes, ha="center")
        fig.text(.5, .055, note, ha="center", fontsize=9)
        fig.text(.5, .018, f"Expected-T bin width = {args.bin_width:g}. Amplification costs use stored metadata; {np.count_nonzero(~unaffected):,} circuits benefit.", ha="center", fontsize=9, color="#555555")
        for extension in ("png", "pdf", "svg"):
            path = args.output / f"fig5_{mode}.{extension}"
            fig.savefig(path, dpi=240, facecolor="white")
            output_files.append(path.name)
        plt.close(fig)
    stats = dict(source_database=str(args.database.resolve()), source_sha256=hashlib.sha256(source).hexdigest(),
                 entries=len(entries), source_axial=sum(e["axial"] for e in entries),
                 source_non_axial=sum(not e["axial"] for e in entries),
                 amplified_entries=int((~unaffected).sum()), unaffected_entries=int(unaffected.sum()),
                 unamplified_over_100=int((before > 100).sum()), amplified_over_100=int((after > 100).sum()),
                 max_unamplified=float(before.max()), max_amplified=float(after.max()),
                 expected_t_bin_width=args.bin_width, probability_bin_edges=probability_edges.tolist(),
                 amplitude_amplification="Stored effective_t_count checked against (2j+1)T/sin²((2j+1)asin(sqrt(p))); no circuit synthesis or class reselection",
                 full_histogram_counts=[int(h.sum()) for h in histograms["all"]],
                 paper_histogram_counts=[int(h.sum()) for h in histograms["paper_range"]],
                 files=output_files)
    (args.output / "plot_summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
