#!/usr/bin/env python3
"""4-way 50-batch end-to-end wallclock comparison + figures (GPU baseline).

Compares per (HW, ds, bs) four sources:
  * sim_off  — LLMServingSim with chunked_prefill / prefix_caching OFF
  * sim_on   — LLMServingSim with both ON
  * vllm_off — measure_vllm.py (CUDA) with both OFF
  * vllm_on  — measure_vllm.py with both ON

Each LENS run = 50 sequential batches; total = sum(per-batch_e2e_ms)
(averaged over n_runs replicates if any).
Sim total = max(end_time) - min(arrival), ns -> ms.

Result tree shape (read by `collect()`):
  results/{sim,lens_vllm}/<hw>/<opt>/<model>/tp<N>/bs<B>/<dataset>.csv
  where <opt> ∈ {"off", "on"}.

Usage:
    python compare.py --hardware L4
    python compare.py --hardware A10G --tps 1
    python compare.py --hardware L4 --no-figs
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent / "results"
FIG_DIR_BASE = Path(__file__).parent / "figures"
DATASETS = ("arxiv", "cnn", "sharegpt", "writing_prompts")

# Each source key maps to (results sub-tree, opt label, plot color, label).
SOURCES = (
    ("sim_off",  "sim",       "off", "tab:blue",   "Sim (off)"),
    ("sim_on",   "sim",       "on",  "tab:cyan",   "Sim (on)"),
    ("vllm_off", "lens_vllm", "off", "tab:green",  "vLLM (off)"),
    ("vllm_on",  "lens_vllm", "on",  "tab:olive",  "vLLM (on)"),
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def lens_total(path: Path) -> Optional[float]:
    """Sum batch_e2e_ms across batch_ids; replicates per batch are averaged."""
    if not path.exists():
        return None
    by_batch: dict[int, list[float]] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            if r.get("status") != "OK":
                continue
            bid = int(r["combo_id"]) if "combo_id" in r else int(r["run_id"])
            by_batch.setdefault(bid, []).append(float(r["batch_e2e_ms"]))
    if not by_batch:
        return None
    return sum(statistics.fmean(v) for v in by_batch.values())


def sim_total(path: Path) -> Optional[float]:
    """max(end_time) - min(arrival), ns -> ms."""
    if not path.exists():
        return None
    arrivals: list[int] = []
    ends: list[int] = []
    with path.open() as f:
        for r in csv.DictReader(f):
            arrivals.append(int(r["arrival"]))
            ends.append(int(r["end_time"]))
    if not arrivals:
        return None
    return (max(ends) - min(arrivals)) / 1e6


def collect(hw: str, tp: int, batch_sizes, sim_model: str,
            lens_model: str) -> dict:
    """Return {(ds, bs): {source_key: ms}} for one (hardware, TP)."""
    out = {}
    for ds in DATASETS:
        for bs in batch_sizes:
            row = {}
            for key, subtree, opt, _color, _label in SOURCES:
                model = sim_model if subtree == "sim" else lens_model
                path = ROOT / subtree / hw / opt / model / f"tp{tp}" / f"bs{bs}" / f"{ds}.csv"
                loader = sim_total if subtree == "sim" else lens_total
                row[key] = loader(path)
            out[(ds, bs)] = row
    return out


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def print_table(hw: str, tp: int, data, batch_sizes):
    print()
    print("=" * 110)
    print(f"{hw}  TP={tp}  —  50-batch e2e total wallclock (ms)  "
          f"+ sim/vllm diff% (matched opts)")
    print("=" * 110)
    header_cells = ["dataset", "bs"] + [lbl for _, _, _, _, lbl in SOURCES] \
                   + ["sim/vllm off", "sim/vllm on"]
    widths = [16, 3] + [11] * 4 + [12, 12]
    print("  ".join(f"{c:<{w}}" if i < 2 else f"{c:>{w}}"
                    for i, (c, w) in enumerate(zip(header_cells, widths))))
    print("-" * 110)

    for ds in DATASETS:
        for bs in batch_sizes:
            d = data[(ds, bs)]
            cells = [f"{ds:<16}", f"{bs:>3}"]
            for key, *_ in SOURCES:
                v = d[key]
                cells.append(f"{v:>11.1f}" if v is not None else f"{'-':>11}")
            for opt in ("off", "on"):
                s, v = d[f"sim_{opt}"], d[f"vllm_{opt}"]
                if s is not None and v:
                    cells.append(f"{(s - v) / v * 100:+11.1f}%")
                else:
                    cells.append(f"{'-':>12}")
            print("  ".join(cells))
        print()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(hw: str, tps, batch_sizes, all_data, model, fig_dir):
    """Grouped bar charts: per (TP, bs) subplot, x = 4 datasets,
    4 bars per dataset (sim_off / sim_on / vllm_off / vllm_on)."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir.mkdir(parents=True, exist_ok=True)

    n_methods = len(SOURCES)
    width = 0.20

    FS_SUPTITLE   = 24
    FS_SUBTITLE   = 20
    FS_AXISLABEL  = 18
    FS_TICKLABEL  = 16
    FS_LEGEND     = 16
    FS_NA_TEXT    = 11

    def _plot_one(ax, data, tp, bs):
        x = np.arange(len(DATASETS))
        for i, (key, _subtree, _opt, color, label) in enumerate(SOURCES):
            offset = (i - (n_methods - 1) / 2) * width
            vals = []
            for ds in DATASETS:
                v = data[(ds, bs)][key]
                vals.append(v / 1000 if v is not None else np.nan)
            ax.bar(x + offset, vals, width, color=color,
                   label=label, edgecolor="black", linewidth=0.5)
            for j, v in enumerate(vals):
                if not np.isfinite(v):
                    ax.text(x[j] + offset, 0, "N/A", ha="center", va="bottom",
                            rotation=90, fontsize=FS_NA_TEXT, color=color, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS, rotation=20, ha="right", fontsize=FS_TICKLABEL)
        ax.set_title(f"TP={tp}  bs={bs}", fontsize=FS_SUBTITLE)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="y", labelsize=FS_TICKLABEL)

    # ---- Figure 1: grid (rows = TP, cols = bs). Single PNG. ----
    n_rows = len(tps)
    n_cols = len(batch_sizes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5.5 * n_rows),
                             sharey=False, squeeze=False)
    handles_for_legend = None
    for r, tp in enumerate(tps):
        data = all_data[tp]
        for c, bs in enumerate(batch_sizes):
            ax = axes[r, c]
            _plot_one(ax, data, tp, bs)
            if c == 0:
                ax.set_ylabel("50-batch e2e total (s)", fontsize=FS_AXISLABEL)
            if handles_for_legend is None:
                handles_for_legend = ax.get_legend_handles_labels()
    fig.legend(*handles_for_legend, loc="upper center",
               bbox_to_anchor=(0.5, 1.01), ncol=n_methods,
               fontsize=FS_LEGEND, frameon=True, edgecolor="black")
    fig.suptitle(f"{hw} — {model} end-to-end offline latency",
                 fontsize=FS_SUPTITLE, y=1.04)
    plt.tight_layout()
    out = fig_dir / "e2e_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

    # ---- Figure 2: per (TP, bs) separate PNGs. ----
    per_dir = fig_dir / "per_tp_bs"
    per_dir.mkdir(parents=True, exist_ok=True)
    for tp in tps:
        data = all_data[tp]
        for bs in batch_sizes:
            fig, ax = plt.subplots(figsize=(10, 6.5))
            _plot_one(ax, data, tp, bs)
            ax.set_ylabel("50-batch e2e total (s)", fontsize=FS_AXISLABEL)
            ax.legend(loc="best", fontsize=FS_LEGEND,
                      frameon=True, edgecolor="black")
            fig.suptitle(f"{hw} — {model} end-to-end offline latency",
                         fontsize=FS_SUPTITLE)
            plt.tight_layout()
            out = per_dir / f"tp{tp}_bs{bs}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
    print(f"  saved {per_dir}/tp{{{','.join(str(t) for t in tps)}}}_"
          f"bs{{{','.join(str(b) for b in batch_sizes)}}}.png "
          f"({len(tps) * len(batch_sizes)} files)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hardware", required=True,
                   help="GPU label matching results/{sim,lens_vllm}/<hw>/<opt>/... "
                        "(e.g. L4, A10G).")
    p.add_argument("--sim-model", default="Llama-3.2-1B-Instruct",
                   help="Sim folder name (default: Llama-3.2-1B-Instruct).")
    p.add_argument("--lens-model", default=None,
                   help="LENS folder name. Default: same as --sim-model.")
    p.add_argument("--tps", default="1", help="Comma list (default: 1)")
    p.add_argument("--batch-sizes", default="1,2,4,8,16,32",
                   help="Comma list (default: 1,2,4,8,16,32)")
    p.add_argument("--no-table", action="store_true")
    p.add_argument("--no-figs", action="store_true")
    args = p.parse_args()

    lens_model = args.lens_model or args.sim_model
    tps = [int(x) for x in args.tps.split(",") if x]
    bs_list = [int(x) for x in args.batch_sizes.split(",") if x]

    all_data = {tp: collect(args.hardware, tp, bs_list, args.sim_model, lens_model)
                for tp in tps}

    if not args.no_table:
        for tp in tps:
            print_table(args.hardware, tp, all_data[tp], bs_list)

    if not args.no_figs:
        fig_dir = FIG_DIR_BASE / args.hardware
        print()
        print("=" * 90)
        print(f"Figures → {fig_dir}")
        print("=" * 90)
        make_figures(args.hardware, tps, bs_list, all_data, args.sim_model, fig_dir)


if __name__ == "__main__":
    main()
