#!/usr/bin/env python3
"""3-way 50-batch end-to-end wallclock comparison + figures.

LENS-NxD vs LENS-vLLM vs LLMServingSim.

Each LENS run = 50 sequential batches; total = sum(per-batch_e2e_ms)
(averaged over n_runs replicates if any).
Sim total = max(end_time) - min(arrival), ns -> ms.

Usage:
    python compare.py                       # all TPs, all bs, all ds + figures
    python compare.py --tps 1               # TP=1 only
    python compare.py --no-figs             # table only
    python compare.py --no-table            # figures only
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent / "results"
FIG_DIR = Path(__file__).parent / "figures"
DATASETS = ("arxiv", "cnn", "sharegpt", "writing_prompts")


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


def collect(tp: int, batch_sizes, model: str, lens_model: str) -> dict:
    """Return {(ds, bs): {'sim', 'tpu', 'vllm'}} for one TP."""
    out = {}
    for ds in DATASETS:
        for bs in batch_sizes:
            out[(ds, bs)] = {
                "sim":  sim_total(ROOT / f"sim/{model}/tp{tp}/bs{bs}/{ds}.csv"),
                "tpu":  lens_total(ROOT / f"lens_tpu/{lens_model}/tp{tp}/bs{bs}/{ds}.csv"),
                "vllm": lens_total(ROOT / f"lens_vllm/{lens_model}/tp{tp}/bs{bs}/{ds}.csv"),
            }
    return out


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def print_table(tp: int, data, batch_sizes):
    print()
    print("=" * 90)
    print(f"TP={tp}  —  50-batch e2e total wallclock (ms)")
    print("=" * 90)
    h_ds, h_bs = "dataset", "bs"
    h_sim, h_tpu, h_vllm = "sim", "tpu", "vllm"
    h_sn, h_sv = "sim/maxtext", "sim/vllm"
    print(f"{h_ds:<16} {h_bs:>3}  {h_sim:>10}  {h_tpu:>10}  {h_vllm:>10}  {h_sn:>9}  {h_sv:>9}")
    print("-" * 90)
    for ds in DATASETS:
        for bs in batch_sizes:
            d = data[(ds, bs)]
            s, n, v = d["sim"], d["tpu"], d["vllm"]
            ss = f"{s:>10.1f}" if s is not None else f'{"-":>10}'
            ns = f"{n:>10.1f}" if n is not None else f'{"-":>10}'
            vs = f"{v:>10.1f}" if v is not None else f'{"-":>10}'
            sn = f"{(s-n)/n*100:+8.1f}%" if (s is not None and n) else f'{"-":>9}'
            sv = f"{(s-v)/v*100:+8.1f}%" if (s is not None and v) else f'{"-":>9}'
            print(f"{ds:<16} {bs:>3}  {ss}  {ns}  {vs}  {sn}  {sv}")
        print()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(tps, batch_sizes, all_data, model):
    """Grouped bar charts: per (TP, bs) subplot, x = 4 datasets,
    3 bars per dataset (sim / lens_tpu / lens_vllm)."""
    import matplotlib.pyplot as plt
    import numpy as np

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    METHODS = (("sim", "tab:blue"), ("tpu", "tab:orange"), ("vllm", "tab:green"))
    METHOD_LABEL = {"sim": "LLMServingSim2.0", "tpu": "MaxText", "vllm": "vLLM"}
    n_methods = len(METHODS)
    width = 0.27   # per-bar width within a dataset group

    # Font sizes
    FS_SUPTITLE   = 24
    FS_SUBTITLE   = 20
    FS_AXISLABEL  = 18
    FS_TICKLABEL  = 16
    FS_LEGEND     = 18
    FS_NA_TEXT    = 12

    def _plot_one(ax, data, tp, bs):
        x = np.arange(len(DATASETS))
        for i, (key, color) in enumerate(METHODS):
            offset = (i - (n_methods - 1) / 2) * width
            vals = []
            for ds in DATASETS:
                v = data[(ds, bs)][key]
                vals.append(v / 1000 if v is not None else np.nan)
            ax.bar(x + offset, vals, width, color=color,
                   label=METHOD_LABEL[key], edgecolor="black", linewidth=0.5)
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
    fig.suptitle(f"{model} end-to-end offline latency",
                 fontsize=FS_SUPTITLE, y=1.04)
    plt.tight_layout()
    out = FIG_DIR / "e2e_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

    # ---- Figure 2: per (TP, bs) separate PNGs. ----
    per_dir = FIG_DIR / "per_tp_bs"
    per_dir.mkdir(parents=True, exist_ok=True)
    for tp in tps:
        data = all_data[tp]
        for bs in batch_sizes:
            fig, ax = plt.subplots(figsize=(10, 6.5))
            _plot_one(ax, data, tp, bs)
            ax.set_ylabel("50-batch e2e total (s)", fontsize=FS_AXISLABEL)
            ax.legend(loc="best", fontsize=FS_LEGEND,
                      frameon=True, edgecolor="black")
            fig.suptitle(f"{model} end-to-end offline latency",
                         fontsize=FS_SUPTITLE)
            plt.tight_layout()
            out = per_dir / f"tp{tp}_bs{bs}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
    print(f"  saved {per_dir}/tp{{1,2}}_bs{{1,2,4,8,16,32}}.png "
          f"({len(tps) * len(batch_sizes)} files)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Llama-3.2-1B",
                   help="Sim folder name (e.g. 'Llama-3.2-1B').")
    p.add_argument("--lens-model", default=None,
                   help="LENS folder name. Default: '<model>-Instruct'.")
    p.add_argument("--tps", default="1,2", help="Comma list (default: 1,2)")
    p.add_argument("--batch-sizes", default="1,2,4,8,16,32",
                   help="Comma list (default: 1,2,4,8,16,32)")
    p.add_argument("--no-table", action="store_true")
    p.add_argument("--no-figs", action="store_true")
    args = p.parse_args()

    lens_model = args.lens_model or f"{args.model}-Instruct"
    tps = [int(x) for x in args.tps.split(",") if x]
    bs_list = [int(x) for x in args.batch_sizes.split(",") if x]

    all_data = {tp: collect(tp, bs_list, args.model, lens_model) for tp in tps}

    if not args.no_table:
        for tp in tps:
            print_table(tp, all_data[tp], bs_list)

    if not args.no_figs:
        print()
        print("=" * 90)
        print("Figures")
        print("=" * 90)
        make_figures(tps, bs_list, all_data, args.model)


if __name__ == "__main__":
    main()
