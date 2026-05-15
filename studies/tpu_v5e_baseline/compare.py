#!/usr/bin/env python3
"""3-way 50-batch end-to-end wallclock comparison + figures.

LLMServingSim vs vLLM (per-run) vs vLLM (continuous-batched).

Each LENS run = 50 sequential batches; total = sum(per-batch_e2e_ms)
(averaged over n_runs replicates if any).
Sim total = max(end_time) - min(arrival), ns -> ms.

For each TP we plot 3 rows:
    row 0 : absolute wallclock
    row 1 : sim scaled by SF_vllm      = median(vllm / sim)
    row 2 : sim scaled by SF_vllm_cont = median(vllm_cont / sim)

Usage:
    python compare.py                       # default TPs (1), all bs + figures
    python compare.py --tps 1,2             # multi-TP
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
    """Return {(ds, bs): {'sim', 'vllm', 'vllm_cont'}} for one TP."""
    out = {}
    for ds in DATASETS:
        for bs in batch_sizes:
            out[(ds, bs)] = {
                "sim":       sim_total(ROOT / f"sim/{model}/tp{tp}/bs{bs}/{ds}.csv"),
                "vllm":      lens_total(ROOT / f"lens_vllm/{lens_model}/tp{tp}/bs{bs}/{ds}.csv"),
                "vllm_cont": lens_total(ROOT / f"lens_vllm_continuous/{lens_model}/tp{tp}/bs{bs}/{ds}.csv"),
            }
    return out


def compute_sf(data: dict, ref_key: str, sim_key: str = "sim",
               exclude_dataset: Optional[str] = None,
               include_only_dataset: Optional[str] = None) -> float:
    """Global scaling factor: median(ref / sim) across (ds, bs) cells where
    both ref and sim are present.

    - exclude_dataset       : drop that dataset (leave-one-dataset-out)
    - include_only_dataset  : keep only that dataset (per-dataset SF)
    """
    ratios = []
    for (ds, bs), d in data.items():
        if exclude_dataset is not None and ds == exclude_dataset:
            continue
        if include_only_dataset is not None and ds != include_only_dataset:
            continue
        if d.get(sim_key) and d.get(ref_key):
            ratios.append(d[ref_key] / d[sim_key])
    return statistics.median(ratios) if ratios else 1.0


def leave_out_sf_table(data: dict) -> dict:
    """Return {None: (sf_vllm, sf_vc), 'arxiv': (...), ...} — all-included + LOO."""
    out = {None: (compute_sf(data, "vllm"), compute_sf(data, "vllm_cont"))}
    for ds in DATASETS:
        out[ds] = (compute_sf(data, "vllm", exclude_dataset=ds),
                   compute_sf(data, "vllm_cont", exclude_dataset=ds))
    return out


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def print_table(tp: int, data, batch_sizes, sf_vllm: float, sf_vllm_cont: float):
    print()
    print("=" * 100)
    print(f"TP={tp}  —  50-batch e2e total wallclock (ms)")
    print(f"SF_vllm = {sf_vllm:.3f}   SF_vllm_cont = {sf_vllm_cont:.3f}")
    print("=" * 100)
    h_ds, h_bs = "dataset", "bs"
    h_sim, h_vllm, h_vc = "sim", "vllm", "vllm_cont"
    h_sv, h_svc = "sim/vllm", "sim/vc"
    print(f"{h_ds:<16} {h_bs:>3}  {h_sim:>10}  {h_vllm:>10}  {h_vc:>10}  {h_sv:>9}  {h_svc:>9}")
    print("-" * 80)
    for ds in DATASETS:
        for bs in batch_sizes:
            d = data[(ds, bs)]
            s, v, vc = d["sim"], d["vllm"], d["vllm_cont"]
            ss  = f"{s:>10.1f}"  if s  is not None else f'{"-":>10}'
            vs  = f"{v:>10.1f}"  if v  is not None else f'{"-":>10}'
            vcs = f"{vc:>10.1f}" if vc is not None else f'{"-":>10}'
            sv  = f"{(s-v)/v*100:+8.1f}%"   if (s is not None and v)  else f'{"-":>9}'
            svc = f"{(s-vc)/vc*100:+8.1f}%" if (s is not None and vc) else f'{"-":>9}'
            print(f"{ds:<16} {bs:>3}  {ss}  {vs}  {vcs}  {sv}  {svc}")
        print()


def print_leave_out_sf(tp: int, loo: dict):
    """Print leave-one-dataset-out SF table for one TP.
    loo is the output of leave_out_sf_table()."""
    print()
    print("=" * 60)
    print(f"TP={tp}  —  Leave-one-dataset-out SF")
    print("=" * 60)
    print(f"{'excluded':<24} {'SF_vllm':>10} {'SF_vllm_cont':>14}")
    print("-" * 60)
    sf_v, sf_vc = loo[None]
    print(f"{'(none — all datasets)':<24} {sf_v:>10.3f} {sf_vc:>14.3f}")
    for ds in DATASETS:
        sf_v, sf_vc = loo[ds]
        print(f"{('exclude ' + ds):<24} {sf_v:>10.3f} {sf_vc:>14.3f}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

ROW_KINDS = ("abs", "scale_vllm", "scale_vllm_cont")
ROW_LABEL = {
    "abs":              "Wallclock (s)",
    "scale_vllm":       "Scaled by vLLM (s)",
    "scale_vllm_cont":  "Scaled by vLLM-cont (s)",
}
METHODS = (("sim", "tab:blue"), ("vllm", "tab:green"), ("vllm_cont", "tab:red"))
METHOD_LABEL = {"sim": "LLMServingSim2.0", "vllm": "vLLM", "vllm_cont": "vLLM-cont"}


def apply_methods_filter(methods_to_show):
    """Restrict plotting to a subset of {sim, vllm, vllm_cont}.

    Mutates module globals:
    - METHODS    : drops bars not in methods_to_show
    - ROW_KINDS  : drops the scale_* row for any ref method that was dropped
    - FIG_DIR    : appends a `_no_<m>` suffix to avoid clobbering the default
                   run's output

    `sim` is forced into the set (we always plot sim as the bar being scaled)."""
    global METHODS, ROW_KINDS, FIG_DIR
    keep = set(methods_to_show) | {"sim"}
    METHODS = tuple((k, c) for k, c in METHODS if k in keep)
    rks = ["abs"]
    if "vllm"      in keep: rks.append("scale_vllm")
    if "vllm_cont" in keep: rks.append("scale_vllm_cont")
    ROW_KINDS = tuple(rks)
    canonical = {"sim", "vllm", "vllm_cont"}
    excluded = canonical - keep
    if excluded:
        suffix = "_no_" + "_".join(sorted(excluded))
        FIG_DIR = FIG_DIR.parent / (FIG_DIR.name + suffix)


def _sim_value(s: Optional[float], row_kind: str, sf_vllm: float, sf_vllm_cont: float) -> Optional[float]:
    if s is None:
        return None
    if row_kind == "abs":              return s
    if row_kind == "scale_vllm":       return s * sf_vllm
    if row_kind == "scale_vllm_cont":  return s * sf_vllm_cont
    raise ValueError(row_kind)


def _plot_one(ax, data, tp, bs, row_kind, sf_vllm, sf_vllm_cont, datasets=None):
    import numpy as np
    if datasets is None:
        datasets = DATASETS
    n_methods = len(METHODS)
    width = 0.27
    x = np.arange(len(datasets))

    FS_SUBTITLE  = 18
    FS_TICKLABEL = 14
    FS_NA_TEXT   = 11

    for i, (key, color) in enumerate(METHODS):
        offset = (i - (n_methods - 1) / 2) * width
        vals = []
        for ds in datasets:
            v = data[(ds, bs)][key]
            if key == "sim":
                v = _sim_value(v, row_kind, sf_vllm, sf_vllm_cont)
            vals.append(v / 1000 if v is not None else float("nan"))
        ax.bar(x + offset, vals, width, color=color,
               label=METHOD_LABEL[key], edgecolor="black", linewidth=0.5)
        for j, v in enumerate(vals):
            if v != v:  # NaN
                ax.text(x[j] + offset, 0, "N/A", ha="center", va="bottom",
                        rotation=90, fontsize=FS_NA_TEXT, color=color, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right", fontsize=FS_TICKLABEL)

    # title: bs always; for scale rows include the SF used
    if row_kind == "abs":
        title = f"TP={tp}  bs={bs}"
    elif row_kind == "scale_vllm":
        title = f"TP={tp}  bs={bs}  (SF={sf_vllm:.2f})"
    else:
        title = f"TP={tp}  bs={bs}  (SF={sf_vllm_cont:.2f})"
    ax.set_title(title, fontsize=FS_SUBTITLE)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=FS_TICKLABEL)


def make_figures(tps, batch_sizes, all_data, all_sf, model):
    """For each TP, draw 3 rows × len(batch_sizes) cols.
    Combined figure stacks all TPs vertically (3 * len(tps) total rows).
    Per (tp, bs) PNG: 3 rows × 1 col."""
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    n_methods = len(METHODS)

    FS_SUPTITLE  = 24
    FS_AXISLABEL = 18
    FS_LEGEND    = 18

    # ---- Figure 1: combined grid (rows = tps × 3 row_kinds, cols = bs). ----
    n_rows = len(tps) * len(ROW_KINDS)
    n_cols = len(batch_sizes)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 5.5 * n_rows),
                             sharey=False, squeeze=False)
    handles_for_legend = None
    for ti, tp in enumerate(tps):
        data = all_data[tp]
        sf_vllm, sf_vllm_cont = all_sf[tp]
        for ki, row_kind in enumerate(ROW_KINDS):
            r = ti * len(ROW_KINDS) + ki
            for c, bs in enumerate(batch_sizes):
                ax = axes[r, c]
                _plot_one(ax, data, tp, bs, row_kind, sf_vllm, sf_vllm_cont)
                if c == 0:
                    ax.set_ylabel(ROW_LABEL[row_kind], fontsize=FS_AXISLABEL)
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

    # ---- Figure 2: per (tp, bs) separate PNGs — 3 rows × 1 col each. ----
    per_dir = FIG_DIR / "per_tp_bs"
    per_dir.mkdir(parents=True, exist_ok=True)
    for tp in tps:
        data = all_data[tp]
        sf_vllm, sf_vllm_cont = all_sf[tp]
        for bs in batch_sizes:
            fig, axes = plt.subplots(len(ROW_KINDS), 1, figsize=(10, 6.5 * len(ROW_KINDS)),
                                     sharey=False, squeeze=False)
            for ki, row_kind in enumerate(ROW_KINDS):
                ax = axes[ki, 0]
                _plot_one(ax, data, tp, bs, row_kind, sf_vllm, sf_vllm_cont)
                ax.set_ylabel(ROW_LABEL[row_kind], fontsize=FS_AXISLABEL)
            axes[0, 0].legend(loc="best", fontsize=FS_LEGEND,
                              frameon=True, edgecolor="black")
            fig.suptitle(f"{model} end-to-end offline latency",
                         fontsize=FS_SUPTITLE)
            plt.tight_layout()
            out = per_dir / f"tp{tp}_bs{bs}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
    print(f"  saved {per_dir}/tp{{...}}_bs{{...}}.png "
          f"({len(tps) * len(batch_sizes)} files)")


def _plot_per_dataset(ax, data, tp, ds, batch_sizes, row_kind, sf_vllm, sf_vllm_cont):
    """Same as _plot_one but x-axis is batch_size (one dataset per axes)."""
    import numpy as np
    n_methods = len(METHODS)
    width = 0.27
    x = np.arange(len(batch_sizes))

    FS_SUBTITLE  = 18
    FS_TICKLABEL = 14
    FS_NA_TEXT   = 11

    for i, (key, color) in enumerate(METHODS):
        offset = (i - (n_methods - 1) / 2) * width
        vals = []
        for bs in batch_sizes:
            v = data[(ds, bs)][key]
            if key == "sim":
                v = _sim_value(v, row_kind, sf_vllm, sf_vllm_cont)
            vals.append(v / 1000 if v is not None else float("nan"))
        ax.bar(x + offset, vals, width, color=color,
               label=METHOD_LABEL[key], edgecolor="black", linewidth=0.5)
        for j, v in enumerate(vals):
            if v != v:
                ax.text(x[j] + offset, 0, "N/A", ha="center", va="bottom",
                        rotation=90, fontsize=FS_NA_TEXT, color=color, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"bs={bs}" for bs in batch_sizes], fontsize=FS_TICKLABEL)

    if row_kind == "abs":
        title = f"TP={tp}  {ds}"
    elif row_kind == "scale_vllm":
        title = f"TP={tp}  {ds}  (SF={sf_vllm:.2f})"
    else:
        title = f"TP={tp}  {ds}  (SF={sf_vllm_cont:.2f})"
    ax.set_title(title, fontsize=FS_SUBTITLE)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=FS_TICKLABEL)


def make_per_dataset_figures(tps, batch_sizes, all_data, model):
    """One PNG per dataset, per TP.  3 rows × 1 col, each cell shows all
    bs at once for that dataset (3 method-bars per bs).  SF for the two
    scaled rows is the **per-dataset** median(ref/sim) — not the global
    SF — so each PNG aligns sim to its own dataset's vLLM target."""
    import matplotlib.pyplot as plt

    out_dir = FIG_DIR / "per_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    FS_SUPTITLE  = 24
    FS_AXISLABEL = 18
    FS_LEGEND    = 18

    files = 0
    for tp in tps:
        data = all_data[tp]
        for ds in DATASETS:
            # per-dataset SF — use only this dataset's cells, not the global pool
            sf_vllm      = compute_sf(data, "vllm",      include_only_dataset=ds)
            sf_vllm_cont = compute_sf(data, "vllm_cont", include_only_dataset=ds)
            fig, axes = plt.subplots(len(ROW_KINDS), 1,
                                     figsize=(max(10, 1.4 * len(batch_sizes) + 4),
                                              6.5 * len(ROW_KINDS)),
                                     sharey=False, squeeze=False)
            for ki, row_kind in enumerate(ROW_KINDS):
                ax = axes[ki, 0]
                _plot_per_dataset(ax, data, tp, ds, batch_sizes,
                                  row_kind, sf_vllm, sf_vllm_cont)
                ax.set_ylabel(ROW_LABEL[row_kind], fontsize=FS_AXISLABEL)
            axes[0, 0].legend(loc="best", fontsize=FS_LEGEND,
                              frameon=True, edgecolor="black")
            fig.suptitle(f"{model}  TP={tp}  —  {ds}", fontsize=FS_SUPTITLE)
            plt.tight_layout()
            out = out_dir / f"tp{tp}_{ds}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            files += 1
    print(f"  saved {out_dir}/tp{{...}}_{{ds}}.png ({files} files)")


def make_leave_out_figures(tps, batch_sizes, all_data, all_loo, model):
    """One PNG per excluded dataset, per TP.
    Each PNG = 3 rows (abs, scale_vllm, scale_vllm_cont) × len(bs) cols,
    using the leave-one-out SF for that excluded dataset.  The 'abs' row
    is identical across leave-out variants but kept for direct in-figure
    visual comparison against the scaled rows."""
    import matplotlib.pyplot as plt

    out_dir = FIG_DIR / "leave_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_methods = len(METHODS)

    FS_SUPTITLE  = 24
    FS_AXISLABEL = 18
    FS_LEGEND    = 18

    n_cols = len(batch_sizes)
    n_rows = len(ROW_KINDS)

    files = 0
    for tp in tps:
        data = all_data[tp]
        loo = all_loo[tp]
        for ex_ds in DATASETS:
            sf_vllm, sf_vllm_cont = loo[ex_ds]
            kept_datasets = tuple(d for d in DATASETS if d != ex_ds)
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(5.5 * n_cols, 5.5 * n_rows),
                                     sharey=False, squeeze=False)
            handles_for_legend = None
            for ki, row_kind in enumerate(ROW_KINDS):
                for c, bs in enumerate(batch_sizes):
                    ax = axes[ki, c]
                    _plot_one(ax, data, tp, bs, row_kind, sf_vllm, sf_vllm_cont,
                              datasets=kept_datasets)
                    if c == 0:
                        ax.set_ylabel(ROW_LABEL[row_kind], fontsize=FS_AXISLABEL)
                    if handles_for_legend is None:
                        handles_for_legend = ax.get_legend_handles_labels()
            fig.legend(*handles_for_legend, loc="upper center",
                       bbox_to_anchor=(0.5, 1.01), ncol=n_methods,
                       fontsize=FS_LEGEND, frameon=True, edgecolor="black")
            fig.suptitle(f"{model}  TP={tp}  —  SF excluding '{ex_ds}'  "
                         f"(SF_vllm={sf_vllm:.2f}, SF_vllm_cont={sf_vllm_cont:.2f})",
                         fontsize=FS_SUPTITLE, y=1.04)
            plt.tight_layout()
            out = out_dir / f"tp{tp}_exclude_{ex_ds}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            files += 1
    print(f"  saved {out_dir}/tp{{...}}_exclude_{{...}}.png ({files} files)")


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
    p.add_argument("--tps", default="1", help="Comma list (default: 1)")
    p.add_argument("--batch-sizes", default="1,2,4,8,16,32",
                   help="Comma list (default: 1,2,4,8,16,32)")
    p.add_argument("--methods", default="sim,vllm,vllm_cont",
                   help="Comma-separated subset of {sim, vllm, vllm_cont}. "
                        "Drops the corresponding bar + scaling row from "
                        "figures and routes output to a separate figures "
                        "subdir.  sim is always included.  "
                        "E.g. '--methods sim,vllm_cont' compares sim only "
                        "against vLLM-continuous.")
    p.add_argument("--no-table", action="store_true")
    p.add_argument("--no-figs", action="store_true")
    args = p.parse_args()

    lens_model = args.lens_model or f"{args.model}-Instruct"
    tps = [int(x) for x in args.tps.split(",") if x]
    bs_list = [int(x) for x in args.batch_sizes.split(",") if x]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    apply_methods_filter(methods)

    all_data = {tp: collect(tp, bs_list, args.model, lens_model) for tp in tps}
    all_loo  = {tp: leave_out_sf_table(all_data[tp]) for tp in tps}
    all_sf   = {tp: all_loo[tp][None] for tp in tps}    # all-included SF

    if not args.no_table:
        for tp in tps:
            sf_v, sf_vc = all_sf[tp]
            print_table(tp, all_data[tp], bs_list, sf_v, sf_vc)
            print_leave_out_sf(tp, all_loo[tp])

    if not args.no_figs:
        print()
        print("=" * 90)
        print("Figures")
        print("=" * 90)
        make_figures(tps, bs_list, all_data, all_sf, args.model)
        make_per_dataset_figures(tps, bs_list, all_data, args.model)
        make_leave_out_figures(tps, bs_list, all_data, all_loo, args.model)


if __name__ == "__main__":
    main()
