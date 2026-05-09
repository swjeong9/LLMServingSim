#!/usr/bin/env python3
"""Compare LLMServingSim simulator output vs LENS measurement on Inf2.

For Phase 5 of the inf2.xlarge baseline validation. Joins request-level
predictions from the simulator with the corresponding LENS measurements
(both use the same workload, so request order matches), then reports
per-request abs/rel errors and a summary the user can act on:

  * mean abs error < 15%   → simulator baseline is useful, scale up
  * mean abs error 15-30%  → partial value, frame as approximate
  * mean abs error > 30%   → bucketing gap too large, paper pivot

Usage
-----
    python scripts/compare_inf2_baseline.py \\
        --sim outputs/inf2_llama1b_tp2_arxiv.csv \\
        --lens /Users/swjeong/Desktop/npu_chip_project/module4/realtime_llm_inference/inferentia2/Llama-3.2-1B-Instruct/results_arxiv.csv \\
        --output outputs/inf2_llama1b_tp2_arxiv_vs_lens.csv

Or to compare all three datasets at once for one TP:

    python scripts/compare_inf2_baseline.py --tp 2 --all
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
LENS_RESULTS_DIR = Path(
    "/Users/swjeong/Desktop/npu_chip_project/"
    "module4/realtime_llm_inference/inferentia2/Llama-3.2-1B-Instruct"
)
DATASETS = ("arxiv", "cnn", "sharegpt")


def load_lens(path: Path) -> List[Dict]:
    """Load LENS results CSV. Returns rows in order, OK-only."""
    out = []
    with path.open("r") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "OK":
                continue
            out.append({
                "id":         int(row["id"]),
                "input_len":  int(row["input_len"]),
                "output_len": int(row["output_len"]),
                "ttft_ms":    float(row["ttft_ms"]),
                "tpot_ms":    float(row["tpot_ms"]),
                "e2e_ms":     float(row["e2e_ms"]),
            })
    return out


def load_sim(path: Path) -> List[Dict]:
    """Load LLMServingSim per-request CSV. Times in ns → convert to ms."""
    out = []
    with path.open("r") as f:
        for row in csv.DictReader(f):
            out.append({
                "request_id": int(row["request id"]),
                "input":      int(row["input"]),
                "output":     int(row["output"]),
                "ttft_ms":    float(row["TTFT"])    / 1e6,
                "tpot_ms":    float(row["TPOT"])    / 1e6,
                "e2e_ms":     float(row["latency"]) / 1e6,
            })
    return out


def join_and_compare(lens: List[Dict], sim: List[Dict]) -> List[Dict]:
    """Match by request order (both have same workload) and verify
    (input_len, output_len) align as a sanity check."""
    if len(lens) != len(sim):
        print(f"  [warn] length mismatch: lens={len(lens)} sim={len(sim)}; "
              f"truncating to min")
    n = min(len(lens), len(sim))
    rows = []
    for i in range(n):
        l, s = lens[i], sim[i]
        if (l["input_len"], l["output_len"]) != (s["input"], s["output"]):
            print(f"  [warn] request {i}: shape mismatch "
                  f"lens=({l['input_len']},{l['output_len']}) "
                  f"sim=({s['input']},{s['output']}) — joining anyway")
        rows.append({
            "id":          i,
            "input_len":   l["input_len"],
            "output_len":  l["output_len"],
            "lens_ttft_ms": l["ttft_ms"],
            "sim_ttft_ms":  s["ttft_ms"],
            "ttft_abs_err_pct":    100 * abs(s["ttft_ms"] - l["ttft_ms"]) / l["ttft_ms"]
                                   if l["ttft_ms"] > 0 else 0.0,
            "ttft_signed_err_pct": 100 * (s["ttft_ms"] - l["ttft_ms"])    / l["ttft_ms"]
                                   if l["ttft_ms"] > 0 else 0.0,
            "lens_tpot_ms": l["tpot_ms"],
            "sim_tpot_ms":  s["tpot_ms"],
            "tpot_abs_err_pct":    100 * abs(s["tpot_ms"] - l["tpot_ms"]) / l["tpot_ms"]
                                   if l["tpot_ms"] > 0 else 0.0,
            "tpot_signed_err_pct": 100 * (s["tpot_ms"] - l["tpot_ms"])    / l["tpot_ms"]
                                   if l["tpot_ms"] > 0 else 0.0,
            "lens_e2e_ms": l["e2e_ms"],
            "sim_e2e_ms":  s["e2e_ms"],
            "e2e_abs_err_pct":    100 * abs(s["e2e_ms"] - l["e2e_ms"]) / l["e2e_ms"]
                                  if l["e2e_ms"] > 0 else 0.0,
            "e2e_signed_err_pct": 100 * (s["e2e_ms"] - l["e2e_ms"])    / l["e2e_ms"]
                                  if l["e2e_ms"] > 0 else 0.0,
        })
    return rows


def summarize(rows: List[Dict], label: str) -> Dict[str, float]:
    """Mean / median / p95 of abs and signed error per metric."""
    summary = {"label": label, "n": len(rows)}
    for metric in ("ttft", "tpot", "e2e"):
        abs_errs    = [r[f"{metric}_abs_err_pct"]    for r in rows]
        signed_errs = [r[f"{metric}_signed_err_pct"] for r in rows]
        summary[f"{metric}_mean_abs"]   = statistics.fmean(abs_errs)
        summary[f"{metric}_median_abs"] = statistics.median(abs_errs)
        summary[f"{metric}_p95_abs"]    = sorted(abs_errs)[int(0.95 * len(abs_errs))] \
                                          if abs_errs else 0
        summary[f"{metric}_mean_signed"] = statistics.fmean(signed_errs)
    return summary


def print_summary(s: Dict[str, float]) -> None:
    print(f"\n=== {s['label']}  (n={s['n']}) ===")
    print(f"{'metric':<8}  {'mean abs':>10}  {'median abs':>11}  {'p95 abs':>9}  {'mean signed':>12}")
    print("-" * 60)
    for metric in ("ttft", "tpot", "e2e"):
        print(f"{metric:<8}  "
              f"{s[f'{metric}_mean_abs']:>9.2f}%  "
              f"{s[f'{metric}_median_abs']:>10.2f}%  "
              f"{s[f'{metric}_p95_abs']:>8.2f}%  "
              f"{s[f'{metric}_mean_signed']:>+11.2f}%")


def verdict(summaries: List[Dict[str, float]]) -> None:
    """Aggregate verdict across datasets — Phase 5 decision."""
    overall_e2e_mean = statistics.fmean(s["e2e_mean_abs"] for s in summaries)
    print()
    print("=" * 60)
    print(f"Overall mean e2e abs error across {len(summaries)} datasets: "
          f"{overall_e2e_mean:.2f}%")
    print("=" * 60)
    if overall_e2e_mean < 15:
        print("✓ baseline is useful — proceed to Phase 6 (scale up to bigger inf2)")
    elif overall_e2e_mean < 30:
        print("△ partial value — frame as 'approximate baseline' in paper")
    else:
        print("✗ bucketing gap too large — paper framing pivot recommended")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim", help="LLMServingSim output CSV")
    p.add_argument("--lens", help="LENS results CSV")
    p.add_argument("--output", help="Combined comparison CSV (per-request)")
    p.add_argument("--tp", type=int, choices=[1, 2], help="TP (used with --all)")
    p.add_argument("--all", action="store_true",
                   help="Compare all 3 datasets (arxiv, cnn, sharegpt) for given --tp")
    p.add_argument("--sim-dir", default="outputs",
                   help="Directory holding sim output CSVs (when --all)")
    args = p.parse_args()

    if args.all:
        if args.tp is None:
            p.error("--all requires --tp {1|2}")
        summaries = []
        for ds in DATASETS:
            sim_path  = Path(args.sim_dir) / f"inf2_llama1b_tp{args.tp}_{ds}.csv"
            lens_path = LENS_RESULTS_DIR / f"results_{ds}.csv"
            out_path  = Path(args.sim_dir) / f"inf2_llama1b_tp{args.tp}_{ds}_vs_lens.csv"
            if not sim_path.exists():
                print(f"[skip] {sim_path} not found — run simulator first")
                continue
            print(f"\n--- {ds} ---")
            print(f"  sim : {sim_path}")
            print(f"  lens: {lens_path}")
            lens = load_lens(lens_path)
            sim  = load_sim(sim_path)
            rows = join_and_compare(lens, sim)
            with out_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            summary = summarize(rows, f"TP={args.tp} {ds}")
            print_summary(summary)
            summaries.append(summary)
            print(f"  → wrote {out_path}")
        if summaries:
            verdict(summaries)
    else:
        if not (args.sim and args.lens and args.output):
            p.error("--sim, --lens, --output required (or use --all --tp N)")
        lens = load_lens(Path(args.lens))
        sim  = load_sim(Path(args.sim))
        rows = join_and_compare(lens, sim)
        with Path(args.output).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        summary = summarize(rows, Path(args.sim).stem)
        print_summary(summary)
        verdict([summary])


if __name__ == "__main__":
    main()
