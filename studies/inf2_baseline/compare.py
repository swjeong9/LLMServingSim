#!/usr/bin/env python3
"""3-way compare for the inf2 baseline study.

Joins three measurements of the same (dataset, tp, batch_size) by
request order and reports per-batch and aggregated abs/signed errors:

  * LENS-NxD     — LENS run_profiling.py output (NxD-direct, no vLLM)
  * LENS-vLLM    — LENS run_profiling_vllm.py output (vLLM-Neuron)
  * Sim          — LLMServingSim per-request CSV

Each LENS row is one batch of size B → 50 batches × n_runs rows.
Sim's per-request output is grouped into batches of B consecutive
requests; batch_e2e = max(latency) across the B, batch_ttft = max(TTFT).

The directory layout under results/ is::

    results/lens_nxd/<model>/tp<N>/bs<B>/<dataset>.csv
    results/lens_vllm/<model>/tp<N>/bs<B>/<dataset>.csv
    results/sim/<model>/tp<N>/bs<B>/<dataset>.csv

Status: SKELETON. Will be finalized once we have the first real
LENS measurement output and confirm column names. Wire-up TODO marked
with ``# TODO`` comments.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Dict, List, Tuple


STUDY_ROOT = Path(__file__).resolve().parent
RESULTS = STUDY_ROOT / "results"
DATASETS = ("arxiv", "cnn", "sharegpt", "writing_prompts")
NUM_BATCHES = 50


def load_lens(path: Path) -> List[Dict]:
    """Parse LENS measurement CSV. Two formats supported:
      * measure_nxd.py / measure_vllm.py (dataset-driven, our path):
          run_id, status, batch_size, sample_ids, input_lens, output_lens,
          max_input_len, max_output_len, max_n_generated,
          batch_ttft_ms, batch_e2e_ms, error
        → run_id IS the batch_id.
      * LENS run_profiling.py (uniform-batch combo, legacy):
          run_id, combo_id, combo_il, combo_ol, batch_size, status, ...
        → combo_id is the batch_id; multiple run_id per combo (n_runs replicates).
    """
    out = []
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("status") != "OK":
                continue
            # batch_id resolution: combo_id if present (legacy),
            # else run_id (dataset-driven format).
            batch_id = int(row["combo_id"]) if "combo_id" in row else int(row["run_id"])
            out.append({
                "run_id":      int(row["run_id"]),
                "batch_id":    batch_id,
                "batch_ttft":  float(row["batch_ttft_ms"] or 0),
                "batch_e2e":   float(row["batch_e2e_ms"]),
            })
    return out


def load_sim(path: Path, batch_size: int) -> Tuple[List[Dict], float]:
    """Parse LLMServingSim per-request CSV. Returns (per_batch_rows, total_ms).

    Sim emits ns; convert to ms.

    * total_ms = max(end_time) - min(arrival): wall time to process all
      requests (the headline number — equivalent to LENS's sum-of-batches
      under sequential batches, and to vLLM's overall sweep time under
      continuous batching).
    * per-batch rows are an analytical view: chunk every batch_size
      consecutive requests (arrival order) and report per-chunk
      max(latency), max(TTFT). Useful for inspection but not for the
      headline error since chunk boundaries don't line up with vLLM's
      actual batching.
    """
    with path.open() as f:
        rows = list(csv.DictReader(f))
    arrivals  = [int(r["arrival"])  for r in rows]
    end_times = [int(r["end_time"]) for r in rows]
    total_ms = (max(end_times) - min(arrivals)) / 1e6

    batches = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        if len(chunk) < batch_size:
            break
        # Subtract queuing_delay so per-batch e2e is pure processing
        # time of that batch — comparable to LENS NxD's batch_e2e_ms
        # (each NxD batch starts fresh from 0, queueing-free). Without
        # this, sim's `latency` accumulates wait time for queued reqs
        # at small batch sizes, blowing up the per-batch error to
        # thousands of percent.
        def proc(r):
            return (float(r["latency"]) - float(r["queuing_delay"])) / 1e6
        batches.append({
            "batch_id":   i // batch_size,
            "batch_ttft": max(float(r["TTFT"]) / 1e6 for r in chunk),
            "batch_e2e":  max(proc(r) for r in chunk),
        })
    return batches, total_ms


def lens_total_ms(rows: List[Dict]) -> float:
    """LENS sweep is 50 sequential batches per measurement file →
    sum of per-batch e2e is the total time to process all requests.
    Average across replicate runs (n_runs) per batch_id first so we
    don't double-count when LENS recorded n_runs > 1."""
    by_batch: Dict[int, List[float]] = {}
    for r in rows:
        by_batch.setdefault(r["batch_id"], []).append(r["batch_e2e"])
    return sum(statistics.fmean(v) for v in by_batch.values())


def join_3way(lens_nxd, lens_vllm, sim) -> List[Dict]:
    """Match by batch_id. LENS rows include n_runs replicates; average them
    per batch_id before joining with Sim (which has one entry per batch)."""
    def avg_by_batch(rows):
        groups: Dict[int, List[Dict]] = {}
        for r in rows:
            groups.setdefault(r["batch_id"], []).append(r)
        return {bid: {
            "batch_ttft": statistics.fmean(r["batch_ttft"] for r in rs),
            "batch_e2e":  statistics.fmean(r["batch_e2e"]  for r in rs),
        } for bid, rs in groups.items()}

    nxd  = avg_by_batch(lens_nxd)  if lens_nxd  else {}
    vllm = avg_by_batch(lens_vllm) if lens_vllm else {}
    sim_by_id = {b["batch_id"]: b for b in sim}

    rows = []
    for bid in sorted(sim_by_id):
        s = sim_by_id[bid]
        n = nxd.get(bid)
        v = vllm.get(bid)
        rows.append({
            "batch_id":   bid,
            "sim_e2e":    s["batch_e2e"],
            "nxd_e2e":    n["batch_e2e"] if n else None,
            "vllm_e2e":   v["batch_e2e"] if v else None,
            "sim_ttft":   s["batch_ttft"],
            "nxd_ttft":   n["batch_ttft"] if n else None,
            "vllm_ttft":  v["batch_ttft"] if v else None,
        })
    return rows


def summarize(rows: List[Dict], totals: Dict[str, float], label: str):
    """Print TOTAL (headline) + per-batch error stats of sim vs each ref."""
    def err_pct(s, g):
        if g is None or g == 0:
            return None
        return abs(s - g) / g * 100

    print(f"\n=== {label}  (n_batches={len(rows)}) ===")

    # Headline: TOTAL time to process all requests.
    sim_total = totals["sim"]
    print(f"  TOTAL e2e (all-reqs wall time, ms):")
    print(f"    sim   = {sim_total:>10.1f}")
    for ref in ("nxd", "vllm"):
        rt = totals.get(ref)
        if rt is None:
            continue
        err = err_pct(sim_total, rt)
        sign = "+" if sim_total >= rt else "-"
        print(f"    {ref:<5} = {rt:>10.1f}    sim/{ref} err = "
              f"{sign}{err:5.2f}%  (sim {'over' if sim_total>rt else 'under'})")

    # Analytical: per-batch distribution.
    if rows:
        print(f"  per-batch (analytical):")
        for ref in ("nxd", "vllm"):
            for metric in ("e2e", "ttft"):
                errs = [err_pct(r[f"sim_{metric}"], r[f"{ref}_{metric}"])
                        for r in rows]
                errs = [e for e in errs if e is not None]
                if not errs:
                    continue
                print(f"    sim vs {ref:<4} {metric:<4}  "
                      f"mean={statistics.fmean(errs):6.2f}%  "
                      f"median={statistics.median(errs):6.2f}%  "
                      f"p95={sorted(errs)[int(0.95*len(errs))]:6.2f}%")


def run_one(model: str, tp: int, bs: int, dataset: str, write_csv: bool,
            sim_subdir: str = "sim", lens_model: str | None = None):
    base = RESULTS
    lm = lens_model or f"{model}-Instruct"   # measure_*.py uses basename (Instruct included)
    nxd_path  = base / "lens_nxd"  / lm    / f"tp{tp}" / f"bs{bs}" / f"{dataset}.csv"
    vllm_path = base / "lens_vllm" / lm    / f"tp{tp}" / f"bs{bs}" / f"{dataset}.csv"
    sim_path  = base / sim_subdir  / model / f"tp{tp}" / f"bs{bs}" / f"{dataset}.csv"

    if not sim_path.exists():
        print(f"[skip] sim missing: {sim_path}")
        return None
    nxd  = load_lens(nxd_path)  if nxd_path.exists()  else []
    vllm = load_lens(vllm_path) if vllm_path.exists() else []
    sim, sim_total = load_sim(sim_path, bs)
    if not (nxd or vllm):
        print(f"[warn] no ground truth for {dataset} tp{tp} bs{bs}")

    totals = {"sim": sim_total}
    if nxd:  totals["nxd"]  = lens_total_ms(nxd)
    if vllm: totals["vllm"] = lens_total_ms(vllm)

    rows = join_3way(nxd, vllm, sim)
    summarize(rows, totals, f"{dataset}  tp={tp}  bs={bs}")

    if write_csv:
        out = STUDY_ROOT / "comparison" / f"{model}_tp{tp}_bs{bs}_{dataset}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            # First row: TOTAL (headline). Then per-batch rows.
            fieldnames = ["batch_id", "sim_e2e", "nxd_e2e", "vllm_e2e",
                          "sim_ttft", "nxd_ttft", "vllm_ttft"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow({
                "batch_id": "TOTAL",
                "sim_e2e":  totals["sim"],
                "nxd_e2e":  totals.get("nxd",  ""),
                "vllm_e2e": totals.get("vllm", ""),
                "sim_ttft": "", "nxd_ttft": "", "vllm_ttft": "",
            })
            w.writerows(rows)
        print(f"  → {out}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Llama-3.2-1B")
    p.add_argument("--tp", type=int, choices=[1, 2], required=True)
    p.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--no-csv", action="store_true",
                   help="skip writing per-batch comparison CSV")
    p.add_argument("--sim-subdir", default="sim",
                   help="results/<sim-subdir>/<model>/... — use 'parallel_sim' "
                        "if results were collected under a different subtree.")
    p.add_argument("--lens-model", default=None,
                   help="LENS measurement folder name (lens_nxd/<this>/...). "
                        "Default: <model>-Instruct, since measure_*.py uses "
                        "the model_path basename which carries the -Instruct "
                        "suffix while the simulator side typically does not.")
    args = p.parse_args()

    bsl = [int(x) for x in args.batch_sizes.split(",")]
    dsl = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for ds in dsl:
        for bs in bsl:
            run_one(args.model, args.tp, bs, ds,
                    write_csv=not args.no_csv,
                    sim_subdir=args.sim_subdir,
                    lens_model=args.lens_model)


if __name__ == "__main__":
    main()
