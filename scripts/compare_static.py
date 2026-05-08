#!/usr/bin/env python3
"""
compare_static.py — Compare LLMServingSim simulator output (sim CSV)
against measured ground truth (meas CSV) for a static offline-batch
workload.

Both CSVs follow the simulator's schema (instance id, request id,
input, output, arrival, end_time, latency, queuing_delay, TTFT, TPOT,
ITL). Rows are matched by ``request id``.

Subtracts ``queuing_delay`` from TTFT and latency on both sides to
obtain "execution-only" timings (the user's stated target metric).

Outputs:

* ``<output-dir>/per_request.csv`` — per-request side-by-side
  comparison with absolute and percent errors.
* ``<output-dir>/summary.txt``     — distribution stats (median, p50,
  p90, p99, mean, std) of measured / predicted / error_pct for each
  metric (exec_TTFT, exec_latency, TPOT, throughput).
* ``<output-dir>/scatter.png``     — scatter plot of measured vs
  predicted exec_latency (if matplotlib available).

Multiple (sim_csv, meas_csv) pairs can be passed in one invocation so
you can compare A and B in a single report:

    python scripts/compare_static.py \\
        --pair sim_A=outputs/sim_modeA.csv,meas_A=outputs/meas_modeA.csv \\
        --pair sim_B=outputs/sim_modeB.csv,meas_B=outputs/meas_modeB.csv \\
        --output-dir outputs/compare_b4
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple


METRICS = [
    ("exec_TTFT_us",    "TTFT - queuing_delay (µs)"),
    ("exec_latency_us", "latency - queuing_delay (µs)"),
    ("TPOT_us",         "TPOT (µs)"),
]


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def derived(rows: List[Dict[str, str]]) -> Dict[int, Dict[str, float]]:
    """Map request_id -> {exec_TTFT_us, exec_latency_us, TPOT_us}."""
    out: Dict[int, Dict[str, float]] = {}
    for r in rows:
        rid = int(r["request id"])
        ttft_ns = int(r["TTFT"])
        latency_ns = int(r["latency"])
        queue_ns = int(r["queuing_delay"])
        tpot_ns = int(r["TPOT"])
        out[rid] = {
            "exec_TTFT_us":    (ttft_ns - queue_ns) / 1000.0,
            "exec_latency_us": (latency_ns - queue_ns) / 1000.0,
            "TPOT_us":         tpot_ns / 1000.0,
            "input":           int(r["input"]),
            "output":          int(r["output"]),
        }
    return out


def compute_pair(sim_csv: Path, meas_csv: Path
                 ) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """Match sim and meas by request_id; return per-request rows + summary."""
    sim = derived(load_csv(sim_csv))
    meas = derived(load_csv(meas_csv))
    common = sorted(set(sim.keys()) & set(meas.keys()))
    if len(common) != len(sim) or len(common) != len(meas):
        print(f"[!] {sim_csv.name} vs {meas_csv.name}: "
              f"{len(common)} matched / {len(sim)} sim / {len(meas)} meas")

    per_req: List[Dict[str, float]] = []
    for rid in common:
        row: Dict[str, float] = {"request id": rid,
                                 "input": sim[rid]["input"],
                                 "output": sim[rid]["output"]}
        for key, _ in METRICS:
            s = sim[rid][key]
            m = meas[rid][key]
            row[f"{key}_sim"] = s
            row[f"{key}_meas"] = m
            row[f"{key}_abs_err"] = s - m
            row[f"{key}_pct_err"] = (s - m) / m * 100 if m else float("nan")
        per_req.append(row)

    # Summary stats
    summary: Dict[str, float] = {}
    for key, _ in METRICS:
        sims = [r[f"{key}_sim"] for r in per_req]
        meass = [r[f"{key}_meas"] for r in per_req]
        errs = [r[f"{key}_pct_err"] for r in per_req
                if not (r[f"{key}_pct_err"] != r[f"{key}_pct_err"])]
        abs_errs = [abs(e) for e in errs]
        summary[f"{key}_meas_p50"] = statistics.median(meass)
        summary[f"{key}_meas_p90"] = sorted(meass)[int(0.9 * (len(meass)-1))]
        summary[f"{key}_meas_p99"] = sorted(meass)[int(0.99 * (len(meass)-1))]
        summary[f"{key}_sim_p50"]  = statistics.median(sims)
        summary[f"{key}_sim_p90"]  = sorted(sims)[int(0.9 * (len(sims)-1))]
        summary[f"{key}_pct_err_median"] = statistics.median(abs_errs)
        summary[f"{key}_pct_err_p90"]    = sorted(abs_errs)[int(0.9 * (len(abs_errs)-1))]
        summary[f"{key}_pct_err_signed_median"] = statistics.median(errs)
    return per_req, summary


def write_per_request(per_req: List[Dict[str, float]], path: Path,
                      label: str) -> None:
    if not per_req:
        return
    fields = list(per_req[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in per_req:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"[✓] {label}: per-request -> {path}")


def render_summary(label: str, summary: Dict[str, float]) -> str:
    lines = [f"=== {label} ==="]
    for key, desc in METRICS:
        lines.append(f"  {desc}")
        lines.append(f"    measured  p50={summary[f'{key}_meas_p50']:9.1f}  "
                     f"p90={summary[f'{key}_meas_p90']:9.1f}  "
                     f"p99={summary[f'{key}_meas_p99']:9.1f}")
        lines.append(f"    predicted p50={summary[f'{key}_sim_p50']:9.1f}  "
                     f"p90={summary[f'{key}_sim_p90']:9.1f}")
        lines.append(f"    |pct err| median={summary[f'{key}_pct_err_median']:6.2f}%  "
                     f"p90={summary[f'{key}_pct_err_p90']:6.2f}%   "
                     f"(signed median {summary[f'{key}_pct_err_signed_median']:+6.2f}%)")
    return "\n".join(lines)


def render_scatter(per_reqs: Dict[str, List[Dict[str, float]]],
                   out_path: Path, key: str = "exec_latency_us") -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[!] matplotlib not available ({e}); skipping scatter")
        return False
    fig, ax = plt.subplots(figsize=(6, 6))
    all_vals: List[float] = []
    for label, rows in per_reqs.items():
        sims = [r[f"{key}_sim"] for r in rows]
        meass = [r[f"{key}_meas"] for r in rows]
        ax.scatter(meass, sims, s=12, alpha=0.6, label=label)
        all_vals.extend(meass + sims)
    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
    ax.set_xlabel(f"measured {key}")
    ax.set_ylabel(f"predicted {key}")
    ax.set_title("Predicted vs measured")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[✓] scatter -> {out_path}")
    return True


def parse_pair(spec: str) -> Tuple[str, Path, str, Path]:
    """Parse 'sim_A=path,meas_A=path' -> ('sim_A', Path, 'meas_A', Path)."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--pair must have exactly two K=V comma-separated entries, got {spec!r}")
    sim_label = sim_path = meas_label = meas_path = None
    for p in parts:
        k, v = p.split("=", 1)
        if k.startswith("sim"):
            sim_label, sim_path = k, Path(v)
        elif k.startswith("meas"):
            meas_label, meas_path = k, Path(v)
        else:
            raise argparse.ArgumentTypeError(f"Unknown key in --pair: {k!r}")
    if not (sim_label and meas_label):
        raise argparse.ArgumentTypeError(
            f"--pair must include both sim* and meas* entries: {spec!r}")
    return sim_label, sim_path, meas_label, meas_path


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--pair", action="append", required=True,
                   help="One pair, e.g. 'sim_A=path/to/sim.csv,"
                        "meas_A=path/to/meas.csv'. Repeat for multiple pairs.")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Tuple[str, Dict[str, float]]] = []
    per_reqs: Dict[str, List[Dict[str, float]]] = {}
    for spec in args.pair:
        sim_label, sim_csv, meas_label, meas_csv = parse_pair(spec)
        label = sim_label.replace("sim_", "")
        per_req, summary = compute_pair(sim_csv, meas_csv)
        write_per_request(per_req, out_dir / f"per_request_{label}.csv", label)
        summaries.append((label, summary))
        per_reqs[label] = per_req

    summary_path = out_dir / "summary.txt"
    with summary_path.open("w") as f:
        for label, s in summaries:
            block = render_summary(label, s)
            print(block); print()
            f.write(block); f.write("\n\n")
    print(f"[✓] summary -> {summary_path}")

    render_scatter(per_reqs, out_dir / "scatter_exec_latency.png",
                   key="exec_latency_us")


if __name__ == "__main__":
    main()
