#!/usr/bin/env python3
"""Transform a workload jsonl so each batch's requests arrive after the
previous batch has plenty of time to finish — forcing the simulator's
router into batch-isolation mode without per-run subprocess overhead.

Why: measure_vllm.py runs 50 separate .generate() calls (one per batch),
blocking continuous batching. The simulator side normally takes the
entire jsonl in one process — the scheduler then folds the 50 batches
into continuous batching, masking the per-batch behavior we want to
compare against.

Trick: simulator's `serving/__main__.py` advances simulated time to the
next pending arrival when all instances are idle (the agentic
sub-request release mechanism). Stagger arrival_time_ns by batch:

    batch 0 : arrival = 0
    batch 1 : arrival = 1 × T_GAP_NS    (e.g. 100s = 1e11 ns)
    batch 2 : arrival = 2 × T_GAP_NS
    ...
    batch B : arrival = B × T_GAP_NS

With T_GAP_NS larger than the worst expected batch latency, batch i+1
never begins before batch i finishes. The simulator's idle-time jump
absorbs the wait — per-request latency (end - arrival) stays accurate.

Metric note: `max(end) - min(arrival)` becomes meaningless (dominated by
the staggering gap). Use **sum(end - arrival) over rows** = sum of
per-request latency, which matches measure_vllm.py's sum-of-batch-e2e.

Usage:
    python studies/gpu_baseline/stagger_workload.py \\
        --input  studies/gpu_baseline/workloads/arxiv_bs4.jsonl \\
        --output studies/gpu_baseline/workloads_staggered/arxiv_bs4.jsonl \\
        --batch-size 4 \\
        --gap-ns 100000000000    # 100s, default

Bulk mode:
    python studies/gpu_baseline/stagger_workload.py --all
        # → reads workloads/<ds>_bs<B>.jsonl, writes
        #   workloads_staggered/<ds>_bs<B>.jsonl for all (ds, bs) combos
"""
import argparse
import json
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR  = STUDY_ROOT / "workloads"              # symlinked from inf2_baseline
DEFAULT_OUTPUT_DIR = STUDY_ROOT / "workloads_staggered"
DATASETS    = ("arxiv", "cnn", "sharegpt", "writing_prompts")
BATCH_SIZES = (1, 2, 4, 8, 16, 32)

DEFAULT_GAP_NS = 100_000_000_000   # 100s — larger than any expected batch latency


def stagger_one(src: Path, dst: Path, batch_size: int, gap_ns: int) -> int:
    """Read src jsonl, rewrite each row's arrival_time_ns as
    (row_index // batch_size) × gap_ns, write to dst. Returns row count."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open() as fin, dst.open("w") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line: continue
            row = json.loads(line)
            batch_idx = i // batch_size
            row["arrival_time_ns"] = batch_idx * gap_ns
            fout.write(json.dumps(row) + "\n")
            n += 1
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input",  help="single jsonl input path")
    p.add_argument("--output", help="single jsonl output path")
    p.add_argument("--batch-size", type=int,
                   help="batch_size (rows are grouped consecutively into batches)")
    p.add_argument("--gap-ns", type=int, default=DEFAULT_GAP_NS,
                   help=f"per-batch arrival gap in ns (default: {DEFAULT_GAP_NS} "
                        f"= {DEFAULT_GAP_NS/1e9:.0f}s)")
    p.add_argument("--all", action="store_true",
                   help=f"bulk-transform all {DEFAULT_INPUT_DIR}/<ds>_bs<B>.jsonl "
                        f"to {DEFAULT_OUTPUT_DIR}/<ds>_bs<B>.jsonl")
    p.add_argument("--input-dir",  default=str(DEFAULT_INPUT_DIR),
                   help=f"--all input dir (default: {DEFAULT_INPUT_DIR})")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                   help=f"--all output dir (default: {DEFAULT_OUTPUT_DIR})")
    args = p.parse_args()

    if args.all:
        in_dir  = Path(args.input_dir)
        out_dir = Path(args.output_dir)
        for ds in DATASETS:
            for bs in BATCH_SIZES:
                src = in_dir / f"{ds}_bs{bs}.jsonl"
                dst = out_dir / f"{ds}_bs{bs}.jsonl"
                if not src.exists():
                    print(f"  [skip] {src}  (not found)")
                    continue
                n = stagger_one(src, dst, bs, args.gap_ns)
                print(f"  {ds}_bs{bs}:  {n} rows  ({n // bs} batches × {bs}, "
                      f"gap={args.gap_ns/1e9:.0f}s)  → {dst}")
        return

    if not (args.input and args.output and args.batch_size):
        p.error("either --all, or all of --input + --output + --batch-size")
    n = stagger_one(Path(args.input), Path(args.output),
                    args.batch_size, args.gap_ns)
    print(f"{n} rows → {args.output}  ({n // args.batch_size} batches × "
          f"{args.batch_size}, gap={args.gap_ns/1e9:.0f}s)")


if __name__ == "__main__":
    main()
