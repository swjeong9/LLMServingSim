#!/usr/bin/env python3
"""Convert LENS dataset CSV → LLMServingSim JSONL workload.

LENS dataset format (run_id, sample_id, input_len, output_len): the
top batch_size * NUM_BATCHES rows are taken so each (dataset, batch)
pair always uses the same prefix of requests for reproducibility.

Each emitted JSONL line carries dummy token ids whose offset is
derived from sample_id, so distinct samples never share token ids
(prefix cache hits are limited to within a sample, not across).

Usage
-----
    # Single (dataset, batch_size)
    python studies/inf2_baseline/convert_workload.py \\
        --dataset arxiv --batch-size 4

    # Sweep — all listed batch sizes for the dataset, capped at row count
    python studies/inf2_baseline/convert_workload.py \\
        --dataset cnn --batch-sizes 1,2,4,8,16,32

    # All datasets × all batches (driven by row counts in each CSV)
    python studies/inf2_baseline/convert_workload.py --all
"""
import argparse
import csv
import json
from pathlib import Path

NUM_BATCHES = 50
SAMPLE_TOKEN_OFFSET = 1_000_000   # spacing between samples' token id ranges
DATASETS = ("arxiv", "cnn", "sharegpt", "writing_prompts")
ALL_BATCH_SIZES = (1, 2, 4, 8, 16, 32)


def convert_one(dataset: str, batch_size: int, data_dir: Path,
                output_dir: Path) -> int:
    """Convert one (dataset, batch_size) pair. Returns request count.

    Skips silently with a warning if the source CSV has fewer than
    batch_size * NUM_BATCHES rows.
    """
    src = data_dir / f"{dataset}.csv"
    dst = output_dir / f"{dataset}_bs{batch_size}.jsonl"
    n_reqs = batch_size * NUM_BATCHES

    with src.open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) < n_reqs:
        print(f"[skip] {dataset} bs={batch_size}: have {len(rows)} rows, "
              f"need {n_reqs}")
        return 0

    rows = rows[:n_reqs]
    output_dir.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        for row in rows:
            sample_id = int(row["sample_id"])
            in_len = int(row["input_len"])
            out_len = int(row["output_len"])
            offset = sample_id * SAMPLE_TOKEN_OFFSET
            f.write(json.dumps({
                "input_toks": in_len,
                "output_toks": out_len,
                "arrival_time_ns": 0,    # all arrive at t=0
                "input_tok_ids":  list(range(offset, offset + in_len)),
                "output_tok_ids": list(range(offset + in_len,
                                             offset + in_len + out_len)),
            }) + "\n")
    print(f"[ok]  {dataset} bs={batch_size}: {n_reqs} requests → {dst}")
    return n_reqs


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                 description=__doc__)
    p.add_argument("--dataset", choices=DATASETS,
                   help="single dataset; with --batch-size or --batch-sizes")
    p.add_argument("--batch-size", type=int,
                   help="single batch size (use with --dataset)")
    p.add_argument("--batch-sizes",
                   help="comma-separated batch sizes "
                        "(default: 1,2,4,8,16,32)")
    p.add_argument("--all", action="store_true",
                   help="convert all datasets × all batch sizes")
    p.add_argument("--data-dir",
                   default="studies/inf2_baseline/data/datasets",
                   help="dataset CSV directory")
    p.add_argument("--output-dir",
                   default="studies/inf2_baseline/workloads",
                   help="JSONL output directory")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if args.all:
        targets = [(ds, bs) for ds in DATASETS for bs in ALL_BATCH_SIZES]
    elif args.dataset and args.batch_size:
        targets = [(args.dataset, args.batch_size)]
    elif args.dataset and args.batch_sizes:
        bsl = [int(x) for x in args.batch_sizes.split(",")]
        targets = [(args.dataset, bs) for bs in bsl]
    else:
        p.error("specify --all, or --dataset with --batch-size / --batch-sizes")

    total = sum(convert_one(ds, bs, data_dir, output_dir) for ds, bs in targets)
    print(f"\nDone. {total} total requests across "
          f"{sum(1 for _ in targets)} workloads.")


if __name__ == "__main__":
    main()
