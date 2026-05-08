#!/usr/bin/env python3
"""
make_static_workload.py — Generate a static (offline-batch) workload
JSONL for the LLMServingSim simulator and matching measurements on
inf2.

Two modes:

* ``--mode A`` (continuous batching): all B*M requests have
  ``arrival_time_ns = 0``. The scheduler (LLMServingSim or vLLM-Neuron)
  with ``--max-num-seqs B`` keeps B requests in flight, replacing
  finished slots from the queue. This matches what
  ``vllm.LLM.generate(all_prompts, ...)`` does by default.

* ``--mode B`` (strict static batches): batch i has all B requests
  arriving at ``i * gap_ns``. With a large enough gap (default 10s),
  the next batch arrives only after the previous batch finishes →
  no overlap, no slot replacement. The simulator skips idle wall-clock
  via fast-forward, so the gap costs nothing.

Length distribution:

* ``fixed``        — every request has the same input/output length
* ``uniform``      — uniformly random within given ranges
* ``sampled``      — sample from a JSONL dataset (e.g., a real
                     ShareGPT trace) and reuse its lengths

Output schema mirrors the existing simulator workloads
(``input_toks``, ``output_toks``, ``arrival_time_ns``,
``input_tok_ids``, ``output_tok_ids``).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple


def _make_pair(in_len: int, out_len: int, base_id: int) -> dict:
    """Build one JSONL entry with sequential token IDs (prefix-cache safe)."""
    return {
        "input_toks": in_len,
        "output_toks": out_len,
        "input_tok_ids": list(range(base_id, base_id + in_len)),
        "output_tok_ids": list(range(base_id + in_len,
                                     base_id + in_len + out_len)),
    }


def make_lengths_fixed(n: int, in_len: int, out_len: int) -> List[Tuple[int, int]]:
    return [(in_len, out_len) for _ in range(n)]


def make_lengths_uniform(n: int, in_lo: int, in_hi: int,
                         out_lo: int, out_hi: int, rng: random.Random
                         ) -> List[Tuple[int, int]]:
    return [(rng.randint(in_lo, in_hi), rng.randint(out_lo, out_hi))
            for _ in range(n)]


def make_lengths_sampled(n: int, source_jsonl: Path, rng: random.Random
                         ) -> List[Tuple[int, int]]:
    pool: List[Tuple[int, int]] = []
    with source_jsonl.open() as f:
        for line in f:
            obj = json.loads(line)
            if "input_toks" in obj and "output_toks" in obj:
                pool.append((int(obj["input_toks"]), int(obj["output_toks"])))
            elif "sub_requests" in obj:
                # Agentic dataset — pick first sub-request only
                sr = obj["sub_requests"][0]
                pool.append((int(sr["input_toks"]), int(sr["output_toks"])))
    if not pool:
        raise ValueError(f"No usable rows in {source_jsonl}")
    return [pool[rng.randrange(len(pool))] for _ in range(n)]


def write_workload(lengths: List[Tuple[int, int]], out_path: Path,
                   batch_size: int, mode: str, gap_ns: int) -> None:
    with out_path.open("w") as f:
        base_id = 1
        for i, (in_len, out_len) in enumerate(lengths):
            entry = _make_pair(in_len, out_len, base_id)
            if mode == "A":
                entry["arrival_time_ns"] = 0
            elif mode == "B":
                batch_idx = i // batch_size
                entry["arrival_time_ns"] = batch_idx * gap_ns
            else:
                raise ValueError(f"Unknown mode: {mode}")
            f.write(json.dumps(entry) + "\n")
            # Increment base_id beyond this request's tokens so the next
            # request's IDs are disjoint (no accidental prefix-cache hit
            # between requests, even in case prefix caching is on).
            base_id += in_len + out_len


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--out", required=True, help="Output JSONL path")
    p.add_argument("--batch-size", type=int, required=True,
                   help="B (target concurrent batch size)")
    p.add_argument("--num-batches", type=int, default=50,
                   help="M (number of batches; total requests = B*M)")
    p.add_argument("--mode", choices=["A", "B"], required=True,
                   help="A=continuous (all t=0), B=strict (i*gap)")
    p.add_argument("--gap-seconds", type=float, default=10.0,
                   help="Mode B: gap between batch arrivals in seconds. "
                        "Must exceed the longest expected per-batch latency.")
    p.add_argument("--seed", type=int, default=42)

    sub = p.add_subparsers(dest="length_mode", required=True)

    pf = sub.add_parser("fixed", help="all requests same length")
    pf.add_argument("--in-len", type=int, required=True)
    pf.add_argument("--out-len", type=int, required=True)

    pu = sub.add_parser("uniform", help="uniform random lengths")
    pu.add_argument("--in-lo", type=int, required=True)
    pu.add_argument("--in-hi", type=int, required=True)
    pu.add_argument("--out-lo", type=int, required=True)
    pu.add_argument("--out-hi", type=int, required=True)

    ps = sub.add_parser("sampled", help="sample lengths from a dataset")
    ps.add_argument("--source", required=True,
                    help="Path to a workload JSONL whose lengths to sample")

    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    n = args.batch_size * args.num_batches

    if args.length_mode == "fixed":
        lengths = make_lengths_fixed(n, args.in_len, args.out_len)
    elif args.length_mode == "uniform":
        lengths = make_lengths_uniform(n, args.in_lo, args.in_hi,
                                       args.out_lo, args.out_hi, rng)
    elif args.length_mode == "sampled":
        lengths = make_lengths_sampled(n, Path(args.source), rng)
    else:
        raise ValueError(args.length_mode)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_workload(lengths, out_path, args.batch_size, args.mode,
                   int(args.gap_seconds * 1e9))

    in_lens = [l[0] for l in lengths]
    out_lens = [l[1] for l in lengths]
    print(f"[✓] wrote {out_path} ({n} requests, batch_size={args.batch_size}, "
          f"mode={args.mode})")
    print(f"    input_toks  : min={min(in_lens):5d}  max={max(in_lens):5d}  "
          f"mean={sum(in_lens)/n:7.1f}")
    print(f"    output_toks : min={min(out_lens):5d}  max={max(out_lens):5d}  "
          f"mean={sum(out_lens)/n:7.1f}")
    if args.mode == "B":
        print(f"    arrival gap : {args.gap_seconds}s between batches "
              f"(total span {args.num_batches * args.gap_seconds}s)")


if __name__ == "__main__":
    main()
