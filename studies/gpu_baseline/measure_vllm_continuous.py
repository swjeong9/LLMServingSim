#!/usr/bin/env python3
"""Single-shot continuous-batched vLLM measurement on NVIDIA GPUs.

Counterpart to measure_vllm.py — that script splits NUM_BATCHES=50 batches
of `batch_size` requests and runs them sequentially (one .generate() call
per batch), blocking vLLM's scheduler from folding requests across batch
boundaries. This script does the OPPOSITE: feed the entire
batch_size × NUM_BATCHES = N_total request set to vLLM in **one
.generate() call**, letting the continuous-batching scheduler interleave
prefill/decode across all of them subject to `max_num_seqs=batch_size`.

Output : studies/gpu_baseline/results/lens_vllm_continuous/<hw>/<opt>/<model>/tp<N>/bs<B>/<dataset>.csv
         (mirrors measure_vllm.py's path tree under `lens_vllm_continuous/`)

Schema : single-row CSV per sweep, compatible with compare.py's
         lens_total() loader (column `batch_e2e_ms` = total wallclock,
         `run_id`=0 placeholder, `status=OK`).

Usage
-----
    python studies/gpu_baseline/measure_vllm_continuous.py \\
        --dataset arxiv --batch-size 4 \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --hardware L4 \\
        --tp-degree 1 --max-model-len 8192
"""
import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

BUCKETS = [128, 256, 512, 1024, 2048, 4096, 8192]
WARMUP_SHAPES = [(64, 5), (130, 10), (260, 20), (520, 40),
                 (1030, 80), (2050, 160), (4100, 300)]
NUM_BATCHES = 50    # total requests per sweep = batch_size × NUM_BATCHES

STUDY_ROOT = Path(__file__).resolve().parent
DATA_DIR = STUDY_ROOT / "data" / "datasets"
RESULTS_ROOT = STUDY_ROOT / "results" / "lens_vllm_continuous"


def opt_label(enable_chunked_prefill: bool, enable_prefix_caching: bool) -> str:
    if enable_chunked_prefill and enable_prefix_caching:
        return "on"
    if (not enable_chunked_prefill) and (not enable_prefix_caching):
        return "off"
    return f"cp{int(enable_chunked_prefill)}_pc{int(enable_prefix_caching)}"


def _model_name(model_path: str) -> str:
    p = model_path.rstrip("/")
    if "/snapshots/" in p and "/models--" in p:
        for part in p.split("/"):
            if part.startswith("models--"):
                segs = part[len("models--"):].split("--")
                if len(segs) >= 2:
                    return "--".join(segs[1:])
    return os.path.basename(p)


def _buckets_for(max_model_len: int):
    bs = [b for b in BUCKETS if b <= max_model_len]
    if max_model_len not in bs:
        bs.append(max_model_len)
    return bs


def load_dataset_csv(path: Path, batch_size: int):
    """Return the full batch_size × NUM_BATCHES request list as a single
    flat array — no per-batch grouping. Same input rows as measure_vllm.py
    but unfolded into one continuous stream."""
    n_reqs = batch_size * NUM_BATCHES
    with path.open() as f:
        rows = list(csv.DictReader(f))[:n_reqs]
    if len(rows) < n_reqs:
        raise ValueError(
            f"{path.name}: need {n_reqs} rows ({batch_size} × {NUM_BATCHES}), "
            f"have {len(rows)}")
    return [{
        "sample_id":  int(r["sample_id"]),
        "input_len":  int(r["input_len"]),
        "output_len": int(r["output_len"]),
    } for r in rows]


def init_llm(model, tp_degree, batch_size, max_model_len,
             enable_chunked_prefill, enable_prefix_caching):
    """vLLM-CUDA LLM init — same toggles as measure_vllm.py.
    max_num_seqs=batch_size enforces the concurrent slot cap so continuous
    batching folds the full request stream into batch_size at a time."""
    from vllm import LLM
    print(f"[init_llm] model={model}")
    print(f"  tp={tp_degree} batch={batch_size} max_model_len={max_model_len}")
    print(f"  dtype=bfloat16  prefix_caching={enable_prefix_caching}  "
          f"chunked_prefill={enable_chunked_prefill}")
    return LLM(
        model=model,
        tensor_parallel_size=tp_degree,
        max_model_len=max_model_len,
        max_num_seqs=batch_size,
        dtype="bfloat16",
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
    )


def make_input_ids(tokenizer, target_len: int):
    if target_len <= 0:
        target_len = 1
    text = "The quick brown fox jumps over the lazy dog. " * (target_len // 8 + 1)
    return tokenizer.encode(text, add_special_tokens=True)[:target_len]


def warmup(llm, tokenizer, batch_size, max_model_len):
    """Warm each bucket with uniform-batch generate (same as measure_vllm.py)."""
    from vllm import SamplingParams
    for il, ol in WARMUP_SHAPES:
        if il + ol > max_model_len:
            continue
        ids = make_input_ids(tokenizer, il)
        prompts = [{"prompt_token_ids": ids} for _ in range(batch_size)]
        params = [SamplingParams(max_tokens=ol, min_tokens=ol,
                                 temperature=0, ignore_eos=True)
                  for _ in range(batch_size)]
        try:
            t0 = time.perf_counter()
            llm.generate(prompts, sampling_params=params)
            print(f"  warmup (il={il}, ol={ol}): {time.perf_counter()-t0:.2f}s")
        except Exception as e:
            print(f"  [WARN] warmup (il={il}, ol={ol}) failed: {e}")


def measure_continuous(llm, tokenizer, samples, max_model_len, batch_size):
    """Fire all N_total samples in a single .generate() call. vLLM's
    continuous batching scheduler interleaves them subject to
    max_num_seqs=batch_size. Returns one summary row."""
    from vllm import SamplingParams
    input_lens  = [s["input_len"]  for s in samples]
    output_lens = [s["output_len"] for s in samples]
    sample_ids  = [s["sample_id"]  for s in samples]
    max_il, max_ol = max(input_lens), max(output_lens)
    # vLLM rejects when a SINGLE sequence's il+ol > max_model_len. Aggregate
    # max_il + max_ol can come from different rows and gives a false positive,
    # so check per-row instead.
    max_pair = max(il + ol for il, ol in zip(input_lens, output_lens))

    base = {
        "run_id":      0,                 # placeholder — single sweep
        "batch_size":  batch_size,
        "n_requests":  len(samples),
        "sample_ids":  json.dumps(sample_ids),
        "input_lens":  json.dumps(input_lens),
        "output_lens": json.dumps(output_lens),
        "max_input_len":  max_il,
        "max_output_len": max_ol,
        "max_il_plus_ol": max_pair,
    }
    if max_pair > max_model_len:
        return {**base, "status": "SKIPPED_TOO_LONG",
                "max_n_generated": "",
                "batch_ttft_ms": "", "batch_e2e_ms": "",
                "error": f"max(il+ol) any row ({max_pair}) > "
                         f"max_model_len({max_model_len})"}

    prompts = [{"prompt_token_ids": make_input_ids(tokenizer, il)}
               for il in input_lens]
    params = [SamplingParams(max_tokens=ol, min_tokens=ol,
                             temperature=0, ignore_eos=True)
              for ol in output_lens]
    try:
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sampling_params=params)
        e2e_ms = round((time.perf_counter() - t0) * 1000, 3)
        gen_lens = [len(o.outputs[0].token_ids) for o in outs]
        return {**base, "status": "OK",
                "max_n_generated": max(gen_lens),
                "batch_ttft_ms": "", "batch_e2e_ms": e2e_ms,
                "error": ""}
    except Exception as e:
        return {**base, "status": "ERROR",
                "max_n_generated": "",
                "batch_ttft_ms": "", "batch_e2e_ms": "",
                "error": str(e)}


def write_csv(path: Path, rows):
    fields = ["run_id", "status", "batch_size", "n_requests",
              "sample_ids", "input_lens", "output_lens",
              "max_input_len", "max_output_len", "max_n_generated",
              "batch_ttft_ms", "batch_e2e_ms", "error"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    p.add_argument("--dataset", required=True,
                   choices=["arxiv", "cnn", "sharegpt", "writing_prompts"])
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--hardware", required=True,
                   help="GPU label for the output folder (e.g. L4, A10G).")
    p.add_argument("--tp-degree", type=int, required=True, choices=[1, 2, 4, 8])
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--enable-chunked-prefill",
                   action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--enable-prefix-caching",
                   action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--output-dir", default=None,
                   help="default: results/lens_vllm_continuous/<hw>/<opt>/<model>/tp<N>/bs<B>/")
    p.add_argument("--skip-warmup", action="store_true")
    args = p.parse_args()

    src_csv = DATA_DIR / f"{args.dataset}.csv"
    if not src_csv.exists():
        raise FileNotFoundError(f"dataset CSV missing: {src_csv}")

    opt = opt_label(args.enable_chunked_prefill, args.enable_prefix_caching)
    if args.output_dir is None:
        args.output_dir = (RESULTS_ROOT / args.hardware / opt
                           / _model_name(args.model)
                           / f"tp{args.tp_degree}" / f"bs{args.batch_size}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] {out_dir}  (opt={opt})")

    try:
        samples = load_dataset_csv(src_csv, args.batch_size)
    except ValueError as e:
        print(f"[skip] {e}")
        return
    n_reqs = len(samples)
    print(f"[{datetime.now()}] loaded {n_reqs} requests "
          f"(batch={args.batch_size} × {NUM_BATCHES}) from {src_csv}")

    llm = init_llm(args.model, args.tp_degree, args.batch_size, args.max_model_len,
                   args.enable_chunked_prefill, args.enable_prefix_caching)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if not args.skip_warmup:
        print(f"\n[{datetime.now()}] warming up buckets ...")
        warmup(llm, tokenizer, args.batch_size, args.max_model_len)

    print(f"\n[{datetime.now()}] single-shot continuous-batched sweep: "
          f"{n_reqs} requests, max_num_seqs={args.batch_size}")
    t_sweep = time.perf_counter()
    row = measure_continuous(llm, tokenizer, samples,
                             args.max_model_len, args.batch_size)
    e2e = row["batch_e2e_ms"]
    e2e_str = f"e2e={e2e:>9.0f}ms" if isinstance(e2e, (int, float)) else ""
    print(f"  [sweep]  n_reqs={n_reqs}  max_il={row['max_input_len']}  "
          f"max_ol={row['max_output_len']}  {row['status']:<20} {e2e_str}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv  = out_dir / f"{args.dataset}_{ts}.csv"
    out_json = out_dir / f"{args.dataset}_{ts}.json"
    write_csv(out_csv, [row])
    with out_json.open("w") as f:
        json.dump({
            "framework": "lens_vllm_continuous",
            "runtime": "vLLM (CUDA backend, single-shot continuous batching)",
            "hardware": args.hardware,
            "dataset": args.dataset,
            "model": args.model,
            "tp_degree": args.tp_degree,
            "batch_size": args.batch_size,
            "max_model_len": args.max_model_len,
            "n_requests": n_reqs,
            "buckets": _buckets_for(args.max_model_len),
            "enable_chunked_prefill": args.enable_chunked_prefill,
            "enable_prefix_caching":  args.enable_prefix_caching,
            "opt_label": opt,
            "input_csv": str(src_csv),
            "run_timestamp": datetime.now().isoformat(),
            "total_sweep_s": round(time.perf_counter() - t_sweep, 1),
        }, f, indent=2)

    stable = out_dir / f"{args.dataset}.csv"
    if stable.is_symlink() or stable.exists():
        stable.unlink()
    stable.symlink_to(out_csv.name)

    print(f"\n[DONE]\n  {out_csv}\n  {out_json}\n  latest -> {stable.name}")


if __name__ == "__main__":
    main()
