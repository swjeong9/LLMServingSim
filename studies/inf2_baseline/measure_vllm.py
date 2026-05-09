#!/usr/bin/env python3
"""vLLM-Neuron measurement on Inferentia 2 — vLLM framework path.

Combines two LENS scripts:
  * run_profiling_vllm.py — vLLM-Neuron init (LLM(...) with
    override_neuron_config for explicit buckets, no chunked prefill,
    no prefix caching). vLLM-Neuron's continuous batching is auto-on
    when max_num_seqs > 1.
  * run_eval.py — dataset CSV loop: 50 runs (= batches), each batch
    has heterogeneous (input_len, output_len) per sample.

Per-batch llm.generate(prompts, sampling_params) call so the measured
batch_e2e_ms is comparable to measure_nxd.py's. Within one batch,
vLLM's scheduler may early-evict short outputs (the seat opens but
no new request joins inside the same generate() call), so the
observed e2e ≈ max per-req latency, just like NxD.

Input  : studies/inf2_baseline/data/datasets/<dataset>.csv
Output : studies/inf2_baseline/results/lens_vllm/<model>/tp<N>/bs<B>/<dataset>.csv

Usage
-----
    python studies/inf2_baseline/measure_vllm.py \\
        --dataset arxiv --batch-size 4 \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --tp-degree 1 --max-model-len 8192 \\
        --compiled-dir /home/ubuntu/compiled_models_inf2_baseline_vllm
"""
import argparse
import csv
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

NEURON_BUCKETS = [128, 256, 512, 1024, 2048, 4096, 8192]
WARMUP_SHAPES = [(64, 5), (130, 10), (260, 20), (520, 40),
                 (1030, 80), (2050, 160), (4100, 300)]
NUM_BATCHES = 50

STUDY_ROOT = Path(__file__).resolve().parent
DATA_DIR = STUDY_ROOT / "data" / "datasets"
RESULTS_ROOT = STUDY_ROOT / "results" / "lens_vllm"


def _model_name(model_path: str) -> str:
    p = model_path.rstrip("/")
    if "/snapshots/" in p and "/models--" in p:
        for part in p.split("/"):
            if part.startswith("models--"):
                segs = part[len("models--"):].split("--")
                if len(segs) >= 2:
                    return "--".join(segs[1:])
    return os.path.basename(p)


def _buckets_for(max_model_len: int, override=None):
    base = override if override else NEURON_BUCKETS
    bs = [b for b in base if b <= max_model_len]
    if max_model_len not in bs:
        bs.append(max_model_len)
    return sorted(set(bs))


def load_dataset_csv(path: Path, batch_size: int):
    """Take top batch_size * NUM_BATCHES rows in source order, re-group
    into NUM_BATCHES batches of batch_size each. The dataset CSV's own
    run_id is bs32-anchored — ignored so the same workload rows are
    reused across batch sizes (sweep reproducibility)."""
    n_reqs = batch_size * NUM_BATCHES
    with path.open() as f:
        rows = list(csv.DictReader(f))[:n_reqs]
    if len(rows) < n_reqs:
        raise ValueError(
            f"{path.name}: need {n_reqs} rows ({batch_size} × {NUM_BATCHES}), "
            f"have {len(rows)}")
    runs = {}
    for i in range(NUM_BATCHES):
        chunk = rows[i * batch_size:(i + 1) * batch_size]
        runs[i] = [{
            "sample_id":  int(r["sample_id"]),
            "input_len":  int(r["input_len"]),
            "output_len": int(r["output_len"]),
        } for r in chunk]
    return runs


def init_llm(model, tp_degree, batch_size, max_model_len, compiled_dir,
             bucket_override=None):
    """vLLM-Neuron LLM with explicit buckets + no chunked prefill."""
    from vllm import LLM
    os.environ["NEURON_COMPILED_ARTIFACTS"] = compiled_dir
    os.makedirs(compiled_dir, exist_ok=True)
    buckets = _buckets_for(max_model_len, override=bucket_override)
    print(f"[init_llm] model={model}")
    print(f"  tp={tp_degree} batch={batch_size} max_model_len={max_model_len}")
    print(f"  buckets={buckets}  compiled_dir={compiled_dir}")
    print(f"  is_continuous_batching: auto (vllm-neuron sets True iff "
          f"max_num_seqs > 1)")
    return LLM(
        model=model,
        tensor_parallel_size=tp_degree,
        max_model_len=max_model_len,
        max_num_seqs=batch_size,
        dtype="bfloat16",
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        additional_config={
            "override_neuron_config": {
                "save_sharded_checkpoint": True,
                "context_encoding_buckets": buckets,
                "token_generation_buckets": buckets,
            },
        },
    )


def make_input_ids(tokenizer, target_len: int):
    if target_len <= 0:
        target_len = 1
    text = "The quick brown fox jumps over the lazy dog. " * (target_len // 8 + 1)
    return tokenizer.encode(text, add_special_tokens=True)[:target_len]


def warmup(llm, tokenizer, batch_size, max_model_len):
    """Warm each bucket with uniform-batch generate."""
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


def measure_run(llm, tokenizer, samples, batch_size, max_model_len, run_id):
    """One batch (run): heterogeneous (input_len, output_len) per sample."""
    from vllm import SamplingParams
    input_lens  = [s["input_len"]  for s in samples]
    output_lens = [s["output_len"] for s in samples]
    sample_ids  = [s["sample_id"]  for s in samples]
    max_il, max_ol = max(input_lens), max(output_lens)

    base = {
        "run_id": run_id, "batch_size": batch_size,
        "sample_ids":  json.dumps(sample_ids),
        "input_lens":  json.dumps(input_lens),
        "output_lens": json.dumps(output_lens),
        "max_input_len":  max_il,
        "max_output_len": max_ol,
    }
    if max_il + max_ol > max_model_len:
        return {**base, "status": "SKIPPED_TOO_LONG",
                "max_n_generated": "", "batch_ttft_ms": "",
                "batch_e2e_ms": "",
                "error": f"max_il({max_il}) + max_ol({max_ol}) > "
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
                "max_n_generated": "", "batch_ttft_ms": "",
                "batch_e2e_ms": "", "error": str(e)}


def write_csv(path: Path, rows):
    fields = ["run_id", "status", "batch_size",
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
    p.add_argument("--tp-degree", type=int, required=True, choices=[1, 2, 4, 8])
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--compiled-dir",
                   default="/home/ubuntu/compiled_models_inf2_baseline_vllm")
    p.add_argument("--output-dir", default=None,
                   help="default: results/lens_vllm/<model>/tp<N>/bs<B>/")
    p.add_argument("--skip-warmup", action="store_true")
    p.add_argument("--max-runs", type=int, default=None,
                   help="Run only first N runs (sanity). default: all")
    p.add_argument("--buckets", default=None,
                   help="comma-separated bucket override, see measure_nxd.py")
    args = p.parse_args()
    bucket_override = ([int(x) for x in args.buckets.split(",")]
                       if args.buckets else None)

    src_csv = DATA_DIR / f"{args.dataset}.csv"
    if not src_csv.exists():
        raise FileNotFoundError(f"dataset CSV missing: {src_csv}")

    if args.output_dir is None:
        args.output_dir = (RESULTS_ROOT / _model_name(args.model)
                           / f"tp{args.tp_degree}" / f"bs{args.batch_size}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] {out_dir}")

    try:
        runs = load_dataset_csv(src_csv, args.batch_size)
    except ValueError as e:
        print(f"[skip] {e}")
        return
    if args.max_runs is not None:
        keep = sorted(runs)[:args.max_runs]
        runs = {k: runs[k] for k in keep}
    n_runs = len(runs)
    print(f"[{datetime.now()}] loaded {n_runs} runs (batch={args.batch_size}) "
          f"from {src_csv}")

    llm = init_llm(args.model, args.tp_degree, args.batch_size,
                   args.max_model_len, args.compiled_dir,
                   bucket_override=bucket_override)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if not args.skip_warmup:
        print(f"\n[{datetime.now()}] warming up buckets ...")
        warmup(llm, tokenizer, args.batch_size, args.max_model_len)

    print(f"\n[{datetime.now()}] eval sweep: {n_runs} runs × batch={args.batch_size}")
    rows = []
    t_sweep = time.perf_counter()
    for rid in sorted(runs):
        t0 = time.perf_counter()
        row = measure_run(llm, tokenizer, runs[rid], args.batch_size,
                          args.max_model_len, rid)
        rows.append(row)
        e2e = row["batch_e2e_ms"]
        e2e_str = f"e2e={e2e:>9.0f}ms" if isinstance(e2e, (int, float)) else ""
        print(f"  [run {rid:>2d}]  max_il={row['max_input_len']:>5}  "
              f"max_ol={row['max_output_len']:>5}  {row['status']:<20} "
              f"{e2e_str}  ({time.perf_counter()-t0:.1f}s)")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"{args.dataset}_{ts}.csv"
    out_json = out_dir / f"{args.dataset}_{ts}.json"
    write_csv(out_csv, rows)
    with out_json.open("w") as f:
        json.dump({
            "framework": "lens_vllm",
            "hardware": "inf2",
            "dataset": args.dataset,
            "model": args.model,
            "tp_degree": args.tp_degree,
            "batch_size": args.batch_size,
            "max_model_len": args.max_model_len,
            "n_runs": n_runs,
            "buckets": _buckets_for(args.max_model_len,
                                     override=bucket_override),
            "is_continuous_batching":
                "auto (vllm-neuron sets True iff max_num_seqs > 1)",
            "input_csv": str(src_csv),
            "run_timestamp": datetime.now().isoformat(),
            "total_sweep_s": round(time.perf_counter() - t_sweep, 1),
        }, f, indent=2)

    stable = out_dir / f"{args.dataset}.csv"
    if stable.is_symlink() or stable.exists():
        stable.unlink()
    stable.symlink_to(out_csv.name)

    print(f"\n[DONE]\n  {out_csv}  ({len(rows)} rows)\n  {out_json}\n  "
          f"latest -> {stable.name}")


if __name__ == "__main__":
    main()
