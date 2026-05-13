#!/usr/bin/env python3
"""vLLM measurement on NVIDIA GPUs (L4, A10G, ...) — vLLM framework path.

GPU equivalent of studies/inf2_baseline/measure_vllm.py and
studies/tpu_baseline/measure_vllm.py. Same heterogeneous batch eval loop
and output schema; differences:

  * No `NEURON_COMPILED_ARTIFACTS` / `override_neuron_config` —
    vLLM's CUDA backend uses its native attention kernels
    (FlashAttention/xFormers, auto-selected).
  * `dtype="bfloat16"`. Prefix cache + chunked prefill default off
    (parity with measure_vllm.py on inf2 / tpu — same per-batch,
    no in-batch scheduler eviction). Unlike inf2/tpu these are
    **toggleable on GPU**: `--enable-chunked-prefill` and
    `--enable-prefix-caching` (or their `--no-…` counterparts) flip the
    LLM init. Mirror the matching `--{no-,}enable-{chunked-prefill,
    prefix-caching}` flags on the simulator side for an apples-to-apples
    comparison under the new setting.

Input  : studies/gpu_baseline/data/datasets/<dataset>.csv
Output : studies/gpu_baseline/results/lens_vllm/<hw>/<opt>/<model>/tp<N>/bs<B>/<dataset>.csv
         where <opt> is derived from the toggles:
           "off"   = both off (default; parity with inf2/tpu)
           "on"    = both on
           "cp{0,1}_pc{0,1}" = any mixed combination

Hardware label (`--hardware`) selects the output sub-folder so L4 and
A10G runs from the same script don't overwrite each other. It's a free
string — match it to the cluster config name (e.g. "L4" → folder
matches `configs/cluster/l4_llama1b_tp1.json`'s `hardware` field).

Usage
-----
    python studies/gpu_baseline/measure_vllm.py \\
        --dataset arxiv --batch-size 4 \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --hardware L4 \\
        --tp-degree 1 --max-model-len 8192

vLLM CUDA prerequisites on the host:
    pip install vllm          # standard CUDA build
    # NVIDIA driver + CUDA toolkit + matching torch (handled by the vllm
    # docker image: vllm/vllm-openai:v0.19.0). nvidia-smi must list the GPU.
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
NUM_BATCHES = 50

STUDY_ROOT = Path(__file__).resolve().parent
DATA_DIR = STUDY_ROOT / "data" / "datasets"
RESULTS_ROOT = STUDY_ROOT / "results" / "lens_vllm"


def opt_label(enable_chunked_prefill: bool, enable_prefix_caching: bool) -> str:
    """Canonical sub-folder name for the (chunked_prefill, prefix_caching) pair.
    Shared with the sim side of the comparison so paths line up."""
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
    """Same logic as inf2_baseline / tpu_baseline — top batch_size * NUM_BATCHES
    rows, re-grouped into NUM_BATCHES batches."""
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


def init_llm(model, tp_degree, batch_size, max_model_len,
             enable_chunked_prefill, enable_prefix_caching):
    """vLLM-CUDA LLM init. No vendor-specific overrides — the CUDA backend
    auto-selects FlashAttention / xFormers. bfloat16. Chunked prefill +
    prefix caching are toggleable here (GPU-only — inf2/tpu baselines
    keep both off for apples-to-apples comparison)."""
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
    """One batch (run): heterogeneous (input_len, output_len) per sample.
    Matches inf2_baseline / tpu_baseline schema."""
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
    p.add_argument("--hardware", required=True,
                   help="GPU label for the output folder (e.g. L4, A10G). "
                        "Match the cluster config's `hardware` field.")
    p.add_argument("--tp-degree", type=int, required=True, choices=[1, 2, 4, 8])
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--enable-chunked-prefill",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="GPU-only toggle. Default off (parity with "
                        "inf2/tpu baselines).")
    p.add_argument("--enable-prefix-caching",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="GPU-only toggle. Default off.")
    p.add_argument("--output-dir", default=None,
                   help="default: results/lens_vllm/<hw>/<model>/tp<N>/bs<B>/")
    p.add_argument("--skip-warmup", action="store_true")
    p.add_argument("--max-runs", type=int, default=None)
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

    llm = init_llm(args.model, args.tp_degree, args.batch_size, args.max_model_len,
                   args.enable_chunked_prefill, args.enable_prefix_caching)
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
            "runtime": "vLLM (CUDA backend)",
            "hardware": args.hardware,
            "dataset": args.dataset,
            "model": args.model,
            "tp_degree": args.tp_degree,
            "batch_size": args.batch_size,
            "max_model_len": args.max_model_len,
            "n_runs": n_runs,
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

    print(f"\n[DONE]\n  {out_csv}  ({len(rows)} rows)\n  {out_json}\n  "
          f"latest -> {stable.name}")


if __name__ == "__main__":
    main()
