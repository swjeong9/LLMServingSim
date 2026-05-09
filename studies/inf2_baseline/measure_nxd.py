#!/usr/bin/env python3
"""NxD-direct measurement on Inferentia 2 — LENS framework path.

Self-contained port of LENS/inference_source/run_eval.py — same
heterogeneous-batch eval logic, but reads our study's dataset CSV
and writes to our study's results tree, no LENS repo dependency.

Input  : studies/inf2_baseline/data/datasets/<dataset>.csv
         columns: run_id, sample_id, input_len, output_len
         (50 runs × batch_size samples per file at bs<=32)

Output : studies/inf2_baseline/results/lens_nxd/<model>/tp<N>/bs<B>/<dataset>.csv
         + sibling .json with run config snapshot

What it measures
----------------
For each of the 50 runs (= 50 batches in the dataset CSV):
  * uniform output_len = max(output_len in the batch)
  * heterogeneous input_len within the batch (right-padded)
  * batch_e2e_ms via time.perf_counter around gen_model.generate()
  * NxDI's NeuronConfig sets is_continuous_batching=False so all
    batch members run as a single uniform batch (no eviction).

Usage
-----
    python studies/inf2_baseline/measure_nxd.py \\
        --dataset arxiv --batch-size 4 \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --tp-degree 1 --max-model-len 8192 \\
        --compiled-dir /home/ubuntu/compiled_models_inf2_baseline

If --output-dir is omitted, defaults to:
    studies/inf2_baseline/results/lens_nxd/<model_name>/tp<N>/bs<B>/<dataset>.csv
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
RESULTS_ROOT = STUDY_ROOT / "results" / "lens_nxd"


def _model_name(model_path: str) -> str:
    """Pretty model name from local path or HF id."""
    p = model_path.rstrip("/")
    if "/snapshots/" in p and "/models--" in p:
        for part in p.split("/"):
            if part.startswith("models--"):
                segs = part[len("models--"):].split("--")
                if len(segs) >= 2:
                    return "--".join(segs[1:])
    return os.path.basename(p)


def _buckets_for(max_model_len: int):
    bs = [b for b in NEURON_BUCKETS if b <= max_model_len]
    if max_model_len not in bs:
        bs.append(max_model_len)
    return bs


def load_dataset_csv(path: Path, batch_size: int):
    """Take the top ``batch_size * NUM_BATCHES`` rows in source order
    and re-group into ``NUM_BATCHES`` batches of ``batch_size`` each.

    The dataset CSV's own ``run_id`` is bs32-anchored (32 samples per
    run_id) — we ignore it and re-derive batches so the same workload
    rows are reused across batch sizes (sweep reproducibility).
    """
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


def _get_model_cls(model_path: str):
    """Dispatch family → NxDI Neuron*ForCausalLM class."""
    name = os.path.basename(model_path.rstrip("/")).lower()
    full = model_path.lower()
    if "qwen3" in name or "qwen3" in full:
        from neuronx_distributed_inference.models.qwen3.modeling_qwen3 import (
            NeuronQwen3ForCausalLM,
        )
        return NeuronQwen3ForCausalLM
    if "mistral" in name or "mistral" in full:
        from neuronx_distributed_inference.models.mistral.modeling_mistral import (
            NeuronMistralForCausalLM,
        )
        return NeuronMistralForCausalLM
    if "llama" in name or "llama" in full:
        from neuronx_distributed_inference.models.llama.modeling_llama import (
            NeuronLlamaForCausalLM,
        )
        return NeuronLlamaForCausalLM
    raise ValueError(f"Unknown model family for path: {model_path}")


def init_model(model_path, tp_degree, batch_size, max_model_len,
               compiled_dir, skip_compile=False):
    """Load + compile NxDI model with is_continuous_batching=False."""
    import torch
    from transformers import AutoTokenizer
    from neuronx_distributed_inference.models.config import NeuronConfig
    from neuronx_distributed_inference.utils.hf_adapter import (
        HuggingFaceGenerationAdapter, load_pretrained_config,
    )

    buckets = _buckets_for(max_model_len)
    print(f"[init_model] model={model_path}")
    print(f"  tp={tp_degree} batch={batch_size} max_model_len={max_model_len}")
    print(f"  buckets={buckets}  is_continuous_batching=False")
    print(f"  compiled_dir={compiled_dir}")
    os.makedirs(compiled_dir, exist_ok=True)

    neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        batch_size=batch_size,
        ctx_batch_size=1,
        seq_len=max_model_len,
        max_context_length=max_model_len,
        context_encoding_buckets=buckets,
        token_generation_buckets=buckets,
        is_continuous_batching=False,
        enable_bucketing=True,
        torch_dtype=torch.bfloat16,
        padding_side="right",
    )
    model_cls = _get_model_cls(model_path)
    config = model_cls.get_config_cls()(
        neuron_config, load_config=load_pretrained_config(model_path),
    )
    model = model_cls(model_path, config)

    if not skip_compile:
        print("\n[compile] compiling ...")
        t0 = time.monotonic()
        model.compile(compiled_dir)
        print(f"  compile time: {time.monotonic()-t0:.1f}s")

    print("\n[load] loading to Neuron ...")
    t0 = time.monotonic()
    model.load(compiled_dir)
    print(f"  load time: {time.monotonic()-t0:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, HuggingFaceGenerationAdapter(model), tokenizer


def make_input_ids(tokenizer, target_len: int):
    """Dummy prompt of length target_len (post-tokenization, padded if short)."""
    if target_len <= 0:
        target_len = 1
    text = "The quick brown fox jumps over the lazy dog. " * (target_len // 8 + 1)
    ids = tokenizer.encode(text, add_special_tokens=True)[:target_len]
    while len(ids) < target_len:
        ids.append(tokenizer.pad_token_id)
    return ids


def build_batch_tensors(tokenizer, input_lens, batch_size):
    """Heterogeneous il list → right-padded (B, max_il) ids + mask."""
    import torch
    assert len(input_lens) == batch_size
    max_il = max(input_lens)
    pad = tokenizer.pad_token_id
    ids = torch.full((batch_size, max_il), pad, dtype=torch.int64)
    mask = torch.zeros((batch_size, max_il), dtype=torch.int32)
    for i, il in enumerate(input_lens):
        row = make_input_ids(tokenizer, il)
        ids[i, :il] = torch.tensor(row, dtype=torch.int64)
        mask[i, :il] = 1
    return ids, mask


def warmup_buckets(gen_model, neuron_model, tokenizer, batch_size, max_model_len):
    """Warm up each prefill bucket once (uniform batch)."""
    for il, ol in WARMUP_SHAPES:
        if il + ol > max_model_len:
            continue
        ids, mask = build_batch_tensors(tokenizer, [il] * batch_size, batch_size)
        try:
            t0 = time.perf_counter()
            gen_model.generate(input_ids=ids, attention_mask=mask,
                               max_new_tokens=ol, min_new_tokens=ol,
                               do_sample=False)
            neuron_model.reset()
            print(f"  warmup (il={il}, ol={ol}): {time.perf_counter()-t0:.2f}s")
        except Exception as e:
            print(f"  [WARN] warmup (il={il}, ol={ol}) failed: {e}")


def measure_run(gen_model, neuron_model, tokenizer, samples, batch_size,
                max_model_len, run_id):
    """One batch (run) measurement: heterogeneous il, uniform max_ol output."""
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

    ids, mask = build_batch_tensors(tokenizer, input_lens, batch_size)
    try:
        t0 = time.perf_counter()
        outputs = gen_model.generate(
            input_ids=ids, attention_mask=mask,
            max_new_tokens=max_ol, min_new_tokens=max_ol,
            do_sample=False, return_dict_in_generate=True)
        e2e_ms = round((time.perf_counter() - t0) * 1000, 3)
        neuron_model.reset()
        gen_len = int(outputs.sequences.shape[1] - max_il)
        return {**base, "status": "OK",
                "max_n_generated": gen_len,
                "batch_ttft_ms": "", "batch_e2e_ms": e2e_ms,
                "error": ""}
    except Exception as e:
        try: neuron_model.reset()
        except Exception: pass
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
    p.add_argument("--model", required=True,
                   help="Local model path or HF id (local recommended)")
    p.add_argument("--tp-degree", type=int, required=True, choices=[1, 2, 4, 8])
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--compiled-dir",
                   default="/home/ubuntu/compiled_models_inf2_baseline")
    p.add_argument("--output-dir", default=None,
                   help="default: results/lens_nxd/<model>/tp<N>/bs<B>/")
    p.add_argument("--skip-compile", action="store_true")
    p.add_argument("--skip-warmup", action="store_true")
    p.add_argument("--max-runs", type=int, default=None,
                   help="Run only first N runs (sanity check). default: all")
    args = p.parse_args()

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

    neuron_model, gen_model, tokenizer = init_model(
        args.model, args.tp_degree, args.batch_size, args.max_model_len,
        args.compiled_dir, skip_compile=args.skip_compile,
    )

    if not args.skip_warmup:
        print(f"\n[{datetime.now()}] warming up buckets ...")
        warmup_buckets(gen_model, neuron_model, tokenizer,
                       args.batch_size, args.max_model_len)

    print(f"\n[{datetime.now()}] eval sweep: {n_runs} runs × batch={args.batch_size}")
    rows = []
    t_sweep = time.perf_counter()
    for rid in sorted(runs):
        t0 = time.perf_counter()
        row = measure_run(gen_model, neuron_model, tokenizer, runs[rid],
                          args.batch_size, args.max_model_len, rid)
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
            "framework": "lens_nxd",
            "hardware": "inf2",
            "dataset": args.dataset,
            "model": args.model,
            "tp_degree": args.tp_degree,
            "batch_size": args.batch_size,
            "max_model_len": args.max_model_len,
            "n_runs": n_runs,
            "buckets": _buckets_for(args.max_model_len),
            "is_continuous_batching": False,
            "input_csv": str(src_csv),
            "run_timestamp": datetime.now().isoformat(),
            "total_sweep_s": round(time.perf_counter() - t_sweep, 1),
        }, f, indent=2)

    # Convenience: also leave a stable "<dataset>.csv" symlink to latest run
    stable = out_dir / f"{args.dataset}.csv"
    if stable.is_symlink() or stable.exists():
        stable.unlink()
    stable.symlink_to(out_csv.name)

    print(f"\n[DONE]\n  {out_csv}  ({len(rows)} rows)\n  {out_json}\n  "
          f"latest -> {stable.name}")


if __name__ == "__main__":
    main()
