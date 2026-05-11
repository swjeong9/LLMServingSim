#!/usr/bin/env python3
"""MaxText measurement on TPU — LENS framework path (TPU equivalent of measure_nxd.py).

Self-contained port of LENS/inference_source/run_eval_tpu.py — same
heterogeneous-batch eval logic (multi-engine, sample-by-sample prefill,
max_ol generate steps), but reads our study's dataset CSV and writes
to our study's results tree, no LENS repo dependency.

Input  : studies/tpu_baseline/data/datasets/<dataset>.csv  (linked from inf2_baseline)
         columns: run_id, sample_id, input_len, output_len

Output : studies/tpu_baseline/results/lens_tpu/<model>/tp<N>/bs<B>/<dataset>.csv
         + sibling .json with run config snapshot

Dependencies (must be installed on the TPU host):
    jax / jaxlib (TPU build)
    maxtext (from google/maxtext, with `maxtext.configs.pyconfig`
             and `maxtext.inference.maxengine.maxengine.MaxEngine`)
    transformers (for tokenizer)

Usage
-----
    python studies/tpu_baseline/measure_tpu.py \\
        --dataset arxiv --batch-size 4 \\
        --maxtext-model-name llama3.2-1b \\
        --tokenizer-path meta-llama/Llama-3.2-1B-Instruct \\
        --load-parameters-path /home/<user>/maxtext_ckpts/llama3.2-1b/0/items \\
        --tp-degree 1 --max-model-len 8192

If --output-dir is omitted, defaults to:
    studies/tpu_baseline/results/lens_tpu/<model_name>/tp<N>/bs<B>/<dataset>.csv
"""
import argparse
import csv
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Bucket grid — same as LENS run_eval_tpu.py
BUCKETS = [128, 256, 512, 1024, 2048, 4096, 8192]
NUM_BATCHES = 50

STUDY_ROOT = Path(__file__).resolve().parent
DATA_DIR = STUDY_ROOT / "data" / "datasets"
RESULTS_ROOT = STUDY_ROOT / "results" / "lens_tpu"


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
    bs = [b for b in BUCKETS if b <= max_model_len]
    if max_model_len not in bs:
        bs.append(max_model_len)
    return bs


def load_dataset_csv(path: Path, batch_size: int):
    """Same logic as inf2_baseline/measure_nxd.py — top `batch_size*NUM_BATCHES`
    rows in source order, re-grouped into `NUM_BATCHES` batches of `batch_size`."""
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


def _build_engine(maxtext_model_name, tokenizer_path, load_parameters_path,
                  batch_size, max_prefill, max_target, tp_degree):
    """Single MaxEngine init (config + engine). Lazy imports inside so the
    module is importable on hosts without maxtext."""
    import jax
    from maxtext.configs import pyconfig
    from maxtext.inference.maxengine.maxengine import MaxEngine

    base_yml = os.path.join(os.path.dirname(pyconfig.__file__), "base.yml")
    devices = max(jax.device_count(), 1)
    per_device_bs = (batch_size / devices) if (batch_size % devices) else (batch_size // devices)
    args_list = [
        "to_maxtext.py", base_yml,
        f"model_name={maxtext_model_name}",
        f"tokenizer_path={tokenizer_path}",
        f"load_parameters_path={load_parameters_path}",
        f"per_device_batch_size={per_device_bs}",
        f"max_prefill_predict_length={max_prefill}",
        f"max_target_length={max_target}",
        f"ici_tensor_parallelism={tp_degree}",
        "scan_layers=true", "weight_dtype=bfloat16",
        "attention=dot_product", "async_checkpointing=false",
        "skip_jax_distributed_system=true",
    ]
    config = pyconfig.initialize(args_list)
    engine = MaxEngine(config)
    return engine, config


def init_engines_multi(maxtext_model_name, tokenizer_path, load_parameters_path,
                       batch_size, tp_degree, jax_cache, decode_buckets):
    """One MaxEngine per decode bucket. Shares params across engines after
    the first (reshard only, not re-read from disk)."""
    import jax
    os.makedirs(jax_cache, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", jax_cache)

    print(f"[init_engines_multi] decode_buckets={decode_buckets}")
    print(f"[init_engines_multi] devices={jax.devices()}")

    engines_by_bucket = {}
    params = None
    rng = jax.random.PRNGKey(0)

    for bucket in decode_buckets:
        max_prefill = bucket
        max_target = bucket + 128
        engine, config = _build_engine(
            maxtext_model_name, tokenizer_path, load_parameters_path,
            batch_size, max_prefill, max_target, tp_degree,
        )
        t0 = time.perf_counter()
        rng, rng_load = jax.random.split(rng)
        if params is None:
            params = engine.load_params(rng_load)
        else:
            engine.load_params(rng_load, params=params)   # reshard only
        engines_by_bucket[bucket] = (engine, config)
        print(f"  engine[bucket={bucket}] ready (max_target={max_target}, "
              f"{time.perf_counter()-t0:.1f}s)")
    return engines_by_bucket, params, rng


def _pick_decode_bucket(target, decode_buckets):
    """target = max(il+ol) — smallest decode bucket that fits."""
    return next((b for b in decode_buckets if target <= b), decode_buckets[-1])


def run_heterogeneous_batch(engines_by_bucket, params, rng, samples, batch_size,
                            decode_buckets, max_steps_override=None):
    """Heterogeneous batch (per-sample il/ol) measurement.

    Dispatch path:
      * decode bucket = smallest bucket ≥ max(il+ol)
      * prefill per-sample (sequential, NxD-direct ctx_batch_size=1 equivalent)
      * insert into decode_state, then generate `max_ol` steps
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    input_lens  = [s["input_len"]  for s in samples]
    output_lens = [s["output_len"] for s in samples]
    max_ol = max(output_lens)
    max_il = max(input_lens)

    target = max_il + max_ol
    decode_bucket = _pick_decode_bucket(target, decode_buckets)
    engine, config = engines_by_bucket[decode_bucket]
    max_prefill_engine = config.max_prefill_predict_length

    t_start = time.perf_counter()

    # 1) Prefill — sample by sample
    prefill_results = []
    for slot, il in enumerate(input_lens):
        rng, rng_p = jax.random.split(rng)
        prefill_bucket = next(
            (b for b in BUCKETS if il <= b and b <= max_prefill_engine),
            max_prefill_engine,
        )
        tokens = np.ones((prefill_bucket,), dtype=np.int32)
        tokens[:il] = np.arange(1, il + 1, dtype=np.int32)
        tokens = jnp.array(tokens)
        prefill_result, _first_token = engine.prefill(
            params=params, padded_tokens=tokens, true_length=il,
            rng=rng_p, slot=slot,
        )
        prefill_results.append(prefill_result)

    # 2) init decode_state + insert
    rng, rng_init = jax.random.split(rng)
    decode_state = engine.init_decode_state(rng_init)
    for slot, pr in enumerate(prefill_results):
        decode_state = engine.insert(pr, decode_state, slot=slot)

    # 3) Generate (cap with max_steps_override during warmup)
    n_steps = min(max_ol, max_steps_override) if max_steps_override else max_ol
    sampled_tokens = None
    for _ in range(n_steps):
        rng, rng_g = jax.random.split(rng)
        decode_state, sampled_tokens = engine.generate(
            params, decode_state, rng=rng_g,
        )

    # 4) wait for async
    if sampled_tokens is not None:
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            sampled_tokens,
        )
    return (time.perf_counter() - t_start) * 1000  # ms


def _dataset_unique_pairs(runs, max_model_len):
    """For warmup: collect unique (decode_bucket, prefill_bucket) pairs the
    sweep will actually need. Avoids cross-product overkill."""
    pairs = set()
    for samples in runs.values():
        max_io = max(s["input_len"] + s["output_len"] for s in samples)
        if max_io > max_model_len:
            continue
        decode_b = next((b for b in BUCKETS if max_io <= b), BUCKETS[-1])
        for s in samples:
            il = s["input_len"]
            prefill_b = next(
                (b for b in BUCKETS if il <= b and b <= decode_b), decode_b)
            pairs.add((decode_b, prefill_b))
    return sorted(pairs)


def warmup_buckets(engines_by_bucket, params, rng, batch_size, max_model_len,
                   decode_buckets, runs, max_warmup_steps=2):
    """Compile-only warmup. Generate steps capped at `max_warmup_steps`."""
    pairs = _dataset_unique_pairs(runs, max_model_len)
    shapes = [(prefill_b - 1 if prefill_b > 1 else 1, decode_b - max(1, prefill_b - 1))
              for (decode_b, prefill_b) in pairs]
    shapes = [(il, ol) for il, ol in shapes if il + ol <= max_model_len and ol >= 1]
    print(f"  [warmup] {len(shapes)} dataset-adaptive (decode, prefill) pairs "
          f"(max_steps={max_warmup_steps})")
    for il, ol in shapes:
        try:
            samples = [{"sample_id": -1, "input_len": il, "output_len": ol}
                       for _ in range(batch_size)]
            t0 = time.perf_counter()
            run_heterogeneous_batch(engines_by_bucket, params, rng, samples,
                                    batch_size, decode_buckets,
                                    max_steps_override=max_warmup_steps)
            print(f"  warmup (il={il}, ol={ol}): {time.perf_counter()-t0:.2f}s")
        except Exception as e:
            print(f"  [WARN] warmup (il={il}, ol={ol}) failed: {e}")


def measure_run(engines_by_bucket, params, rng, samples, batch_size,
                max_model_len, run_id, decode_buckets):
    """One batch (run) — matches inf2_baseline/measure_nxd.py output schema."""
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
    try:
        e2e_ms = run_heterogeneous_batch(
            engines_by_bucket, params, rng, samples,
            batch_size, decode_buckets,
        )
        return {**base, "status": "OK",
                "max_n_generated": max_ol,
                "batch_ttft_ms": "", "batch_e2e_ms": round(e2e_ms, 3),
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
    p.add_argument("--maxtext-model-name", required=True,
                   help="MaxText model_name (yaml key, e.g. llama3.2-1b)")
    p.add_argument("--tokenizer-path", required=True,
                   help="HF id or local path for AutoTokenizer (e.g. meta-llama/Llama-3.2-1B-Instruct)")
    p.add_argument("--load-parameters-path", required=True,
                   help="MaxText Orbax checkpoint dir (e.g. ~/maxtext_ckpts/<m>/0/items)")
    p.add_argument("--tp-degree", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--output-dir", default=None,
                   help="default: results/lens_tpu/<model>/tp<N>/bs<B>/")
    p.add_argument("--jax-cache",
                   default=os.path.expanduser("~/jax_cache_lens_profiling"))
    p.add_argument("--skip-warmup", action="store_true")
    p.add_argument("--max-runs", type=int, default=None)
    p.add_argument("--per-batch-runs", type=int, default=1,
                   help="Median over N trials per batch (outlier guard)")
    args = p.parse_args()

    src_csv = DATA_DIR / f"{args.dataset}.csv"
    if not src_csv.exists():
        raise FileNotFoundError(f"dataset CSV missing: {src_csv}")

    model_label = _model_name(args.tokenizer_path)
    if args.output_dir is None:
        args.output_dir = (RESULTS_ROOT / model_label
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

    decode_buckets = _buckets_for(args.max_model_len)
    engines_by_bucket, params, rng = init_engines_multi(
        args.maxtext_model_name, args.tokenizer_path, args.load_parameters_path,
        args.batch_size, args.tp_degree, args.jax_cache, decode_buckets,
    )

    if not args.skip_warmup:
        print(f"\n[{datetime.now()}] warming up buckets ...")
        warmup_buckets(engines_by_bucket, params, rng,
                       args.batch_size, args.max_model_len,
                       decode_buckets, runs=runs)

    print(f"\n[{datetime.now()}] eval sweep: {n_runs} runs × batch={args.batch_size}"
          + (f" × {args.per_batch_runs} trials (median)" if args.per_batch_runs > 1 else ""))
    import statistics as _st
    rows = []
    t_sweep = time.perf_counter()
    for rid in sorted(runs):
        trials = []
        last_row = None
        for _ in range(args.per_batch_runs):
            row = measure_run(engines_by_bucket, params, rng, runs[rid],
                              args.batch_size, args.max_model_len, rid,
                              decode_buckets)
            last_row = row
            if row.get("status") == "OK":
                trials.append(row.get("batch_e2e_ms"))
        if trials and args.per_batch_runs > 1:
            last_row["batch_e2e_ms"] = round(_st.median(trials), 3)
            last_row["per_batch_trials"] = json.dumps(
                [round(t, 3) for t in trials])
        e2e = last_row.get("batch_e2e_ms")
        e2e_str = f"e2e={e2e:>9.0f}ms" if isinstance(e2e, (int, float)) else ""
        print(f"  [run {rid:>2d}]  max_il={last_row['max_input_len']:>5}  "
              f"max_ol={last_row['max_output_len']:>5}  "
              f"{last_row['status']:<20} {e2e_str}")
        rows.append(last_row)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"{args.dataset}_{ts}.csv"
    out_json = out_dir / f"{args.dataset}_{ts}.json"
    write_csv(out_csv, rows)
    with out_json.open("w") as f:
        json.dump({
            "framework": "lens_tpu",
            "runtime": "MaxText",
            "hardware": "tpu",
            "dataset": args.dataset,
            "maxtext_model_name": args.maxtext_model_name,
            "tokenizer_path": args.tokenizer_path,
            "load_parameters_path": args.load_parameters_path,
            "tp_degree": args.tp_degree,
            "batch_size": args.batch_size,
            "max_model_len": args.max_model_len,
            "n_runs": n_runs,
            "buckets": decode_buckets,
            "input_csv": str(src_csv),
            "run_timestamp": datetime.now().isoformat(),
            "total_sweep_s": round(time.perf_counter() - t_sweep, 1),
        }, f, indent=2)

    # Convenience: stable "<dataset>.csv" symlink to latest run (matches inf2_baseline)
    stable = out_dir / f"{args.dataset}.csv"
    if stable.is_symlink() or stable.exists():
        stable.unlink()
    stable.symlink_to(out_csv.name)

    print(f"\n[DONE]\n  {out_csv}  ({len(rows)} rows)\n  {out_json}\n  "
          f"latest -> {stable.name}")


if __name__ == "__main__":
    main()
