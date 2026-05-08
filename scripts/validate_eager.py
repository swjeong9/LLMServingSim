#!/usr/bin/env python3
"""
validate_eager.py — Fit a global scaling factor to a perf bundle by
comparing measured e2e generation latency (full-layer model on the same
eager runtime as profile_neuron.py) against the analytical estimate
predicted by the CSV bundle.

Mirrors the LLMServingSim 2.0 paper's TPU-v6e-1 notebook
``validate_and_scale`` flow (cells 5, 6, 11 of
``references/ispass26-artifact/llm_profile/perf_models/TPU-v6e-1/llm_profiler_tpu.ipynb``),
ported to AWS Inferentia 2.

Why this matters
----------------
``profile_neuron.py`` measures one-layer module latencies in isolation.
The simulator multiplies by ``num_hidden_layers`` to predict full-model
latency. But the sum of N isolated layer timings is *not* the same as
running the actual N-layer model end to end — fusion, cache reuse,
scheduling overhead all change. A single global scaling factor absorbs
this systematic gap.

Workflow
--------
1. Re-load the HF model with ``num_hidden_layers=<full or override>``
   on the same Neuron device as the profiler.
2. For each (input_len, output_len) shape:
     a. measured = e2e wall time of one prefill + (output_len-1) decode
        steps (warmup + median of repeats).
     b. estimated = analytical sum of per-layer CSV lookups × layer
        count + per_sequence + attention + sampler + embedding.
3. Scaling factor = median(measured / estimated) over all shapes.
4. Multiply every ``time_us`` in {dense, per_sequence, attention}.csv
   for every TP folder by the scaling factor; back up originals to
   ``*.pre_calib.csv``.
5. Record the scaling factor + per-shape ratios in
   ``meta.yaml::calibration``.

Important
---------
* Validation needs the FULL model on a single NeuronCore. Llama 3.2 1B
  (~2.5 GB) and Mistral 7B v0.3 (~14 GB) fit. Qwen3 14B (~30 GB) does
  not fit on one core; lower ``--num-layers`` (e.g. ``4``) for that
  case, which still captures inter-layer bias adequately.
* The scaling factor is shape-agnostic. We fit it on TP=1 (smallest
  shards, lowest memory), then apply the same factor to every TP folder.
  This mirrors the paper.

Usage
-----
::

    python scripts/validate_eager.py \\
        --model meta-llama/Llama-3.2-1B \\
        --variant-root profiler/perf/Inferentia2/meta-llama/Llama-3.2-1B/bf16 \\
        --shapes 128:32,512:32,1024:64,2048:128

    # Memory-tight model:
    python scripts/validate_eager.py \\
        --model Qwen/Qwen3-14B \\
        --variant-root profiler/perf/Inferentia2/Qwen/Qwen3-14B/bf16 \\
        --shapes 128:32,512:32,1024:64 \\
        --num-layers 4

Output
------
* ``<variant-root>/validation_data.json``  — per-shape measured + estimated
* ``<variant-root>/validation_fit.json``   — scaling factor + ratios
* Updated CSVs (each ``time_us`` ×scaling), with backups
* Updated ``<variant-root>/meta.yaml::calibration``
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
# Lazy imports
# ----------------------------------------------------------------------
def _lazy_import_runtime():
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_neuronx  # noqa: F401
    from transformers import AutoConfig, AutoModelForCausalLM
    return {
        "torch": torch, "torch_xla": torch_xla, "xm": xm,
        "AutoConfig": AutoConfig, "AutoModelForCausalLM": AutoModelForCausalLM,
    }


def _xla_device(rt):
    txla = rt["torch_xla"]
    if hasattr(txla, "device"):
        try:
            return txla.device()
        except TypeError:
            pass
    return rt["xm"].xla_device()


def _get_sync(rt):
    """Return a zero-arg sync callable. Prefers torch_xla.sync() (>= 2.4)
    over the deprecated xm.mark_step() + xm.wait_device_ops() pair."""
    txla = rt["torch_xla"]
    if hasattr(txla, "sync"):
        return lambda: txla.sync()
    xm = rt["xm"]
    def _legacy():
        xm.mark_step()
        xm.wait_device_ops()
    return _legacy


def _model_from_config(AutoModelForCausalLM, cfg, dtype):
    """Prefer ``dtype=`` kwarg (transformers >= 4.50) over deprecated ``torch_dtype=``."""
    try:
        return AutoModelForCausalLM.from_config(cfg, dtype=dtype).eval()
    except TypeError:
        return AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype).eval()


def _set_cfg_dtype(cfg, dtype):
    """Set config dtype using non-deprecated attribute when available."""
    if hasattr(cfg, "dtype"):
        try:
            cfg.dtype = dtype
            return
        except Exception:
            pass
    cfg.torch_dtype = dtype


# ======================================================================
# Layer occurrence counts per architecture
# ======================================================================
# Maps (catalog layer) -> (per-block count, per-model count). Total
# occurrences in a forward pass = per_block * num_layers + per_model.
LAYER_OCCURRENCES: Dict[str, Dict[str, Tuple[int, int]]] = {
    "llama": {
        "embedding":       (0, 1),
        "layernorm":       (2, 0),       # input + post_attention
        "qkv_proj":        (1, 0),
        "rotary_emb":      (1, 0),
        "o_proj":          (1, 0),
        "gate_up_proj":    (1, 0),
        "act_fn":          (1, 0),
        "down_proj":       (1, 0),
        "final_layernorm": (0, 1),
    },
    "mistral": {
        "embedding":       (0, 1),
        "layernorm":       (2, 0),
        "qkv_proj":        (1, 0),
        "rotary_emb":      (1, 0),
        "o_proj":          (1, 0),
        "gate_up_proj":    (1, 0),
        "act_fn":          (1, 0),
        "down_proj":       (1, 0),
        "final_layernorm": (0, 1),
    },
    "qwen3": {
        "embedding":       (0, 1),
        "layernorm":       (2, 0),
        "qkv_proj":        (1, 0),
        "qk_norm":         (1, 0),
        "rotary_emb":      (1, 0),
        "o_proj":          (1, 0),
        "gate_up_proj":    (1, 0),
        "act_fn":          (1, 0),
        "down_proj":       (1, 0),
        "final_layernorm": (0, 1),
    },
}


# ======================================================================
# CSV loaders + lookup helpers
# ======================================================================
def load_dense_csv(path: Path) -> Dict[str, List[Tuple[int, float]]]:
    """{layer_name: sorted [(tokens, time_us), ...]}."""
    out: Dict[str, List[Tuple[int, float]]] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            out.setdefault(r["layer"], []).append(
                (int(r["tokens"]), float(r["time_us"])))
    for k in out:
        out[k].sort()
    return out


def load_per_seq_csv(path: Path) -> Dict[str, List[Tuple[int, float]]]:
    out: Dict[str, List[Tuple[int, float]]] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            out.setdefault(r["layer"], []).append(
                (int(r["sequences"]), float(r["time_us"])))
    for k in out:
        out[k].sort()
    return out


def load_attention_csv(path: Path) -> List[Tuple[int, int, int, int, float]]:
    """List of (pc, kv_p, n, kv_d, time_us). Empty file returns []."""
    rows: List[Tuple[int, int, int, int, float]] = []
    if not path.exists() or path.stat().st_size == 0:
        return rows
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append((
                int(r["prefill_chunk"]), int(r["kv_prefill"]),
                int(r["n_decode"]), int(r["kv_decode"]),
                float(r["time_us"]),
            ))
    return rows


def lookup_1d(table: List[Tuple[int, float]], x: int) -> float:
    """Linear interpolation, with clamping at endpoints."""
    if not table:
        return 0.0
    if x <= table[0][0]:
        return table[0][1]
    if x >= table[-1][0]:
        return table[-1][1]
    for i in range(len(table) - 1):
        a, b = table[i], table[i + 1]
        if a[0] <= x <= b[0]:
            if b[0] == a[0]:
                return a[1]
            return a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0])
    return table[-1][1]


def lookup_attention(rows: List[Tuple[int, int, int, int, float]],
                     pc: int, kv_p: int, n: int, kv_d: int) -> float:
    """Nearest-neighbor on (pc, n) + bilinear on (kv_p, kv_d). Mirrors
    the simulator's _lookup_attention semantics for first-pass accuracy.
    Pure-prefill rows have n=0, kv_d=0; pure-decode rows have pc=0, kv_p=0.
    """
    if not rows:
        return 0.0
    # Filter to matching regime (pure prefill / pure decode).
    if n == 0 and kv_d == 0:
        cands = [r for r in rows if r[2] == 0 and r[3] == 0]
        if not cands:
            return 0.0
        # Pick nearest in (pc, kv_p) Manhattan distance.
        cands.sort(key=lambda r: abs(r[0]-pc) + abs(r[1]-kv_p)/64)
        return cands[0][4]
    if pc == 0 and kv_p == 0:
        cands = [r for r in rows if r[0] == 0 and r[1] == 0]
        if not cands:
            return 0.0
        cands.sort(key=lambda r: abs(r[2]-n) + abs(r[3]-kv_d)/64)
        return cands[0][4]
    # Mixed regime fallback: sum nearest pure-prefill + nearest pure-decode.
    return (lookup_attention(rows, pc, kv_p, 0, 0)
            + lookup_attention(rows, 0, 0, n, kv_d))


# ======================================================================
# Analytical estimator (replicates the simulator's per-iteration logic)
# ======================================================================
def estimate_prefill_us(input_len: int, num_layers: int, arch: str,
                        dense, per_seq, attn) -> float:
    """One prefill step over input_len tokens."""
    occ = LAYER_OCCURRENCES[arch]
    total = 0.0
    for layer, (per_block, per_model) in occ.items():
        if layer not in dense:
            continue
        t = lookup_1d(dense[layer], input_len)
        total += t * (per_block * num_layers + per_model)
    # Attention kernel
    total += lookup_attention(attn, pc=input_len, kv_p=0, n=0, kv_d=0) * num_layers
    # Head
    total += lookup_1d(per_seq.get("lm_head", []), 1)
    total += lookup_1d(per_seq.get("sampler", []), 1)
    return total


def estimate_decode_step_us(kv_so_far: int, num_layers: int, arch: str,
                            dense, per_seq, attn) -> float:
    """One decode step, generating one token, with kv_so_far KV history."""
    occ = LAYER_OCCURRENCES[arch]
    total = 0.0
    for layer, (per_block, per_model) in occ.items():
        if layer not in dense:
            continue
        t = lookup_1d(dense[layer], 1)
        total += t * (per_block * num_layers + per_model)
    total += lookup_attention(attn, pc=0, kv_p=0, n=1, kv_d=kv_so_far) * num_layers
    total += lookup_1d(per_seq.get("lm_head", []), 1)
    total += lookup_1d(per_seq.get("sampler", []), 1)
    return total


def estimate_total_us(input_len: int, output_len: int, num_layers: int,
                      arch: str, dense, per_seq, attn) -> float:
    """Estimated wall-time for prefill + (output_len-1) decodes.
    Matches the timing scope of measure_generation_latency below."""
    t = estimate_prefill_us(input_len, num_layers, arch, dense, per_seq, attn)
    for k in range(1, max(1, output_len)):
        kv_so_far = input_len + k - 1
        t += estimate_decode_step_us(kv_so_far, num_layers, arch,
                                     dense, per_seq, attn)
    return t


# ======================================================================
# Eager full-model measurement (mirrors notebook cell 5)
# ======================================================================
def load_full_model(model_id: str, num_layers: Optional[int],
                    dtype_str: str, max_pos: int, hf_token: str):
    """Load HF model with NUM_LAYERS = full (or override). Random weights."""
    rt = _lazy_import_runtime()
    torch = rt["torch"]
    sync = _get_sync(rt)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[dtype_str]
    cfg = rt["AutoConfig"].from_pretrained(model_id, token=hf_token or None)
    if num_layers is not None and num_layers > 0:
        cfg.num_hidden_layers = num_layers
    cfg.max_position_embeddings = min(
        getattr(cfg, "max_position_embeddings", max_pos), max_pos)
    _set_cfg_dtype(cfg, dtype)

    model = _model_from_config(rt["AutoModelForCausalLM"], cfg, dtype)
    device = _xla_device(rt)
    model = model.to(device)
    sync()
    return model, cfg, device


def measure_generation_latency(model, cfg, device, input_len: int,
                               output_len: int, warmup: int, repeat: int,
                               ) -> float:
    """One prefill + (output_len-1) decode steps, summed wall time. Returns us."""
    rt = _lazy_import_runtime()
    torch = rt["torch"]
    sync = _get_sync(rt)

    input_ids = torch.randint(0, cfg.vocab_size, (1, input_len),
                              dtype=torch.long, device=device)

    # ---- Warmup: prefill + (output_len-1) decode steps with same shapes
    #              the timed phase will use, so caches are populated.
    for _ in range(warmup):
        with torch.no_grad():
            out = model(input_ids=input_ids, past_key_values=None,
                        use_cache=True)
        sync()
        pkv = out.past_key_values
        last = input_ids[:, -1:]
        for _ in range(max(0, output_len - 1)):
            with torch.no_grad():
                out = model(input_ids=last, past_key_values=pkv,
                            use_cache=True)
            sync()
            pkv = out.past_key_values
            logits = out.logits[:, -1, :]
            last = torch.argmax(logits, dim=-1, keepdim=True)

    # ---- Timed
    samples_us: List[float] = []
    for _ in range(repeat):
        # Prefill
        sync()
        t0 = time.perf_counter_ns()
        with torch.no_grad():
            out = model(input_ids=input_ids, past_key_values=None,
                        use_cache=True)
        sync()
        iter_ns = time.perf_counter_ns() - t0

        pkv = out.past_key_values
        last = input_ids[:, -1:]
        for _ in range(1, max(1, output_len)):
            sync()
            t1 = time.perf_counter_ns()
            with torch.no_grad():
                out = model(input_ids=last, past_key_values=pkv,
                            use_cache=True)
            sync()
            iter_ns += time.perf_counter_ns() - t1
            pkv = out.past_key_values
            logits = out.logits[:, -1, :]
            last = torch.argmax(logits, dim=-1, keepdim=True)
        samples_us.append(iter_ns / 1000.0)
    return statistics.median(samples_us)


# ======================================================================
# Apply + record
# ======================================================================
def apply_scaling(variant_root: Path, scaling: float) -> None:
    """Multiply every time_us by scaling, in every tp<N>/{...}.csv.
    Saves originals to *.pre_calib.csv on first apply."""
    for tp_dir in sorted(variant_root.glob("tp*")):
        for csv_name in ("dense.csv", "per_sequence.csv", "attention.csv"):
            p = tp_dir / csv_name
            if not p.exists() or p.stat().st_size == 0:
                continue
            backup = tp_dir / f"{csv_name}.pre_calib.csv"
            if not backup.exists():
                shutil.copy2(p, backup)

            with p.open() as f:
                rows = list(csv.reader(f))
            header, *data = rows
            tcol = header.index("time_us")
            for row in data:
                row[tcol] = f"{float(row[tcol]) * scaling:.6g}"
            with p.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(data)
            print(f"  scaled {p}  (backup: {backup.name})")


def update_meta(variant_root: Path, scaling: float,
                ratios: Dict[str, float], num_layers: int,
                shapes: List[Tuple[int, int]]) -> None:
    import yaml
    meta_path = variant_root / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text()) or {}
    meta["calibration"] = {
        "scaling_factor": scaling,
        "scaled_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "method": "median(measured_eager_e2e / analytical_estimate)",
        "num_layers_validated": num_layers,
        "shapes": [f"{i}:{o}" for i, o in shapes],
        "per_shape_ratios": ratios,
    }
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"  updated {meta_path}")


# ======================================================================
# CLI
# ======================================================================
def parse_shapes(spec: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, b = chunk.split(":")
        out.append((int(a), int(b)))
    return out


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--model", required=True)
    p.add_argument("--variant-root", required=True,
                   help="profiler/perf/<HW>/<MODEL>/<variant>/")
    p.add_argument("--validate-tp", type=int, default=1,
                   help="Which tp<N>/ folder to validate against. "
                        "Use the smallest TP that fits memory (default 1).")
    p.add_argument("--num-layers", type=int, default=0,
                   help="Override num_hidden_layers for full-model load. "
                        "0 = use model config's full count. Lower this for "
                        "memory-tight models (Qwen3 14B: try 4).")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--shapes", default="128:32,512:32,1024:64,2048:128",
                   help="Comma list of input_len:output_len anchors")
    p.add_argument("--max-position-embeddings", type=int, default=8192)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--hf-token", default=os.getenv("HF_TOKEN", ""))
    p.add_argument("--metric", default="e2e", choices=["e2e"],
                   help="(reserved) Currently only e2e is supported.")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute scaling but do NOT modify CSVs.")
    return p.parse_args()


def main():
    args = parse_args()
    variant_root = Path(args.variant_root).resolve()
    if not variant_root.exists():
        print(f"[!] variant_root not found: {variant_root}")
        sys.exit(2)
    shapes = parse_shapes(args.shapes)

    # ---- Load CSVs from the chosen TP folder ----
    tp_dir = variant_root / f"tp{args.validate_tp}"
    if not tp_dir.exists():
        print(f"[!] {tp_dir} not found. Re-check --validate-tp.")
        sys.exit(2)
    print(f"[*] reading CSV bundle from {tp_dir}")
    dense = load_dense_csv(tp_dir / "dense.csv")
    per_seq = load_per_seq_csv(tp_dir / "per_sequence.csv")
    attn = load_attention_csv(tp_dir / "attention.csv")

    # Validation-time accounting (saved as validation_timing.json).
    timing_run = {
        "schema": "validation_timing-v1",
        "model": args.model, "variant_root": str(variant_root),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "machine": _capture_machine_info(),
        "shapes": [],   # filled per shape: {key, measured_us, estimated_us, ratio, wall_sec}
        "stages": {"load_sec": 0.0, "shapes_sec": 0.0, "apply_sec": 0.0,
                   "total_sec": 0.0},
    }
    run_t0 = time.perf_counter()

    # ---- Determine arch + num_layers_validate ----
    rt = _lazy_import_runtime()
    cfg0 = rt["AutoConfig"].from_pretrained(args.model,
                                            token=args.hf_token or None)
    arch = cfg0.model_type
    if arch not in LAYER_OCCURRENCES:
        print(f"[!] arch {arch!r} not registered. Add to LAYER_OCCURRENCES.")
        sys.exit(2)
    full_layers = cfg0.num_hidden_layers
    nlv = args.num_layers if args.num_layers > 0 else full_layers
    if nlv < full_layers:
        print(f"[!] using num_layers={nlv} (< full {full_layers}). "
              "Inter-layer bias still captured but absolute estimate slightly "
              "shifted; scaling factor remains valid as long as we use the "
              "same nlv on both sides.")
    timing_run["num_layers_validated"] = nlv

    # ---- Load full model ----
    print(f"[*] loading model with num_hidden_layers={nlv} on Neuron core")
    load_t0 = time.perf_counter()
    model, cfg, device = load_full_model(args.model, nlv, args.dtype,
                                         args.max_position_embeddings,
                                         args.hf_token)
    timing_run["stages"]["load_sec"] = time.perf_counter() - load_t0
    print(f"  loaded in {timing_run['stages']['load_sec']:.1f}s")

    # ---- Per-shape: measure + estimate ----
    print(f"[*] validating on {len(shapes)} shape(s)")
    rows: Dict[str, Dict[str, float]] = {}
    ratios: Dict[str, float] = {}
    shapes_t0 = time.perf_counter()
    for inp_len, out_len in shapes:
        if inp_len + out_len > cfg.max_position_embeddings:
            print(f"  shape {inp_len}:{out_len} exceeds max_position_embeddings; skip")
            continue
        shape_t0 = time.perf_counter()
        try:
            measured = measure_generation_latency(
                model, cfg, device, inp_len, out_len,
                args.warmup, args.repeat)
        except Exception as e:
            print(f"  [WARN] measurement {inp_len}:{out_len} failed: {e}")
            continue
        shape_wall = time.perf_counter() - shape_t0
        estimated = estimate_total_us(inp_len, out_len, nlv, arch,
                                      dense, per_seq, attn)
        ratio = measured / estimated if estimated > 0 else float("nan")
        key = f"{inp_len}:{out_len}"
        rows[key] = {"measured_us": measured, "estimated_us": estimated,
                     "ratio": ratio}
        ratios[key] = ratio
        timing_run["shapes"].append({
            "key": key, "input_len": inp_len, "output_len": out_len,
            "measured_us": measured, "estimated_us": estimated,
            "ratio": ratio, "wall_sec": shape_wall,
        })
        print(f"  {key:>14s}   measured={measured:10.1f} us   "
              f"estimated={estimated:10.1f} us   ratio={ratio:.4f}   "
              f"(wall {shape_wall:.1f}s)")
    timing_run["stages"]["shapes_sec"] = time.perf_counter() - shapes_t0

    if not ratios:
        print("[!] no valid shapes; aborting.")
        sys.exit(2)

    # ---- Fit ----
    s = statistics.median(ratios.values())
    print(f"\n[*] median scaling factor s = {s:.4f}")

    # ---- Save validation_data.json + validation_fit.json ----
    (variant_root / "validation_data.json").write_text(
        json.dumps(rows, indent=2))
    (variant_root / "validation_fit.json").write_text(
        json.dumps({"scaling_factor": s, "method": "median",
                    "num_layers_validated": nlv,
                    "shapes": shapes,
                    "per_shape_ratios": ratios}, indent=2))
    print(f"[✓] wrote validation_data.json + validation_fit.json")

    apply_t0 = time.perf_counter()
    if args.dry_run:
        print("[*] --dry-run: not modifying CSVs.")
    else:
        # ---- Apply ----
        import yaml
        meta = yaml.safe_load((variant_root / "meta.yaml").read_text()) or {}
        already = (meta.get("calibration", {}) or {}).get("scaling_factor", 1.0)
        if abs(already - 1.0) > 1e-6:
            print(f"[!] CSVs already scaled by {already:.4f}; "
                  f"undoing first by ×{1.0/already:.4f}, then ×{s:.4f}.")
            apply_scaling(variant_root, 1.0 / already)
        apply_scaling(variant_root, s)
        update_meta(variant_root, s, ratios, nlv, shapes)
        print(f"[✓] calibration applied")
    timing_run["stages"]["apply_sec"] = time.perf_counter() - apply_t0

    # Save timing artifact
    timing_run["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    timing_run["stages"]["total_sec"] = time.perf_counter() - run_t0
    timing_run["scaling_factor"] = s
    timing_path = variant_root / "validation_timing.json"
    timing_path.write_text(json.dumps(timing_run, indent=2, default=str))
    print(f"[✓] validation_timing.json written ({timing_path}, "
          f"total {timing_run['stages']['total_sec']/60:.1f} min)")


def _capture_machine_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00",
                                                       time.gmtime())}
    try:
        import platform
        info["python"] = platform.python_version()
        info["platform"] = platform.platform()
    except Exception:
        pass
    for mod_name in ("torch", "torch_xla", "torch_neuronx", "transformers"):
        try:
            mod = __import__(mod_name)
            info[mod_name] = getattr(mod, "__version__", "unknown")
        except Exception:
            info[mod_name] = "not-installed"
    try:
        import subprocess
        out = subprocess.run(["neuron-ls"], capture_output=True, text=True, timeout=5)
        for line in out.stdout.splitlines():
            if "instance-type" in line:
                info["instance_type"] = line.split(":", 1)[-1].strip()
                break
    except Exception:
        pass
    return info


if __name__ == "__main__":
    main()
