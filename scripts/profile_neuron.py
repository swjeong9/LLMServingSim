#!/usr/bin/env python3
"""
profile_neuron.py — Layer-wise latency profiler for LLMServingSim 2.0
on AWS Inferentia 2 (eager mode, transformers + torch_neuronx).

Methodology
-----------
Mirrors the LLMServingSim 2.0 paper's TPU-v6e-1 notebook
(see references/ispass26-artifact/.../llm_profiler_tpu.ipynb), adapted
for AWS Inferentia 2:

* Load the HF model with ``num_hidden_layers=1`` so a single
  NeuronCore-v2 hosts any model size up to ~30B params. The simulator
  multiplies per-layer latency by the real layer count via the model
  config at runtime.
* Emulate TP > 1 on a single core by sharding head/intermediate
  dimensions in the loaded HF config (matches the GPU profiler's
  ``hf_overrides`` approach).
* Wrap each catalog layer's ``forward`` with a timing wrapper that
  brackets ``mark_step + wait_device_ops`` for synchronous Neuron
  execution and records ``time.perf_counter_ns()``.
* Sweep input shapes:
    - dense:        a 1D token-count grid
    - per_sequence: a 1D batch-size grid (lm_head)
    - attention:    1D × 2 sweeps (pure prefill, pure decode), with
      2D mixed regime turned off by default. Output rows are the
      kernel-only time (full self_attn time minus per-projection time
      from the dense sweep).
* Run ``warmup`` warm-ups + ``repeat`` timed iterations per shape;
  report median.
* Write CSVs in the v1 schema directly: ``dense.csv``,
  ``per_sequence.csv``, ``attention.csv``, plus one ``meta.yaml`` per
  variant.

Production deployment is expected to use NxDI (compiled NEFF). Eager
profile timings have a systematic offset vs compiled execution; run
``calibrate_with_nxdi.py`` afterwards to fit a global scaling factor.

Environment
-----------
Run on an inf2 instance (e.g., inf2.24xlarge) with the AWS Neuron
DLAMI. A single core is enough thanks to the NUM_LAYERS=1 trick.

Required packages: torch, torch_xla, torch_neuronx, transformers,
PyYAML.

Usage
-----
::

    # On inf2:
    source /opt/aws_neuronx_venv_pytorch_2_5/bin/activate
    pip install transformers pyyaml

    python scripts/profile_neuron.py \\
        --model meta-llama/Llama-3.2-1B \\
        --tp 1,2,4,8 \\
        --output-root profiler/perf

    python scripts/profile_neuron.py \\
        --model mistralai/Mistral-7B-v0.3 \\
        --tp 1,2,4,8 \\
        --output-root profiler/perf

    python scripts/profile_neuron.py \\
        --model Qwen/Qwen3-14B \\
        --tp 2,4,8 \\
        --output-root profiler/perf

Output
------
::

    <output-root>/Inferentia2/<MODEL>/<variant>/
        meta.yaml
        tp1/{dense,per_sequence,attention}.csv
        tp2/...
        ...
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ----------------------------------------------------------------------
# Lazy imports — let `--help` work on a dev machine without Neuron SDK
# ----------------------------------------------------------------------
def _lazy_import_runtime():
    """Import torch / torch_xla / torch_neuronx / transformers lazily."""
    import torch
    import torch_xla.core.xla_model as xm
    import torch_neuronx  # noqa: F401  — wires Neuron into torch_xla
    from transformers import AutoConfig, AutoModelForCausalLM
    return {
        "torch": torch,
        "xm": xm,
        "AutoConfig": AutoConfig,
        "AutoModelForCausalLM": AutoModelForCausalLM,
    }


# ======================================================================
# Architecture descriptors
# ======================================================================
# For each supported model_type, declare:
#   * dense_layers: list of (catalog_name, list_of_dotted_paths_under_model,
#                            input_shape_kind)
#   * per_seq_layers: ditto for per_sequence catalog
#   * attention_module_path: dotted path to the self_attn module of layer 0
#
# input_shape_kind values used by build_dummy_input():
#   "ids"     -> (1, n) int64 token ids
#   "hidden"  -> (1, n, hidden_size) bf16
#   "qkv_in"  -> (1, n, hidden_size) bf16   (input to qkv projections)
#   "o_in"    -> (1, n, num_heads*head_dim) bf16
#   "mlp_in"  -> (1, n, hidden_size) bf16
#   "down_in" -> (1, n, intermediate_size) bf16
#   "qknorm"  -> (1, num_heads, n, head_dim) bf16
#   "headhid" -> (1, n, hidden_size) bf16   (lm_head input; same as hidden)

ARCH_DESC: Dict[str, Dict[str, Any]] = {
    "llama": {
        "dense_layers": [
            ("embedding",       ["model.embed_tokens"],                  "ids"),
            ("layernorm",       ["model.layers.0.input_layernorm",
                                 "model.layers.0.post_attention_layernorm"], "hidden"),
            ("qkv_proj",        ["model.layers.0.self_attn.q_proj",
                                 "model.layers.0.self_attn.k_proj",
                                 "model.layers.0.self_attn.v_proj"],     "qkv_in"),
            ("o_proj",          ["model.layers.0.self_attn.o_proj"],     "o_in"),
            ("gate_up_proj",    ["model.layers.0.mlp.gate_proj",
                                 "model.layers.0.mlp.up_proj"],          "mlp_in"),
            ("down_proj",       ["model.layers.0.mlp.down_proj"],        "down_in"),
            ("final_layernorm", ["model.norm"],                          "hidden"),
        ],
        "per_seq_layers": [("lm_head", ["lm_head"], "headhid")],
        "attention_module": "model.layers.0.self_attn",
    },
    "mistral": {
        # Same Decoder layer structure as Llama.
        "dense_layers": [
            ("embedding",       ["model.embed_tokens"],                  "ids"),
            ("layernorm",       ["model.layers.0.input_layernorm",
                                 "model.layers.0.post_attention_layernorm"], "hidden"),
            ("qkv_proj",        ["model.layers.0.self_attn.q_proj",
                                 "model.layers.0.self_attn.k_proj",
                                 "model.layers.0.self_attn.v_proj"],     "qkv_in"),
            ("o_proj",          ["model.layers.0.self_attn.o_proj"],     "o_in"),
            ("gate_up_proj",    ["model.layers.0.mlp.gate_proj",
                                 "model.layers.0.mlp.up_proj"],          "mlp_in"),
            ("down_proj",       ["model.layers.0.mlp.down_proj"],        "down_in"),
            ("final_layernorm", ["model.norm"],                          "hidden"),
        ],
        "per_seq_layers": [("lm_head", ["lm_head"], "headhid")],
        "attention_module": "model.layers.0.self_attn",
    },
    "qwen3": {
        # Qwen3 adds q_norm / k_norm inside self_attn. Catalog name is
        # qk_norm (sum of both). Other layers identical to Llama.
        "dense_layers": [
            ("embedding",       ["model.embed_tokens"],                  "ids"),
            ("layernorm",       ["model.layers.0.input_layernorm",
                                 "model.layers.0.post_attention_layernorm"], "hidden"),
            ("qkv_proj",        ["model.layers.0.self_attn.q_proj",
                                 "model.layers.0.self_attn.k_proj",
                                 "model.layers.0.self_attn.v_proj"],     "qkv_in"),
            ("qk_norm",         ["model.layers.0.self_attn.q_norm",
                                 "model.layers.0.self_attn.k_norm"],     "qknorm"),
            ("o_proj",          ["model.layers.0.self_attn.o_proj"],     "o_in"),
            ("gate_up_proj",    ["model.layers.0.mlp.gate_proj",
                                 "model.layers.0.mlp.up_proj"],          "mlp_in"),
            ("down_proj",       ["model.layers.0.mlp.down_proj"],        "down_in"),
            ("final_layernorm", ["model.norm"],                          "hidden"),
        ],
        "per_seq_layers": [("lm_head", ["lm_head"], "headhid")],
        "attention_module": "model.layers.0.self_attn",
    },
}


# Layers that the simulator wants but we don't directly time on Neuron:
# rotary_emb, act_fn, sampler. Approximate with cheap analytic forms;
# refine later if precision demands.
ROTARY_BASE_US = 1.0
ROTARY_PER_TOK_US = 0.001
ACT_FN_BASE_US   = 0.5
ACT_FN_PER_TOK_US = 0.0008
SAMPLER_BASE_US  = 1.0
SAMPLER_PER_SEQ_US = 0.05


# ======================================================================
# Helpers
# ======================================================================
def _csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def get_module(model, dotted: str):
    obj = model
    for part in dotted.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def build_dummy_input(kind: str, n: int, cfg, dtype, device, batch: int = 1):
    """Build a synthetic input tensor for a given input_shape_kind."""
    rt = _lazy_import_runtime()
    torch = rt["torch"]
    H = cfg.hidden_size
    nh = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", H // nh)
    inter = cfg.intermediate_size

    if kind == "ids":
        return torch.zeros((batch, n), dtype=torch.long, device=device)
    if kind in ("hidden", "qkv_in", "mlp_in", "headhid"):
        return torch.zeros((batch, n, H), dtype=dtype, device=device)
    if kind == "o_in":
        return torch.zeros((batch, n, nh * head_dim), dtype=dtype, device=device)
    if kind == "down_in":
        return torch.zeros((batch, n, inter), dtype=dtype, device=device)
    if kind == "qknorm":
        return torch.zeros((batch, nh, n, head_dim), dtype=dtype, device=device)
    raise ValueError(f"Unknown input kind: {kind}")


# ======================================================================
# Synchronous Neuron-aware timing
# ======================================================================
def time_callable(fn: Callable[[], Any], warmup: int, repeat: int) -> float:
    """Time fn() on Neuron via XLA mark_step + wait. Return median microseconds.

    fn() must wrap whatever forward call we want to measure. Inputs
    should already live on the Neuron device.
    """
    rt = _lazy_import_runtime()
    xm = rt["xm"]

    # Warmup. Mirror the timed phase exactly: each fn() call is bracketed
    # by mark_step + wait_device_ops so the compiled graph in the cache
    # matches what the timed phase will look up. Without per-call
    # mark_step, the warmup would build one big N-fn-calls graph and the
    # timed phase's single-fn-call graph would be a fresh cache miss.
    for _ in range(warmup):
        xm.mark_step()
        xm.wait_device_ops()
        out = fn()
        xm.mark_step()
        xm.wait_device_ops()
        del out

    samples: List[float] = []
    for _ in range(repeat):
        xm.mark_step()
        xm.wait_device_ops()
        t0 = time.perf_counter_ns()
        out = fn()
        xm.mark_step()
        xm.wait_device_ops()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1000.0)  # ns -> us
        del out
    return statistics.median(samples)


# ======================================================================
# Model loading: NUM_LAYERS=1 + TP shard emulation
# ======================================================================
def shard_config(cfg, tp: int):
    """Mutate cfg in-place to emulate TP=tp by dividing shardable dims."""
    if tp == 1:
        return cfg
    nh = cfg.num_attention_heads
    nkv = getattr(cfg, "num_key_value_heads", nh)
    inter = cfg.intermediate_size
    if nh % tp or nkv % tp or inter % tp:
        raise ValueError(
            f"TP={tp} doesn't evenly divide one of "
            f"(num_attention_heads={nh}, num_key_value_heads={nkv}, "
            f"intermediate_size={inter})"
        )
    cfg.num_attention_heads = nh // tp
    cfg.num_key_value_heads = nkv // tp
    cfg.intermediate_size = inter // tp
    return cfg


def load_model(model_id: str, tp: int, dtype_str: str, max_pos: int,
               hf_token: str):
    """Load HF model with num_hidden_layers=1, sharded for TP, on Neuron."""
    rt = _lazy_import_runtime()
    torch = rt["torch"]
    xm = rt["xm"]
    AutoConfig = rt["AutoConfig"]
    AutoModelForCausalLM = rt["AutoModelForCausalLM"]

    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[dtype_str]

    cfg = AutoConfig.from_pretrained(model_id, token=hf_token or None)
    cfg.num_hidden_layers = 1   # the trick
    cfg.max_position_embeddings = min(
        getattr(cfg, "max_position_embeddings", max_pos), max_pos)
    cfg = shard_config(cfg, tp)
    cfg.torch_dtype = dtype

    # Random weights are fine: we time arithmetic, not correctness.
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype).eval()
    device = xm.xla_device()
    model = model.to(device)
    xm.mark_step()
    xm.wait_device_ops()
    return model, cfg, device, dtype


def free_model(model):
    rt = _lazy_import_runtime()
    xm = rt["xm"]
    del model
    gc.collect()
    try:
        xm.mark_step()
    except Exception:
        pass


# ======================================================================
# Sweeps
# ======================================================================
def sweep_dense(model, cfg, dtype, device, arch: str,
                tokens_grid: Sequence[int], warmup: int, repeat: int
                ) -> List[Tuple[str, int, float]]:
    """Time each catalog dense layer at each token count.

    Returns rows of (layer_name, tokens, time_us).
    """
    rows: List[Tuple[str, int, float]] = []
    for layer_name, paths, kind in ARCH_DESC[arch]["dense_layers"]:
        modules = [get_module(model, p) for p in paths]
        for n in tokens_grid:
            x = build_dummy_input(kind, n, cfg, dtype, device)

            def call(mods=modules, x=x):
                # Sum the outputs to keep them live until mark_step.
                outs = [m(x) for m in mods]
                return outs

            try:
                t_us = time_callable(call, warmup, repeat)
            except Exception as e:
                print(f"    [WARN] dense {layer_name} n={n} failed: {e}")
                continue
            rows.append((layer_name, n, t_us))
            print(f"    dense  {layer_name:18s} tokens={n:5d}  -> {t_us:9.3f} us")

    # Synthesize rotary_emb and act_fn (not directly times-able as standalone
    # modules without HF-version-specific glue).
    for n in tokens_grid:
        rows.append(("rotary_emb", n, ROTARY_BASE_US + ROTARY_PER_TOK_US * n))
        rows.append(("act_fn",     n, ACT_FN_BASE_US + ACT_FN_PER_TOK_US * n))

    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def sweep_per_sequence(model, cfg, dtype, device, arch: str,
                       sequences_grid: Sequence[int], warmup: int, repeat: int
                       ) -> List[Tuple[str, int, float]]:
    """Time lm_head at each sequence count; synthesize sampler row."""
    rows: List[Tuple[str, int, float]] = []
    for layer_name, paths, kind in ARCH_DESC[arch]["per_seq_layers"]:
        modules = [get_module(model, p) for p in paths]
        for s in sequences_grid:
            x = build_dummy_input(kind, n=1, cfg=cfg, dtype=dtype,
                                  device=device, batch=s)

            def call(mods=modules, x=x):
                return [m(x) for m in mods]

            try:
                t_us = time_callable(call, warmup, repeat)
            except Exception as e:
                print(f"    [WARN] per_seq {layer_name} s={s} failed: {e}")
                continue
            rows.append((layer_name, s, t_us))
            print(f"    per_s  {layer_name:18s} seqs={s:5d}    -> {t_us:9.3f} us")

    for s in sequences_grid:
        rows.append(("sampler", s, SAMPLER_BASE_US + SAMPLER_PER_SEQ_US * s))

    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


# Attention: time the full self_attn forward at various (pc, kv_p, n, kv_d).
# Subtract the matching q_proj + k_proj + v_proj + o_proj cost (looked up
# from the dense sweep) to leave the kernel-only residual.
def _build_kv_cache(cfg, dtype, device, kv_len: int, batch: int):
    """Construct a HF DynamicCache populated with zero K/V tensors at layer 0."""
    rt = _lazy_import_runtime()
    torch = rt["torch"]
    try:
        from transformers import DynamicCache
    except ImportError:
        from transformers.cache_utils import DynamicCache

    nkv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    cache = DynamicCache()
    if kv_len <= 0:
        return cache
    k = torch.zeros((batch, nkv, kv_len, head_dim), dtype=dtype, device=device)
    v = torch.zeros((batch, nkv, kv_len, head_dim), dtype=dtype, device=device)
    cache.update(k, v, layer_idx=0,
                 cache_kwargs={"cache_position": torch.arange(kv_len, device=device)})
    return cache


def _projection_us_at(dense_rows: List[Tuple[str, int, float]],
                      tokens: int) -> float:
    """Sum qkv_proj + o_proj at the given token count (linear interp)."""
    def lookup(layer: str, n: int) -> float:
        pts = sorted([(t, us) for (l, t, us) in dense_rows if l == layer])
        if not pts:
            return 0.0
        if n <= pts[0][0]:
            return pts[0][1]
        if n >= pts[-1][0]:
            return pts[-1][1]
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            if a[0] <= n <= b[0]:
                return a[1] + (b[1] - a[1]) * (n - a[0]) / (b[0] - a[0])
        return pts[-1][1]

    return lookup("qkv_proj", tokens) + lookup("o_proj", tokens)


def sweep_attention(model, cfg, dtype, device, arch: str,
                    prefill_grid: Sequence[int],
                    kv_prefill_grid: Sequence[int],
                    decode_n_grid: Sequence[int],
                    kv_decode_grid: Sequence[int],
                    dense_rows: List[Tuple[str, int, float]],
                    warmup: int, repeat: int,
                    ) -> List[Tuple[int, int, int, int, float]]:
    """4D-axis attention sweep. Pure prefill and pure decode regimes only.

    Returns rows of (prefill_chunk, kv_prefill, n_decode, kv_decode, time_us).
    The time_us is the kernel-only residual: full self_attn time minus
    the (qkv_proj + o_proj) cost looked up from dense_rows at the matching
    token count.
    """
    rt = _lazy_import_runtime()
    torch = rt["torch"]
    self_attn = get_module(model, ARCH_DESC[arch]["attention_module"])
    H = cfg.hidden_size
    nh = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", H // nh)

    def call_self_attn(hidden, position_ids, past_key_value):
        """Try a few HF-version-friendly call signatures."""
        try:
            return self_attn(
                hidden_states=hidden,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=False,
            )
        except TypeError:
            # newer HF wants position_embeddings tuple
            seq_len = hidden.shape[1] + (
                past_key_value.get_seq_length(0) if past_key_value else 0)
            cos = torch.zeros((1, seq_len, head_dim), dtype=hidden.dtype, device=hidden.device)
            sin = torch.zeros_like(cos)
            return self_attn(
                hidden_states=hidden,
                position_embeddings=(cos, sin),
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=False,
            )

    rows: List[Tuple[int, int, int, int, float]] = []

    # ---- Pure prefill (pc, kv_p, 0, 0) ----
    for pc in prefill_grid:
        for kv_p in kv_prefill_grid:
            if pc + kv_p > cfg.max_position_embeddings:
                continue
            hidden = torch.zeros((1, pc, H), dtype=dtype, device=device)
            past_kv = _build_kv_cache(cfg, dtype, device, kv_p, batch=1)
            position_ids = torch.arange(kv_p, kv_p + pc, device=device).unsqueeze(0)

            def call(h=hidden, pi=position_ids, pkv=past_kv):
                return call_self_attn(h, pi, pkv)

            try:
                t_full = time_callable(call, warmup, repeat)
            except Exception as e:
                print(f"    [WARN] attn prefill pc={pc} kv_p={kv_p} failed: {e}")
                continue
            t_proj = _projection_us_at(dense_rows, pc)
            t_kernel = max(t_full - t_proj, 0.5)
            rows.append((pc, kv_p, 0, 0, t_kernel))
            print(f"    attn   prefill pc={pc:5d} kv_p={kv_p:5d}                 "
                  f"-> {t_full:9.3f} - {t_proj:7.3f} = {t_kernel:9.3f} us")

    # ---- Pure decode (0, 0, n, kv_d) ----
    for n in decode_n_grid:
        for kv_d in kv_decode_grid:
            if kv_d > cfg.max_position_embeddings:
                continue
            hidden = torch.zeros((n, 1, H), dtype=dtype, device=device)
            past_kv = _build_kv_cache(cfg, dtype, device, kv_d, batch=n)
            position_ids = torch.full((n, 1), kv_d, dtype=torch.long, device=device)

            def call(h=hidden, pi=position_ids, pkv=past_kv):
                return call_self_attn(h, pi, pkv)

            try:
                t_full = time_callable(call, warmup, repeat)
            except Exception as e:
                print(f"    [WARN] attn decode n={n} kv_d={kv_d} failed: {e}")
                continue
            t_proj = _projection_us_at(dense_rows, n)
            t_kernel = max(t_full - t_proj, 0.5)
            rows.append((0, 0, n, kv_d, t_kernel))
            print(f"    attn   decode                       n={n:4d} kv_d={kv_d:5d}"
                  f" -> {t_full:9.3f} - {t_proj:7.3f} = {t_kernel:9.3f} us")

    rows.sort()
    return rows


# ======================================================================
# CSV / meta.yaml writers (v1 schema)
# ======================================================================
def write_dense_csv(rows: List[Tuple[str, int, float]], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "tokens", "time_us"])
        for layer, n, t in rows:
            w.writerow([layer, n, f"{t:.6g}"])


def write_per_sequence_csv(rows: List[Tuple[str, int, float]], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "sequences", "time_us"])
        for layer, s, t in rows:
            w.writerow([layer, s, f"{t:.6g}"])


def write_attention_csv(rows: List[Tuple[int, int, int, int, float]], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prefill_chunk", "kv_prefill", "n_decode", "kv_decode", "time_us"])
        for pc, kvp, n, kvd, t in rows:
            w.writerow([pc, kvp, n, kvd, f"{t:.6g}"])


def write_meta(out_dir: Path, hardware: str, model: str, variant: str,
               tps: List[int], arch: str, dtype_str: str,
               max_kv: int,
               tokens_grid: Sequence[int], sequences_grid: Sequence[int],
               prefill_grid: Sequence[int], kv_prefill_grid: Sequence[int],
               decode_n_grid: Sequence[int], kv_decode_grid: Sequence[int],
               max_num_batched_tokens: int, max_num_seqs: int) -> None:
    import yaml

    def _spec(values: Sequence[int]) -> str:
        return ", ".join(str(v) for v in values)

    meta = {
        "profiler_version": "neuron-eager-v1",
        "vllm_version": "n/a",
        "gpu": hardware,
        "hardware": hardware,
        "profiled_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "architecture": arch,
        "model": model,
        "variant": variant,
        "tp_degrees": tps,
        "engine_effective": {
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_num_seqs": max_num_seqs,
            "dtype": dtype_str,
            "kv_cache_dtype": "auto",
            "block_size": 16,
            "num_hidden_layers_profiled": 1,
        },
        "attention_grid": {
            "max_kv": max_kv,
            "tokens": _spec(tokens_grid),
            "sequences": _spec(sequences_grid),
            "prefill_chunk": _spec(prefill_grid),
            "kv_prefill": _spec(kv_prefill_grid),
            "n_decode": _spec(decode_n_grid),
            "kv_decode": _spec(kv_decode_grid),
        },
        "skew_fit": {
            "per_tp": {
                tp: {"method": "synthetic-constant", "alpha_default": 0.3}
                for tp in tps
            }
        },
        "calibration": {
            "scaling_factor": 1.0,
            "scaled_by": "raw eager profiler output (no NxDI calibration applied yet)",
        },
    }
    (out_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))


# ======================================================================
# CLI
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--model", required=True,
                   help="HF id, e.g. meta-llama/Llama-3.2-1B")
    p.add_argument("--tp", required=True,
                   help="Comma list of TP degrees (e.g. 1,2,4,8)")
    p.add_argument("--output-root", required=True,
                   help="Base perf dir (writes <root>/<HARDWARE>/<MODEL>/<variant>/...)")
    p.add_argument("--hardware", default="Inferentia2")
    p.add_argument("--variant", default="",
                   help="Variant name. Default: bf16/fp16/fp32 by --dtype.")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--hf-token", default=os.getenv("HF_TOKEN", ""),
                   help="HF token for gated models. Defaults to $HF_TOKEN.")

    # Sweep grids (lean defaults)
    p.add_argument("--tokens-grid", default="1,16,64,256,1024,2048")
    p.add_argument("--sequences-grid", default="1,8,32,128")
    p.add_argument("--prefill-grid", default="16,64,256,1024,2048")
    p.add_argument("--kv-prefill-grid", default="0,1024,4096,8192")
    p.add_argument("--decode-n-grid", default="1,4,16,64")
    p.add_argument("--kv-decode-grid", default="64,256,1024,4096,8192,16384")

    # Engine knobs (recorded in meta.yaml)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--max-num-seqs", type=int, default=128)
    p.add_argument("--max-position-embeddings", type=int, default=16384)

    # Measurement
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=30)

    p.add_argument("--skip-attention", action="store_true",
                   help="Write empty attention.csv (debug; speeds up first run)")
    return p.parse_args()


def main():
    args = parse_args()
    tps = _csv_ints(args.tp)
    tokens_grid = _csv_ints(args.tokens_grid)
    sequences_grid = _csv_ints(args.sequences_grid)
    prefill_grid = _csv_ints(args.prefill_grid)
    kv_prefill_grid = _csv_ints(args.kv_prefill_grid)
    decode_n_grid = _csv_ints(args.decode_n_grid)
    kv_decode_grid = _csv_ints(args.kv_decode_grid)
    variant = args.variant or {"bfloat16": "bf16",
                               "float16": "fp16",
                               "float32": "fp32"}[args.dtype]

    # Read architecture once (without sharding) to dispatch.
    rt = _lazy_import_runtime()
    cfg0 = rt["AutoConfig"].from_pretrained(args.model,
                                            token=args.hf_token or None)
    arch = cfg0.model_type
    if arch not in ARCH_DESC:
        print(f"[!] Architecture {arch!r} not supported; "
              f"add it to ARCH_DESC at the top of this script.")
        sys.exit(2)

    out_root = Path(args.output_root) / args.hardware / args.model / variant
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[*] Output: {out_root}")
    print(f"[*] Architecture: {arch}, base dtype={args.dtype}, "
          f"hidden={cfg0.hidden_size}, layers={cfg0.num_hidden_layers}, "
          f"heads={cfg0.num_attention_heads}, kv_heads={cfg0.num_key_value_heads}, "
          f"inter={cfg0.intermediate_size}")

    for tp in tps:
        print(f"\n========== TP={tp} ==========")
        out_tp = out_root / f"tp{tp}"
        out_tp.mkdir(exist_ok=True)
        try:
            model, cfg, device, dtype = load_model(
                args.model, tp, args.dtype,
                args.max_position_embeddings, args.hf_token)
        except Exception as e:
            traceback.print_exc()
            print(f"[!] tp{tp} model load failed: {e}")
            continue
        print(f"  loaded model with sharded dims: "
              f"heads={cfg.num_attention_heads}, kv={cfg.num_key_value_heads}, "
              f"inter={cfg.intermediate_size}")

        # --- dense ---
        print("  -- dense sweep --")
        dense_rows = sweep_dense(model, cfg, dtype, device, arch,
                                 tokens_grid, args.warmup, args.repeat)
        write_dense_csv(dense_rows, out_tp / "dense.csv")
        print(f"  [✓] dense.csv ({len(dense_rows)} rows)")

        # --- per_sequence ---
        print("  -- per_sequence sweep --")
        ps_rows = sweep_per_sequence(model, cfg, dtype, device, arch,
                                     sequences_grid, args.warmup, args.repeat)
        write_per_sequence_csv(ps_rows, out_tp / "per_sequence.csv")
        print(f"  [✓] per_sequence.csv ({len(ps_rows)} rows)")

        # --- attention ---
        if args.skip_attention:
            (out_tp / "attention.csv").write_text(
                "prefill_chunk,kv_prefill,n_decode,kv_decode,time_us\n")
            print("  [-] attention.csv (skipped via --skip-attention)")
        else:
            print("  -- attention sweep --")
            attn_rows = sweep_attention(model, cfg, dtype, device, arch,
                                        prefill_grid, kv_prefill_grid,
                                        decode_n_grid, kv_decode_grid,
                                        dense_rows, args.warmup, args.repeat)
            write_attention_csv(attn_rows, out_tp / "attention.csv")
            print(f"  [✓] attention.csv ({len(attn_rows)} rows)")

        free_model(model)

    write_meta(out_root, args.hardware, args.model, variant, tps,
               arch, args.dtype, args.max_position_embeddings,
               tokens_grid, sequences_grid,
               prefill_grid, kv_prefill_grid,
               decode_n_grid, kv_decode_grid,
               args.max_num_batched_tokens, args.max_num_seqs)
    print(f"\n[✓] meta.yaml written")
    print(f"[✓] Done. Variant root: {out_root}")
    print()
    print("Next step: calibrate against NxDI e2e measurements:")
    print(f"    python scripts/calibrate_with_nxdi.py "
          f"--model {args.model} --tp {','.join(map(str, tps))} "
          f"--variant-root {out_root}")


if __name__ == "__main__":
    main()
