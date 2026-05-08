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
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_neuronx  # noqa: F401  — wires Neuron into torch_xla
    from transformers import AutoConfig, AutoModelForCausalLM
    return {
        "torch": torch,
        "torch_xla": torch_xla,
        "xm": xm,
        "AutoConfig": AutoConfig,
        "AutoModelForCausalLM": AutoModelForCausalLM,
    }


def _xla_device(rt):
    """Return the current XLA (NeuronCore) device, preferring the new
    torch_xla.device() over the deprecated xm.xla_device()."""
    txla = rt["torch_xla"]
    if hasattr(txla, "device"):
        try:
            return txla.device()
        except TypeError:
            pass
    return rt["xm"].xla_device()


def _get_sync(rt):
    """Return a zero-arg callable that flushes pending device ops and
    waits for completion. Prefers torch_xla.sync() (torch_xla >= 2.4)
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
    """Instantiate model from config, preferring ``dtype=`` kwarg
    (transformers >= 4.50) over the deprecated ``torch_dtype=``."""
    try:
        return AutoModelForCausalLM.from_config(cfg, dtype=dtype).eval()
    except TypeError:
        return AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype).eval()


def _set_cfg_dtype(cfg, dtype):
    """Set the config's dtype using the non-deprecated attribute when
    available. Newer transformers expose ``cfg.dtype``; older ones use
    ``cfg.torch_dtype``."""
    if hasattr(cfg, "dtype"):
        try:
            cfg.dtype = dtype
            return
        except Exception:
            pass
    cfg.torch_dtype = dtype


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
def time_callable(fn: Callable[[], Any], warmup: int, repeat: int
                  ) -> Tuple[float, Dict[str, Any]]:
    """Time fn() on Neuron via XLA mark_step + wait.

    Returns ``(median_microseconds, meta)`` where ``meta`` has:

      * ``first_call_us``  — wall time of the *first* warmup iteration.
        On Neuron this is dominated by the first-time NEFF compile for
        this shape (or near-zero if cache hit).
      * ``median_warmup_us`` — median of warmup[1:] (post-compile),
        useful as a sanity check vs ``median_us``.
      * ``median_us``       — median of the timed phase (the value
        used as the official measurement).
      * ``n_warmup`` / ``n_timed``
      * ``wall_us``         — total wall time invested in this shot
        (warmup + timed). Sums ≈ total profiling cost.
      * ``compile_us``      — first_call_us - median_us (best-effort
        compile cost estimate; clamped to >= 0).

    fn() must wrap whatever forward call we want to measure. Inputs
    should already live on the Neuron device.
    """
    rt = _lazy_import_runtime()
    sync = _get_sync(rt)

    warmup_samples: List[float] = []
    timed_samples: List[float] = []

    # Warmup. Mirror the timed phase exactly: each fn() call is bracketed
    # by sync() (= mark_step + wait_device_ops) so the compiled graph in
    # the cache matches what the timed phase will look up. Without per-call
    # sync, the warmup would build one big N-fn-calls graph and the timed
    # phase's single-fn-call graph would be a fresh cache miss.
    for _ in range(warmup):
        sync()
        t0 = time.perf_counter_ns()
        out = fn()
        sync()
        t1 = time.perf_counter_ns()
        warmup_samples.append((t1 - t0) / 1000.0)
        del out

    for _ in range(repeat):
        sync()
        t0 = time.perf_counter_ns()
        out = fn()
        sync()
        t1 = time.perf_counter_ns()
        timed_samples.append((t1 - t0) / 1000.0)  # ns -> us
        del out

    median_us = statistics.median(timed_samples)
    first_us = warmup_samples[0] if warmup_samples else median_us
    median_warmup_us = (statistics.median(warmup_samples[1:])
                        if len(warmup_samples) >= 2 else first_us)
    wall_us = sum(warmup_samples) + sum(timed_samples)
    compile_us = max(first_us - median_us, 0.0)

    meta = {
        "first_call_us":   first_us,
        "median_warmup_us": median_warmup_us,
        "median_us":       median_us,
        "n_warmup":        warmup,
        "n_timed":         repeat,
        "wall_us":         wall_us,
        "compile_us":      compile_us,
    }
    return median_us, meta


# ======================================================================
# Model loading: NUM_LAYERS=1 + TP shard emulation
# ======================================================================
def shard_config(cfg, tp: int):
    """Mutate cfg in-place to emulate TP=tp by dividing shardable dims.

    Also pins ``cfg.head_dim`` to its pre-shard value so post-shard
    code can read it without going through the (now wrong)
    ``hidden_size // num_attention_heads`` shortcut.
    """
    nh = cfg.num_attention_heads
    nkv = getattr(cfg, "num_key_value_heads", nh)
    # Materialize head_dim before we touch num_attention_heads — otherwise
    # `H // num_attention_heads` after sharding overestimates by a factor of tp.
    if not getattr(cfg, "head_dim", None):
        cfg.head_dim = cfg.hidden_size // nh
    if tp == 1:
        return cfg
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
    AutoConfig = rt["AutoConfig"]
    AutoModelForCausalLM = rt["AutoModelForCausalLM"]
    sync = _get_sync(rt)

    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[dtype_str]

    cfg = AutoConfig.from_pretrained(model_id, token=hf_token or None)
    cfg.num_hidden_layers = 1   # the trick
    cfg.max_position_embeddings = min(
        getattr(cfg, "max_position_embeddings", max_pos), max_pos)
    cfg = shard_config(cfg, tp)
    _set_cfg_dtype(cfg, dtype)

    # Random weights are fine: we time arithmetic, not correctness.
    model = _model_from_config(AutoModelForCausalLM, cfg, dtype)
    device = _xla_device(rt)
    model = model.to(device)
    sync()
    return model, cfg, device, dtype


def free_model(model):
    rt = _lazy_import_runtime()
    sync = _get_sync(rt)
    del model
    gc.collect()
    try:
        sync()
    except Exception:
        pass


class _RuntimeState:
    """Mutable holder for (model, cfg, dtype, device) so sweep functions
    can replace them on-the-fly via reload() without breaking caller refs.

    Periodic reload is needed because Neuron's runtime keeps every
    compiled NEFF resident in NeuronCore HBM until the model object is
    freed. With ~hundreds of unique input shapes during a sweep, HBM
    fills up (~5-10 GB per 100 shapes) and subsequent compiles fail with
        Failed to allocate ... (alignment: ..., usage: model constants)
    Free + reload drops the resident NEFFs but keeps the disk-side
    compile cache, so re-encountering a shape is ~immediate.
    """

    def __init__(self, model, cfg, dtype, device,
                 reload_fn=None, reload_every=0):
        self.model = model
        self.cfg = cfg
        self.dtype = dtype
        self.device = device
        self.reload_fn = reload_fn
        self.reload_every = reload_every
        self.shots = 0
        self.reload_count = 0
        # Reload wall-time accounting: total across the run, plus a
        # per-stage marker that callers drain via take_reload_sec().
        self.reload_sec_total = 0.0
        self._reload_sec_marker = 0.0

    def tick(self) -> bool:
        """Increment shot counter; reload if threshold reached. Returns
        True when a reload happened so callers can re-fetch module
        references against the new model."""
        self.shots += 1
        if (self.reload_every > 0
                and self.shots >= self.reload_every
                and self.reload_fn is not None):
            t0 = time.perf_counter()
            print(f"    [reload] freeing model after {self.shots} shots "
                  f"to reset Neuron HBM...")
            free_model(self.model)
            self.model, self.cfg, self.device, self.dtype = self.reload_fn()
            self.shots = 0
            self.reload_count += 1
            elapsed = time.perf_counter() - t0
            self.reload_sec_total += elapsed
            self._reload_sec_marker += elapsed
            print(f"    [reload] reloaded in {elapsed:.1f}s")
            return True
        return False

    def take_reload_sec(self) -> float:
        """Drain reload-time accumulated since the last call. Use after
        a sweep stage to attribute reload cost to that stage."""
        s = self._reload_sec_marker
        self._reload_sec_marker = 0.0
        return s


# ======================================================================
# Sweeps
# ======================================================================
def sweep_dense(state: "_RuntimeState", arch: str,
                tokens_grid: Sequence[int], warmup: int, repeat: int
                ) -> Tuple[List[Tuple[str, int, float]], List[Dict[str, Any]]]:
    """Time each catalog dense layer at each token count.

    Uses ``state`` (mutable holder of model/cfg/dtype/device) so periodic
    reloads can free Neuron HBM mid-sweep without breaking caller refs.

    Returns (rows, shot_timings).
      rows: list of (layer_name, tokens, time_us)  ← simulator-facing
      shot_timings: list of per-shot timing dicts  ← profile_timing.json
    """
    rows: List[Tuple[str, int, float]] = []
    shot_timings: List[Dict[str, Any]] = []
    for layer_name, paths, kind in ARCH_DESC[arch]["dense_layers"]:
        modules = [get_module(state.model, p) for p in paths]
        for n in tokens_grid:
            x = build_dummy_input(kind, n, state.cfg, state.dtype, state.device)

            def call(mods=modules, x=x):
                # Sum the outputs to keep them live until mark_step.
                outs = [m(x) for m in mods]
                return outs

            try:
                t_us, meta = time_callable(call, warmup, repeat)
            except Exception as e:
                print(f"    [WARN] dense {layer_name} n={n} failed: {e}")
                continue
            rows.append((layer_name, n, t_us))
            shot_timings.append({
                "category": "dense",
                "layer": layer_name,
                "key": {"tokens": n},
                **meta,
            })
            print(f"    dense  {layer_name:18s} tokens={n:5d}  -> {t_us:9.3f} us  "
                  f"(compile~{meta['compile_us']/1000:6.1f}ms, wall {meta['wall_us']/1e6:6.2f}s)")

            # Periodic reload to release accumulated Neuron HBM. After
            # reload, model identity changes — re-fetch module references
            # against the fresh model.
            if state.tick():
                modules = [get_module(state.model, p) for p in paths]

    # Synthesize rotary_emb and act_fn (not directly times-able as standalone
    # modules without HF-version-specific glue).
    for n in tokens_grid:
        rows.append(("rotary_emb", n, ROTARY_BASE_US + ROTARY_PER_TOK_US * n))
        rows.append(("act_fn",     n, ACT_FN_BASE_US + ACT_FN_PER_TOK_US * n))

    rows.sort(key=lambda r: (r[0], r[1]))
    return rows, shot_timings


def sweep_per_sequence(state: "_RuntimeState", arch: str,
                       sequences_grid: Sequence[int], warmup: int, repeat: int
                       ) -> Tuple[List[Tuple[str, int, float]], List[Dict[str, Any]]]:
    """Time lm_head at each sequence count; synthesize sampler row."""
    rows: List[Tuple[str, int, float]] = []
    shot_timings: List[Dict[str, Any]] = []
    for layer_name, paths, kind in ARCH_DESC[arch]["per_seq_layers"]:
        modules = [get_module(state.model, p) for p in paths]
        for s in sequences_grid:
            x = build_dummy_input(kind, n=1, cfg=state.cfg, dtype=state.dtype,
                                  device=state.device, batch=s)

            def call(mods=modules, x=x):
                return [m(x) for m in mods]

            try:
                t_us, meta = time_callable(call, warmup, repeat)
            except Exception as e:
                print(f"    [WARN] per_seq {layer_name} s={s} failed: {e}")
                continue
            rows.append((layer_name, s, t_us))
            shot_timings.append({
                "category": "per_sequence",
                "layer": layer_name,
                "key": {"sequences": s},
                **meta,
            })
            print(f"    per_s  {layer_name:18s} seqs={s:5d}    -> {t_us:9.3f} us  "
                  f"(compile~{meta['compile_us']/1000:6.1f}ms, wall {meta['wall_us']/1e6:6.2f}s)")

            if state.tick():
                modules = [get_module(state.model, p) for p in paths]

    for s in sequences_grid:
        rows.append(("sampler", s, SAMPLER_BASE_US + SAMPLER_PER_SEQ_US * s))

    rows.sort(key=lambda r: (r[0], r[1]))
    return rows, shot_timings


# Attention sweep: time `F.scaled_dot_product_attention` directly with
# synthesised (q, k, v) tensors of the right shapes for each (pc, kv_p,
# n, kv_d) combination. Bypasses HF's self_attn.forward + DynamicCache
# entirely — Neuron's static-shape compiler can't capture pre-populated
# DynamicCache state across calls (cache.update's torch.cat is traced
# inside but the past tensors aren't visible as graph inputs, leading
# to the V/K being treated as length-1 in the compiled graph). The
# simulator's attention.csv expects kernel-only timing anyway (qkv_proj
# / o_proj are timed separately and live in dense.csv).
def sweep_attention(state: "_RuntimeState", arch: str,
                    prefill_grid: Sequence[int],
                    kv_prefill_grid: Sequence[int],
                    decode_n_grid: Sequence[int],
                    kv_decode_grid: Sequence[int],
                    dense_rows: List[Tuple[str, int, float]],
                    warmup: int, repeat: int,
                    ) -> Tuple[List[Tuple[int, int, int, int, float]], List[Dict[str, Any]]]:
    """SDPA kernel sweep over (prefill_chunk, kv_prefill, n_decode, kv_decode).

    Returns (rows, shot_timings).
      rows: (prefill_chunk, kv_prefill, n_decode, kv_decode, time_us) — pure
            scaled_dot_product_attention kernel time, ready for the simulator's
            attention.csv lookup.
      shot_timings: per-shot timing dicts for profile_timing.json.
    """
    del dense_rows  # no longer used; kernel time is measured directly

    rt = _lazy_import_runtime()
    torch = rt["torch"]
    F = torch.nn.functional

    # Cached config dims — these are stable across reloads (same TP
    # shard, same dtype) so we can capture once.
    nh = state.cfg.num_attention_heads
    nkv = state.cfg.num_key_value_heads
    head_dim = state.cfg.head_dim   # set explicitly by shard_config()
    n_rep = max(nh // max(nkv, 1), 1)

    def _build_qkv(B: int, q_len: int, k_len: int):
        q = torch.zeros((B, nh, q_len, head_dim),
                        dtype=state.dtype, device=state.device)
        k = torch.zeros((B, nkv, k_len, head_dim),
                        dtype=state.dtype, device=state.device)
        v = torch.zeros((B, nkv, k_len, head_dim),
                        dtype=state.dtype, device=state.device)
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        return q, k, v

    def _build_prefill_mask(pc: int, kv_p: int):
        if pc + kv_p == 0:
            return None
        mask = torch.zeros((pc, kv_p + pc),
                           dtype=state.dtype, device=state.device)
        idx_q = torch.arange(pc, device=state.device).unsqueeze(1)
        idx_k = torch.arange(kv_p + pc, device=state.device).unsqueeze(0)
        blocked = idx_k > (kv_p + idx_q)
        mask = mask.masked_fill(blocked, float("-inf"))
        return mask

    rows: List[Tuple[int, int, int, int, float]] = []
    shot_timings: List[Dict[str, Any]] = []

    # ---- Pure prefill (pc, kv_p, 0, 0) ----
    for pc in prefill_grid:
        for kv_p in kv_prefill_grid:
            if pc + kv_p > state.cfg.max_position_embeddings:
                continue
            q, k, v = _build_qkv(B=1, q_len=pc, k_len=pc + kv_p)
            attn_mask = _build_prefill_mask(pc, kv_p)

            def call(q=q, k=k, v=v, m=attn_mask):
                return F.scaled_dot_product_attention(q, k, v, attn_mask=m,
                                                      is_causal=False)

            try:
                t_kernel, meta = time_callable(call, warmup, repeat)
            except Exception as e:
                print(f"    [WARN] attn prefill pc={pc} kv_p={kv_p} failed: {e}")
                continue
            rows.append((pc, kv_p, 0, 0, t_kernel))
            shot_timings.append({
                "category": "attention",
                "regime": "prefill",
                "key": {"prefill_chunk": pc, "kv_prefill": kv_p,
                        "n_decode": 0, "kv_decode": 0},
                "t_kernel_us": t_kernel,
                **meta,
            })
            print(f"    attn   prefill pc={pc:5d} kv_p={kv_p:5d}                 "
                  f"-> {t_kernel:9.3f} us  "
                  f"(compile~{meta['compile_us']/1000:6.1f}ms, wall {meta['wall_us']/1e6:6.2f}s)")
            state.tick()  # state.cfg/device still valid after reload

    # ---- Pure decode (0, 0, n, kv_d) ----
    for n in decode_n_grid:
        for kv_d in kv_decode_grid:
            if kv_d + 1 > state.cfg.max_position_embeddings:
                continue
            q, k, v = _build_qkv(B=n, q_len=1, k_len=kv_d + 1)

            def call(q=q, k=k, v=v):
                return F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                      is_causal=False)

            try:
                t_kernel, meta = time_callable(call, warmup, repeat)
            except Exception as e:
                print(f"    [WARN] attn decode n={n} kv_d={kv_d} failed: {e}")
                continue
            rows.append((0, 0, n, kv_d, t_kernel))
            shot_timings.append({
                "category": "attention",
                "regime": "decode",
                "key": {"prefill_chunk": 0, "kv_prefill": 0,
                        "n_decode": n, "kv_decode": kv_d},
                "t_kernel_us": t_kernel,
                **meta,
            })
            print(f"    attn   decode                       n={n:4d} kv_d={kv_d:5d}"
                  f" -> {t_kernel:9.3f} us  "
                  f"(compile~{meta['compile_us']/1000:6.1f}ms, wall {meta['wall_us']/1e6:6.2f}s)")
            state.tick()

    rows.sort()
    return rows, shot_timings


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
    p.add_argument("--reload-every", type=int, default=30,
                   help="Free + re-load the model after every N timed shots "
                        "to release accumulated Neuron NEFFs from HBM. "
                        "0 = never reload (will OOM on large sweeps).")

    p.add_argument("--skip-attention", action="store_true",
                   help="Write empty attention.csv (debug; speeds up first run)")
    p.add_argument("--run-tag", default="",
                   help="Suffix for profile_timing.json (e.g. 'cold', 'hot'). "
                        "If set, writes profile_timing_<tag>.json — useful when "
                        "running the same sweep twice (cold cache vs hot cache) "
                        "and wanting to keep both cost breakdowns side by side.")
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

    # Profile-time accounting (saved as profile_timing.json at end)
    timing_run = {
        "schema": "profile_timing-v1",
        "model": args.model, "hardware": args.hardware, "variant": variant,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "machine": _capture_machine_info(),
        "args": _capture_runtime_args(args),
        "tp_stages": {},   # tp → {load_sec, dense_sec, per_seq_sec, attn_sec, write_sec, total_sec, shots: [...]}
    }
    run_t0 = time.perf_counter()

    def _stage_split(shots):
        """Split shot wall-time into compile (cold-cache first call) vs.
        measure (post-compile forward calls)."""
        comp = sum(s.get("compile_us", 0.0) for s in shots) / 1e6
        wall = sum(s.get("wall_us",    0.0) for s in shots) / 1e6
        return comp, max(wall - comp, 0.0)

    for tp in tps:
        print(f"\n========== TP={tp} ==========")
        out_tp = out_root / f"tp{tp}"
        out_tp.mkdir(exist_ok=True)
        stage_t = {"load_sec": 0.0,
                   "dense_sec": 0.0, "dense_compile_sec": 0.0,
                   "dense_measure_sec": 0.0, "dense_reload_sec": 0.0,
                   "per_seq_sec": 0.0, "per_seq_compile_sec": 0.0,
                   "per_seq_measure_sec": 0.0, "per_seq_reload_sec": 0.0,
                   "attn_sec": 0.0, "attn_compile_sec": 0.0,
                   "attn_measure_sec": 0.0, "attn_reload_sec": 0.0,
                   "write_sec": 0.0, "total_sec": 0.0,
                   "shots": []}
        tp_t0 = time.perf_counter()

        load_t0 = time.perf_counter()
        try:
            model, cfg, device, dtype = load_model(
                args.model, tp, args.dtype,
                args.max_position_embeddings, args.hf_token)
        except Exception as e:
            traceback.print_exc()
            print(f"[!] tp{tp} model load failed: {e}")
            continue
        stage_t["load_sec"] = time.perf_counter() - load_t0
        print(f"  loaded model in {stage_t['load_sec']:.1f}s with sharded dims: "
              f"heads={cfg.num_attention_heads}, kv={cfg.num_key_value_heads}, "
              f"inter={cfg.intermediate_size}")

        # Build a closure that recreates the model with the same args, so
        # _RuntimeState can periodically reload to release Neuron HBM.
        def _reload_for_tp(_tp=tp):
            return load_model(args.model, _tp, args.dtype,
                              args.max_position_embeddings, args.hf_token)

        state = _RuntimeState(model, cfg, dtype, device,
                              reload_fn=_reload_for_tp,
                              reload_every=args.reload_every)

        # --- dense ---
        print("  -- dense sweep --")
        sweep_t0 = time.perf_counter()
        dense_rows, dense_shots = sweep_dense(state, arch,
                                              tokens_grid, args.warmup, args.repeat)
        stage_t["dense_sec"] = time.perf_counter() - sweep_t0
        stage_t["dense_reload_sec"] = state.take_reload_sec()
        stage_t["dense_compile_sec"], stage_t["dense_measure_sec"] = _stage_split(dense_shots)
        write_t0 = time.perf_counter()
        write_dense_csv(dense_rows, out_tp / "dense.csv")
        stage_t["write_sec"] += time.perf_counter() - write_t0
        stage_t["shots"].extend(dense_shots)
        print(f"  [✓] dense.csv ({len(dense_rows)} rows; wall {stage_t['dense_sec']:.1f}s; "
              f"compile {stage_t['dense_compile_sec']:.1f}s, "
              f"measure {stage_t['dense_measure_sec']:.1f}s, "
              f"reload {stage_t['dense_reload_sec']:.1f}s)")

        # --- per_sequence ---
        print("  -- per_sequence sweep --")
        sweep_t0 = time.perf_counter()
        ps_rows, ps_shots = sweep_per_sequence(state, arch,
                                               sequences_grid, args.warmup, args.repeat)
        stage_t["per_seq_sec"] = time.perf_counter() - sweep_t0
        stage_t["per_seq_reload_sec"] = state.take_reload_sec()
        stage_t["per_seq_compile_sec"], stage_t["per_seq_measure_sec"] = _stage_split(ps_shots)
        write_t0 = time.perf_counter()
        write_per_sequence_csv(ps_rows, out_tp / "per_sequence.csv")
        stage_t["write_sec"] += time.perf_counter() - write_t0
        stage_t["shots"].extend(ps_shots)
        print(f"  [✓] per_sequence.csv ({len(ps_rows)} rows; wall {stage_t['per_seq_sec']:.1f}s; "
              f"compile {stage_t['per_seq_compile_sec']:.1f}s, "
              f"measure {stage_t['per_seq_measure_sec']:.1f}s, "
              f"reload {stage_t['per_seq_reload_sec']:.1f}s)")

        # --- attention ---
        if args.skip_attention:
            (out_tp / "attention.csv").write_text(
                "prefill_chunk,kv_prefill,n_decode,kv_decode,time_us\n")
            print("  [-] attention.csv (skipped via --skip-attention)")
        else:
            print("  -- attention sweep --")
            sweep_t0 = time.perf_counter()
            attn_rows, attn_shots = sweep_attention(
                state, arch,
                prefill_grid, kv_prefill_grid,
                decode_n_grid, kv_decode_grid,
                dense_rows, args.warmup, args.repeat)
            stage_t["attn_sec"] = time.perf_counter() - sweep_t0
            stage_t["attn_reload_sec"] = state.take_reload_sec()
            stage_t["attn_compile_sec"], stage_t["attn_measure_sec"] = _stage_split(attn_shots)
            write_t0 = time.perf_counter()
            write_attention_csv(attn_rows, out_tp / "attention.csv")
            stage_t["write_sec"] += time.perf_counter() - write_t0
            stage_t["shots"].extend(attn_shots)
            print(f"  [✓] attention.csv ({len(attn_rows)} rows; wall {stage_t['attn_sec']:.1f}s; "
                  f"compile {stage_t['attn_compile_sec']:.1f}s, "
                  f"measure {stage_t['attn_measure_sec']:.1f}s, "
                  f"reload {stage_t['attn_reload_sec']:.1f}s)")

        stage_t["reload_count"] = state.reload_count
        stage_t["reload_sec_total"] = state.reload_sec_total
        free_model(state.model)
        stage_t["total_sec"] = time.perf_counter() - tp_t0

        # Per-stage breakdown. Each *_sec is the stage's outer wall time;
        # compile + measure + reload are extracted from per-shot timings;
        # whatever's left is loop overhead (input build, dispatch, sync,
        # prints).
        for stage in ("dense", "per_seq", "attn"):
            stage_wall = stage_t[f"{stage}_sec"]
            tracked = (stage_t[f"{stage}_compile_sec"]
                       + stage_t[f"{stage}_measure_sec"]
                       + stage_t[f"{stage}_reload_sec"])
            stage_t[f"{stage}_loop_overhead_sec"] = max(stage_wall - tracked, 0.0)

        compile_total = (stage_t["dense_compile_sec"]
                         + stage_t["per_seq_compile_sec"]
                         + stage_t["attn_compile_sec"])
        measure_total = (stage_t["dense_measure_sec"]
                         + stage_t["per_seq_measure_sec"]
                         + stage_t["attn_measure_sec"])
        reload_total = stage_t["reload_sec_total"]
        loop_total = (stage_t["dense_loop_overhead_sec"]
                      + stage_t["per_seq_loop_overhead_sec"]
                      + stage_t["attn_loop_overhead_sec"])

        # Top-level reconciliation: total = load + (compile + measure +
        # reload + loop_overhead across stages) + write + unaccounted.
        accounted = (stage_t["load_sec"]
                     + compile_total + measure_total
                     + reload_total + loop_total
                     + stage_t["write_sec"])
        unaccounted = max(stage_t["total_sec"] - accounted, 0.0)
        stage_t["unaccounted_sec"] = unaccounted

        timing_run["tp_stages"][str(tp)] = stage_t

        T = stage_t["total_sec"]
        def pct(x): return f"({100*x/T:5.1f}%)" if T > 0 else "(  -  )"
        print(f"\n  [⏱] tp{tp} breakdown — total {T:.1f}s")
        print(f"        initial load   : {stage_t['load_sec']:8.1f}s   {pct(stage_t['load_sec'])}")
        print(f"        compile        : {compile_total:8.1f}s   {pct(compile_total)}")
        print(f"        measure        : {measure_total:8.1f}s   {pct(measure_total)}")
        print(f"        reload         : {reload_total:8.1f}s   {pct(reload_total)}   "
              f"({stage_t['reload_count']} reloads)")
        print(f"        loop overhead  : {loop_total:8.1f}s   {pct(loop_total)}   "
              f"(input build, dispatch sync, prints)")
        print(f"        write CSV      : {stage_t['write_sec']:8.1f}s   {pct(stage_t['write_sec'])}")
        print(f"        unaccounted    : {unaccounted:8.1f}s   {pct(unaccounted)}   "
              f"(setup, free_model)")

    write_meta(out_root, args.hardware, args.model, variant, tps,
               arch, args.dtype, args.max_position_embeddings,
               tokens_grid, sequences_grid,
               prefill_grid, kv_prefill_grid,
               decode_n_grid, kv_decode_grid,
               args.max_num_batched_tokens, args.max_num_seqs)
    print(f"\n[✓] meta.yaml written")

    # Save timing artifact (re-loadable for comparison via show_profile_timing.py)
    timing_run["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    timing_run["wall_clock_total_sec"] = time.perf_counter() - run_t0
    timing_run["run_tag"] = args.run_tag or None
    timing_filename = (f"profile_timing_{args.run_tag}.json"
                       if args.run_tag else "profile_timing.json")
    timing_path = out_root / timing_filename
    import json as _json
    timing_path.write_text(_json.dumps(timing_run, indent=2, default=str))
    print(f"[✓] {timing_filename} written ({timing_path}, "
          f"total {timing_run['wall_clock_total_sec']/60:.1f} min)")

    print(f"[✓] Done. Variant root: {out_root}")


def _capture_machine_info() -> Dict[str, Any]:
    """Best-effort: capture which machine / SDK we're running on."""
    info: Dict[str, Any] = {}
    info["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    try:
        import platform
        info["python"] = platform.python_version()
        info["platform"] = platform.platform()
    except Exception:
        pass
    # Module versions
    for mod_name in ("torch", "torch_xla", "torch_neuronx", "transformers"):
        try:
            mod = __import__(mod_name)
            info[mod_name] = getattr(mod, "__version__", "unknown")
        except Exception:
            info[mod_name] = "not-installed"
    # Inferentia 2 instance type
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


def _capture_runtime_args(args) -> Dict[str, Any]:
    """Subset of CLI args worth recording for reproducibility / comparison."""
    return {
        "tp": args.tp, "dtype": args.dtype,
        "tokens_grid": args.tokens_grid,
        "sequences_grid": args.sequences_grid,
        "prefill_grid": args.prefill_grid,
        "kv_prefill_grid": args.kv_prefill_grid,
        "decode_n_grid": args.decode_n_grid,
        "kv_decode_grid": args.kv_decode_grid,
        "warmup": args.warmup, "repeat": args.repeat,
        "max_position_embeddings": args.max_position_embeddings,
        "skip_attention": args.skip_attention,
    }


if __name__ == "__main__":
    main()
