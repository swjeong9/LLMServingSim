"""JAX TPU profiling core — shared between profile_jax.py (subprocess entry)
and profile_full_jax.ipynb (sanity / interactive).

All measurement primitives, kernel builders, grid definitions, and input
binders live here so there is one source of truth.

Paper SHARD_FIELDS sharding for TP profiling (`resolve_mcfg`) follows
scripts/prompt_tp_methodology_for_tpu_agent.md — only `num_attention_heads`,
`num_key_value_heads`, `intermediate_size`, `vocab_size` get divided by TP.
`hidden_size` and `head_dim` stay raw. Layers marked tp_stable in
llama.yaml (layernorm, final_layernorm, sampler) use raw V via `V_raw`.
"""

import os
import sys
import gzip
import json
import glob
import shutil
import tempfile
import statistics
import time
from contextlib import contextmanager
from itertools import product
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.nn import dot_product_attention as jax_sdpa


# tp_stable layers (from profiler/models/llama.yaml). The simulator's writer
# normally profiles these once at tp=1 and replicates into tp{N}/ folders.
# Here we just measure them at every TP — values come out identical (same
# input shape) but keeps the orchestrator stupid-simple.
TP_STABLE_LAYERS = {'layernorm', 'final_layernorm', 'sampler'}


# ===== Tensor init =====
def randn(shape, dtype=jnp.bfloat16, seed=0):
    arr = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    return jnp.asarray(arr, dtype=dtype)


def randint(shape, low, high, seed=0):
    arr = np.random.default_rng(seed).integers(low, high, size=shape).astype(np.int32)
    return jnp.asarray(arr)


# ===== stderr filter (drops the noisy TF-profiler hook line, keeps real errors) =====
@contextmanager
def _filter_stderr(drop_substrings):
    """Capture fd 2 into a temp file; on exit, replay only the lines that do
    NOT contain any of `drop_substrings` back to the original stderr. Real
    errors and Python tracebacks are preserved — only matching noise is
    dropped. Works for C++ stderr (XLA/JAX log lines) too."""
    tmp = tempfile.TemporaryFile(mode='w+b')
    saved_fd = os.dup(2)
    os.dup2(tmp.fileno(), 2)
    try:
        yield
    finally:
        try:
            os.fsync(2)
        except OSError:
            pass
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        tmp.seek(0)
        for raw in tmp:
            line = raw.decode('utf-8', errors='replace')
            if not any(p in line for p in drop_substrings):
                sys.stderr.write(line)
        tmp.close()


# ===== Device-trace event parsing =====
def _parse_device_events(trace_dir, jit_func_name):
    """Read chrome trace JSON from <trace_dir>/plugins/profile/<ts>/*.trace.json.gz
    and return device-side `dur` (microseconds) values for events whose name
    starts with `jit_<jit_func_name>(`. Returns [] if none found."""
    chrome_files = sorted(glob.glob(os.path.join(trace_dir, 'plugins', 'profile', '*', '*.trace.json.gz')))
    if not chrome_files:
        return []
    with gzip.open(chrome_files[-1], 'rt') as f:
        data = json.load(f)
    events = data.get('traceEvents', data if isinstance(data, list) else [])

    tpu_pids = set()
    for e in events:
        if isinstance(e, dict) and e.get('name') == 'process_name':
            n = e.get('args', {}).get('name', '')
            if '/device:TPU' in n:
                tpu_pids.add(e.get('pid'))
    if not tpu_pids:
        return []

    prefix = f'jit_{jit_func_name}('
    return [float(e['dur']) for e in events
            if isinstance(e, dict) and e.get('pid') in tpu_pids
            and e.get('ph') == 'X' and 'dur' in e
            and e.get('name', '').startswith(prefix)]


def measure(fn, jit_name, warmup, repeat):
    """Measure device time of `fn` via JAX profiler trace.

    `jit_name` = name of the @jax.jit-decorated function (e.g. '_qkv_proj').
    Device event name is `jit__qkv_proj(...)`.

    Returns dict with mean_ns / p50_ns / p90_ns / max_ns from device `dur` (us).
    Falls back to host wallclock (with [WARN]) if no device events are found."""
    for _ in range(warmup):
        fn().block_until_ready()

    trace_dir = tempfile.mkdtemp(prefix='jax_trace_')
    drop = ("Can't import tensorflow.python.profiler.trace",)
    try:
        with _filter_stderr(drop), jax.profiler.trace(trace_dir):
            for _ in range(repeat):
                fn().block_until_ready()
        durs_us = _parse_device_events(trace_dir, jit_name)
        if not durs_us:
            print(f'  [WARN] no device events for jit_{jit_name} — host wallclock fallback')
            shots = []
            for _ in range(repeat):
                t0 = time.perf_counter_ns()
                fn().block_until_ready()
                shots.append(time.perf_counter_ns() - t0)
            shots.sort()
            return {'mean_ns': int(statistics.fmean(shots)), 'p50_ns': shots[len(shots) // 2],
                    'p90_ns': shots[int(len(shots) * 0.9)], 'max_ns': shots[-1],
                    'source': 'host_wallclock_fallback'}
        durs_ns = sorted(int(d * 1000) for d in durs_us)
        return {'mean_ns': int(statistics.fmean(durs_ns)), 'p50_ns': durs_ns[len(durs_ns) // 2],
                'p90_ns': durs_ns[int(len(durs_ns) * 0.9)], 'max_ns': durs_ns[-1],
                'source': 'jax_device_trace'}
    finally:
        shutil.rmtree(trace_dir, ignore_errors=True)


# ===== Model config resolution (paper SHARD_FIELDS) =====
def resolve_mcfg(model_name, tp):
    """Build per-rank model config matching paper's SHARD_FIELDS.

    Sharded by TP: num_attention_heads, num_key_value_heads, intermediate_size, vocab_size.
    Not sharded:    hidden_size, head_dim.
    Plus:           V_raw = unsharded vocab_size for tp_stable layers (sampler)."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_name)

    NH_raw  = cfg.num_attention_heads
    NKV_raw = getattr(cfg, 'num_key_value_heads', NH_raw)
    H       = cfg.hidden_size
    HD      = getattr(cfg, 'head_dim', H // NH_raw)
    I_raw   = cfg.intermediate_size
    V_raw   = cfg.vocab_size

    for name, val in [('num_attention_heads', NH_raw),
                      ('num_key_value_heads', NKV_raw),
                      ('intermediate_size',   I_raw),
                      ('vocab_size',          V_raw)]:
        if val % tp != 0:
            raise ValueError(f'{name}={val} not divisible by tp={tp}')

    return {
        'NH':    NH_raw  // tp,
        'NKV':   NKV_raw // tp,
        'H':     H,
        'HD':    HD,
        'I':     I_raw   // tp,
        'V':     V_raw   // tp,
        'V_raw': V_raw,
        'tp':    tp,
    }


# ===== Kernel + weight builder =====
def build_kernels(mcfg, dtype=jnp.bfloat16):
    """Random weights + jit-decorated kernels. `mcfg` already has TP-sharded
    dimensions (from `resolve_mcfg`). Returns dict name → jit function."""
    H, NH, NKV, HD, I, V = mcfg['H'], mcfg['NH'], mcfg['NKV'], mcfg['HD'], mcfg['I'], mcfg['V']
    Q = NH * HD
    K_dim = NKV * HD

    W_embed    = randn((V, H), seed=1)
    W_qkv      = randn((H, Q + 2 * K_dim), seed=2)
    W_o        = randn((Q, H), seed=3)
    W_gu       = randn((H, 2 * I), seed=4)
    W_down     = randn((I, H), seed=5)
    W_lm       = randn((H, V), seed=6)
    LN_W       = randn((H,), seed=7)
    FINAL_LN_W = randn((H,), seed=8)

    @jax.jit
    def _embedding(ids): return jnp.take(W_embed, ids, axis=0)
    @jax.jit
    def _layernorm(x):
        var = jnp.mean(x * x, axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(var + 1e-5) * LN_W
    @jax.jit
    def _final_layernorm(x):
        var = jnp.mean(x * x, axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(var + 1e-5) * FINAL_LN_W
    @jax.jit
    def _qkv_proj(x): return jnp.dot(x, W_qkv)
    @jax.jit
    def _o_proj(x): return jnp.dot(x, W_o)
    @jax.jit
    def _gate_up_proj(x): return jnp.dot(x, W_gu)
    @jax.jit
    def _act_fn(x): return jax.nn.silu(x)
    @jax.jit
    def _down_proj(x): return jnp.dot(x, W_down)
    @jax.jit
    def _lm_head(x): return jnp.dot(x, W_lm)
    @jax.jit
    def _sampler(logits): return jnp.argmax(logits, axis=-1)
    @jax.jit
    def _rotary_emb(x, cos, sin):
        half = HD // 2
        x1, x2 = x[..., :half], x[..., half:]
        rotated = jnp.concatenate([-x2, x1], axis=-1)
        return x * cos + rotated * sin
    @partial(jax.jit, static_argnames=('is_causal',))
    def _attn(q, k, v, is_causal=False):
        return jax_sdpa(q, k, v, is_causal=is_causal)

    return {
        'embedding': _embedding, 'layernorm': _layernorm,
        'final_layernorm': _final_layernorm, 'qkv_proj': _qkv_proj,
        'o_proj': _o_proj, 'gate_up_proj': _gate_up_proj, 'act_fn': _act_fn,
        'down_proj': _down_proj, 'lm_head': _lm_head, 'sampler': _sampler,
        'rotary_emb': _rotary_emb, 'attn': _attn,
    }


# ===== Grids (paper batch_sampling.py + user-adjusted stepped token grid) =====
def token_grid(max_tokens):
    """Dense / per_seq token axis. Stepped grid:
    1-16 step 1, 16-64 step 4, 64-512 step 16, 512-2048 step 32,
    2048-4096 step 64, 4096-8192 step 128."""
    g = (list(range(1,    16   + 1, 1))
         + list(range(16,   64   + 1, 4))
         + list(range(64,   512  + 1, 16))
         + list(range(512,  2048 + 1, 32))
         + list(range(2048, 4096 + 1, 64))
         + list(range(4096, 8192 + 1, 128)))
    return sorted({v for v in g if v <= max_tokens})


def seq_grid(max_tokens):
    """per_sequence sequence axis (lm_head / sampler)."""
    g = list(range(1, 33)) + list(range(32, max_tokens + 1, 32))
    return sorted({v for v in g if v <= max_tokens})


def _seq_len_grid_kv(max_kv):
    """attention decode kv axis."""
    space = (list(range(0, 1024 + 1, 32)) + list(range(1024, 4 * 1024 + 1, 64))
             + list(range(4 * 1024, 64 * 1024 + 1, 256)))
    return [s for s in space if 0 < s < max_kv]


def _prefill_chunk_grid(max_len):
    """attention prefill chunk axis (= prompt length under single-shot prefill)."""
    space = (list(range(32, 128 + 1, 32)) + list(range(128, 1024 + 1, 32))
             + list(range(1024, 4 * 1024 + 1, 64)) + list(range(4 * 1024, 16 * 1024 + 1, 128)))
    return [pc for pc in space if pc <= max_len]


def _batch_grid(min_b, max_b):
    """Power-of-two decode batches: 1, 2, 4, 8, 16, 32, ..."""
    space = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    return [b for b in space if min_b <= b <= max_b]


def attention_combos(max_prefill, max_decode_tokens, min_batch, max_batch):
    """List of (pc, kv_prefill, batch, is_prefill).

    Assumption — chunked prefill OFF + prefix caching OFF, matching
    studies/tpu_baseline/measure_vllm.py engine config:
      * prefill is single-shot: kv_prefill = 0, pc = full sequence length
      * decode: pc = 0, kv_decode = current cache size, batch varies
    The chunked_prefill_cumulative_kv axis is not swept."""
    pc_vals = _prefill_chunk_grid(max_prefill)
    kv_vals = _seq_len_grid_kv(max_decode_tokens)
    batches = _batch_grid(min_batch, max_batch)
    pre = [(pc, 0, 1, True) for pc in pc_vals]
    dec = list(product([0], kv_vals, batches, [False]))
    return pre + dec


# ===== Input binders =====
def _make_dense_input(layer, n, mcfg):
    H, NH, HD, I, V = mcfg['H'], mcfg['NH'], mcfg['HD'], mcfg['I'], mcfg['V']
    Q = NH * HD
    if layer == 'embedding':
        return (randint((1, n), 0, V, seed=200 + n),)
    if layer in ('layernorm', 'final_layernorm', 'qkv_proj', 'gate_up_proj'):
        return (randn((1, n, H), seed=200 + n),)
    if layer == 'o_proj':
        return (randn((1, n, Q), seed=200 + n),)
    if layer in ('act_fn', 'down_proj'):
        return (randn((1, n, I), seed=200 + n),)
    if layer == 'rotary_emb':
        return (randn((1, NH, n, HD), seed=200 + n),
                randn((1, 1, n, HD), seed=300 + n),
                randn((1, 1, n, HD), seed=400 + n))
    raise ValueError(f'unknown dense layer: {layer}')


def _make_per_seq_input(layer, n, mcfg):
    if layer == 'lm_head':
        return (randn((1, n, mcfg['H']), seed=500 + n),)
    if layer == 'sampler':
        # tp_stable — sampler operates on raw vocab (post all-gather)
        return (randn((1, n, mcfg['V_raw']), seed=500 + n),)
    raise ValueError(f'unknown per_seq layer: {layer}')


def _make_attn_input(pc, kv, batch, is_prefill, mcfg):
    NH, NKV, HD = mcfg['NH'], mcfg['NKV'], mcfg['HD']
    if is_prefill:
        q = randn((1, pc, NH, HD), seed=100)
        k = randn((1, pc + kv, NKV, HD), seed=101)
        v = randn((1, pc + kv, NKV, HD), seed=102)
    else:
        q = randn((batch, 1, NH, HD), seed=100)
        k = randn((batch, kv, NKV, HD), seed=101)
        v = randn((batch, kv, NKV, HD), seed=102)
    return (q, k, v, is_prefill)


# ===== Sanity check (used by ipynb cell) =====
def sanity_check(mcfg, kernels, warmup=3, repeat=10):
    """Quick measurement of every layer at a representative size — print only."""
    print('=== sanity check ===')
    print(f'  mcfg: {mcfg}')
    print('  dense layers @ tokens=128:')
    for layer in ('embedding', 'layernorm', 'qkv_proj', 'rotary_emb', 'o_proj',
                  'gate_up_proj', 'act_fn', 'down_proj', 'final_layernorm'):
        args = _make_dense_input(layer, 128, mcfg)
        fn_ = kernels[layer]
        s = measure(lambda: fn_(*args), jit_name=f'_{layer}', warmup=warmup, repeat=repeat)
        print(f'    {layer:<18} p50={s["p50_ns"]/1000:8.3f}us  src={s["source"]}')

    print('  per_sequence @ sequences=1:')
    for layer in ('lm_head', 'sampler'):
        args = _make_per_seq_input(layer, 1, mcfg)
        fn_ = kernels[layer]
        s = measure(lambda: fn_(*args), jit_name=f'_{layer}', warmup=warmup, repeat=repeat)
        print(f'    {layer:<18} p50={s["p50_ns"]/1000:8.3f}us  src={s["source"]}')

    print('  attention:')
    for label, (pc, kv, b, is_pre) in [
        ('prefill (pc=128, kv=0)',  (128, 0, 1, True)),
        ('decode  (B=32, kv=1024)', (0, 1024, 32, False)),
    ]:
        q, k, v, is_causal = _make_attn_input(pc, kv, b, is_pre, mcfg)
        fn_ = kernels['attn']
        s = measure(lambda: fn_(q, k, v, is_causal=is_causal), jit_name='_attn',
                    warmup=warmup, repeat=repeat)
        print(f'    {label:<30}  p50={s["p50_ns"]/1000:8.3f}us  src={s["source"]}')
