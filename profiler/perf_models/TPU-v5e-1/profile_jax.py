"""Single-stage JAX TPU profiler — designed to be invoked from
scripts/profile_tpu.sh in stages so each fresh Python process releases
JAX runtime + jit compilation cache between stages.

Parallels scripts/profile_inf2_v2.sh's subprocess split for Inf2 (where
Neuron Runtime HBM only releases on process exit). On TPU the issue is
host RAM accumulation from JAX's jit cache — ipynb-style monolithic
sweep hits the ceiling around stage 2.

Usage (called by profile_tpu.sh; can also be run standalone):

    cd profiler/perf_models/TPU-v5e-1
    python profile_jax.py --stage dense --layer qkv_proj \\
        --model meta-llama/Llama-3.2-1B-Instruct --tp 1 --hw TPU-v5e-1 \\
        --token-list "1,2,4,8,16,32,64,128" \\
        --output-dir ../../../profiler/perf/TPU-v5e-1/meta-llama/Llama-3.2-1B-Instruct/bf16/tp1
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# tqdm is optional — falls back to plain iteration if missing
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **kw): return it
    tqdm.write = print

from profile_jax_core import (
    resolve_mcfg, build_kernels, measure,
    _make_dense_input, _make_per_seq_input, _make_attn_input,
)


def parse_list_int(s):
    return [int(x) for x in s.split(',') if x.strip()]


def _write_v2_dense(path, rows, append):
    mode = 'a' if append else 'w'
    with path.open(mode, newline='') as f:
        w = csv.writer(f)
        if not append:
            w.writerow(['layer', 'tokens', 'time_us'])
        for layer, n, t in rows:
            w.writerow([layer, n, f'{t:.6f}'])


def _write_v2_per_seq(path, rows, append):
    mode = 'a' if append else 'w'
    with path.open(mode, newline='') as f:
        w = csv.writer(f)
        if not append:
            w.writerow(['layer', 'sequences', 'time_us'])
        for layer, n, t in rows:
            w.writerow([layer, n, f'{t:.6f}'])


def _write_attn(out_dir, v2_rows, full_rows, append):
    """Write to attention.csv (v2 schema) + attention_full_stats.csv (paper schema)."""
    v2_path = out_dir / 'attention.csv'
    full_path = out_dir / 'attention_full_stats.csv'
    mode = 'a' if append else 'w'
    with v2_path.open(mode, newline='') as f:
        w = csv.writer(f)
        if not append:
            w.writerow(['prefill_chunk', 'kv_prefill', 'n_decode', 'kv_decode', 'time_us'])
        for r in v2_rows:
            w.writerow([r[0], r[1], r[2], r[3], f'{r[4]:.6f}'])
    cols = ['prefill_chunk_size', 'kv_cache_size', 'batch_size', 'is_prefill',
            'mean_ns', 'p50_ns', 'p90_ns', 'max_ns']
    with full_path.open(mode, newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not append:
            w.writeheader()
        w.writerows(full_rows)


def run_dense(args, mcfg, kernels):
    assert args.layer, '--layer required for stage=dense'
    assert args.token_list, '--token-list required for stage=dense'
    tokens = parse_list_int(args.token_list)
    layer = args.layer
    fn_ = kernels[layer]
    rows = []
    for n in tqdm(tokens, desc=f'dense:{layer}', unit='m'):
        try:
            a = _make_dense_input(layer, n, mcfg)
            s = measure(lambda: fn_(*a), jit_name=f'_{layer}',
                        warmup=args.warmup, repeat=args.repeat)
            rows.append((layer, n, s['p50_ns'] / 1000))
        except Exception as e:
            tqdm.write(f'  [WARN] {layer} n={n} failed: {type(e).__name__}: {e}')
    _write_v2_dense(Path(args.output_dir) / 'dense.csv', rows, args.append)
    print(f'  wrote {len(rows)} rows → dense.csv  ({"append" if args.append else "fresh"})')


def run_per_seq(args, mcfg, kernels):
    assert args.layer, '--layer required for stage=per_seq'
    assert args.token_list, '--token-list required for stage=per_seq'
    seqs = parse_list_int(args.token_list)
    layer = args.layer
    fn_ = kernels[layer]
    rows = []
    for n in tqdm(seqs, desc=f'per_seq:{layer}', unit='m'):
        try:
            a = _make_per_seq_input(layer, n, mcfg)
            s = measure(lambda: fn_(*a), jit_name=f'_{layer}',
                        warmup=args.warmup, repeat=args.repeat)
            rows.append((layer, n, s['p50_ns'] / 1000))
        except Exception as e:
            tqdm.write(f'  [WARN] {layer} n={n} failed: {type(e).__name__}: {e}')
    _write_v2_per_seq(Path(args.output_dir) / 'per_sequence.csv', rows, args.append)
    print(f'  wrote {len(rows)} rows → per_sequence.csv')


def run_attn_prefill(args, mcfg, kernels):
    assert args.pc_list, '--pc-list required for stage=attn_prefill'
    pcs = parse_list_int(args.pc_list)
    fn_attn = kernels['attn']
    v2_rows, full_rows = [], []
    for pc in tqdm(pcs, desc='attn_prefill', unit='m'):
        try:
            q, k, v, is_causal = _make_attn_input(pc, 0, 1, True, mcfg)
            s = measure(lambda: fn_attn(q, k, v, is_causal=is_causal),
                        jit_name='_attn', warmup=args.warmup, repeat=args.repeat)
            v2_rows.append((pc, 0, 0, 0, s['p50_ns'] / 1000))
            full_rows.append({'prefill_chunk_size': pc, 'kv_cache_size': 0,
                              'batch_size': 1, 'is_prefill': True,
                              **{k_: s[k_] for k_ in ('mean_ns', 'p50_ns', 'p90_ns', 'max_ns')}})
        except Exception as e:
            tqdm.write(f'  [WARN] attn_prefill pc={pc} failed: {type(e).__name__}: {e}')
    _write_attn(Path(args.output_dir), v2_rows, full_rows, args.append)
    print(f'  wrote {len(v2_rows)} prefill rows')


def run_attn_decode(args, mcfg, kernels):
    assert args.batch is not None, '--batch required for stage=attn_decode'
    assert args.kv_list, '--kv-list required for stage=attn_decode'
    kvs = parse_list_int(args.kv_list)
    b = args.batch
    fn_attn = kernels['attn']
    v2_rows, full_rows = [], []
    for kv in tqdm(kvs, desc=f'attn_decode:b={b}', unit='m'):
        try:
            q, k, v, is_causal = _make_attn_input(0, kv, b, False, mcfg)
            s = measure(lambda: fn_attn(q, k, v, is_causal=is_causal),
                        jit_name='_attn', warmup=args.warmup, repeat=args.repeat)
            v2_rows.append((0, 0, b, kv, s['p50_ns'] / 1000))
            full_rows.append({'prefill_chunk_size': 0, 'kv_cache_size': kv,
                              'batch_size': b, 'is_prefill': False,
                              **{k_: s[k_] for k_ in ('mean_ns', 'p50_ns', 'p90_ns', 'max_ns')}})
        except Exception as e:
            tqdm.write(f'  [WARN] attn_decode b={b} kv={kv} failed: {type(e).__name__}: {e}')
    _write_attn(Path(args.output_dir), v2_rows, full_rows, args.append)
    print(f'  wrote {len(v2_rows)} decode rows (b={b})')


_STAGES = {
    'dense': run_dense,
    'per_seq': run_per_seq,
    'attn_prefill': run_attn_prefill,
    'attn_decode': run_attn_decode,
}


def main():
    p = argparse.ArgumentParser(description='Single-stage JAX TPU profiler')
    p.add_argument('--model', required=True)
    p.add_argument('--hw', default='TPU-v5e-1')
    p.add_argument('--variant', default='bf16')
    p.add_argument('--tp', type=int, default=1)
    p.add_argument('--stage', required=True, choices=list(_STAGES))
    p.add_argument('--layer', default=None, help='Required for dense / per_seq stages')
    p.add_argument('--token-list', default=None,
                   help='Comma-separated token (or seq) values for dense / per_seq')
    p.add_argument('--pc-list', default=None,
                   help='Comma-separated prefill chunk values for attn_prefill')
    p.add_argument('--batch', type=int, default=None,
                   help='Single batch value for attn_decode (sweep across --kv-list)')
    p.add_argument('--kv-list', default=None,
                   help='Comma-separated kv values for attn_decode')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeat', type=int, default=30)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--append', action='store_true',
                   help='Append to CSV (skip header). Default: fresh + header.')
    args = p.parse_args()

    t0 = time.perf_counter()
    print(f'[stage={args.stage}] {datetime.now().isoformat(timespec="seconds")}  '
          f'model={args.model} tp={args.tp} layer={args.layer or "-"}')

    mcfg = resolve_mcfg(args.model, args.tp)
    print(f'  mcfg: NH={mcfg["NH"]} NKV={mcfg["NKV"]} H={mcfg["H"]} HD={mcfg["HD"]} '
          f'I={mcfg["I"]} V={mcfg["V"]}  (V_raw={mcfg["V_raw"]})')

    kernels = build_kernels(mcfg)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    _STAGES[args.stage](args, mcfg, kernels)

    print(f'[stage={args.stage}] done in {time.perf_counter() - t0:.0f}s')


if __name__ == '__main__':
    main()
