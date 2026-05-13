"""Expand bucket-only CSVs in place: fill 1..max_bucket with ceiling
bucket's latency.

Input state (already prepared in this directory):
  - dense.csv:     7 tokens per layer (128, 256, ..., 8192)
  - attention.csv: dedupe된 prefill (1 row per BUCKETS pc) + decode (1 row per (batch, BUCKETS kv))

Output (overwrite):
  - dense.csv:     1..8192 per layer, time_us = bucket_of(tokens) 측정값
  - attention.csv: prefill 1..8192 + decode per batch (1..batch's max measured kv)
                   each time_us = bucket_of(key) 측정값
"""
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
BUCKETS = [128, 256, 512, 1024, 2048, 4096, 8192]


def bucket_of(n: int) -> int:
    for b in BUCKETS:
        if n <= b:
            return b
    return BUCKETS[-1]


def expand_dense(path: Path):
    df = pl.read_csv(path)
    lookup = {(r['layer'], r['tokens']): r['time_us']
              for r in df.iter_rows(named=True)}
    layers = df['layer'].unique(maintain_order=True).to_list()

    rows = []
    for layer in layers:
        for n in range(1, BUCKETS[-1] + 1):
            t = lookup.get((layer, bucket_of(n)))
            if t is None: continue
            rows.append({'layer': layer, 'tokens': n, 'time_us': t})

    out = pl.DataFrame(rows).sort(['layer', 'tokens'])
    out.write_csv(path)
    print(f'  dense.csv: {df.height} → {out.height} rows')


def expand_attention(path: Path):
    df = pl.read_csv(path)
    prefill_lookup = {r['prefill_chunk']: r['time_us']
                      for r in df.iter_rows(named=True) if r['prefill_chunk'] > 0}
    decode_lookup = {(r['n_decode'], r['kv_decode']): r['time_us']
                     for r in df.iter_rows(named=True) if r['prefill_chunk'] == 0}

    rows = []
    # prefill: pc 1..max_bucket
    for pc in range(1, BUCKETS[-1] + 1):
        t = prefill_lookup.get(bucket_of(pc))
        if t is None: continue
        rows.append({'prefill_chunk': pc, 'kv_prefill': 0,
                     'n_decode': 0, 'kv_decode': 0, 'time_us': t})

    # decode: per batch, kv 1..max_measured_for_this_batch
    batches = sorted({nd for (nd, _) in decode_lookup})
    for batch in batches:
        max_kv = max(kv for (nd, kv) in decode_lookup if nd == batch)
        for kv in range(1, max_kv + 1):
            t = decode_lookup.get((batch, bucket_of(kv)))
            if t is None: continue
            rows.append({'prefill_chunk': 0, 'kv_prefill': 0,
                         'n_decode': batch, 'kv_decode': kv, 'time_us': t})

    out = pl.DataFrame(rows).sort(['prefill_chunk', 'n_decode', 'kv_decode'])
    out.write_csv(path)
    print(f'  attention.csv: {df.height} → {out.height} rows')


if __name__ == '__main__':
    print(f'dir: {HERE}')
    print()
    expand_dense(HERE / 'dense.csv')
    expand_attention(HERE / 'attention.csv')
