"""Synthesize bucketized profile from ../tp<N>/ → ./ (this tp<N>-bucket/).

Source dir is derived from this script's own dir name:
  .../tp1-bucket/  →  source ../tp1/
  .../tp2-bucket/  →  source ../tp2/

Process per CSV:
  1. Filter — keep only rows whose key axis ∈ BUCKETS (measurement at
     exact bucket boundary).
  2. Dedupe — for attention: 1 row per unique prefill_chunk (prefill)
     and per (n_decode, kv_decode) (decode).
  3. Expand — fill every integer key in 1..max_bucket using the
     ceiling bucket's measurement. Decode rows expanded per-batch
     independently. Missing bucket measurements (e.g. hardware doesn't
     reach 8192) are silently skipped — that key range has no row.
"""
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent              # .../tp<N>-bucket/
SRC = HERE.parent / HERE.name.replace('-bucket', '')   # .../tp<N>/

BUCKETS = [128, 256, 512, 1024, 2048, 4096, 8192]


def bucket_of(n: int) -> int:
    """Ceil n to the nearest BUCKETS member; n > max bucket → max bucket."""
    for b in BUCKETS:
        if n <= b:
            return b
    return BUCKETS[-1]


def synth_dense(src: Path, dst: Path):
    df = pl.read_csv(src)
    meas = df.filter(pl.col('tokens').is_in(BUCKETS))
    lookup = {(r['layer'], r['tokens']): r['time_us']
              for r in meas.iter_rows(named=True)}
    layers = meas['layer'].unique(maintain_order=True).to_list()

    rows = []
    for layer in layers:
        for n in range(1, BUCKETS[-1] + 1):
            t = lookup.get((layer, bucket_of(n)))
            if t is None: continue
            rows.append({'layer': layer, 'tokens': n, 'time_us': t})

    out = pl.DataFrame(rows).sort(['layer', 'tokens'])
    out.write_csv(dst)
    print(f'  dense.csv: {df.height} (raw) → {meas.height} (filter) → {out.height} (expand) rows')


def synth_attention(src: Path, dst: Path):
    df = pl.read_csv(src)

    filtered = df.filter(
        pl.col('prefill_chunk').is_in(BUCKETS)
        | pl.col('kv_decode').is_in(BUCKETS)
    )
    prefill_meas = (filtered
                    .filter(pl.col('prefill_chunk') > 0)
                    .unique(subset=['prefill_chunk'], keep='first'))
    decode_meas = (filtered
                   .filter(pl.col('prefill_chunk') == 0)
                   .unique(subset=['n_decode', 'kv_decode'], keep='first'))

    prefill_lookup = {r['prefill_chunk']: r['time_us']
                      for r in prefill_meas.iter_rows(named=True)}
    decode_lookup = {(r['n_decode'], r['kv_decode']): r['time_us']
                     for r in decode_meas.iter_rows(named=True)}

    rows = []
    # prefill: pc 1..max_bucket
    for pc in range(1, BUCKETS[-1] + 1):
        t = prefill_lookup.get(bucket_of(pc))
        if t is None: continue
        rows.append({'prefill_chunk': pc, 'kv_prefill': 0,
                     'n_decode': 0, 'kv_decode': 0, 'time_us': t})

    # decode: per batch, kv 1..batch's_max_measured_bucket
    batches = sorted({nd for (nd, _) in decode_lookup})
    for batch in batches:
        avail = sorted(kv for (nd, kv) in decode_lookup if nd == batch)
        if not avail: continue
        max_kv = avail[-1]
        for kv in range(1, max_kv + 1):
            t = decode_lookup.get((batch, bucket_of(kv)))
            if t is None: continue
            rows.append({'prefill_chunk': 0, 'kv_prefill': 0,
                         'n_decode': batch, 'kv_decode': kv, 'time_us': t})

    out = pl.DataFrame(rows).sort(['prefill_chunk', 'n_decode', 'kv_decode'])
    out.write_csv(dst)
    print(f'  attention.csv: {df.height} (raw) → '
          f'{prefill_meas.height} prefill + {decode_meas.height} decode (filter+dedupe) → '
          f'{out.height} (expand) rows')


if __name__ == '__main__':
    print(f'src: {SRC}')
    print(f'dst: {HERE}')
    print()
    synth_dense(SRC / 'dense.csv', HERE / 'dense.csv')
    synth_attention(SRC / 'attention.csv', HERE / 'attention.csv')
