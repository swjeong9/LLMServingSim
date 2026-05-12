#!/usr/bin/env python3
"""Analyse a captured profile_tpu.sh log + the resulting CSVs.

Usage:
    python scripts/analyze_tpu_log.py \\
        --log /path/to/sweep.log \\
        --out-dir profiler/perf/TPU-v6e-1/meta-llama/Llama-3.2-1B-Instruct/bf16/tp1

Extracts from the log:
  - per-subprocess elapsed time   (line: "[stage=X] done in <N>s")
  - per-layer total                (group by --layer)
  - per-stage total                (sum across all chunks)
  - total wallclock                (max end_time - min start_time)
  - peak chunk + slowest chunk     (anomaly inspection)

Verifies the CSVs:
  - dense.csv / per_sequence.csv / attention.csv / attention_full_stats.csv
  - row counts, layer / pc / batch coverage
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# log patterns
RE_START = re.compile(r'^\[stage=(\w+)\] (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\s+model=(\S+) tp=(\d+) layer=(\S+)')
RE_END   = re.compile(r'^\[stage=(\w+)\] done in (\d+)s')
RE_MCFG  = re.compile(r'^  mcfg: NH=(\d+) NKV=(\d+) H=(\d+) HD=(\d+) I=(\d+) V=(\d+)')
RE_WROTE = re.compile(r'^  wrote (\d+) rows')


def parse_log(path):
    """Walk a sweep log line by line, pairing [stage=X] start blocks with their
    matching `done in <N>s` lines. Returns a list of dicts (one per subprocess)
    with stage / layer / start / elapsed / rows / mcfg."""
    runs = []
    cur = None
    with open(path) as f:
        for line in f:
            m = RE_START.match(line)
            if m:
                stage, ts, model, tp, layer = m.groups()
                cur = {
                    'stage':   stage,
                    'layer':   layer if layer != '-' else None,
                    'start':   datetime.fromisoformat(ts),
                    'tp':      int(tp),
                    'model':   model,
                    'elapsed': None,
                    'rows':    None,
                    'mcfg':    None,
                }
                continue
            m = RE_MCFG.match(line)
            if m and cur is not None:
                cur['mcfg'] = {'NH': int(m[1]), 'NKV': int(m[2]), 'H': int(m[3]),
                               'HD': int(m[4]), 'I': int(m[5]), 'V': int(m[6])}
                continue
            m = RE_WROTE.match(line)
            if m and cur is not None:
                cur['rows'] = int(m[1])
                continue
            m = RE_END.match(line)
            if m and cur is not None:
                cur['elapsed'] = int(m[2])
                runs.append(cur)
                cur = None
    return runs


def summarize(runs):
    """Aggregate runs by stage and layer."""
    by_stage = defaultdict(list)
    by_layer = defaultdict(list)
    for r in runs:
        by_stage[r['stage']].append(r)
        if r['layer']:
            by_layer[(r['stage'], r['layer'])].append(r)

    print('=' * 60)
    print(f'  {len(runs)} subprocesses logged')
    if not runs:
        return
    t0 = min(r['start'] for r in runs)
    t1 = max(r['start'].timestamp() + r['elapsed'] for r in runs)
    total_wall = t1 - t0.timestamp()
    cpu_sum = sum(r['elapsed'] for r in runs)
    print(f'  wallclock total : {total_wall:.0f}s  ({total_wall/60:.1f}m, {total_wall/3600:.2f}h)')
    print(f'  cpu-time sum    : {cpu_sum}s        ({cpu_sum/60:.1f}m)')
    print(f'  startup overhead: {total_wall - cpu_sum:.0f}s '
          f'(estimate: {(total_wall - cpu_sum) / max(len(runs),1):.1f}s / subprocess)')
    print()

    print('  per-stage breakdown:')
    print(f'  {"stage":<14}  {"n":>4}  {"total_s":>8}  {"mean_s":>7}  {"p50_s":>6}  {"p95_s":>6}')
    for stage, rs in by_stage.items():
        ts = sorted(r['elapsed'] for r in rs)
        total = sum(ts); mean = total / len(ts); p50 = ts[len(ts)//2]
        p95 = ts[min(int(len(ts) * 0.95), len(ts) - 1)]
        print(f'  {stage:<14}  {len(rs):>4}  {total:>8}  {mean:>7.1f}  {p50:>6}  {p95:>6}')
    print()

    print('  per-(stage, layer) breakdown:')
    print(f'  {"stage/layer":<32}  {"n":>4}  {"total_s":>8}  {"rows":>5}')
    for (st, lay), rs in sorted(by_layer.items()):
        total = sum(r['elapsed'] for r in rs)
        rows = sum(r['rows'] or 0 for r in rs)
        print(f'  {(st + "/" + lay):<32}  {len(rs):>4}  {total:>8}  {rows:>5}')
    print()

    # outliers
    runs_sorted = sorted(runs, key=lambda r: r['elapsed'], reverse=True)
    print('  slowest 5 chunks:')
    for r in runs_sorted[:5]:
        lay = r['layer'] or '-'
        print(f'    {r["start"].strftime("%H:%M:%S")}  {r["stage"]:<12}  '
              f'{lay:<18}  {r["elapsed"]}s  ({r["rows"]} rows)')


def verify_csvs(out_dir):
    out_dir = Path(out_dir)
    if not out_dir.exists():
        print(f'\n[verify] output dir not found: {out_dir}')
        return
    print()
    print('=' * 60)
    print(f'  CSV verification — {out_dir}')
    print('=' * 60)
    files = ['dense.csv', 'per_sequence.csv', 'attention.csv', 'attention_full_stats.csv']
    for fname in files:
        p = out_dir / fname
        if not p.exists():
            print(f'  {fname:<28}  MISSING')
            continue
        with p.open() as f:
            reader = csv.reader(f)
            header = next(reader, None)
            rows = list(reader)
        print(f'  {fname:<28}  {len(rows)} rows  {header}')

        if fname == 'dense.csv':
            by_layer = defaultdict(int)
            for r in rows:
                by_layer[r[0]] += 1
            print(f'    layers ({len(by_layer)}):')
            for lay in sorted(by_layer):
                print(f'      {lay:<20}  {by_layer[lay]} rows')

        elif fname == 'per_sequence.csv':
            by_layer = defaultdict(int)
            for r in rows:
                by_layer[r[0]] += 1
            print(f'    layers ({len(by_layer)}):')
            for lay in sorted(by_layer):
                print(f'      {lay:<20}  {by_layer[lay]} rows')

        elif fname == 'attention.csv':
            n_pre = sum(1 for r in rows if int(r[0]) > 0)
            n_dec = sum(1 for r in rows if int(r[0]) == 0)
            unique_pc = sorted({int(r[0]) for r in rows if int(r[0]) > 0})
            unique_kv = sorted({int(r[3]) for r in rows if int(r[0]) == 0})
            unique_b  = sorted({int(r[2]) for r in rows if int(r[0]) == 0})
            print(f'    prefill rows: {n_pre}  (pc range {unique_pc[0] if unique_pc else "-"}..{unique_pc[-1] if unique_pc else "-"})')
            print(f'    decode rows:  {n_dec}  ({len(unique_kv)} kv × {len(unique_b)} batch)')
            print(f'    batch list:   {unique_b}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log', required=True, help='Captured stdout of profile_tpu.sh')
    p.add_argument('--out-dir', required=True, help='profiler/perf/<HW>/<MODEL>/<VARIANT>/tp<N>/')
    args = p.parse_args()

    runs = parse_log(args.log)
    summarize(runs)
    verify_csvs(args.out_dir)


if __name__ == '__main__':
    main()
