#!/usr/bin/env python3
"""Regenerate meta.yaml for a profile variant folder from the actual
tp<N>/*.csv contents.

For each variant root passed on the CLI:
1. Back up an existing meta.yaml to meta_old.yaml (if any).
2. Read the tp<N>/{dense,per_sequence,attention}.csv files and derive
   `engine_effective`, `dense_grid`, `per_sequence_grid`, `attention_grid`
   from what was actually swept.
3. Preserve manually-edited fields when present (skew_fit, calibration,
   notes, profiled_at, architecture, ...).  If meta.yaml does not exist
   yet (e.g. fresh hardware folder), infer the bare minimum from the
   path: hardware / model / variant.

Usage:
    python profiler/perf/regen_meta.py <variant_root> [<variant_root> ...]

Variant root = folder containing tp<N>/ subfolders, e.g.
    profiler/perf/TPU-v6e-1/meta-llama/Llama-3.2-1B-Instruct/bf16
"""
import argparse
import shutil
import sys
from pathlib import Path

import polars as pl
import yaml


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def summarize_values(values, list_threshold: int = 20):
    """Summarise a set of swept values.

    - ≤ `list_threshold` unique values → returned as a sorted list.
    - Otherwise → a dict with min / max / n_unique (and `step` if the
      values form an arithmetic progression).
    """
    values = sorted(set(values))
    n = len(values)
    if n == 0:
        return None
    if n <= list_threshold:
        return values
    diffs = {values[i + 1] - values[i] for i in range(n - 1)}
    if len(diffs) == 1:
        step = diffs.pop()
        return {"min": values[0], "max": values[-1], "step": step, "n_unique": n}
    return {"min": values[0], "max": values[-1], "n_unique": n}


def _max_of(summary):
    """Extract the maximum value out of either a list or a {min,max,...} dict."""
    if summary is None:
        return 0
    if isinstance(summary, dict):
        return summary.get("max", 0)
    return max(summary) if summary else 0


def infer_path_meta(variant_root: Path) -> dict:
    """Pull (hardware, model, variant) out of a path that follows
    profiler/perf/<hw>/<org>/<model>/<variant>/ .  Falls back gracefully
    when the path is shorter than expected."""
    parts = variant_root.resolve().parts
    try:
        i = parts.index("perf")
    except ValueError:
        return {}
    rest = parts[i + 1:]
    if len(rest) < 4:
        return {}
    hardware, org, model_name, variant = rest[0], rest[1], rest[2], rest[3]
    return {
        "hardware": hardware,
        "gpu": hardware,
        "model": f"{org}/{model_name}",
        "variant": variant,
    }


# ---------------------------------------------------------------------------
# grid extractors (one per CSV)
# ---------------------------------------------------------------------------

def dense_grid_from_csv(path: Path):
    if not path.exists():
        return None
    d = pl.read_csv(path)
    return {
        "tokens": summarize_values(d["tokens"].unique().to_list()),
        "layers": sorted(d["layer"].unique().to_list()),
    }


def per_sequence_grid_from_csv(path: Path):
    if not path.exists():
        return None
    p = pl.read_csv(path)
    return {
        "sequences": summarize_values(p["sequences"].unique().to_list()),
        "layers": sorted(p["layer"].unique().to_list()),
    }


def attention_grid_from_csv(path: Path):
    if not path.exists():
        return None
    a = pl.read_csv(path)
    prefill = a.filter(pl.col("prefill_chunk") > 0)
    decode = a.filter(pl.col("prefill_chunk") == 0)
    grid: dict = {}
    if prefill.height:
        grid["prefill_chunk"] = summarize_values(prefill["prefill_chunk"].unique().to_list())
        grid["kv_prefill"]    = summarize_values(prefill["kv_prefill"].unique().to_list())
    if decode.height:
        grid["n_decode"]  = summarize_values(decode["n_decode"].unique().to_list())
        grid["kv_decode"] = summarize_values(decode["kv_decode"].unique().to_list())
        grid["kv_decode_max_per_batch"] = {
            int(nd): int(decode.filter(pl.col("n_decode") == nd)["kv_decode"].max())
            for nd in sorted(decode["n_decode"].unique().to_list())
        }
        grid["max_kv"] = int(decode["kv_decode"].max())
    return grid


# ---------------------------------------------------------------------------
# main regenerate logic
# ---------------------------------------------------------------------------

def regen(variant_root: Path):
    print(f"\n=== {variant_root} ===")
    if not variant_root.is_dir():
        print(f"  [skip] not a directory")
        return

    meta_path = variant_root / "meta.yaml"
    if meta_path.exists():
        backup = variant_root / "meta_old.yaml"
        shutil.copy(meta_path, backup)
        print(f"  backup: meta.yaml → meta_old.yaml")
        with meta_path.open() as f:
            base = yaml.safe_load(f) or {}
    else:
        print(f"  meta.yaml not present — will create one from scratch")
        base = {}

    # path-derived defaults (fill in only what's missing)
    inferred = infer_path_meta(variant_root)
    for k, v in inferred.items():
        base.setdefault(k, v)

    # find tp<N>/ folders
    tp_dirs = sorted(d for d in variant_root.iterdir()
                     if d.is_dir() and d.name.startswith("tp") and d.name[2:].isdigit())
    if not tp_dirs:
        print(f"  [skip] no tp<N>/ folder")
        return
    tp_degrees = [int(d.name[2:]) for d in tp_dirs]
    base["tp_degrees"] = tp_degrees

    # use the first tp dir as the canonical sweep grid (all should match
    # in practice; tp_stable layers replicate identically across TPs)
    tp_dir = tp_dirs[0]
    dense_grid    = dense_grid_from_csv(tp_dir / "dense.csv")
    per_seq_grid  = per_sequence_grid_from_csv(tp_dir / "per_sequence.csv")
    attn_grid     = attention_grid_from_csv(tp_dir / "attention.csv")

    # derive engine_effective from the actual sweep
    eff = dict(base.get("engine_effective") or {})
    max_tokens = max(
        _max_of(dense_grid["tokens"]) if dense_grid else 0,
        _max_of(attn_grid["prefill_chunk"]) if (attn_grid and "prefill_chunk" in attn_grid) else 0,
    )
    if max_tokens:
        eff["max_num_batched_tokens"] = int(max_tokens)
    if attn_grid and "n_decode" in attn_grid:
        eff["max_num_seqs"] = int(_max_of(attn_grid["n_decode"]))
    if per_seq_grid:
        eff["per_sequence_max_sequences"] = int(_max_of(per_seq_grid["sequences"]))
    # dtype / kv_cache_dtype / block_size: keep prior values when present,
    # else fall back to sensible defaults inferred from variant name.
    variant = base.get("variant", "")
    eff.setdefault("dtype", "bfloat16" if "bf16" in variant else "float16")
    eff.setdefault("kv_cache_dtype", "fp8" if "kvfp8" in variant else "auto")
    eff.setdefault("block_size", 16)
    eff.setdefault("num_hidden_layers_profiled", 1)

    # default skew_fit if missing (matches what simulator falls back to anyway)
    skew_fit = base.get("skew_fit") or {
        "per_tp": {tp: {"method": "synthetic-constant", "alpha_default": 0.3}
                   for tp in tp_degrees},
    }
    # if per_tp does not include all profiled TPs, add stubs
    if isinstance(skew_fit, dict):
        per_tp = skew_fit.setdefault("per_tp", {})
        for tp in tp_degrees:
            per_tp.setdefault(tp, {"method": "synthetic-constant", "alpha_default": 0.3})

    calibration = base.get("calibration") or {
        "scaling_factor": 1.0,
        "scaled_by": "raw profiler output (no scaling applied)",
    }

    # assemble in a stable order
    ordered = {}
    for k in ("profiler_version", "vllm_version", "gpu", "hardware",
              "profiled_at", "architecture", "model", "variant",
              "tp_degrees"):
        if k in base:
            ordered[k] = base[k]

    ordered["engine_effective"] = eff
    if dense_grid:   ordered["dense_grid"]        = dense_grid
    if per_seq_grid: ordered["per_sequence_grid"] = per_seq_grid
    if attn_grid:    ordered["attention_grid"]    = attn_grid
    ordered["skew_fit"]    = skew_fit
    ordered["calibration"] = calibration

    if "notes" in base:
        ordered["notes"] = base["notes"]

    with meta_path.open("w") as f:
        yaml.safe_dump(ordered, f, default_flow_style=None,
                       sort_keys=False, width=120)

    # one-line summary
    mnt = eff.get("max_num_batched_tokens", "?")
    mns = eff.get("max_num_seqs", "?")
    nd  = attn_grid["n_decode"] if attn_grid and "n_decode" in attn_grid else "?"
    print(f"  wrote meta.yaml  "
          f"(max_num_batched_tokens={mnt}, max_num_seqs={mns}, n_decode={nd}, tps={tp_degrees})")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("variant_root", nargs="+",
                   help="Path(s) to variant folder(s) containing tp<N>/ subfolders.")
    args = p.parse_args()
    for path in args.variant_root:
        regen(Path(path))


if __name__ == "__main__":
    main()
