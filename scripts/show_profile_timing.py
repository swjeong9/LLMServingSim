#!/usr/bin/env python3
"""
show_profile_timing.py — Read profile_timing.json (and/or
validation_timing.json) artifacts and print stage / shot-level cost
breakdowns. Useful for:

* tracking how long a profiling run actually took, broken down by stage
* comparing cost across (model, hardware, TP) combinations
* identifying compile-vs-measure bottlenecks (the per-shot meta keeps
  ``compile_us`` and ``wall_us`` separately)

Usage
-----
::

    # Show one run
    python scripts/show_profile_timing.py \\
        profiler/perf/Inferentia2/meta-llama/Llama-3.2-1B/bf16/profile_timing.json

    # Compare multiple runs side by side (auto-discovers profile_timing.json
    # under given roots)
    python scripts/show_profile_timing.py \\
        profiler/perf/Inferentia2/meta-llama/Llama-3.2-1B/bf16 \\
        profiler/perf/Inferentia2/mistralai/Mistral-7B-v0.3/bf16 \\
        profiler/perf/Inferentia2/Qwen/Qwen3-14B/bf16

    # JSON dump for further processing
    python scripts/show_profile_timing.py --json \\
        profiler/perf/Inferentia2/meta-llama/Llama-3.2-1B/bf16

Modes
-----
* default                — pretty stage table per run + summary
* ``--shots``            — also list per-shot times (long output)
* ``--json``             — emit machine-readable summary
* ``--by-category``      — aggregate shots by category (dense / per_seq /
                           attention) within each TP
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------
def find_timing_files(path: Path) -> List[Tuple[str, Path]]:
    """Return [(label, file), ...] for either a JSON file or a directory.

    Picks up tagged variants like profile_timing_cold.json /
    profile_timing_hot.json alongside the canonical names.
    """
    out: List[Tuple[str, Path]] = []
    if path.is_file() and path.suffix == ".json":
        out.append((path.parent.name, path))
        return out
    if path.is_dir():
        # Glob both tagged + untagged. Sort for deterministic ordering.
        candidates = sorted(set(
            list(path.glob("profile_timing*.json"))
            + list(path.glob("validation_timing*.json"))
        ))
        for f in candidates:
            rel = "/".join(path.parts[-3:])
            out.append((f"{rel}::{f.stem}", f))
        if not out:
            print(f"[!] no timing JSONs found under {path}", file=sys.stderr)
    return out


# ----------------------------------------------------------------------
# Loaders + summaries
# ----------------------------------------------------------------------
def load_timing(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def summarize_profile(timing: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate a profile_timing.json into stage + category summaries."""
    out: Dict[str, Any] = {
        "model": timing.get("model"),
        "hardware": timing.get("hardware"),
        "variant": timing.get("variant"),
        "started_at": timing.get("started_at"),
        "wall_clock_total_sec": timing.get("wall_clock_total_sec"),
        "machine": timing.get("machine", {}),
        "tp_summary": {},
    }
    for tp, st in timing.get("tp_stages", {}).items():
        shots = st.get("shots", [])
        wall_us_sum = sum(s.get("wall_us", 0) for s in shots)
        compile_us_sum = sum(s.get("compile_us", 0) for s in shots)
        # By category
        by_cat: Dict[str, Dict[str, float]] = {}
        for s in shots:
            cat = s.get("category", "?")
            d = by_cat.setdefault(cat, {"shots": 0, "wall_us": 0,
                                        "compile_us": 0,
                                        "median_us_sum": 0})
            d["shots"] += 1
            d["wall_us"] += s.get("wall_us", 0)
            d["compile_us"] += s.get("compile_us", 0)
            d["median_us_sum"] += s.get("median_us", 0)
        out["tp_summary"][tp] = {
            "load_sec": st.get("load_sec"),
            "dense_sec": st.get("dense_sec"),
            "per_seq_sec": st.get("per_seq_sec"),
            "attn_sec": st.get("attn_sec"),
            "write_sec": st.get("write_sec"),
            "total_sec": st.get("total_sec"),
            "n_shots": len(shots),
            "wall_us_sum": wall_us_sum,
            "compile_us_sum": compile_us_sum,
            "compile_pct_of_wall": (compile_us_sum / wall_us_sum * 100
                                    if wall_us_sum else 0.0),
            "by_category": by_cat,
        }
    return out


def summarize_validation(timing: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "model": timing.get("model"),
        "scaling_factor": timing.get("scaling_factor"),
        "num_layers_validated": timing.get("num_layers_validated"),
        "stages": timing.get("stages", {}),
        "shapes": timing.get("shapes", []),
    }
    return out


def is_profile(timing: Dict[str, Any]) -> bool:
    return timing.get("schema", "").startswith("profile_timing")


# ----------------------------------------------------------------------
# Renderers
# ----------------------------------------------------------------------
def fmt_dur(seconds: Optional[float]) -> str:
    if seconds is None:
        return "    ?  "
    if seconds < 60:
        return f"{seconds:6.1f}s"
    return f"{seconds/60:6.1f}m"


def render_profile_summary(label: str, summary: Dict[str, Any], shots: bool,
                            by_category: bool) -> str:
    lines: List[str] = [f"=== {label} ==="]
    lines.append(f"  model:    {summary['model']}")
    lines.append(f"  hardware: {summary['hardware']}  variant: {summary['variant']}")
    lines.append(f"  started:  {summary['started_at']}")
    lines.append(f"  total wall: {fmt_dur(summary['wall_clock_total_sec'])}")

    m = summary.get("machine", {}) or {}
    if m:
        lines.append(f"  machine: {m.get('instance_type','?')}  "
                     f"py={m.get('python','?')}  "
                     f"transformers={m.get('transformers','?')}  "
                     f"torch={m.get('torch','?')}  "
                     f"torch_xla={m.get('torch_xla','?')}")

    # Stage table per TP
    lines.append("")
    lines.append("  TP    load     dense    per_seq   attn      write    total    "
                 "n_shots  compile%")
    lines.append("  ─────────────────────────────────────────────────────────"
                 "──────────────────")
    for tp in sorted(summary["tp_summary"].keys(), key=lambda x: int(x)):
        s = summary["tp_summary"][tp]
        lines.append(f"  {tp:<5s} {fmt_dur(s['load_sec'])} "
                     f"{fmt_dur(s['dense_sec'])} "
                     f"{fmt_dur(s['per_seq_sec'])} "
                     f"{fmt_dur(s['attn_sec'])} "
                     f"{fmt_dur(s['write_sec'])} "
                     f"{fmt_dur(s['total_sec'])} "
                     f" {s['n_shots']:5d}    "
                     f"{s['compile_pct_of_wall']:5.1f}%")

    if by_category:
        lines.append("")
        lines.append("  By category (per TP):")
        for tp in sorted(summary["tp_summary"].keys(), key=lambda x: int(x)):
            s = summary["tp_summary"][tp]
            lines.append(f"    tp{tp}:")
            for cat, d in s.get("by_category", {}).items():
                wall_s = d["wall_us"] / 1e6
                comp_s = d["compile_us"] / 1e6
                meas_s = (d["wall_us"] - d["compile_us"]) / 1e6
                avg_med_us = (d["median_us_sum"] / d["shots"]) if d["shots"] else 0
                lines.append(f"      {cat:14s} shots={d['shots']:4d}   "
                             f"wall={wall_s:7.1f}s   "
                             f"compile={comp_s:7.1f}s   "
                             f"measure={meas_s:7.1f}s   "
                             f"median t≈{avg_med_us:9.2f} us")

    if shots:
        lines.append("")
        # We need the original timing to access shots (summary aggregates).
        # Skipped here; --shots requires render_shots() called separately.

    return "\n".join(lines)


def render_shots(label: str, timing: Dict[str, Any]) -> str:
    lines: List[str] = [f"=== {label} :: shots ==="]
    for tp, st in timing.get("tp_stages", {}).items():
        lines.append(f"  tp{tp}:")
        lines.append(f"    {'category':14s} {'layer/regime':18s} {'key':50s}  "
                     f"{'first_us':>10s} {'median_us':>10s} {'compile_us':>10s}  "
                     f"{'wall_us':>10s}")
        for s in st.get("shots", []):
            cat = s.get("category", "?")
            lay = s.get("layer", s.get("regime", "?"))
            key = json.dumps(s.get("key", {}), separators=(",", ":"))
            lines.append(f"    {cat:14s} {lay:18s} {key[:50]:50s}  "
                         f"{s.get('first_call_us',0):10.1f} "
                         f"{s.get('median_us',0):10.1f} "
                         f"{s.get('compile_us',0):10.1f}  "
                         f"{s.get('wall_us',0):10.1f}")
    return "\n".join(lines)


def render_validation_summary(label: str, summary: Dict[str, Any]) -> str:
    lines = [f"=== {label} (validation) ==="]
    lines.append(f"  model: {summary['model']}")
    lines.append(f"  scaling factor: {summary['scaling_factor']:.4f}"
                 if summary.get("scaling_factor") is not None else "  scaling factor: ?")
    lines.append(f"  num_layers_validated: {summary['num_layers_validated']}")
    st = summary.get("stages", {})
    lines.append(f"  stages: load={fmt_dur(st.get('load_sec'))}  "
                 f"shapes={fmt_dur(st.get('shapes_sec'))}  "
                 f"apply={fmt_dur(st.get('apply_sec'))}  "
                 f"total={fmt_dur(st.get('total_sec'))}")
    if summary.get("shapes"):
        lines.append("")
        lines.append("  shape   measured_us   estimated_us  ratio   wall_sec")
        for s in summary["shapes"]:
            lines.append(f"  {s['key']:>10s}  {s['measured_us']:11.1f}   "
                         f"{s['estimated_us']:11.1f}   "
                         f"{s['ratio']:5.3f}   {s['wall_sec']:6.1f}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("paths", nargs="+",
                   help="profile_timing.json or validation_timing.json files, "
                        "or variant-root directories that contain them")
    p.add_argument("--shots", action="store_true",
                   help="Print per-shot timing (long output)")
    p.add_argument("--by-category", action="store_true",
                   help="Aggregate shots by category within each TP")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable summary (skip pretty render)")
    return p.parse_args()


def main():
    args = parse_args()

    discovered: List[Tuple[str, Path]] = []
    for raw in args.paths:
        p = Path(raw)
        if not p.exists():
            print(f"[!] not found: {p}", file=sys.stderr)
            continue
        discovered.extend(find_timing_files(p))

    if not discovered:
        sys.exit(2)

    if args.json:
        out: Dict[str, Any] = {}
        for label, f in discovered:
            t = load_timing(f)
            out[label] = (summarize_profile(t) if is_profile(t)
                          else summarize_validation(t))
        print(json.dumps(out, indent=2, default=str))
        return

    for label, f in discovered:
        t = load_timing(f)
        if is_profile(t):
            summary = summarize_profile(t)
            print(render_profile_summary(label, summary, args.shots,
                                         args.by_category))
            if args.shots:
                print(render_shots(label, t))
        else:
            summary = summarize_validation(t)
            print(render_validation_summary(label, summary))
        print()


if __name__ == "__main__":
    main()
