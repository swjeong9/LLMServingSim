#!/usr/bin/env python3
"""
measure_static_neuron.py — Real measurement on inf2 of an offline batch
workload, in two scheduler modes:

* ``--mode A`` (continuous batching): one ``llm.generate(all_prompts, ...)``
  call. vLLM-Neuron pumps requests through with ``max_num_seqs=B`` keeping
  B in flight, replacing finished slots from the queue. Matches what
  ``vllm.LLM.generate(prompt_list, ...)`` does by default.

* ``--mode B`` (strict static batches): batch-by-batch loop. We slice the
  workload into M batches of B requests each and call ``llm.generate``
  once per batch. The next batch only starts after the previous batch
  fully completes — no slot replacement.

Outputs a CSV with the same schema the simulator emits, so
``compare_static.py`` can merge them by ``request id``:

    instance id, request id, model, input, output,
    arrival, end_time, latency, queuing_delay, TTFT, TPOT, ITL

(All times in ns. ``instance id`` is always 0 for offline batch.)

Per-request stats come from vLLM's ``RequestOutput.metrics``
(``RequestStateStats``-style fields). ``arrival`` is set to the
workload's ``arrival_time_ns`` field for round-trip with the simulator
side; ``queuing_delay`` is computed as ``scheduled_ts - effective_arrival``
where ``effective_arrival`` is when we actually enqueued to vLLM.

Run on inf2 with the AWS Neuron DLAMI's vLLM-Neuron venv:

    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
    pip install -U pandas

Then:

    python scripts/measure_static_neuron.py \\
        --workload workloads/static_b4_50.jsonl \\
        --model meta-llama/Llama-3.2-1B \\
        --tp 4 --batch-size 4 \\
        --mode A \\
        --output outputs/meas_b4_modeA.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


# ----------------------------------------------------------------------
# Lazy imports
# ----------------------------------------------------------------------
def _lazy_vllm():
    from vllm import LLM, SamplingParams
    return LLM, SamplingParams


def load_workload(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def make_llm(model: str, tp: int, batch_size: int, max_seq: int,
             dtype: str, hf_token: str):
    """Boot vLLM-Neuron engine. NEFF compile happens inside .compile()/first generate."""
    LLM, _ = _lazy_vllm()
    print(f"[*] booting vLLM(model={model}, tp={tp}, max_num_seqs={batch_size}, "
          f"max_model_len={max_seq})")
    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        device="neuron",
        max_num_seqs=batch_size,
        max_model_len=max_seq,
        block_size=16,
        dtype=dtype,
        enforce_eager=False,        # use NEFF (production-realistic)
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        # If the model is gated, pass token via env HF_TOKEN — vLLM picks it up.
    )
    return llm


# ----------------------------------------------------------------------
# Per-request output normalization
# ----------------------------------------------------------------------
def _ns(x: float) -> int:
    """Convert a possibly-float seconds timestamp into integer ns."""
    return int(round(x * 1e9))


def _extract_metrics(req_id: int, model: str, in_len: int, out_len: int,
                     arrival_ns: int, output, ref_t0_ns: int) -> Dict[str, Any]:
    """Pull RequestStateStats off a vLLM RequestOutput and compute the
    same fields the LLMServingSim simulator emits.

    ``ref_t0_ns`` is the wall-clock perf_counter_ns when the workload
    was enqueued to vLLM (mode A: when the single .generate started;
    mode B: when this batch's .generate started). It anchors the
    arrival timestamp so per-request waits make sense.
    """
    m = output.metrics
    # All vLLM metrics are seconds-since-engine-epoch (perf_counter style).
    # Translate them into ns relative to ref_t0_ns (which is also captured
    # in perf_counter_ns).
    arrival_pc_ns = _ns(m.arrival_time)
    scheduled_pc_ns = _ns(m.scheduled_ts) if m.scheduled_ts else arrival_pc_ns
    first_token_pc_ns = (_ns(m.first_token_time)
                         if getattr(m, "first_token_time", None)
                         else _ns(m.first_scheduled_time))
    last_token_pc_ns = (_ns(m.last_token_time)
                        if getattr(m, "last_token_time", None)
                        else _ns(m.finished_time))

    # Effective arrival = the workload's declared arrival_time_ns
    # (mode B: i*gap, mode A: 0). queuing_delay is the gap between
    # effective arrival and when the engine actually scheduled the request.
    eff_arrival_ns = arrival_ns
    sched_rel_ns = scheduled_pc_ns - arrival_pc_ns      # vLLM-internal queuing
    queuing_delay_ns = sched_rel_ns                     # report engine-side queue
    ttft_ns = first_token_pc_ns - arrival_pc_ns
    e2e_ns = last_token_pc_ns - arrival_pc_ns
    if out_len > 1:
        tpot_ns = (last_token_pc_ns - first_token_pc_ns) // (out_len - 1)
    else:
        tpot_ns = 0

    # ITL is per-token gap; vLLM doesn't emit per-token timestamps in
    # offline batch by default. Approximate as repeated TPOT.
    itl_list = [tpot_ns] * max(0, out_len - 1)

    return {
        "instance id": 0,
        "request id": req_id,
        "model": model,
        "input": in_len,
        "output": out_len,
        "arrival": eff_arrival_ns,
        "end_time": eff_arrival_ns + e2e_ns,
        "latency": e2e_ns,
        "queuing_delay": queuing_delay_ns,
        "TTFT": ttft_ns,
        "TPOT": tpot_ns,
        "ITL": str(itl_list),
    }


# ----------------------------------------------------------------------
# Mode A: single offline batch
# ----------------------------------------------------------------------
def run_mode_A(llm, workload: List[Dict[str, Any]], model: str,
               warmup_each_shape: bool) -> List[Dict[str, Any]]:
    _, SamplingParams = _lazy_vllm()
    prompt_ids = [r["input_tok_ids"] for r in workload]
    sps = [SamplingParams(min_tokens=r["output_toks"],
                          max_tokens=r["output_toks"],
                          ignore_eos=True, temperature=0.0, top_p=1.0)
           for r in workload]

    if warmup_each_shape:
        # Warm Neuron NEFF cache with the unique (input_len, output_len) pairs
        # so the timed run doesn't pay first-time compile inside one shape.
        seen = set()
        for r in workload:
            key = (r["input_toks"], r["output_toks"])
            if key in seen:
                continue
            seen.add(key)
            print(f"  [warmup] shape ({key[0]}, {key[1]})")
            llm.generate(prompt_token_ids=[r["input_tok_ids"][:key[0]]],
                         sampling_params=[SamplingParams(
                             min_tokens=key[1], max_tokens=key[1],
                             ignore_eos=True, temperature=0.0)],
                         use_tqdm=False)

    print(f"[mode A] generating {len(workload)} requests in one batch...")
    t0 = time.perf_counter_ns()
    outputs = llm.generate(prompt_token_ids=prompt_ids,
                           sampling_params=sps, use_tqdm=False)
    t1 = time.perf_counter_ns()
    print(f"[mode A] done in {(t1-t0)/1e9:.2f}s")

    # Match outputs back to requests by index (order preserved).
    results: List[Dict[str, Any]] = []
    for i, (req, out) in enumerate(zip(workload, outputs)):
        results.append(_extract_metrics(
            req_id=i, model=llm.llm_engine.model_config.model,
            in_len=req["input_toks"], out_len=req["output_toks"],
            arrival_ns=req.get("arrival_time_ns", 0),
            output=out, ref_t0_ns=t0))
    return results


# ----------------------------------------------------------------------
# Mode B: strict static batches
# ----------------------------------------------------------------------
def run_mode_B(llm, workload: List[Dict[str, Any]], batch_size: int,
               warmup_each_shape: bool) -> List[Dict[str, Any]]:
    _, SamplingParams = _lazy_vllm()
    M = len(workload) // batch_size
    if len(workload) % batch_size != 0:
        print(f"[!] {len(workload)} not divisible by batch_size={batch_size}; "
              f"last partial batch will be dropped.")

    if warmup_each_shape:
        seen = set()
        for r in workload[:batch_size]:
            key = (r["input_toks"], r["output_toks"])
            if key in seen:
                continue
            seen.add(key)
            print(f"  [warmup] shape ({key[0]}, {key[1]})")
            llm.generate(prompt_token_ids=[r["input_tok_ids"][:key[0]]],
                         sampling_params=[SamplingParams(
                             min_tokens=key[1], max_tokens=key[1],
                             ignore_eos=True, temperature=0.0)],
                         use_tqdm=False)

    results: List[Dict[str, Any]] = []
    for b in range(M):
        batch = workload[b*batch_size:(b+1)*batch_size]
        prompt_ids = [r["input_tok_ids"] for r in batch]
        sps = [SamplingParams(min_tokens=r["output_toks"],
                              max_tokens=r["output_toks"],
                              ignore_eos=True, temperature=0.0)
               for r in batch]
        t0 = time.perf_counter_ns()
        outputs = llm.generate(prompt_token_ids=prompt_ids,
                               sampling_params=sps, use_tqdm=False)
        t1 = time.perf_counter_ns()
        print(f"[mode B] batch {b+1:3d}/{M} ({batch_size} reqs) "
              f"-> {(t1-t0)/1e6:.1f} ms")
        for j, (req, out) in enumerate(zip(batch, outputs)):
            results.append(_extract_metrics(
                req_id=b*batch_size + j,
                model=llm.llm_engine.model_config.model,
                in_len=req["input_toks"], out_len=req["output_toks"],
                arrival_ns=req.get("arrival_time_ns", 0),
                output=out, ref_t0_ns=t0))
    return results


# ----------------------------------------------------------------------
# Output writer
# ----------------------------------------------------------------------
def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        print("[!] no rows to write")
        return
    fields = ["instance id", "request id", "model", "input", "output",
              "arrival", "end_time", "latency", "queuing_delay",
              "TTFT", "TPOT", "ITL"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[✓] wrote {path}  ({len(rows)} rows)")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--workload", required=True,
                   help="Workload JSONL produced by make_static_workload.py")
    p.add_argument("--model", required=True, help="HF id")
    p.add_argument("--tp", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True,
                   help="max_num_seqs (mode A) / batch group size (mode B)")
    p.add_argument("--mode", choices=["A", "B"], required=True)
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16"])
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--no-shape-warmup", action="store_true",
                   help="Skip per-shape warmup (faster but timed runs may "
                        "include first-time NEFF compile)")
    p.add_argument("--hf-token", default=os.getenv("HF_TOKEN", ""))
    return p.parse_args()


def main():
    args = parse_args()
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    workload = load_workload(Path(args.workload))
    print(f"[*] loaded {len(workload)} requests from {args.workload}")

    llm = make_llm(args.model, args.tp, args.batch_size,
                   args.max_model_len, args.dtype, args.hf_token)

    warmup = not args.no_shape_warmup
    if args.mode == "A":
        results = run_mode_A(llm, workload, args.model, warmup)
    elif args.mode == "B":
        results = run_mode_B(llm, workload, args.batch_size, warmup)
    else:
        raise ValueError(args.mode)

    write_csv(results, Path(args.output))


if __name__ == "__main__":
    main()
