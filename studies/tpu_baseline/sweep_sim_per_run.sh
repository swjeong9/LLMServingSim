#!/usr/bin/env bash
# sweep_sim_per_run.sh — LLMServingSim sweep with **per-run batch isolation**.
#
# Why: studies/tpu_baseline/measure_vllm.py splits each dataset CSV into
# NUM_BATCHES=50 runs of `batch_size` requests each, running them sequentially
# to BLOCK continuous batching (each batch = a fresh LLM call with exactly
# `batch_size` requests, no in-batch scheduler eviction). This is the
# apples-to-apples baseline against vLLM-TPU and NxD on Inferentia2.
#
# The default README sim sweep feeds the entire workload jsonl (100 requests)
# to the simulator in one shot — the scheduler's continuous batching then
# folds them into iterations of up to `max_num_seqs`, masking the static-batch
# behavior we actually want to compare to.
#
# This script reproduces the measure_vllm.py pattern at the simulator level:
#   - Split `workloads/<ds>_bs<B>.jsonl` (100 lines = bs × 50) into 50 chunks
#     of `bs` lines each
#   - For each chunk, run a separate `python -m serving` invocation
#   - Output: results/sim/<model>/tp<N>/bs<B>/<ds>/runs/run_<i>.{csv,log}
#
# Output layout (NEW: one level deeper than before, with runs/):
#   results/sim/Llama-3.2-1B/tp<N>/bs<B>/<dataset>/runs/run_<1..50>.csv
#                                                     /run_<1..50>.log
#
# Aggregation: a separate script reads the runs/*.csv to compute per-(tp, bs,
# ds) summary stats (see compare.py for downstream consumption).
#
# Usage:
#   bash studies/tpu_baseline/sweep_sim_per_run.sh                      # full sweep
#   PARALLEL=16 bash studies/tpu_baseline/sweep_sim_per_run.sh          # higher parallelism
#   TPS="1" BS_LIST="1 2" DATASETS="arxiv" bash sweep_sim_per_run.sh    # narrow sweep
#
# Matrix size: |TPS| × |BS_LIST| × |DATASETS| × NUM_BATCHES
#   default = 1 × 6 × 4 × 50 = 1200 simulator invocations.
#   ~2-3 sec per invocation (container start) + ~5-30 sec simulation =
#   1200 × ~20s / 8-way = ~50 min  at PARALLEL=8.

set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"

MODEL=${MODEL:-Llama-3.2-1B}
TPS=${TPS:-"1"}
BS_LIST=${BS_LIST:-"1 2 4 8 16 32"}
DATASETS=${DATASETS:-"arxiv cnn sharegpt writing_prompts"}
NUM_BATCHES=${NUM_BATCHES:-50}
PARALLEL=${PARALLEL:-8}

WORKLOAD_DIR="studies/inf2_baseline/workloads"   # symlinked from tpu_baseline/workloads
OUT_BASE="studies/tpu_baseline/results/sim/${MODEL}"
TMP_BASE=$(mktemp -d /tmp/sim_chunks.XXXXXX)
trap 'rm -rf "$TMP_BASE"' EXIT

echo "============================================================"
echo "sweep_sim_per_run.sh"
echo "  model:       $MODEL"
echo "  TPS:         $TPS"
echo "  BS_LIST:     $BS_LIST"
echo "  DATASETS:    $DATASETS"
echo "  NUM_BATCHES: $NUM_BATCHES  (runs per (tp, bs, ds))"
echo "  PARALLEL:    $PARALLEL"
echo "  out_base:    $OUT_BASE"
echo "  tmp_chunks:  $TMP_BASE"
echo "============================================================"
echo

# ---- Build task matrix: one line per (tp, ds, bs, run_id) ----

MATRIX_FILE="$TMP_BASE/matrix.txt"
for tp in $TPS; do
    for ds in $DATASETS; do
        for bs in $BS_LIST; do
            WORKLOAD="$REPO/$WORKLOAD_DIR/${ds}_bs${bs}.jsonl"
            if [[ ! -f "$WORKLOAD" ]]; then
                echo "[skip] missing workload: $WORKLOAD" >&2
                continue
            fi
            # Pre-split workload into NUM_BATCHES chunk files (once per (ds, bs)).
            # Each chunk holds exactly `bs` lines (sed is fast on 100-line files).
            CHUNK_DIR="$TMP_BASE/${ds}_bs${bs}"
            mkdir -p "$CHUNK_DIR"
            for run_id in $(seq 1 $NUM_BATCHES); do
                line_start=$(( (run_id - 1) * bs + 1 ))
                line_end=$(( run_id * bs ))
                sed -n "${line_start},${line_end}p" "$WORKLOAD" \
                    > "$CHUNK_DIR/run_${run_id}.jsonl"
                # Verify chunk has exactly `bs` lines (workloads sized for that)
                actual=$(wc -l < "$CHUNK_DIR/run_${run_id}.jsonl")
                if [[ "$actual" -ne "$bs" ]]; then
                    echo "[ERROR] $WORKLOAD chunk $run_id: $actual lines, expected $bs" >&2
                    exit 1
                fi
                echo "$tp $ds $bs $run_id"
            done
        done
    done
done > "$MATRIX_FILE"

n_tasks=$(wc -l < "$MATRIX_FILE")
echo "[matrix] $n_tasks tasks queued"
echo

# ---- Fan out: docker run --rm per task, xargs -P parallelism ----

echo "[sweep] starting ($PARALLEL-way parallel)..."

cat "$MATRIX_FILE" | xargs -n4 -P${PARALLEL} bash -c '
    tp=$0; ds=$1; bs=$2; run_id=$3

    chunk="'"$TMP_BASE"'/${ds}_bs${bs}/run_${run_id}.jsonl"
    rel_out_dir="studies/tpu_baseline/results/sim/'"$MODEL"'/tp${tp}/bs${bs}/${ds}/runs"
    abs_out_dir="'"$REPO"'/${rel_out_dir}"
    mkdir -p "$abs_out_dir"
    rel_chunk="${chunk#'"$REPO"'/}"   # may be /tmp/... not under repo

    docker run --rm \
        -v '"$REPO"':/app/LLMServingSim \
        -v "$chunk":/tmp/chunk.jsonl:ro \
        -v '"$REPO"'/astra-sim/inputs:/tmp/inputs_template:ro \
        -v /app/LLMServingSim/astra-sim/inputs \
        -w /app/LLMServingSim \
        llmservingsim:built \
        bash -c "
            cp -r /tmp/inputs_template/. /app/LLMServingSim/astra-sim/inputs/
            python -m serving \
                --cluster-config configs/cluster/tpu_v6e_llama1b_tp${tp}.json \
                --dataset /tmp/chunk.jsonl \
                --output ${rel_out_dir}/run_${run_id}.csv \
                --max-num-seqs ${bs} \
                --no-enable-chunked-prefill \
                --no-enable-prefix-caching \
                --max-num-batched-tokens 8192 \
                --dtype bfloat16 \
                > ${rel_out_dir}/run_${run_id}.log 2>&1
        "
    echo "[done] tp${tp} bs${bs} ${ds} run_${run_id}"
'

echo
echo "============================================================"
echo "[sweep] all $n_tasks tasks complete"
echo "============================================================"
echo
echo "Output tree:"
find "$OUT_BASE" -maxdepth 5 -type d | sort
