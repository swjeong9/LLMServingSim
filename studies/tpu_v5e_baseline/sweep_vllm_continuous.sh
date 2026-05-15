#!/usr/bin/env bash
# Sweep vLLM-TPU measurements in **single-shot continuous-batched** mode
# across (dataset, batch_size, tp) combinations on TPU.
#
# Counterpart to sweep_vllm.sh — that one runs measure_vllm.py (50 batches
# × bs requests, per-run isolation, no continuous batching). This one runs
# measure_vllm_continuous.py (all bs × NUM_BATCHES requests in one
# .generate() call, full continuous batching under max_num_seqs=bs).
#
# Output: studies/tpu_v5e_baseline/results/lens_vllm_continuous/<model>/tp<N>/bs<B>/<dataset>_<ts>.csv
#         + <dataset>.csv stable symlink to the latest run.
#
# Edit the variables at the top, then run from repo root:
#     bash studies/tpu_v5e_baseline/sweep_vllm_continuous.sh

set -euo pipefail

# ----- user knobs -----
MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
# Per-step token budget. Pin = MAX_MODEL_LEN so any single prefill fits in
# one scheduler step — that is the actual "no chunked prefill" knob on V1
# (enable_chunked_prefill=False is silently ignored — vllm#18547).
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

TPS="${TPS:-1}"                     # space-separated, e.g. "1 4 8"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16 32}"
DATASETS="${DATASETS:-arxiv cnn sharegpt writing_prompts}"
# ----------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_LABEL="$(basename "${MODEL}")"

trap 'echo; echo "[ABORT] sweep stopped at: ${LAST_TASK:-unknown}"' ERR

run_vllm() {
    local tp=$1 bs=$2 ds=$3
    local dir="studies/tpu_v5e_baseline/results/lens_vllm_continuous/${MODEL_LABEL}/tp${tp}/bs${bs}"
    LAST_TASK="tp${tp} bs${bs} ${ds}  (log: ${dir}/${ds}.log)"
    mkdir -p "${dir}"
    echo "=== [tp${tp} bs${bs} ${ds}] start $(date +%T) ==="
    python studies/tpu_v5e_baseline/measure_vllm_continuous.py \
            --dataset "${ds}" --batch-size "${bs}" \
            --model "${MODEL}" \
            --tp-degree "${tp}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
            2>&1 | tee "${dir}/${ds}.log"
    echo "[done] tp${tp} bs${bs} ${ds}"
}

t_sweep=$(date +%s)
for tp in ${TPS}; do
    for bs in ${BATCH_SIZES}; do
        for ds in ${DATASETS}; do
            run_vllm "${tp}" "${bs}" "${ds}"
        done
    done
done

echo
echo "=== sweep done in $(( $(date +%s) - t_sweep ))s ==="
echo "  results: studies/tpu_v5e_baseline/results/lens_vllm_continuous/${MODEL_LABEL}/"
