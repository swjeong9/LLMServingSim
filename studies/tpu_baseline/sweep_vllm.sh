#!/usr/bin/env bash
# Sweep vLLM-TPU measurements across (dataset, batch_size, tp) combinations.
#
# Output: studies/tpu_baseline/results/lens_vllm/<model>/tp<N>/bs<B>/<dataset>_<ts>.csv
#         + <dataset>.csv stable symlink to the latest run.
#
# Edit the variables at the top, then run from repo root:
#     bash studies/tpu_baseline/sweep_vllm.sh
#
# Same tee + pipefail pattern as sweep_tpu.sh.

set -euo pipefail

# ----- user knobs -----
MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

TPS="${TPS:-1}"                     # space-separated, e.g. "1 4 8"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16 32}"
DATASETS="${DATASETS:-arxiv cnn sharegpt writing_prompts}"
# ----------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Pretty model name for output path — mirrors measure_vllm.py's _model_name().
MODEL_LABEL="$(basename "${MODEL}")"

# Fail fast — same pattern as sweep_tpu.sh.
trap 'echo; echo "[ABORT] sweep stopped at: ${LAST_TASK:-unknown}"' ERR

run_vllm() {
    local tp=$1 bs=$2 ds=$3
    local dir="studies/tpu_baseline/results/lens_vllm/${MODEL_LABEL}/tp${tp}/bs${bs}"
    LAST_TASK="tp${tp} bs${bs} ${ds}  (log: ${dir}/${ds}.log)"
    mkdir -p "${dir}"
    echo "=== [tp${tp} bs${bs} ${ds}] start $(date +%T) ==="
    python studies/tpu_baseline/measure_vllm.py \
            --dataset "${ds}" --batch-size "${bs}" \
            --model "${MODEL}" \
            --tp-degree "${tp}" \
            --max-model-len "${MAX_MODEL_LEN}" \
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
echo "  results: studies/tpu_baseline/results/lens_vllm/${MODEL_LABEL}/"
