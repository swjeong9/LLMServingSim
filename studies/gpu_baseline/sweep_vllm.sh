#!/usr/bin/env bash
# Sweep vLLM-CUDA measurements across (dataset, batch_size, tp) combinations
# on a single NVIDIA GPU instance (L4, A10G, ...).
#
# Output: studies/gpu_baseline/results/lens_vllm/<HW>/<model>/tp<N>/bs<B>/<dataset>_<ts>.csv
#         + <dataset>.csv stable symlink to the latest run.
#
# Edit the variables at the top, then run from repo root:
#     HARDWARE=L4 bash studies/gpu_baseline/sweep_vllm.sh
#     HARDWARE=A10G bash studies/gpu_baseline/sweep_vllm.sh
#
# Same tee + pipefail pattern as studies/tpu_baseline/sweep_vllm.sh.

set -euo pipefail

# ----- user knobs -----
HARDWARE="${HARDWARE:?set HARDWARE=L4 or HARDWARE=A10G (matches the cluster config + output folder)}"
MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

TPS="${TPS:-1}"                     # space-separated, e.g. "1 2"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16 32}"
DATASETS="${DATASETS:-arxiv cnn sharegpt writing_prompts}"

# GPU-only toggles (inf2/tpu baselines keep both off). Pass "1" to enable.
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-0}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
# ----------------------

cp_flag="--no-enable-chunked-prefill"; [ "${ENABLE_CHUNKED_PREFILL}" = "1" ] && cp_flag="--enable-chunked-prefill"
pc_flag="--no-enable-prefix-caching";  [ "${ENABLE_PREFIX_CACHING}"  = "1" ] && pc_flag="--enable-prefix-caching"

# Mirror measure_vllm.py::opt_label() so the shell-side log dir matches the
# python-side csv dir.
if   [ "${ENABLE_CHUNKED_PREFILL}" = "1" ] && [ "${ENABLE_PREFIX_CACHING}" = "1" ]; then OPT=on
elif [ "${ENABLE_CHUNKED_PREFILL}" = "0" ] && [ "${ENABLE_PREFIX_CACHING}" = "0" ]; then OPT=off
else OPT="cp${ENABLE_CHUNKED_PREFILL}_pc${ENABLE_PREFIX_CACHING}"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_LABEL="$(basename "${MODEL}")"

trap 'echo; echo "[ABORT] sweep stopped at: ${LAST_TASK:-unknown}"' ERR

run_vllm() {
    local tp=$1 bs=$2 ds=$3
    local dir="studies/gpu_baseline/results/lens_vllm/${HARDWARE}/${OPT}/${MODEL_LABEL}/tp${tp}/bs${bs}"
    LAST_TASK="${HARDWARE}/${OPT} tp${tp} bs${bs} ${ds}  (log: ${dir}/${ds}.log)"
    mkdir -p "${dir}"
    echo "=== [${HARDWARE}/${OPT} tp${tp} bs${bs} ${ds}] start $(date +%T) ==="
    python studies/gpu_baseline/measure_vllm.py \
            --dataset "${ds}" --batch-size "${bs}" \
            --model "${MODEL}" \
            --hardware "${HARDWARE}" \
            --tp-degree "${tp}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            "${cp_flag}" "${pc_flag}" \
            2>&1 | tee "${dir}/${ds}.log"
    echo "[done] ${HARDWARE}/${OPT} tp${tp} bs${bs} ${ds}"
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
echo "  results: studies/gpu_baseline/results/lens_vllm/${HARDWARE}/${OPT}/${MODEL_LABEL}/"
