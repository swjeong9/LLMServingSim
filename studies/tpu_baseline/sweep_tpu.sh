#!/usr/bin/env bash
# Sweep MaxText (TPU) measurements across (dataset, batch_size, tp) combinations.
#
# Output: studies/tpu_baseline/results/lens_tpu/<model>/tp<N>/bs<B>/<dataset>_<ts>.csv
#         + <dataset>.csv stable symlink to the latest run.
#
# Edit the variables at the top, then run from repo root:
#     bash studies/tpu_baseline/sweep_tpu.sh
#
# Per-task stdout/stderr is tee'd to both the terminal AND
# <out_dir>/<dataset>.log so progress is visible live and a copy is left
# for post-mortem (scp results/lens_tpu/ pulls log + CSV together).
# pipefail so the python exit code (not tee's) decides done vs FAIL.

set -euo pipefail

# ----- user knobs -----
MAXTEXT_MODEL_NAME="${MAXTEXT_MODEL_NAME:-llama3.2-1b}"
TOKENIZER_PATH="${TOKENIZER_PATH:-meta-llama/Llama-3.2-1B-Instruct}"
LOAD_PARAMETERS_PATH="${LOAD_PARAMETERS_PATH:-$HOME/maxtext_ckpts/llama3.2-1b/0/items}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
JAX_CACHE="${JAX_CACHE:-$HOME/jax_cache_lens_profiling}"

TPS="${TPS:-1}"                     # space-separated, e.g. "1 4 8"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16 32}"
DATASETS="${DATASETS:-arxiv cnn sharegpt writing_prompts}"
# ----------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Pretty model name for output path — mirrors measure_tpu.py's _model_name().
MODEL_LABEL="$(basename "${TOKENIZER_PATH}")"

# Fail fast — set -e + pipefail propagate the python exit code through tee.
# The first failing (tp, bs, ds) aborts the sweep so we don't burn hours on
# a broken config.
trap 'echo; echo "[ABORT] sweep stopped at: ${LAST_TASK:-unknown}"' ERR

run_tpu() {
    local tp=$1 bs=$2 ds=$3
    local dir="studies/tpu_baseline/results/lens_tpu/${MODEL_LABEL}/tp${tp}/bs${bs}"
    LAST_TASK="tp${tp} bs${bs} ${ds}  (log: ${dir}/${ds}.log)"
    mkdir -p "${dir}"
    echo "=== [tp${tp} bs${bs} ${ds}] start $(date +%T) ==="
    python studies/tpu_baseline/measure_tpu.py \
            --dataset "${ds}" --batch-size "${bs}" \
            --maxtext-model-name "${MAXTEXT_MODEL_NAME}" \
            --tokenizer-path "${TOKENIZER_PATH}" \
            --load-parameters-path "${LOAD_PARAMETERS_PATH}" \
            --tp-degree "${tp}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            --jax-cache "${JAX_CACHE}" \
            2>&1 | tee "${dir}/${ds}.log"
    echo "[done] tp${tp} bs${bs} ${ds}"
}

t_sweep=$(date +%s)
for tp in ${TPS}; do
    for bs in ${BATCH_SIZES}; do
        for ds in ${DATASETS}; do
            run_tpu "${tp}" "${bs}" "${ds}"
        done
    done
done

echo
echo "=== sweep done in $(( $(date +%s) - t_sweep ))s ==="
echo "  results: studies/tpu_baseline/results/lens_tpu/${MODEL_LABEL}/"
