#!/bin/bash
# Launch vLLM-TPU Docker for studies/tpu_v5e_baseline/ sweeps.
#
# Same pattern as scripts/docker-vllm.sh (GPU), but for TPU:
#   * image: vllm/vllm-tpu (Pallas attention + JAX/libtpu pre-installed)
#   * --privileged + --net host so JAX can probe /dev/vfio/* and the TPU
#     runtime can attach. (TPU containers don't expose the device via
#     --gpus all — they need privileged + /dev passthrough.)
#   * PJRT_DEVICE=TPU explicit (vLLM-TPU auto-detects but pinning saves
#     a probe round).
#   * Repo root mounted at /workspace so measure_vllm.py / sweep_vllm.sh
#     and the dataset CSVs are all visible.
#
# Usage (from repo root or anywhere — the script resolves its own paths):
#
#     # default image (latest)
#     bash studies/tpu_v5e_baseline/docker-vllm-tpu.sh
#
#     # pin a specific tag, e.g. nightly or v0.6.x-tpu
#     VLLM_TPU_IMAGE="vllm/vllm-tpu:nightly" \
#         bash studies/tpu_v5e_baseline/docker-vllm-tpu.sh
#
# Inside the container, run the sweep:
#     bash studies/tpu_v5e_baseline/sweep_vllm.sh
#     # or single shot:
#     python studies/tpu_v5e_baseline/measure_vllm.py \
#         --dataset arxiv --batch-size 4 \
#         --model meta-llama/Llama-3.2-1B-Instruct \
#         --tp-degree 1 --max-model-len 8192

set -euo pipefail

# Resolve repo root regardless of where invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../studies/tpu_v5e_baseline
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"                  # .../LLMServingSim

VLLM_TPU_IMAGE="${VLLM_TPU_IMAGE:-vllm/vllm-tpu:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm_tpu_docker}"

# Remove stale container if it lingers from a previous failed run.
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "[docker] removing stale container ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

# Note: --privileged is required for TPU device access. --net host is
# needed for some TPU pod configurations (PJRT distributed init).
# Mount the repo at /app, NOT /workspace. vLLM's TPU image installs
# itself editable from `/workspace/...` — mounting our LLMServingSim
# repo at /workspace overwrites the vllm source tree and breaks
# `import vllm`. Same pattern as scripts/docker-sim.sh (which uses /app).
docker run --name "${CONTAINER_NAME}" \
    --privileged \
    --net host \
    --shm-size=16g \
    -it \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e PJRT_DEVICE=TPU \
    -v "$REPO_ROOT":/app/LLMServingSim \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -v /dev:/dev \
    -w /app/LLMServingSim \
    --entrypoint /bin/bash \
    "${VLLM_TPU_IMAGE}" \
    -c "pip install -q datasets matplotlib 2>/dev/null || true; exec bash"
