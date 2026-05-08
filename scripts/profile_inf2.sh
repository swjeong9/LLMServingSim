#!/usr/bin/env bash
# scripts/profile_inf2.sh — sweep three target models on Inferentia 2
# with a powers-of-2 grid up to max_sequence_length = 8192.
#
# Design philosophy: keep the grid lean. The simulator-vs-measurement
# comparison (compare_static.py) is fair as long as both sides use the
# same profile bundle, so dense paper-matching grids aren't necessary
# for accuracy benchmarking — they only help when you want absolute
# latency predictions to match raw measurements within ±5%. See
# GRID_DENSITY_KO.md for the trade-off discussion.
#
# Skew sweep, mixed prefill+decode shots, and the paper's irregular
# inter-power points are all skipped. kv_prefill is collapsed to {0}
# because we run with --no-enable-chunked-prefill (so kv_prefill is
# always zero in the workload).
#
# Per (model, TP):
#   ~185 shots × (warmup 10 + repeat 30) = ~7400 forward calls
#   Wall-clock: ~30 min to 1 h on first run (Neuron compile populating),
#   minutes on rerun (cache hit).
#
# Usage (on inf2 with the AWS Neuron DLAMI's pytorch_2_9 venv):
#
#     source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
#     export HF_TOKEN="hf_xxx_..."     # for gated models
#     ./scripts/profile_inf2.sh
#
# Resume / partial: comment out finished MODEL_TPS entries below. Each
# (model, TP) combination is independent.

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"      # caps both tokens and kv axes
DTYPE="${DTYPE:-bfloat16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-profiler/perf}"

WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-30}"

# Three target models. Comment out to skip.
# Format: "model_id|tp_csv"
MODEL_TPS=(
    "meta-llama/Llama-3.2-1B|1,2,4,8"
    "mistralai/Mistral-7B-v0.3|1,2,4,8"
    "Qwen/Qwen3-14B|2,4,8"               # TP=1 OOM at full 30 GB; skip
)

# =============================================================================
# Powers-of-2 grids
# =============================================================================

# 14 token points: 1, 2, 4, 8, ..., 8192
TOKENS_GRID="1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"

# 7 sequence (= batch size in lm_head) points: 1, 2, 4, ..., 64
SEQUENCES_GRID="1,2,4,8,16,32,64"

# 10 prefill_chunk points (skip 1/2/4/8 — prefill chunks are >=16 tokens
# in any realistic scheduler).
PREFILL_GRID="16,32,64,128,256,512,1024,2048,4096,8192"

# kv_prefill is always 0 because we disable chunked prefill in the
# baseline workload (see compare_static.py / mode A & B). Single point.
KV_PREFILL_GRID="0"

# 7 n_decode points, matching SEQUENCES_GRID since pure-decode
# n_decode == batch size.
DECODE_N_GRID="1,2,4,8,16,32,64"

# 10 kv_decode points. Note: profile_neuron.py auto-skips entries where
# kv_decode + 1 > max_position_embeddings, so kv_decode=8192 with
# MAX_SEQ_LEN=8192 gets dropped by the script itself.
KV_DECODE_GRID="16,32,64,128,256,512,1024,2048,4096,8192"

# =============================================================================
# Execute
# =============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo "  profile_inf2.sh — Inferentia 2 sweep with powers-of-2 grid"
echo "============================================================"
echo "  output_root : $OUTPUT_ROOT"
echo "  max_seq_len : $MAX_SEQ_LEN"
echo "  dtype       : $DTYPE"
echo "  warmup      : $WARMUP   repeat : $REPEAT"
echo
echo "  tokens_grid    : 14 points (1..8192, powers of 2)"
echo "  sequences_grid :  7 points (1..64, powers of 2)"
echo "  prefill_grid   : 10 points (16..8192, powers of 2)"
echo "  kv_prefill     :  1 point  (0; chunked prefill disabled)"
echo "  decode_n_grid  :  7 points (1..64, powers of 2)"
echo "  kv_decode      : 10 points (16..8192, powers of 2)"
echo "  ~185 shots × $((WARMUP + REPEAT)) forwards ≈ $((185 * (WARMUP + REPEAT))) per (model, TP)"
echo "  Estimate: 30-60 min first run / TP, minutes on rerun"
echo
echo "  models × TPs:"
for spec in "${MODEL_TPS[@]}"; do
    IFS='|' read -r model tps <<<"$spec"
    echo "    - $model  (TP=$tps)"
done
echo
echo "Press Ctrl-C in 5s to abort..."
sleep 5

for spec in "${MODEL_TPS[@]}"; do
    IFS='|' read -r model tps <<<"$spec"
    echo
    echo "############################################################"
    echo "  $model  (TP=$tps)"
    echo "############################################################"
    python scripts/profile_neuron.py \
        --model "$model" \
        --tp "$tps" \
        --output-root "$OUTPUT_ROOT" \
        --dtype "$DTYPE" \
        --max-position-embeddings "$MAX_SEQ_LEN" \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 64 \
        --tokens-grid "$TOKENS_GRID" \
        --sequences-grid "$SEQUENCES_GRID" \
        --prefill-grid "$PREFILL_GRID" \
        --kv-prefill-grid "$KV_PREFILL_GRID" \
        --decode-n-grid "$DECODE_N_GRID" \
        --kv-decode-grid "$KV_DECODE_GRID" \
        --warmup "$WARMUP" \
        --repeat "$REPEAT"
done

echo
echo "============================================================"
echo "  Done. Profile bundles written under $OUTPUT_ROOT/Inferentia2/"
echo "  Inspect timing: python scripts/show_profile_timing.py \\"
echo "      \$OUTPUT_ROOT/Inferentia2/<model>/bf16"
echo "============================================================"
