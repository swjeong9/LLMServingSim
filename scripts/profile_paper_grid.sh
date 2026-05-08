#!/usr/bin/env bash
# scripts/profile_paper_grid.sh — sweep three target models on Inferentia 2
# using the same per-axis grid points the paper used, capped at
# max_sequence_length = 8192 (paper went up to 16384 on KV axes; we
# cap so a single NeuronCore can host the full-layer model for
# downstream validation if needed).
#
# Skew sweep (skew.csv / skew_fit.csv) is intentionally skipped — the
# user's static-offline-batch scenario doesn't need heterogeneous-decode
# correction. Mixed prefill+decode sweeps are also out of scope for
# profile_neuron.py (see GRID_DENSITY_KO.md §3).
#
# Per (model, TP):
#   ~1550 shots × (warmup 10 + repeat 30) = ~62k forward calls
#   Wall-clock: 4-5 h on first run (Neuron compile cache populating),
#   minutes on rerun (cache hit).
#
# Usage (on inf2 with the AWS Neuron DLAMI's pytorch_2_9 venv):
#
#     source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
#     export HF_TOKEN="hf_xxx_..."     # for gated models (Llama, Mistral)
#     ./scripts/profile_paper_grid.sh
#
# Resume / partial: comment out finished MODEL/TP lines below. Each
# (model, TP) combination is independent.
#
# Tweakable knobs at the top of the file (MODELS, TPS, MAX_SEQ_LEN).

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"      # caps both tokens and kv axes
DTYPE="${DTYPE:-bfloat16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-profiler/perf}"

WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-30}"

# Three target models. Comment out if you don't want to sweep one.
# Format: "model_id|tp_csv"
MODEL_TPS=(
    "meta-llama/Llama-3.2-1B|1,2,4,8"
    "mistralai/Mistral-7B-v0.3|1,2,4,8"
    "Qwen/Qwen3-14B|2,4,8"               # TP=1 OOM at full 30 GB; skip
)

# =============================================================================
# Grids — the paper's exact per-axis points, capped at MAX_SEQ_LEN.
# Extracted from profiler/perf/RTXPRO6000/meta-llama/Llama-3.1-8B/bf16/tp1.
# Token / sequence / prefill_chunk / n_decode caps are unchanged from paper
# (already <= 8192). KV axes (kv_prefill / kv_decode) go up to paper's 16384;
# we drop everything > 8192.
# =============================================================================

# 152 token points — every integer 1..16, then step-4 to 64, step-16 to 256,
# step-16 to 2048. Identical to paper's `dense.csv` grid.
TOKENS_GRID="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,28,32,36,40,44,48,52,56,60,64,80,96,112,128,144,160,176,192,208,224,240,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512,528,544,560,576,592,608,624,640,656,672,688,704,720,736,752,768,784,800,816,832,848,864,880,896,912,928,944,960,976,992,1008,1024,1040,1056,1072,1088,1104,1120,1136,1152,1168,1184,1200,1216,1232,1248,1264,1280,1296,1312,1328,1344,1360,1376,1392,1408,1424,1440,1456,1472,1488,1504,1520,1536,1552,1568,1584,1600,1616,1632,1648,1664,1680,1696,1712,1728,1744,1760,1776,1792,1808,1824,1840,1856,1872,1888,1904,1920,1936,1952,1968,1984,2000,2016,2032,2048"

# 40 sequence points — same pattern as TOKENS_GRID but capped at 256
# (= paper's max_num_seqs).
SEQUENCES_GRID="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,28,32,36,40,44,48,52,56,60,64,80,96,112,128,144,160,176,192,208,224,240,256"

# 20 prefill_chunk points — preserved exactly from paper's mixture of
# powers of 2 and irregular intermediates (16, 24, 32, 36, 54, ...).
PREFILL_GRID="0,16,24,32,36,54,64,81,122,128,182,256,273,410,512,615,923,1024,1384,2048"

# 17 kv_prefill points — paper had 20 going to 16384; we drop entries > 8192.
KV_PREFILL_GRID="0,16,32,64,128,256,512,768,1024,1152,1728,2048,2592,3888,4096,5832,8192"

# 10 n_decode points — identical to paper.
DECODE_N_GRID="0,1,2,4,8,16,32,64,128,256"

# 17 kv_decode points — same as KV_PREFILL_GRID (paper used the same axis
# both for prefill and decode KV; we mirror that).
KV_DECODE_GRID="0,16,32,64,128,256,512,768,1024,1152,1728,2048,2592,3888,4096,5832,8192"

# =============================================================================
# Execute
# =============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo "  profile_paper_grid.sh — Inferentia 2 sweep with paper grid"
echo "============================================================"
echo "  output_root : $OUTPUT_ROOT"
echo "  max_seq_len : $MAX_SEQ_LEN"
echo "  dtype       : $DTYPE"
echo "  warmup      : $WARMUP   repeat : $REPEAT"
echo
echo "  tokens_grid    : 152 points (1..2048)"
echo "  sequences_grid :  40 points (1..256)"
echo "  prefill_grid   :  20 points (0..2048)"
echo "  kv_prefill     :  17 points (0..8192)"
echo "  decode_n_grid  :  10 points (0..256)"
echo "  kv_decode      :  17 points (0..8192)"
echo "  ~1550 shots × $((WARMUP + REPEAT)) forwards ≈ $((1550 * (WARMUP + REPEAT))) per (model, TP)"
echo "  Estimate: 4-5h first run / TP, minutes on rerun"
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
        --max-num-batched-tokens 2048 \
        --max-num-seqs 256 \
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
