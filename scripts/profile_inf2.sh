#!/usr/bin/env bash
# scripts/profile_inf2.sh — sweep three target models on Inferentia 2
# using the **LLMServingSim 2.0 paper's profiling grids** (RTXPRO6000
# Llama-3.1-8B/bf16/tp1 reference bundle), with the following exclusions:
#
#   - skew sweep                    (heterogeneous-decode batches; not used)
#   - chunked prefill               (kv_prefill collapsed to {0})
#   - PD disaggregation             (no mixed prefill+decode shots)
#   - mixed regime                  (paper measures pc × kv_p × n × kv_d
#                                     ≈ 19 k cells; we keep only the two
#                                     pure regions)
#
# Scenario tweaks vs. paper (max_num_seqs = 32 in our setup):
#
#   sequences  : capped at 32      (paper: up to 256)
#   n_decode   : capped at 32      (paper: up to 256)
#   tokens     : extended to 8192  (paper: 2048; LENS scenarios reach
#                                    ~8 k iter tokens with 4 k prefill)
#   prefill_chunk : extended to 8192  (paper: 2048; attention is O(pc²)
#                                       so extrapolation isn't safe — we
#                                       must measure)
#
# Final grids:
#   tokens           157 points (paper 152 + 5 sparse extension to 8192)
#   sequences         20 points (1..16 step1, 20..32 step4)
#   prefill_chunk     24 points (paper 19 + 5 sparse to 8192)
#   kv_prefill         1 point  (0)
#   n_decode           6 points (1, 2, 4, 8, 16, 32)
#   kv_decode         16 points (paper, up to 8192; the 8192 entry
#                                 auto-skips at runtime because
#                                 kv_d+1 > max_position_embeddings)
#
# Per (model, TP):
#   Llama / Mistral  : ~1239 shots × 40 forwards = ~50 k forward calls
#   Qwen3 14B        : ~1396 shots × 40 forwards = ~56 k forward calls
#   ~1240 NEFF compiles on first run.
#   Wall clock: ~3-6 h on first run, minutes on rerun (cache hit).
#
# Profiling cost vs. LENS (NxDI bucket profiling): ~100x more NEFFs
# (LENS profiles 14 buckets; this sweep emits ~1300 distinct shapes).
# This is the cost of operator-level coverage — the simulator
# interpolates over fine-grained shape grids whereas LENS measures
# bucket-padded end-to-end latency directly. The trade-off is intentional.
#
# Usage (on inf2 with the AWS Neuron DLAMI's pytorch_2_9 venv):
#
#     source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
#     export HF_TOKEN="hf_xxx_..."     # for gated models
#     ./scripts/profile_inf2.sh
#
# Resume / partial: comment out finished MODEL_TPS entries below. Each
# (model, TP) combination is independent. Within a sweep, --reload-every
# (default 30) bounds peak HBM by reloading the model between shot
# batches.

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"      # caps both tokens and kv axes
DTYPE="${DTYPE:-bfloat16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-profiler/perf}"

WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-30}"
RELOAD_EVERY="${RELOAD_EVERY:-30}"      # NEFF count between HBM reloads

# Three target models. Comment out to skip.
# Format: "model_id|tp_csv"
MODEL_TPS=(
    "meta-llama/Llama-3.2-1B|1,2,4,8"
    "mistralai/Mistral-7B-v0.3|1,2,4,8"
    "Qwen/Qwen3-14B|2,4,8"               # TP=1 OOM at full 30 GB; skip
)

# =============================================================================
# Paper grids (extracted from RTXPRO6000/Llama-3.1-8B/bf16/tp1)
# =============================================================================

# 157 token points (paper 152 up to 2048 + sparse extension to 8192)
TOKENS_GRID="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,\
20,24,28,32,36,40,44,48,52,56,60,64,\
80,96,112,128,144,160,176,192,208,224,240,256,\
272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512,\
528,544,560,576,592,608,624,640,656,672,688,704,720,736,752,768,\
784,800,816,832,848,864,880,896,912,928,944,960,976,992,1008,1024,\
1040,1056,1072,1088,1104,1120,1136,1152,1168,1184,1200,1216,1232,1248,1264,1280,\
1296,1312,1328,1344,1360,1376,1392,1408,1424,1440,1456,1472,1488,1504,1520,1536,\
1552,1568,1584,1600,1616,1632,1648,1664,1680,1696,1712,1728,1744,1760,1776,1792,\
1808,1824,1840,1856,1872,1888,1904,1920,1936,1952,1968,1984,2000,2016,2032,2048,\
2560,3072,4096,6144,8192"

# 20 sequence points (paper pattern up to max_num_seqs = 32)
SEQUENCES_GRID="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,28,32"

# 24 prefill_chunk points (paper 19 up to 2048 + sparse extension to 8192;
# attention is O(pc²) so we cannot rely on extrapolation)
PREFILL_GRID="16,24,32,36,54,64,81,122,128,182,256,273,410,512,615,923,\
1024,1384,2048,2560,3072,4096,6144,8192"

# kv_prefill always 0 — chunked prefill is excluded from this profile
KV_PREFILL_GRID="0"

# 6 n_decode points (capped at max_num_seqs = 32)
DECODE_N_GRID="1,2,4,8,16,32"

# 17 kv_decode points (paper, capped at MAX_SEQ_LEN; 8192 entry will be
# auto-skipped by profile_neuron.py since kv_d+1 > max_position_embeddings).
KV_DECODE_GRID="16,32,64,128,256,512,768,1024,1152,1728,2048,2592,3888,4096,5832,8192"

# =============================================================================
# Execute
# =============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Count points (helper for banner)
count_csv() { tr ',' '\n' <<<"$1" | grep -cv '^$'; }

n_tok=$(count_csv "$TOKENS_GRID")
n_seq=$(count_csv "$SEQUENCES_GRID")
n_pc=$(count_csv "$PREFILL_GRID")
n_kvp=$(count_csv "$KV_PREFILL_GRID")
n_nd=$(count_csv "$DECODE_N_GRID")
n_kvd=$(count_csv "$KV_DECODE_GRID")

n_dense_llama=$(( 7 * n_tok ))            # 7 layers × tokens
n_dense_qwen3=$(( 8 * n_tok ))            # +qk_norm
n_perseq=$(( 1 * n_seq ))
n_attn_pre=$(( n_pc * n_kvp ))
n_attn_dec=$(( n_nd * n_kvd ))
shots_llama=$(( n_dense_llama + n_perseq + n_attn_pre + n_attn_dec ))
shots_qwen3=$(( n_dense_qwen3 + n_perseq + n_attn_pre + n_attn_dec ))

echo "============================================================"
echo "  profile_inf2.sh — Inferentia 2 sweep, paper-grade grids"
echo "============================================================"
echo "  output_root  : $OUTPUT_ROOT"
echo "  max_seq_len  : $MAX_SEQ_LEN"
echo "  dtype        : $DTYPE"
echo "  warmup       : $WARMUP   repeat : $REPEAT   reload_every : $RELOAD_EVERY"
echo
echo "  grids (paper bundle + scenario tweaks for max_num_seqs=32):"
echo "    tokens           : $n_tok pts (1..8192; paper 152 + 5 ext)"
echo "    sequences        : $n_seq pts (1..32; capped from paper 256)"
echo "    prefill_chunk    : $n_pc pts  (16..8192; paper 19 + 5 ext)"
echo "    kv_prefill       : $n_kvp pt  (0 — chunked prefill excluded)"
echo "    n_decode         : $n_nd pts  (1..32; capped from paper 256)"
echo "    kv_decode        : $n_kvd pts (16..8192; paper)"
echo
echo "  shots/(model, TP):"
echo "    Llama / Mistral  = ${shots_llama} (7 dense × ${n_tok} + ${n_perseq} per_seq + ${n_attn_pre} attn_pre + ${n_attn_dec} attn_dec)"
echo "    Qwen3            = ${shots_qwen3} (8 dense × ${n_tok} + …)"
echo
echo "  forward calls/(model, TP) ≈ shots × $((WARMUP + REPEAT))"
echo "    Llama / Mistral  ≈ $((shots_llama * (WARMUP + REPEAT)))"
echo "    Qwen3            ≈ $((shots_qwen3 * (WARMUP + REPEAT)))"
echo
echo "  Estimated wall clock (first run, with NEFF compile):"
echo "    ~3-7 h per (model, TP); subsequent runs minutes (compile cache)."
echo "    Compile cost ~100x LENS (NxDI bucketing); intentional trade-off"
echo "    for operator-level coverage."
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
        --max-num-seqs 256 \
        --tokens-grid "$TOKENS_GRID" \
        --sequences-grid "$SEQUENCES_GRID" \
        --prefill-grid "$PREFILL_GRID" \
        --kv-prefill-grid "$KV_PREFILL_GRID" \
        --decode-n-grid "$DECODE_N_GRID" \
        --kv-decode-grid "$KV_DECODE_GRID" \
        --warmup "$WARMUP" \
        --repeat "$REPEAT" \
        --reload-every "$RELOAD_EVERY"
done

echo
echo "============================================================"
echo "  Done. Profile bundles written under $OUTPUT_ROOT/Inferentia2/"
echo "  Inspect timing: python scripts/show_profile_timing.py \\"
echo "      \$OUTPUT_ROOT/Inferentia2/<model>/bf16"
echo "============================================================"
