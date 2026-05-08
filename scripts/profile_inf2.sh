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
# Scenario tweaks vs. paper (max_num_seqs = 32, compile cost reduction):
#
#   sequences  : capped at 32      (paper: up to 256)
#   n_decode   : capped at 32      (paper: up to 256)
#   tokens     : 4x-coarsened step + extended to 8192 (paper had
#                step 1/4/16; ours step 4/16/64. Token-linear GEMM
#                interpolates well from sparse measurements, and dense
#                makes up the bulk of compile time.)
#   prefill_chunk : extended to 8192 (paper: 2048; attention is O(pc²)
#                                      so extrapolation isn't safe.)
#
# Final grids:
#   tokens            43 points (paper 152 → 4x step coarsened to 38
#                                 in [4..2048] + 5 sparse pts to 8192)
#   sequences         20 points (1..16 step1, 20..32 step4)
#   prefill_chunk     24 points (paper 19 + 5 sparse to 8192)
#   kv_prefill         1 point  (0)
#   n_decode           6 points (1, 2, 4, 8, 16, 32)
#   kv_decode         16 points (paper, up to 8192; the 8192 entry
#                                 auto-skips at runtime because
#                                 kv_d+1 > max_position_embeddings)
#
# Per (model, TP):
#   Llama / Mistral  : 7×43 + 20 + 24 + 96 =  441 shots
#   Qwen3 14B        : 8×43 + 20 + 24 + 96 =  484 shots
#   ~440-480 NEFF compiles on first run.
#   Wall clock: ~30-45 min per (model, TP) at ~2 s/shot compile cost.
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
RELOAD_EVERY="${RELOAD_EVERY:-200}"     # NEFF count between HBM reloads
                                        # (each NEFF ~20-40 MB; HBM is 16 GB/core
                                        # so 200 NEFFs ≈ 4-8 GB headroom — safe.
                                        # Lower if Qwen3 14B OOMs at TP<=2.)

# Cold-cache vs. hot-cache measurement.
#
# This script does NOT touch the Neuron compile cache. Cache clearing
# is a destructive operation (hours of compile work can vanish), so it
# is left to the user to trigger explicitly when they want pure
# first-time compile cost numbers in profile_timing.json::compile_us.
#
# To start from cold (= measure true compile cost across the sweep):
#
#     rm -rf /var/tmp/neuron-compile-cache
#
# To inspect cache state without clearing:
#
#     du -sh /var/tmp/neuron-compile-cache
#     find /var/tmp/neuron-compile-cache -name "model.neff" | wc -l
#
# The script just records the cache directory path in the banner so the
# user can decide. compile_us is captured per shot regardless — when
# cache is hot, compile_us reads near-zero (only disk reload).
NEURON_CACHE_DIR="${NEURON_CACHE_DIR:-/var/tmp/neuron-compile-cache}"

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

# 43 token points: paper grid coarsened by 4x step + sparse extension.
# Compile cost dominates the sweep (~89% wall in cold-cache run), and a
# 4x step is plenty for token-linear dense GEMM interpolation.
#
#   paper step pattern:   1 (in [1,16]) → 4 (in [20,64]) → 16 (in [80,2048])
#   our 4x-coarsened:     4 (in [4,16]) → 16 (in [32,64]) → 64 (in [128,2048])
#                         + 5 sparse pts to 8192
#
# Going below tokens=4 is rarely useful: pure-decode iteration with
# n_decode=1 is the only case, and that's covered by per_seq sweep at
# sequences=1 anyway. Drop tokens={1,2,3} to save 3 × 7 layer = 21 shots.
TOKENS_GRID="4,8,12,16,32,48,64,\
128,192,256,320,384,448,512,576,640,704,768,832,896,960,1024,\
1088,1152,1216,1280,1344,1408,1472,1536,1600,1664,1728,1792,1856,1920,1984,2048,\
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

# Auto-detect cache state to choose a default RUN_TAG so cold and hot
# runs save into separate profile_timing files. User can override via
# RUN_TAG=<anything>.
if [[ -d "$NEURON_CACHE_DIR" ]]; then
    cache_size=$(du -sh "$NEURON_CACHE_DIR" 2>/dev/null | cut -f1 || echo "?")
    cache_neffs=$(find "$NEURON_CACHE_DIR" -name "model.neff" 2>/dev/null | wc -l | tr -d ' ')
else
    cache_size="0B"
    cache_neffs=0
fi
if [[ "$cache_neffs" -eq 0 ]]; then
    DETECTED_STATE="cold"
else
    DETECTED_STATE="hot"
fi
RUN_TAG="${RUN_TAG:-$DETECTED_STATE}"

echo "  neuron cache : $NEURON_CACHE_DIR  (${cache_size}, ${cache_neffs} neff)"
echo "  detected     : $DETECTED_STATE   →  run_tag : $RUN_TAG"
echo "  output       : profile_timing_${RUN_TAG}.json (per (model, variant))"
if [[ "$DETECTED_STATE" = "hot" ]]; then
    echo "                 hot cache → compile_us ≈ disk reload (not true compile)"
    echo "                 to measure true compile cost, manually:"
    echo "                   rm -rf $NEURON_CACHE_DIR"
    echo "                 then set RUN_TAG=cold (or just rerun — auto-detected)"
fi
echo
echo "  grids (paper grids + scenario tweaks for max_num_seqs=32"
echo "         + 4x-coarsened tokens to bound compile budget):"
echo "    tokens           : $n_tok pts (4..8192; 4x step from paper)"
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
        --reload-every "$RELOAD_EVERY" \
        --run-tag "$RUN_TAG"
done

echo
echo "============================================================"
echo "  Done. Profile bundles written under $OUTPUT_ROOT/Inferentia2/"
echo "  Inspect timing: python scripts/show_profile_timing.py \\"
echo "      \$OUTPUT_ROOT/Inferentia2/<model>/bf16"
echo "============================================================"
