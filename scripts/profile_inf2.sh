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
#   prefill_chunk     23 points (paper 19 + 4 sparse to 6144; 8192
#                                 dropped — its NEFF scratchpad ≈ 8 GB
#                                 alone caused HBM fragmentation OOM)
#   kv_prefill         1 point  (0)
#   n_decode           6 points (1, 2, 4, 8, 16, 32)
#   kv_decode         15 points (paper sans 0; 8192 dropped for the
#                                 same scratchpad reason)
#
# Per (model, TP):
#   Llama / Mistral  : 7×43 + 20 + 23 + 90 =  434 shots
#   Qwen3 14B        : 8×43 + 20 + 23 + 90 =  477 shots
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
REPEAT="${REPEAT:-100}"     # raised from 30 — forward latency is microseconds,
                            # so 100 reps × 1 ms ~= 100 ms per shot extra cost
                            # for noticeably tighter mean estimates.
RELOAD_EVERY="${RELOAD_EVERY:-0}"        # 0 = NEVER reload mid-sweep.
                                        # Counter-intuitive but: Neuron Runtime
                                        # doesn't actually release the OLD model's
                                        # HBM on Python-level del+gc+sync, so a
                                        # mid-sweep reload ends up DOUBLING Model
                                        # Constants on-device (old + new = 2x base
                                        # weights). HBM dumps confirmed this. With
                                        # subprocess splitting per (model, TP,
                                        # stage), the OS reclaims everything on
                                        # process exit anyway — reload mid-sweep
                                        # is pure overhead, set to 0.

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

# 23 prefill_chunk points. Capped at 6144 (was 8192) — attention is
# O(pc²) and the pc=8192 NEFF allocates ~8 GB Shared Scratchpad alone,
# which dominates HBM and triggers fragmentation OOM mid-sweep on Inf2.
# LENS scenarios max input length = 4100, so 6144 has comfortable margin.
PREFILL_GRID="16,24,32,36,54,64,81,122,128,182,256,273,410,512,615,923,\
1024,1384,2048,2560,3072,4096,6144"

# kv_prefill always 0 — chunked prefill is excluded from this profile
KV_PREFILL_GRID="0"

# 6 n_decode points (capped at max_num_seqs = 32)
DECODE_N_GRID="1,2,4,8,16,32"

# 15 kv_decode points (paper sans 0; 8192 dropped — was auto-skipped at
# runtime since kv_d+1 > max_pos=8192, and we now also avoid driving up
# the scratchpad with a near-max kv length).
KV_DECODE_GRID="16,32,64,128,256,512,768,1024,1152,1728,2048,2592,3888,4096,5832"

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

# Cache state is informational only — we do NOT branch on it. The user
# decides when to clear (rm -rf $NEURON_CACHE_DIR) for a true cold-cache
# measurement. Profile-timing JSON output uses auto-numbering (0, 1, 2,
# ...) per variant root so repeated runs never overwrite each other —
# this is robust regardless of cache state, intent, or interruption.
if [[ -d "$NEURON_CACHE_DIR" ]]; then
    cache_size=$(du -sh "$NEURON_CACHE_DIR" 2>/dev/null | cut -f1 || echo "?")
    cache_neffs=$(find "$NEURON_CACHE_DIR" -name "model.neff" 2>/dev/null | wc -l | tr -d ' ')
    echo "  neuron cache : $NEURON_CACHE_DIR  (${cache_size}, ${cache_neffs} neff)"
else
    echo "  neuron cache : $NEURON_CACHE_DIR  (not present)"
fi
# Default RUN_TAG to empty string so set -u + later expansions don't blow up.
# profile_neuron.py treats empty --run-tag as "auto-number".
RUN_TAG="${RUN_TAG:-}"
if [[ -n "$RUN_TAG" ]]; then
    echo "  run_tag      : $RUN_TAG (explicit override; profile_timing_${RUN_TAG}.json)"
else
    echo "  run_tag      : auto (profile_timing_<N>.json — N = first unused integer)"
fi
echo "  to measure pure compile cost, manually clear cache before running:"
echo "    rm -rf $NEURON_CACHE_DIR"
echo
echo "  Watch HBM live (separate terminal):"
echo "    neuron-top    # or: neuron-monitor"
echo "  Process exit (Ctrl-C, kill, OOM) cleans HBM down to 0 GB —"
echo "  the per-(TP, stage) subprocess split below relies on that."
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
    IFS='|' read -r model tps_csv <<<"$spec"
    IFS=',' read -ra tp_array <<<"$tps_csv"

    echo
    echo "############################################################"
    echo "  $model  (TP=$tps_csv)"
    echo "############################################################"

    # Process isolation per (model, TP, stage). Neuron Runtime
    # accumulates HBM across NEFFs AND across model-load calls within
    # one process — neither reload_every nor switching TP inside one
    # python invocation actually releases the previous TP's weights /
    # NEFFs. The only reliable HBM reset is a fresh process. So per
    # (model, TP) we run THREE python invocations, each with a SINGLE
    # TP and a SINGLE stage.

    # Base args common to all subprocesses. Grids are passed PER stage
    # below so we can split attention into prefill-only and decode-only
    # (each is the largest individual NEFF cluster on Inf2 — splitting
    # halves the in-process NEFF count and HBM peak).
    base_args=(
        --model "$model"
        --output-root "$OUTPUT_ROOT"
        --dtype "$DTYPE"
        --max-position-embeddings "$MAX_SEQ_LEN"
        --max-num-batched-tokens 8192
        --max-num-seqs 256
        --tokens-grid "$TOKENS_GRID"
        --sequences-grid "$SEQUENCES_GRID"
        --kv-prefill-grid "$KV_PREFILL_GRID"
        --warmup "$WARMUP"
        --repeat "$REPEAT"
        --reload-every "$RELOAD_EVERY"
    )

    for tp in "${tp_array[@]}"; do
        echo
        echo "  ══════════════ $model  TP=$tp  ══════════════"

        # If user set RUN_TAG explicitly, suffix it per stage so the
        # subprocesses don't overwrite each other's profile_timing.json.
        # If RUN_TAG is empty, profile_neuron's auto-numbering picks
        # unique integers per tp folder.
        if [[ -n "$RUN_TAG" ]]; then
            attn_pre_tag=(--run-tag "${RUN_TAG}_attn_prefill")
            attn_dec_tag=(--run-tag "${RUN_TAG}_attn_decode")
            dense_tag_args=(--run-tag "${RUN_TAG}_dense")
            perseq_tag_args=(--run-tag "${RUN_TAG}_per_seq")
        else
            attn_pre_tag=()
            attn_dec_tag=()
            dense_tag_args=()
            perseq_tag_args=()
        fi

        echo "  >>> [1/4] attention PREFILL only  (TP=$tp, fresh process)"
        python scripts/profile_neuron.py "${base_args[@]}" \
            --tp "$tp" \
            --skip-dense --skip-per-seq \
            --prefill-grid "$PREFILL_GRID" \
            --decode-n-grid "" --kv-decode-grid "" \
            "${attn_pre_tag[@]}"

        echo
        echo "  >>> [2/4] attention DECODE only  (TP=$tp, fresh process)"
        python scripts/profile_neuron.py "${base_args[@]}" \
            --tp "$tp" \
            --skip-dense --skip-per-seq \
            --prefill-grid "" \
            --decode-n-grid "$DECODE_N_GRID" \
            --kv-decode-grid "$KV_DECODE_GRID" \
            "${attn_dec_tag[@]}"

        echo
        echo "  >>> [3/4] dense sweep  (TP=$tp, fresh process)"
        python scripts/profile_neuron.py "${base_args[@]}" \
            --tp "$tp" \
            --skip-attention --skip-per-seq \
            --prefill-grid "$PREFILL_GRID" \
            --decode-n-grid "$DECODE_N_GRID" \
            --kv-decode-grid "$KV_DECODE_GRID" \
            "${dense_tag_args[@]}"

        echo
        echo "  >>> [4/4] per_sequence sweep  (TP=$tp, fresh process)"
        python scripts/profile_neuron.py "${base_args[@]}" \
            --tp "$tp" \
            --skip-attention --skip-dense \
            --prefill-grid "$PREFILL_GRID" \
            --decode-n-grid "$DECODE_N_GRID" \
            --kv-decode-grid "$KV_DECODE_GRID" \
            "${perseq_tag_args[@]}"
    done
done

echo
echo "============================================================"
echo "  Done. Profile bundles written under $OUTPUT_ROOT/Inferentia2/"
echo "  Inspect timing: python scripts/show_profile_timing.py \\"
echo "      \$OUTPUT_ROOT/Inferentia2/<model>/bf16"
echo "============================================================"
