#!/usr/bin/env bash
# scripts/profile_lens_aligned.sh — LENS-aligned profile sweep
#
# Targets the same (input_len, output_len, batch_size=32) measurement
# points the LENS profiler uses (~/Desktop/npu_chip_project/LENS/
# inference_profiling/inf2/profile.csv, 60 combos), so the LLMServingSim
# simulator can predict each LENS combo's batch e2e latency with zero
# interpolation error at the LENS measurement points.
#
# Mapping
# -------
# LENS measures uniform batch=32 of (il, ol), with NxDI running
# ctx_batch_size=1 (sequential prefill) + batched decode. So one
# combo executes:
#
#   * 32x sequential prefill iters:
#       dense @ tokens=il    | attn @ (pc=il, kp=0, n=0, kd=0)
#       per_seq @ sequences=1 (lm_head emits 1st decode token)
#
#   * (ol-1) batched decode iters:
#       dense @ tokens=32    | attn @ (pc=0, kp=0, n=32, kd=il+k)  for k in 0..ol-2
#       per_seq @ sequences=32
#
# To predict LENS measurements, the simulator must look up:
#   * dense.tokens at every distinct LENS il + 32 (decode batch)
#   * per_sequence.sequences at {1, 32}
#   * attn pure prefill at every distinct LENS il (kv_p = 0)
#   * attn pure decode at n=32 over kv_d range [min_il=64, max_il+max_ol=8050]
#
# Per (model, TP): ~200 shots × 40 forwards ≈ 8000 forward calls.
# Wall clock: 30-60 min on first run (NEFF compile cache populating),
# minutes on rerun (cache hit), with periodic reloads to bound HBM.
#
# Usage (on inf2 with the AWS Neuron DLAMI's pytorch_2_9 venv):
#
#     source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
#     export HF_TOKEN="hf_xxx_..."
#     ./scripts/profile_lens_aligned.sh
#
# Profile bundles land at  profiler/perf/Inferentia2/<MODEL>/bf16-lens/...
# (separate variant from the powers-of-2 sweep at bf16/...) so both
# coexist for compare_static.py runs.

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
DTYPE="${DTYPE:-bfloat16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-profiler/perf}"
VARIANT_LABEL="${VARIANT_LABEL:-bf16-lens}"   # distinguishes from bf16 powers-of-2 sweep

WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-30}"
RELOAD_EVERY="${RELOAD_EVERY:-30}"

# Three target models. Same set as profile_inf2.sh.
MODEL_TPS=(
    "meta-llama/Llama-3.2-1B|1,2,4,8"
    "mistralai/Mistral-7B-v0.3|1,2,4,8"
    "Qwen/Qwen3-14B|2,4,8"
)

# =============================================================================
# LENS-aligned grids
# =============================================================================
# LENS profile.csv has these 22 unique input_len (il) values across the
# 60 combos (35 bucket_sweep + 10 cross_bucket + 15 boundary). Output
# from `python -c "import csv; print(sorted({int(r['input_len']) for r in
# csv.DictReader(open('LENS/inference_profiling/inf2/profile.csv'))}))"`
LENS_ILS="64,127,128,129,130,255,256,257,260,511,512,513,520,1023,1024,1025,1030,2047,2048,2049,2050,4100"

# dense.tokens: LENS ils ∪ {32} ∪ {1, 6144, 8192}.
#   * LENS ils — every prefill iter at LENS's ctx_batch_size=1 visits one
#   * 32       — every batched decode iter has 32 dense tokens
#   * 1        — endpoint anchor for small extrapolation; cheap to add
#   * 6144, 8192 — upper anchors. The simulator's batched-prefill scheduler
#     can pack multiple short requests into one iter; per-iter token count
#     reaches up to --max-num-batched-tokens (= 8192 here). Without these
#     anchors the simulator would extrapolate from the (2050, 4100) tail
#     for any batched-prefill iter that carries 4100..8192 dense tokens
#     (typical at LENS combos like (260, *)×32 → 31 reqs × 260 = 8060
#     packed in a single iter). Adding two upper points keeps interpolation
#     tight without exploding shot count.
TOKENS_GRID="1,32,${LENS_ILS},6144,8192"

# per_sequence.sequences: powers of 2 from 1 to 32.
#   * 1 = lm_head at the end of each prefill iter (one new token)
#   * 32 = lm_head + sampler in batched decode iter (LENS B=32)
#   * 2,4,8,16 = filled in so the simulator can also predict static-batch
#     scenarios (compare_static.py mode A/B) at intermediate batch sizes
#     without crude 1↔32 interpolation. Add to ${SEQUENCES_GRID} env var
#     to override (e.g. SEQUENCES_GRID=1,32 for strict LENS-only).
SEQUENCES_GRID="${SEQUENCES_GRID:-1,2,4,8,16,32}"

# attn prefill_chunk: LENS ils. kv_prefill = 0 always (no chunked prefill).
PREFILL_GRID="${LENS_ILS}"
KV_PREFILL_GRID="0"

# attn n_decode: powers of 2 to mirror SEQUENCES_GRID. LENS itself only
# exercises n_decode=32, but matching the per_sequence axis keeps the
# attn batch dim symmetric and lets the simulator score smaller batches
# without interpolation error. Override with DECODE_N_GRID=32 for strict
# LENS-only.
#
# kv_decode covers the range [min_il=64, max_il+max_ol=8050]. Powers of 2
# plus LENS-il anchors so interpolation across each combo's il+k decode
# trajectory is accurate. 8192 is omitted because kv_d+1 must fit in
# max_position_embeddings (the script auto-skips kd=8192 anyway).
DECODE_N_GRID="${DECODE_N_GRID:-1,2,4,8,16,32}"
KV_DECODE_GRID="${KV_DECODE_GRID:-64,128,256,512,1024,2048,4096,4100,8050}"

# =============================================================================
# Execute
# =============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Count grid sizes for the banner.
n_tokens=$(awk -F',' '{print NF}' <<<"$TOKENS_GRID")
n_seqs=$(awk -F',' '{print NF}' <<<"$SEQUENCES_GRID")
n_pc=$(awk -F',' '{print NF}' <<<"$PREFILL_GRID")
n_kp=$(awk -F',' '{print NF}' <<<"$KV_PREFILL_GRID")
n_n=$(awk -F',' '{print NF}' <<<"$DECODE_N_GRID")
n_kd=$(awk -F',' '{print NF}' <<<"$KV_DECODE_GRID")

echo "============================================================"
echo "  profile_lens_aligned.sh — LENS-aligned Inferentia 2 sweep"
echo "============================================================"
echo "  output_root : $OUTPUT_ROOT"
echo "  variant     : $VARIANT_LABEL"
echo "  max_seq_len : $MAX_SEQ_LEN"
echo "  dtype       : $DTYPE"
echo "  warmup      : $WARMUP   repeat : $REPEAT   reload-every : $RELOAD_EVERY"
echo
echo "  tokens_grid    : $n_tokens points  ($TOKENS_GRID)"
echo "  sequences_grid : $n_seqs points    ($SEQUENCES_GRID)"
echo "  prefill_grid   : $n_pc points     (LENS il values)"
echo "  kv_prefill     : $n_kp point      (0 only; chunked prefill disabled)"
echo "  decode_n_grid  : $n_n point       (32; LENS batch_size)"
echo "  kv_decode      : $n_kd points     ($KV_DECODE_GRID)"
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
    echo "  $model  (TP=$tps)  variant=$VARIANT_LABEL"
    echo "############################################################"
    python scripts/profile_neuron.py \
        --model "$model" \
        --tp "$tps" \
        --output-root "$OUTPUT_ROOT" \
        --variant "$VARIANT_LABEL" \
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
        --repeat "$REPEAT" \
        --reload-every "$RELOAD_EVERY"
done

echo
echo "============================================================"
echo "  Done. LENS-aligned bundles:"
echo "    $OUTPUT_ROOT/Inferentia2/<model>/$VARIANT_LABEL/"
echo
echo "  Use --variant $VARIANT_LABEL when invoking the simulator,"
echo "  or compare_static.py with --variant flag wired through."
echo "============================================================"
