#!/usr/bin/env bash
# scripts/profile_lens_aligned.sh — LENS-aligned profile sweep
#
# Targets the same (input_len, output_len, batch_size=32) measurement
# points the LENS profiler uses (~/Desktop/npu_chip_project/LENS/
# inference_profiling/inf2/profile_min14.csv, 14 combos = 7 prefill
# buckets × 2 ol values (min, max)), so the LLMServingSim simulator
# can predict each LENS combo's batch e2e latency at exactly the
# (il, ol) points LENS measures.
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
# LENS profile_min14.csv defines 14 (il, ol) combos = 7 prefill buckets
# × 2 ol values (min and max for that bucket). Each combo's decode
# trajectory visits kv_d ∈ {il, il+1, ..., il+ol-2}; the simulator's
# kv_decode lookup needs anchors at start (il) and end (il+ol) of every
# combo's trajectory to keep linear-interp accurate without measuring
# every integer in between.
#
# The 14 combos:
#   ( 64,    5), ( 64,   60)
#   (130,   10), (130,  120)
#   (260,   20), (260,  240)
#   (520,   40), (520,  480)
#   (1030,  80), (1030, 960)
#   (2050, 160), (2050,1950)
#   (4100, 300), (4100,3950)

# 7 unique input lengths
LENS_ILS="64,130,260,520,1030,2050,4100"

# 14 unique (il+ol) decode trajectory endpoints. Each LENS combo's
# decode iters traverse kv_d from il (start) up to il+ol-1 (last iter)
# — we use il+ol as the upper anchor to bracket the full trajectory.
LENS_TRAJECTORY_ENDS="69,124,140,250,280,500,560,1000,1110,1990,2210,4000,4400,8050"

# dense.tokens: LENS_ILS ∪ {32} ∪ {1}.
#   * LENS ils (7) — every prefill iter at LENS's ctx_batch_size=1 visits one
#   * 32           — every batched decode iter has 32 dense tokens
#   * 1            — endpoint anchor for small extrapolation; cheap to add
TOKENS_GRID="1,32,${LENS_ILS}"

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

# attn n_decode: powers of 2 mirroring SEQUENCES_GRID. LENS itself only
# exercises n_decode=32, but matching the per_sequence axis keeps the
# attn batch dim symmetric and lets the simulator score smaller batches
# without interpolation error. Override with DECODE_N_GRID=32 for strict
# LENS-only.
#
# kv_decode = LENS_ILS (decode trajectory starts) ∪ LENS_TRAJECTORY_ENDS
# (decode trajectory endpoints). Each LENS combo's decode integration
# (sum of attn cost over kv_d=il..il+ol-1) is then bracketed by an
# anchor at the trajectory start AND the trajectory end — linear interp
# is exact at the endpoints and (since decode cost ≈ a + b·kv_d, memory
# bound) accurate at every iter in between.
#
# 21 unique kv_d points after sorting:
#   64, 69, 124, 130, 140, 250, 260, 280, 500, 520, 560,
#   1000, 1030, 1110, 1990, 2050, 2210, 4000, 4100, 4400, 8050
DECODE_N_GRID="${DECODE_N_GRID:-1,2,4,8,16,32}"
KV_DECODE_GRID="${KV_DECODE_GRID:-${LENS_ILS},${LENS_TRAJECTORY_ENDS}}"

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
echo "  tokens_grid    : $n_tokens points  ({1, 32} ∪ LENS_ILS)"
echo "  sequences_grid : $n_seqs points    ($SEQUENCES_GRID)"
echo "  prefill_grid   : $n_pc points     (LENS il values: $LENS_ILS)"
echo "  kv_prefill     : $n_kp point      (0 only; chunked prefill disabled)"
echo "  decode_n_grid  : $n_n points     ($DECODE_N_GRID)"
echo "  kv_decode      : $n_kd points     (LENS_ILS ∪ LENS_TRAJECTORY_ENDS)"
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
