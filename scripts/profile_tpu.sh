#!/usr/bin/env bash
# scripts/profile_tpu.sh — JAX TPU profile sweep (stage-by-stage subprocess).
#
# Mirrors profile_inf2_v2.sh: each stage is a fresh `python profile_jax.py`
# invocation so JAX runtime + jit compilation cache get fully released
# between stages. ipynb-style monolithic sweep accumulates compile cache
# in host RAM until OOM (typically after the dense stage on a 128GB host
# at the user-adjusted full grid).
#
# Output layout (paper v2 schema):
#   profiler/perf/<HW>/<MODEL>/<VARIANT>/tp<TP>/
#     dense.csv                  9 layers × token sweep
#     per_sequence.csv           2 layers × seq sweep
#     attention.csv              single-shot prefill + decode sweep
#                                (chunked prefill OFF, prefix caching OFF)
#     attention_full_stats.csv   paper schema with mean / p50 / p90 / max
#
# Stage subprocess split:
#   1. dense              — 9 subprocesses (one per layer)
#   2. per_seq            — 2 subprocesses (lm_head + sampler)
#   3. attn_prefill       — 1 subprocess
#   4. attn_decode        — 1 subprocess per batch (default 6: 1/2/4/8/16/32)
#   Total: ~18 subprocesses. Each pays ~5-10s JAX import + libtpu init
#   overhead. Total overhead ~2 min, well worth not OOM'ing.
#
# Sharding (paper SHARD_FIELDS — handled inside resolve_mcfg):
#   num_attention_heads / num_key_value_heads / intermediate_size / vocab_size
#   are divided by TP. hidden_size and head_dim stay raw. Per-rank shapes
#   on a single chip — collectives modelled by ASTRA-Sim from cluster
#   config's link_bw, not by the profiler.
#
# Usage:
#   ./scripts/profile_tpu.sh                                       # defaults
#   TP=2 ./scripts/profile_tpu.sh                                  # single TP
#   TP_LIST="1 2 4 8" ./scripts/profile_tpu.sh                     # multi TP
#   MODEL=meta-llama/Llama-3.2-3B-Instruct TP=2 ./scripts/profile_tpu.sh
#
# Environment overrides:
#   MODEL TP VARIANT HW WARMUP REPEAT TP_LIST
#   MAX_TOKENS MAX_PREFILL_CHUNK MAX_DECODE_TOKENS BATCH_LIST
#   HF_TOKEN

set -euo pipefail

# ============================================================
# Configuration (override via env)
# ============================================================

MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
TP_LIST="${TP_LIST:-${TP:-1}}"
VARIANT="${VARIANT:-bf16}"
HW="${HW:-TPU-v6e-1}"
WARMUP="${WARMUP:-5}"
REPEAT="${REPEAT:-30}"

# Grid caps (cell 4 of profile_full_jax.ipynb mirrors these defaults)
MAX_TOKENS="${MAX_TOKENS:-8192}"
MAX_PREFILL_CHUNK="${MAX_PREFILL_CHUNK:-8192}"
MAX_DECODE_TOKENS="${MAX_DECODE_TOKENS:-8192}"
BATCH_LIST="${BATCH_LIST:-1 2 4 8 16 32}"

# Stages to run (space-separated). Default = all 4. Override to resume from a
# crash without redoing earlier stages:
#   STAGES="per_seq attn_prefill attn_decode" ./scripts/profile_tpu.sh
STAGES="${STAGES:-dense per_seq attn_prefill attn_decode}"

# Subprocess chunk sizes (each chunk = one fresh `python profile_jax.py`).
# JAX's jit cache + internal trace state leaks host RAM linearly with unique
# input shapes. Empirical anchor: ~48 unique shapes consumed ~113GB on a
# 173GB host (≈ 2GB / unique shape). Chunk size ~40 keeps each subprocess
# under ~80GB peak. Tune down if you OOM; tune up if RAM is spare.
TOKEN_CHUNK="${TOKEN_CHUNK:-40}"
SEQ_CHUNK="${SEQ_CHUNK:-40}"
PC_CHUNK="${PC_CHUNK:-40}"
KV_CHUNK="${KV_CHUNK:-40}"

# ============================================================
# Paths
# ============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SCRIPT_DIR="$REPO_ROOT/profiler/perf_models/TPU-v6e-1"
JAX_PY="$SCRIPT_DIR/profile_jax.py"

if [[ ! -f "$JAX_PY" ]]; then
    echo "ERROR: $JAX_PY not found" >&2
    exit 1
fi

# Build grids by importing the core module — cd into the script dir so the
# import works (folder name contains a dash → not importable as a package).
build_grids() {
    cd "$SCRIPT_DIR"
    TOKEN_LIST="$(python -c "
from profile_jax_core import token_grid
print(','.join(map(str, token_grid($MAX_TOKENS))))
")"
    SEQ_LIST="$(python -c "
from profile_jax_core import seq_grid
print(','.join(map(str, seq_grid($MAX_TOKENS))))
")"
    PC_LIST="$(python -c "
from profile_jax_core import _prefill_chunk_grid
print(','.join(map(str, _prefill_chunk_grid($MAX_PREFILL_CHUNK))))
")"
    KV_LIST="$(python -c "
from profile_jax_core import _seq_len_grid_kv
print(','.join(map(str, _seq_len_grid_kv($MAX_DECODE_TOKENS))))
")"
    cd "$REPO_ROOT"
}

build_grids

N_TOKENS=$(echo "$TOKEN_LIST" | tr ',' ' ' | wc -w | tr -d ' ')
N_SEQS=$(echo "$SEQ_LIST" | tr ',' ' ' | wc -w | tr -d ' ')
N_PC=$(echo "$PC_LIST" | tr ',' ' ' | wc -w | tr -d ' ')
N_KV=$(echo "$KV_LIST" | tr ',' ' ' | wc -w | tr -d ' ')
N_BATCH=$(echo "$BATCH_LIST" | wc -w | tr -d ' ')

# ============================================================
# Banner
# ============================================================

echo "============================================================"
echo "  profile_tpu.sh — JAX TPU sweep (subprocess per stage)"
echo "============================================================"
echo "  model        : $MODEL"
echo "  tp_list      : $TP_LIST   variant : $VARIANT   hw : $HW"
echo "  warmup/repeat: $WARMUP / $REPEAT"
echo
echo "  dense tokens : $N_TOKENS values (max=$MAX_TOKENS)"
echo "  per_seq seqs : $N_SEQS values"
echo "  prefill pc   : $N_PC values (max=$MAX_PREFILL_CHUNK)"
echo "  decode kv    : $N_KV values × $N_BATCH batches = $((N_KV * N_BATCH)) combos"
echo "  batch list   : [$BATCH_LIST]"
echo
echo "Press Ctrl-C in 3s to abort..."
sleep 3

# ============================================================
# Per-TP sweep
# ============================================================

DENSE_LAYERS=(embedding layernorm qkv_proj rotary_emb o_proj gate_up_proj act_fn down_proj final_layernorm)
PER_SEQ_LAYERS=(lm_head sampler)

# Split a comma-separated list into N-sized chunks. Echoes one chunk per line.
chunk_csv() {
    local csv=$1 size=$2
    local arr
    IFS=',' read -ra arr <<< "$csv"
    local n=${#arr[@]}
    local i j end out
    for ((i=0; i<n; i+=size)); do
        end=$((i + size))
        (( end > n )) && end=$n
        out=""
        for ((j=i; j<end; j++)); do
            [[ -n "$out" ]] && out+=","
            out+="${arr[j]}"
        done
        echo "$out"
    done
}

run_one_tp() {
    local TP=$1
    local OUT_DIR="$REPO_ROOT/profiler/perf/${HW}/${MODEL}/${VARIANT}/tp${TP}"
    mkdir -p "$OUT_DIR"

    echo
    echo "############################################################"
    echo "  TP=$TP — output: $OUT_DIR"
    echo "############################################################"

    local STAGE_T0=$(date +%s)

    # ---- Stage 1: dense (subprocess per layer × token chunk) ----
    if [[ " $STAGES " == *" dense "* ]]; then
    echo
    echo "  [1/4] dense — ${#DENSE_LAYERS[@]} layers × $N_TOKENS tokens  (chunk size $TOKEN_CHUNK)"
    local APPEND=""
    for layer in "${DENSE_LAYERS[@]}"; do
        echo
        echo "    --- $layer ---"
        local chunk_idx=0
        while IFS= read -r chunk; do
            [[ -z "$chunk" ]] && continue
            chunk_idx=$((chunk_idx + 1))
            local nv=$(echo "$chunk" | tr ',' ' ' | wc -w | tr -d ' ')
            echo "      chunk $chunk_idx ($nv values)  ${APPEND:-fresh}"
            cd "$SCRIPT_DIR"
            python profile_jax.py \
                --model "$MODEL" --hw "$HW" --variant "$VARIANT" --tp "$TP" \
                --stage dense --layer "$layer" \
                --token-list "$chunk" \
                --warmup "$WARMUP" --repeat "$REPEAT" \
                --output-dir "$OUT_DIR" \
                $APPEND
            cd "$REPO_ROOT"
            APPEND="--append"
        done < <(chunk_csv "$TOKEN_LIST" "$TOKEN_CHUNK")
    done
    else echo; echo "  [1/4] dense — SKIPPED (not in STAGES)"; fi

    # ---- Stage 2: per_seq (subprocess per layer × seq chunk) ----
    if [[ " $STAGES " == *" per_seq "* ]]; then
    echo
    echo "  [2/4] per_sequence — ${#PER_SEQ_LAYERS[@]} layers × $N_SEQS sequences  (chunk size $SEQ_CHUNK)"
    APPEND=""
    for layer in "${PER_SEQ_LAYERS[@]}"; do
        echo
        echo "    --- $layer ---"
        local chunk_idx=0
        while IFS= read -r chunk; do
            [[ -z "$chunk" ]] && continue
            chunk_idx=$((chunk_idx + 1))
            local nv=$(echo "$chunk" | tr ',' ' ' | wc -w | tr -d ' ')
            echo "      chunk $chunk_idx ($nv values)  ${APPEND:-fresh}"
            cd "$SCRIPT_DIR"
            python profile_jax.py \
                --model "$MODEL" --hw "$HW" --variant "$VARIANT" --tp "$TP" \
                --stage per_seq --layer "$layer" \
                --token-list "$chunk" \
                --warmup "$WARMUP" --repeat "$REPEAT" \
                --output-dir "$OUT_DIR" \
                $APPEND
            cd "$REPO_ROOT"
            APPEND="--append"
        done < <(chunk_csv "$SEQ_LIST" "$SEQ_CHUNK")
    done
    else echo; echo "  [2/4] per_sequence — SKIPPED (not in STAGES)"; fi

    # ---- Stage 3: attn_prefill (subprocess per pc chunk) ----
    if [[ " $STAGES " == *" attn_prefill "* ]]; then
    echo
    echo "  [3/4] attn_prefill — $N_PC single-shot prefill combos  (chunk size $PC_CHUNK)"
    APPEND=""
    local chunk_idx=0
    while IFS= read -r chunk; do
        [[ -z "$chunk" ]] && continue
        chunk_idx=$((chunk_idx + 1))
        local nv=$(echo "$chunk" | tr ',' ' ' | wc -w | tr -d ' ')
        echo "      chunk $chunk_idx ($nv values)  ${APPEND:-fresh}"
        cd "$SCRIPT_DIR"
        python profile_jax.py \
            --model "$MODEL" --hw "$HW" --variant "$VARIANT" --tp "$TP" \
            --stage attn_prefill \
            --pc-list "$chunk" \
            --warmup "$WARMUP" --repeat "$REPEAT" \
            --output-dir "$OUT_DIR" \
            $APPEND
        cd "$REPO_ROOT"
        APPEND="--append"
    done < <(chunk_csv "$PC_LIST" "$PC_CHUNK")
    else echo; echo "  [3/4] attn_prefill — SKIPPED (not in STAGES)"; fi

    # ---- Stage 4: attn_decode (subprocess per batch × kv chunk) ----
    if [[ " $STAGES " == *" attn_decode "* ]]; then
    echo
    echo "  [4/4] attn_decode — $N_BATCH batches × $N_KV kv = $((N_BATCH * N_KV)) combos  (kv chunk size $KV_CHUNK)"
    for batch in $BATCH_LIST; do
        echo
        echo "    --- batch=$batch ---"
        local chunk_idx=0
        while IFS= read -r chunk; do
            [[ -z "$chunk" ]] && continue
            chunk_idx=$((chunk_idx + 1))
            local nv=$(echo "$chunk" | tr ',' ' ' | wc -w | tr -d ' ')
            echo "      chunk $chunk_idx ($nv values)"
            cd "$SCRIPT_DIR"
            python profile_jax.py \
                --model "$MODEL" --hw "$HW" --variant "$VARIANT" --tp "$TP" \
                --stage attn_decode --batch "$batch" \
                --kv-list "$chunk" \
                --warmup "$WARMUP" --repeat "$REPEAT" \
                --output-dir "$OUT_DIR" \
                --append
            cd "$REPO_ROOT"
        done < <(chunk_csv "$KV_LIST" "$KV_CHUNK")
    done
    else echo; echo "  [4/4] attn_decode — SKIPPED (not in STAGES)"; fi

    local STAGE_T1=$(date +%s)
    echo
    echo "  TP=$TP done in $((STAGE_T1 - STAGE_T0))s"
    wc -l "$OUT_DIR"/*.csv
}

for tp in $TP_LIST; do
    run_one_tp "$tp"
done

echo
echo "============================================================"
echo "  all TP done"
echo "============================================================"
