#!/usr/bin/env bash
# scripts/profile_inf2_v2.sh — Inf2 profile v2 sweep (chain_v2 + extras).
#
# Replaces scripts/profile_chain_v2.sh + a separate extras run with a
# single per-(model, tp) entry point. Output layout matches what the
# simulator expects:
#
#   profiler/perf_chain_v2/Inferentia2/<MODEL>/<VARIANT>/tp<TP>/
#     dense.csv          chain_v2 (7 layers × L)
#                        + extras dense_extras (embedding + final_layernorm × N)
#     attention.csv      chain_v2 (prefill: pc>0, others=0)
#                        + extras attn_decode (decode: pc=0, n>=1, kvd>0)
#     per_sequence.csv   extras per_sequence (lm_head + sampler × S)
#
# Stage order matters:
#   1. chain_v2          — writes CSV headers (fresh) then per-L rows
#   2. extras dense_extras       — appends embedding/final_layernorm
#   3. extras per_sequence       — creates per_sequence.csv (new file)
#   4. extras attn_decode        — appends decode rows to attention.csv
#
# Subprocess split: each stage is a SEPARATE python invocation so Neuron
# Runtime fully releases HBM between stages (Python-level del+gc+sync
# does NOT release Inf2 HBM, only process exit does — see
# profile_inf2.sh::RELOAD_EVERY note).
#
# chain_v2 sweep structure (Inf2 HBM constraint at tp=1, L_max=4736):
#   chunk 0:     L=1                            (writes header)
#   chunks 1-4:  L=64..2048 in 8-cfg chunks     (--append)
#   singles:     L=2112..L_MAX, 1 cfg/process   (--append)
# At tp>1, per-rank dims are 1/TP smaller → larger L fits, so L_MAX
# can be safely pushed past 4736 (try 8192 first on tp=2).
#
# Usage (on inf2 with the AWS Neuron DLAMI pytorch_2_9 venv):
#
#     source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
#     export HF_TOKEN="hf_xxx_..."     # if model is gated
#
#     # defaults: MODEL=Llama-3.2-1B-Instruct, TP=1, L_MAX=4736
#     ./scripts/profile_inf2_v2.sh
#
#     # tp=2 sweep (inf2.xlarge / inf2.8xlarge — 1 chip × 2 NeuronCores):
#     TP=2 L_MAX=8192 ./scripts/profile_inf2_v2.sh
#
#     # other model / tp:
#     MODEL=mistralai/Mistral-7B-v0.3 TP=4 ./scripts/profile_inf2_v2.sh
#
# Resume / range extension:
#   RESUME=1 RESUME_FROM=<L> [EXTRAS_N_MIN=<N>] L_MAX=<new_max> \
#       ./scripts/profile_inf2_v2.sh
#
#   - Skips output-dir cleanup
#   - Skips chain_v2's small-L stages (L=1, small chunks); only runs
#     singles from RESUME_FROM to L_MAX with --append
#   - extras dense_extras runs from EXTRAS_N_MIN (default RESUME_FROM)
#     to L_MAX  (appends; user must NOT overlap with existing N range)
#   - extras per_sequence + attn_decode SKIPPED (rarely need extension;
#     re-run manually if so)
#
# Example: first run reaches L_MAX=4736 successfully, then extend to 8192:
#   TP=2 L_MAX=4736 ./scripts/profile_inf2_v2.sh                 # initial
#   RESUME=1 RESUME_FROM=4800 L_MAX=8192 TP=2 \
#       ./scripts/profile_inf2_v2.sh                              # extension

set -euo pipefail

# =============================================================================
# Configuration (override via env)
# =============================================================================

MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
TP="${TP:-1}"
VARIANT="${VARIANT:-bf16}"
HW="${HW:-Inferentia2}"

# chain_v2 L sweep
L_MAX="${L_MAX:-4736}"            # tp=1 OOM ceiling. tp>=2: try 8192.
L_STEP="${L_STEP:-64}"
SMALL_CHUNK="${SMALL_CHUNK:-512}" # small-L chunk width (8 cfgs at step 64)
SINGLES_START="${SINGLES_START:-2112}"  # L >= this → 1 cfg / process

# extras dense_extras + per_sequence sweep
N_MIN="${N_MIN:-64}"
N_MAX="${N_MAX:-$L_MAX}"
N_STEP="${N_STEP:-64}"
S_MAX="${S_MAX:-32}"              # per_sequence: N capped at max_num_seqs
S_STEP="${S_STEP:-2}"

# extras attn_decode sweep
BS_LIST="${BS_LIST:-1,2,4,8,16,32}"
KV_LIST="${KV_LIST:-32,64,128,256,512,1024,2048,4096}"

# misc
WARMUP="${WARMUP:-3}"
REPEAT="${REPEAT:-5}"
NEURON_CACHE_DIR="${NEURON_CACHE_DIR:-/var/tmp/neuron-compile-cache}"

# Resume mode (range extension on top of an existing OUT_DIR)
RESUME="${RESUME:-0}"
RESUME_FROM="${RESUME_FROM:-}"           # required when RESUME=1: starting L for chain_v2 singles
EXTRAS_N_MIN="${EXTRAS_N_MIN:-${RESUME_FROM}}"   # extras dense_extras starting N (default = RESUME_FROM)

if [[ "$RESUME" == "1" && -z "$RESUME_FROM" ]]; then
    echo "ERROR: RESUME=1 requires RESUME_FROM (starting L for chain_v2 extension)." >&2
    exit 1
fi

# =============================================================================
# Resolve paths
# =============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="profiler/perf_chain_v2/${HW}/${MODEL}/${VARIANT}"
OUT_DIR="${OUT_ROOT}/tp${TP}"

# =============================================================================
# Banner
# =============================================================================

echo "============================================================"
echo "  profile_inf2_v2.sh — chain_v2 + extras sweep"
echo "============================================================"
echo "  model        : $MODEL"
echo "  tp           : $TP   variant : $VARIANT   hw : $HW"
echo "  out_dir      : $OUT_DIR"
echo
echo "  chain_v2 L   : 1, $L_STEP..$L_MAX  (step $L_STEP, small chunks $SMALL_CHUNK, singles from $SINGLES_START)"
echo "  extras N     : $N_MIN..$N_MAX  step $N_STEP   (dense_extras)"
echo "  extras S     : $S_STEP..$S_MAX  step $S_STEP   (per_sequence)"
echo "  extras bs×kv : [$BS_LIST] × [$KV_LIST]   (attn_decode)"
echo "  warmup/repeat: $WARMUP / $REPEAT"
echo
if [[ -d "$NEURON_CACHE_DIR" ]]; then
    cs=$(du -sh "$NEURON_CACHE_DIR" 2>/dev/null | cut -f1 || echo "?")
    cn=$(find "$NEURON_CACHE_DIR" -name "model.neff" 2>/dev/null | wc -l | tr -d ' ')
    echo "  neuron cache : $NEURON_CACHE_DIR  (${cs}, ${cn} neff)"
else
    echo "  neuron cache : $NEURON_CACHE_DIR  (not present)"
fi
echo "  Clear cache for cold-compile cost: rm -rf $NEURON_CACHE_DIR"
echo
echo "  Watch HBM live (separate terminal): neuron-top"
echo
echo "Press Ctrl-C in 3s to abort..."
sleep 3

# =============================================================================
# Cleanup output dir (skip in RESUME mode)
# =============================================================================

if [[ "$RESUME" == "0" ]]; then
    if [[ -d "$OUT_DIR" ]]; then
        echo "[clean] removing existing $OUT_DIR"
        rm -rf "$OUT_DIR"
    fi
else
    if [[ ! -d "$OUT_DIR" ]]; then
        echo "ERROR: RESUME=1 but $OUT_DIR doesn't exist." >&2
        exit 1
    fi
    echo "[resume] keeping existing $OUT_DIR; extending from L=$RESUME_FROM"
    echo "  existing files:" && wc -l "$OUT_DIR"/*.csv 2>/dev/null || true
fi

# =============================================================================
# Stage 1: chain_v2 — decoder block prefill sweep over L
# =============================================================================

echo
echo "############################################################"
echo "  [1/4] chain_v2 — LlamaDecoderLayer prefill sweep"
echo "############################################################"

run_chain() {
    local Lmin=$1 Lmax=$2 append=${3:-}
    echo
    echo "  --- L=$Lmin..$Lmax  ${append:+(append)}"
    python scripts/profile_chain_v2.py \
        --model "$MODEL" --hw "$HW" --variant "$VARIANT" --tp "$TP" \
        --L-min "$Lmin" --L-max "$Lmax" --L-step "$L_STEP" \
        --warmup "$WARMUP" --repeat "$REPEAT" \
        $append
}

if [[ "$RESUME" == "0" ]]; then
    # chunk 0: L=1 (fresh — writes CSV headers)
    run_chain 1 1

    # small L: $SMALL_CHUNK-token chunks (8 cfgs at step 64)
    end=$SMALL_CHUNK
    while (( end <= 2048 && end <= L_MAX )); do
        start=$(( end - SMALL_CHUNK + L_STEP ))
        if (( start < L_STEP )); then start=$L_STEP; fi
        run_chain "$start" "$end" --append
        end=$(( end + SMALL_CHUNK ))
    done

    # big L: 1-cfg-per-process from SINGLES_START to L_MAX
    if (( L_MAX >= SINGLES_START )); then
        for L in $(seq "$SINGLES_START" "$L_STEP" "$L_MAX"); do
            run_chain "$L" "$L" --append
        done
    fi
else
    # RESUME: only singles from RESUME_FROM to L_MAX (all --append)
    echo "[resume] chain_v2 singles L=$RESUME_FROM..$L_MAX (1-cfg/process)"
    for L in $(seq "$RESUME_FROM" "$L_STEP" "$L_MAX"); do
        run_chain "$L" "$L" --append
    done
fi

# =============================================================================
# Stage 2: extras dense_extras — embedding + final_layernorm
#   RESUME mode: start from EXTRAS_N_MIN (skip already-profiled range),
#   no --include-N1 (already written in initial run).
# =============================================================================

echo
echo "############################################################"
echo "  [2/4] extras dense_extras — embedding + final_layernorm"
echo "############################################################"

if [[ "$RESUME" == "0" ]]; then
    python scripts/profile_extras.py \
        --model "$MODEL" --hw "$HW" --variant "$VARIANT" --tp "$TP" \
        --mode dense_extras \
        --N-min "$N_MIN" --N-max "$N_MAX" --N-step "$N_STEP" --include-N1 \
        --warmup "$WARMUP" --repeat "$REPEAT"
else
    echo "[resume] dense_extras N=$EXTRAS_N_MIN..$N_MAX (append; no N=1)"
    python scripts/profile_extras.py \
        --model "$MODEL" --hw "$HW" --variant "$VARIANT" --tp "$TP" \
        --mode dense_extras \
        --N-min "$EXTRAS_N_MIN" --N-max "$N_MAX" --N-step "$N_STEP" \
        --warmup "$WARMUP" --repeat "$REPEAT"
fi

# =============================================================================
# Stage 3: extras per_sequence — lm_head + sampler
#   RESUME mode: SKIPPED (S range 1..32 already covered in initial run;
#   re-run manually if extension needed).
# =============================================================================

if [[ "$RESUME" == "0" ]]; then
    echo
    echo "############################################################"
    echo "  [3/4] extras per_sequence — lm_head + sampler"
    echo "############################################################"

    python scripts/profile_extras.py \
        --model "$MODEL" --hw "$HW" --variant "$VARIANT" --tp "$TP" \
        --mode per_sequence \
        --N-min "$S_STEP" --N-max "$S_MAX" --N-step "$S_STEP" --include-N1 \
        --warmup "$WARMUP" --repeat "$REPEAT"
else
    echo
    echo "  [3/4] per_sequence — SKIPPED in RESUME mode"
fi

# =============================================================================
# Stage 4: extras attn_decode — batched-decode SDPA
#   RESUME mode: SKIPPED (bs × kv grid already covered; re-run manually
#   if you want to extend kv beyond the initial KV_LIST).
# =============================================================================

if [[ "$RESUME" == "0" ]]; then
    echo
    echo "############################################################"
    echo "  [4/4] extras attn_decode — batched-decode SDPA"
    echo "############################################################"

    python scripts/profile_extras.py \
        --model "$MODEL" --hw "$HW" --variant "$VARIANT" --tp "$TP" \
        --mode attn_decode \
        --bs-list "$BS_LIST" --kv-list "$KV_LIST" \
        --warmup "$WARMUP" --repeat "$REPEAT"
else
    echo
    echo "  [4/4] attn_decode — SKIPPED in RESUME mode"
fi

# =============================================================================
# Done
# =============================================================================

echo
echo "============================================================"
echo "  done — output: $OUT_DIR"
echo "============================================================"
wc -l "$OUT_DIR"/*.csv
