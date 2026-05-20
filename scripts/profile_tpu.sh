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
#
# ============================================================
# 로그 파일 (profiling wall-time 보존용) — 한국어 안내
# ============================================================
#
# 이 스크립트는 stdout/stderr 전체를 `tee` 로 복사해서 아래 경로의 로그
# 파일에 저장한다. profile_jax.py 가 매 subprocess 마다 출력하는
# `[stage=...] done in Ns` 라인과, 본 스크립트가 추가한 stage / TP /
# total 단위의 wall-time 이 모두 이 파일에 남는다.
#
# 로그 파일 경로:
#   profiler/perf/<HW>/<MODEL>/<VARIANT>/log/profile_<YYYYMMDD_HHMMSS>.log
#
# 같은 디렉토리 구조 안에 `meta.yaml`, `tp1/`, `tp2/`, … 와 나란히
# `log/` 폴더가 생긴다. 매 실행마다 timestamp 가 다른 새 파일이
# 만들어지므로 이전 실행의 로그는 그대로 보존된다.
#
# 로그에 포함되는 정보:
#   1) Banner — 이번 실행에 사용된 모든 환경 변수 / config 값
#      (MODEL, TP_LIST, VARIANT, HW, STAGES, WARMUP/REPEAT, MAX_TOKENS,
#       MAX_PREFILL_CHUNK, MAX_DECODE_TOKENS, TOKEN_CHUNK / SEQ_CHUNK /
#       PC_CHUNK / KV_CHUNK, BATCH_LIST) 와 각 stage 의 grid 개수
#      (예: "dense: 9 layers × 168 tokens (~5 chunks/layer)").
#   2) Per-subprocess timing — profile_jax.py 한 번 호출당 1줄
#      ("[stage=dense] done in 87s" 같은 형태).
#   3) Per-stage 합계 — "[1/4] dense done in Ns" 처럼 4 stage 별로 한 줄.
#   4) Per-TP 합계 — "TP=1 done in Ns".
#   5) 전체 wall-time — 스크립트 마지막의 "all TP done in Ns".
#
# 로그에서 정보 빠르게 뽑기:
#   grep "all TP done"          log/profile_*.log    # 전체 시간
#   grep -E "\[[0-9]/4\] .* done in" log/profile_<ts>.log  # stage 별 합계
#   grep "done in"              log/profile_<ts>.log  # 모든 timing 라인
#   head -25                    log/profile_<ts>.log  # 어떤 config 로 돌렸는지
#
# 주의사항:
#   * 로그 파일은 매 실행마다 새로 생기고 자동 삭제되지 않는다. 한
#     변형(<HW>/<MODEL>/<VARIANT>) 을 반복해서 돌리면 log/ 안에 파일이
#     계속 쌓이므로 주기적으로 정리할 것. .gitignore 에 `log/` 가
#     포함되어 있는지도 확인 (대용량 sweep 의 로그도 보통 MB 단위지만
#     repo 에 들어가면 안 됨).
#   * MODEL / HW / VARIANT 환경 변수를 바꿔서 돌리면 로그도 새 디렉토리
#     아래로 가게 되므로, 동일 실행에서 여러 모델/하드웨어를 돈다면
#     `tail -f` 로 추적할 경로를 새 로그 파일 경로에 맞춰 갱신해야 한다.
#   * `tee` 가 background subshell 로 동작한다. 정상 종료 시에는 모든
#     라인이 flush 되지만, 외부에서 강제 종료 (kill -9 등) 하면 마지막
#     몇 줄이 잘릴 수 있다. Ctrl-C (SIGINT) 정도는 안전.
#   * 같은 초 안에 두 번 실행하면 timestamp 가 겹쳐서 같은 파일에 `-a`
#     (append) 로 누적된다. 보통은 안 겹치지만, 짧은 dry-run 을 연속으로
#     돌릴 때 주의.
#   * 로그는 stdout 사본일 뿐이므로 CSV / meta.yaml 같은 profiling 결과
#     자체에는 영향을 주지 않는다. 로그를 지워도 측정 데이터는 안전.
#   * 재실행하면 CSV 는 (chunk 단위로 `--append` 모드라) 기존 파일에 덮어쓰거나
#     이어 붙는다. 이전 sweep 결과를 보존하려면 먼저 tp<N>/ 폴더를
#     백업해 둘 것. (로그는 매 실행마다 별도 파일이라 안전.)

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

# SCRIPT_DIR follows ${HW} so v6e-1 / v5e-1 / future TPU revs each get their
# own perf_models/<HW>/ tree (independent profile_jax.py copies, jit caches).
SCRIPT_DIR="$REPO_ROOT/profiler/perf_models/${HW}"
JAX_PY="$SCRIPT_DIR/profile_jax.py"

if [[ ! -f "$JAX_PY" ]]; then
    echo "ERROR: $JAX_PY not found" >&2
    echo "       (HW=$HW expects $SCRIPT_DIR/profile_jax.py)" >&2
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

DENSE_LAYERS=(embedding layernorm qkv_proj rotary_emb o_proj gate_up_proj act_fn down_proj final_layernorm)
PER_SEQ_LAYERS=(lm_head sampler)

# ============================================================
# Log file — tee everything below this point so wall-time (which
# profile_jax.py and the per-stage summaries print to stdout) is
# persisted alongside the CSVs.
# ============================================================
LOG_DIR="$REPO_ROOT/profiler/perf/${HW}/${MODEL}/${VARIANT}/log"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/profile_$(date +%Y%m%d_%H%M%S).log"
SCRIPT_T0=$(date +%s)
exec > >(tee -a "$LOG_FILE") 2>&1

# ============================================================
# Banner
# ============================================================

echo "============================================================"
echo "  profile_tpu.sh — JAX TPU sweep (subprocess per stage)"
echo "============================================================"
echo "  log file     : $LOG_FILE"
echo "  model        : $MODEL"
echo "  tp_list      : $TP_LIST   variant : $VARIANT   hw : $HW"
echo "  stages       : $STAGES"
echo "  warmup/repeat: $WARMUP / $REPEAT"
echo
echo "  grid caps    : MAX_TOKENS=$MAX_TOKENS  MAX_PREFILL_CHUNK=$MAX_PREFILL_CHUNK  MAX_DECODE_TOKENS=$MAX_DECODE_TOKENS"
echo "  chunk sizes  : TOKEN_CHUNK=$TOKEN_CHUNK  SEQ_CHUNK=$SEQ_CHUNK  PC_CHUNK=$PC_CHUNK  KV_CHUNK=$KV_CHUNK"
echo
echo "  dense        : ${#DENSE_LAYERS[@]} layers × $N_TOKENS tokens  (~$(( (N_TOKENS + TOKEN_CHUNK - 1) / TOKEN_CHUNK )) chunks/layer)"
echo "  per_seq      : ${#PER_SEQ_LAYERS[@]} layers × $N_SEQS sequences  (~$(( (N_SEQS + SEQ_CHUNK - 1) / SEQ_CHUNK )) chunks/layer)"
echo "  attn_prefill : $N_PC values  (~$(( (N_PC + PC_CHUNK - 1) / PC_CHUNK )) chunks, max=$MAX_PREFILL_CHUNK)"
echo "  attn_decode  : $N_BATCH batches × $N_KV kv = $((N_BATCH * N_KV)) combos  (~$(( (N_KV + KV_CHUNK - 1) / KV_CHUNK )) chunks/batch)"
echo "  batch list   : [$BATCH_LIST]"
echo
echo "Press Ctrl-C in 3s to abort..."
sleep 3

# ============================================================
# Per-TP sweep
# ============================================================

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
    local DENSE_T0=$(date +%s)
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
    local DENSE_T1=$(date +%s)
    echo
    echo "  [1/4] dense done in $((DENSE_T1 - DENSE_T0))s"
    else echo; echo "  [1/4] dense — SKIPPED (not in STAGES)"; fi

    # ---- Stage 2: per_seq (subprocess per layer × seq chunk) ----
    if [[ " $STAGES " == *" per_seq "* ]]; then
    local PERSEQ_T0=$(date +%s)
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
    local PERSEQ_T1=$(date +%s)
    echo
    echo "  [2/4] per_sequence done in $((PERSEQ_T1 - PERSEQ_T0))s"
    else echo; echo "  [2/4] per_sequence — SKIPPED (not in STAGES)"; fi

    # ---- Stage 3: attn_prefill (subprocess per pc chunk) ----
    if [[ " $STAGES " == *" attn_prefill "* ]]; then
    local PREFILL_T0=$(date +%s)
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
    local PREFILL_T1=$(date +%s)
    echo
    echo "  [3/4] attn_prefill done in $((PREFILL_T1 - PREFILL_T0))s"
    else echo; echo "  [3/4] attn_prefill — SKIPPED (not in STAGES)"; fi

    # ---- Stage 4: attn_decode (subprocess per batch × kv chunk) ----
    if [[ " $STAGES " == *" attn_decode "* ]]; then
    local DECODE_T0=$(date +%s)
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
    local DECODE_T1=$(date +%s)
    echo
    echo "  [4/4] attn_decode done in $((DECODE_T1 - DECODE_T0))s"
    else echo; echo "  [4/4] attn_decode — SKIPPED (not in STAGES)"; fi

    local STAGE_T1=$(date +%s)
    echo
    echo "  TP=$TP done in $((STAGE_T1 - STAGE_T0))s"
    wc -l "$OUT_DIR"/*.csv
}

for tp in $TP_LIST; do
    run_one_tp "$tp"
done

SCRIPT_T1=$(date +%s)
echo
echo "============================================================"
echo "  all TP done in $((SCRIPT_T1 - SCRIPT_T0))s"
echo "  log file: $LOG_FILE"
echo "============================================================"
