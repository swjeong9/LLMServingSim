# GPU baseline study (L4 / A10G)

LLMServingSim 의 NVIDIA GPU baseline 검증. 같은 (dataset, batch, TP) 조합에
대해 **LENS-vLLM (CUDA)** 와 **LLMServingSim** 두 source 의 latency 를 비교.
inf2 / tpu baseline 의 NVIDIA GPU 버전. 측정 path 는 vLLM 만:

| 측정 path          | inf2_baseline | tpu_baseline      | gpu_baseline (이 study) |
| ------------------ | ------------- | ----------------- | ----------------------- |
| vendor first-party | NxDI          | Jax               | —                      |
| open production    | vLLM-Neuron   | vLLM-TPU (Pallas) | **vLLM-CUDA**     |
| simulator          | LLMServingSim | LLMServingSim     | LLMServingSim           |

NVIDIA GPU 는 vendor SDK 와 vLLM 사이 격차가 거의 없어 vLLM 한 source 면 충분.

## Layout

```
studies/gpu_baseline/
├── data/
│   └── datasets/                   # → symlink to inf2_baseline/data/datasets
│       └── {arxiv,cnn,sharegpt,writing_prompts}.csv
├── workloads/                      # → symlink to inf2_baseline/workloads
├── results/
│   ├── lens_vllm/<HW>/<opt>/<model>/tp<N>/bs<B>/<dataset>.csv   # vLLM 측정
│   └── sim/<HW>/<opt>/<model>/tp<N>/bs<B>/<dataset>.csv         # LLMServingSim
│       # <opt> ∈ {"off", "on"} = (chunked_prefill, prefix_caching)
│       # 두 옵션을 토글한 결과를 한 GPU 에 대해 4-way 비교 가능.
├── figures/<HW>/                                          # compare.py 출력
├── measure_vllm.py        # vLLM-CUDA 측정 (hardware-agnostic, --hardware 로 라벨링)
├── sweep_vllm.sh          # vLLM-CUDA sweep wrapper
├── convert_workload.py    # dataset CSV → sim workload JSONL (gpu_baseline 경로 default)
├── compare.py             # 2-way 비교 + figures
└── README.md
```

`data/datasets` 와 `workloads` 는 inf2_baseline 의 그것을 symlink — 같은 50 batch
× 4 dataset 사용해서 cross-HW 일관성 유지.

## Sweep matrix

| 차원       | 값                                                              |
| ---------- | --------------------------------------------------------------- |
| Hardware   | **L4, A10G** (TP=1 single GPU instance)                   |
| Dataset    | arxiv, cnn, sharegpt, writing_prompts                           |
| Batch size | 1, 2, 4, 8, 16, 32                                              |
| TP         | 1                                                               |
| Source     | LENS-vLLM (CUDA), LLMServingSim                                 |
| opt        | off (default), on — chunked_prefill + prefix_caching 토글 묶음 |

각 (dataset, batch_size) 조합 = `batch × 50` requests, dataset CSV 위에서부터
순서대로. 24 runs / GPU / opt × 2 opts × 2 GPU = **96 runs**. 한 GPU 위에서 4-way
(sim_off, sim_on, vllm_off, vllm_on) 비교 가능.

## Workflow

### 0. Profile bundle 준비

`profiler/perf/<HW>/<model>/<variant>/tp1/` 가 있어야 함. user 가 직접 진행:

```bash
# 예: L4 인스턴스에서
HARDWARE=L4 MODEL=meta-llama/Llama-3.2-1B-Instruct TP_DEGREES=1 \
    bash profiler/profile.sh
# → profiler/perf/L4/meta-llama/Llama-3.2-1B-Instruct/bf16/tp1/
#     {dense,per_sequence,attention,skew,skew_fit}.csv + meta.yaml
```

A10G 도 동일. `HARDWARE` 라벨은 cluster config 의 `hardware` 필드 그리고
`measure_vllm.py --hardware` 와 정확히 일치해야 함 (대소문자 포함).

### 1. Workload 변환 (한 번만)

이미 inf2_baseline 의 workloads 를 symlink 하므로 추가 변환 불필요. 새 batch_size
면:

```bash
python studies/gpu_baseline/convert_workload.py --all
# → studies/gpu_baseline/workloads/{ds}_bs{N}.jsonl 24개
#    (inf2_baseline/workloads 와 같은 dir, symlink)
```

### 2. vLLM-CUDA sweep (GPU 인스턴스에서)

vLLM CUDA build (`vllm/vllm-openai:v0.19.0`) 설치 상태에서 sweep. 4-way 비교를
위해 **각 GPU 에 두 번** 돌림 — opt off + opt on.

```bash
# Docker 통한 path 권장
bash scripts/docker-vllm.sh
# (container 안)
cd /workspace

# L4 GPU 인스턴스 (g6.xlarge)
# 1) opt=off  — chunked_prefill, prefix_caching 둘 다 off (default)
HARDWARE=L4 bash studies/gpu_baseline/sweep_vllm.sh
# 2) opt=on   — 둘 다 on
HARDWARE=L4 ENABLE_CHUNKED_PREFILL=1 ENABLE_PREFIX_CACHING=1 \
    bash studies/gpu_baseline/sweep_vllm.sh

# A10G 도 동일 (24 runs × 2 opts = 48 runs)
HARDWARE=A10G bash studies/gpu_baseline/sweep_vllm.sh
HARDWARE=A10G ENABLE_CHUNKED_PREFILL=1 ENABLE_PREFIX_CACHING=1 \
    bash studies/gpu_baseline/sweep_vllm.sh
```

출력 경로의 `<opt>` 는 두 env 의 조합에서 자동 도출:

- `off`  = 둘 다 `0` (default, inf2/tpu baseline 패리티)
- `on`   = 둘 다 `1`
- `cp{0,1}_pc{0,1}` = mixed (compare.py 가 인식 안 하지만 측정은 가능)

`HARDWARE` 변수는 cluster config 의 `hardware` 필드 + 결과 sub-folder 와 정확히
일치해야 함. `measure_vllm.py` 산출 JSON 의 `opt_label`, `enable_chunked_prefill`,
`enable_prefix_caching` 필드에 어떤 설정으로 측정됐는지 흔적 남음. OOM /
too-long 은 자동 skip / ERROR.

### 3. Simulator 실행 (CPU 인스턴스 + docker, multi-container 병렬)

시뮬레이션은 **결정적** + **CPU bound** (ASTRA-Sim 단일 process 가 core 1개
사용). GPU 자원과 무관 → vLLM sweep 와 동시 진행 가능. 별도 cheap CPU 인스턴스
권장 (c6i.8xlarge / c5.4xlarge) 또는 사용자 로컬.

> ⚠️ **단일 컨테이너 안 multi-process 병렬 (xargs -P) 금지** — 모든 process 가
> 같은 `astra-sim/inputs/` 를 share 해서 race condition. **container 별 file
> system 격리** (multi-container 패턴) 사용. 자세히 inf2/tpu README 참고.

#### 3.a. Built image 한 번 만들기

```bash
./scripts/docker-sim.sh         # 호스트 → 컨테이너 진입
./scripts/compile.sh            # (컨테이너 안) ASTRA-Sim + Chakra 빌드
exit
docker ps -a | grep tutorial-micro2024     # 이름 확인
docker commit <container_name> llmservingsim:built
```

#### 3.b. 시뮬레이션 실행

vLLM sweep 와 같은 toggle (off / on) 으로 sim 도 두 번 돌림. 같은 GPU 의 4
source — sim_off / sim_on / vllm_off / vllm_on — 가 모두 채워져야 compare.py
가 4-way 비교 표 + 그림을 그릴 수 있음.

```bash
# 호스트에서 (LLMServingSim 디렉토리 안)
REPO=$(pwd)
HARDWARE=${HARDWARE:?set HARDWARE=L4 or HARDWARE=A10G}
hw_lc=$(echo "${HARDWARE}" | tr '[:upper:]' '[:lower:]')
PARALLEL=${PARALLEL:-8}

# GPU-only toggles — must match what was passed to sweep_vllm.sh.
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-0}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-0}
cp_flag=--no-enable-chunked-prefill; [ "${ENABLE_CHUNKED_PREFILL}" = "1" ] && cp_flag=--enable-chunked-prefill
pc_flag=--no-enable-prefix-caching;  [ "${ENABLE_PREFIX_CACHING}"  = "1" ] && pc_flag=--enable-prefix-caching

# Mirror measure_vllm.py::opt_label() so the sim output path matches the vllm side.
if   [ "${ENABLE_CHUNKED_PREFILL}" = "1" ] && [ "${ENABLE_PREFIX_CACHING}" = "1" ]; then OPT=on
elif [ "${ENABLE_CHUNKED_PREFILL}" = "0" ] && [ "${ENABLE_PREFIX_CACHING}" = "0" ]; then OPT=off
else OPT="cp${ENABLE_CHUNKED_PREFILL}_pc${ENABLE_PREFIX_CACHING}"
fi

for ds in arxiv cnn sharegpt writing_prompts; do
  for bs in 1 2 4 8 16 32; do
    echo "1 ${ds} ${bs}"
  done
done > /tmp/sim_matrix.txt

cat /tmp/sim_matrix.txt | xargs -n3 -P${PARALLEL} bash -c '
  tp=$0; ds=$1; bs=$2
  rel_out=studies/gpu_baseline/results/sim/'"${HARDWARE}"'/'"${OPT}"'/Llama-3.2-1B-Instruct/tp${tp}/bs${bs}/${ds}.csv
  mkdir -p $(dirname '"${REPO}"'/${rel_out})
  abs_log=/app/LLMServingSim/${rel_out%.csv}.log

  docker run --rm \
    -v '"${REPO}"':/app/LLMServingSim \
    -v '"${REPO}"'/astra-sim/inputs:/tmp/inputs_template:ro \
    -v /app/LLMServingSim/astra-sim/inputs \
    -w /app/LLMServingSim \
    llmservingsim:built \
    bash -c "
      cp -r /tmp/inputs_template/. /app/LLMServingSim/astra-sim/inputs/
      python -m serving \
        --cluster-config configs/cluster/'"${hw_lc}"'_llama1b_tp${tp}.json \
        --dataset studies/gpu_baseline/workloads/${ds}_bs${bs}.jsonl \
        --output ${rel_out} \
        --max-num-seqs ${bs} \
        '"${cp_flag}"' '"${pc_flag}"' \
        --max-num-batched-tokens 8192 \
        --dtype bfloat16 \
        > ${abs_log} 2>&1
    "
  echo "[done] '"${HARDWARE}/${OPT}"' tp${tp} bs${bs} ${ds}"
'
```

4-way 를 위해 같은 docker block 을 두 번 실행 (env 만 변경):

```bash
# pass 1 — sim_off
HARDWARE=L4 bash <위 docker block>
# pass 2 — sim_on
HARDWARE=L4 ENABLE_CHUNKED_PREFILL=1 ENABLE_PREFIX_CACHING=1 bash <위 docker block>
```

Mount 정책은 inf2/tpu README 와 동일 (anonymous volume 으로 `astra-sim/inputs`
격리, repo 전체 bind, base inputs template ro mount).

진행 상황 모니터:

```bash
tail -f studies/gpu_baseline/results/sim/${HARDWARE}/*/Llama-3.2-1B-Instruct/tp*/bs*/*.log
docker ps
```

`--max-num-batched-tokens 8192` 는 vLLM 의 max_model_len 와 일치 시키기 위함.
chunked prefill off + prefix caching off 도 vLLM 측 설정과 align (apples-to-apples).

### 4. 비교

```bash
python studies/gpu_baseline/compare.py --hardware L4
python studies/gpu_baseline/compare.py --hardware A10G
# 표 + figures/<HW>/{e2e_grid.png, per_tp_bs/tp{1}_bs{1,2,4,8,16,32}.png}
```

4 bar / dataset: Sim (off) / Sim (on) / vLLM (off) / vLLM (on). 표는 각 source
ms 값 + 같은 opt 끼리의 `sim/vllm` diff% 두 컬럼 (off / on).

비교 시 살펴볼 항목:

- **sim_off vs vllm_off** — base apples-to-apples accuracy (inf2/tpu baseline 과
  동일 framing)
- **sim_on vs vllm_on** — chunked prefill + prefix caching 켰을 때의 sim accuracy
- **sim_off vs sim_on** — simulator 가 두 기능의 latency 영향을 모델링 하는지
- **vllm_off vs vllm_on** — 실측에서 두 기능이 주는 speedup (ground truth)

## Cluster config

| 파일                                      | hardware | mem_bw (GB/s) | link_bw (GB/s)     | 비고                      |
| ----------------------------------------- | -------- | ------------- | ------------------ | ------------------------- |
| `configs/cluster/l4_llama1b_tp1.json`   | L4       | 300           | 32 (PCIe 4.0 ×16) | Ada Lovelace, 24 GB GDDR6 |
| `configs/cluster/a10g_llama1b_tp1.json` | A10G     | 600           | 32 (PCIe 4.0 ×16) | Ampere GA102, 24 GB GDDR6 |

`link_bw=32` (host PCIe bound) — TP=1 이라 ALLREDUCE 없어 영향 거의 0.
`link_latency=0` 은 inf2/tpu baseline 의 H100 패턴 (intra-node idealize).

## 해석 가이드

평균 e2e abs error 기준:

* `< 15%` — baseline 으로 valid
* `15-30%` — "approximate baseline" framing
* `> 30%` — bucketing / profile mechanism gap 큰 것

L4 vs A10G 차이 패턴:

* 둘 다 paper-level fit → simulator generalization OK
* A10G ≈ paper, L4 diverge → simulator 가 저-BW GPU 에서 부정확. 이는 inf2 의
  820 GB/s claimed BW 도 sensitivity 검토 필요 signal.
* 둘 다 큰 error → simulator 자체 또는 profile mechanism 문제.

## 측정 결과 해석 시 유의

LLMServingSim 의 perf bundle 은 vLLM `layerwise_profile` 의 per-layer CUDA
event 합산. mechanism 자체 inflation 적음 (TPU notebook 같은 lazy host-time
sync 누적 없음). 그래서 GPU baseline 은 inf2/tpu 보다 paper §VI 결과에 더
가까운 fit 이 기대됨 — 이게 우리가 GPU 를 cross-validation reference 로 쓰는
이유.
