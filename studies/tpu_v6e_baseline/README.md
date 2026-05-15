# TPU baseline study

LLMServingSim 의 TPU baseline 검증. 같은 (dataset, batch, TP) 조합에 대해
**LENS-MaxText** / **LENS-vLLM** / **LLMServingSim** 세 source 의 latency 를
비교해 시뮬레이터의 baseline 가치 + framework overhead 를 동시 측정.

inf2_baseline 의 TPU 등가. 차이점은 framework 만:

| 측정 path          | inf2_baseline                           | tpu_v6e_baseline                          |
| ------------------ | --------------------------------------- | ------------------------------------- |
| vendor first-party | NxDI (AWS Neuron)                       | **MaxText** (Google JAX)        |
| open production    | vLLM-Neuron                             | **vLLM-TPU** (Pallas attention) |
| simulator          | LLMServingSim (Inferentia2 perf bundle) | LLMServingSim (TPU-v6e-1 perf bundle) |

## Layout

```
studies/tpu_v6e_baseline/
├── data/
│   └── datasets/                   # → symlink to inf2_baseline/data/datasets
│       └── {arxiv,cnn,sharegpt,writing_prompts}.csv
├── workloads/                      # → symlink to inf2_baseline/workloads
├── results/
│   ├── lens_tpu/<model>/tp<N>/bs<B>/<dataset>.csv    # MaxText 측정
│   ├── lens_vllm/<model>/tp<N>/bs<B>/<dataset>.csv   # vLLM-TPU 측정
│   └── sim/<model>/tp<N>/bs<B>/<dataset>.csv         # LLMServingSim
├── figures/                                          # compare.py 출력
├── measure_tpu.py        # MaxText (LENS run_eval_tpu.py 의 LENS-free port)
├── sweep_tpu.sh          # MaxText sweep wrapper (fail-fast)
├── measure_vllm.py       # vLLM-TPU
├── sweep_vllm.sh         # vLLM-TPU sweep wrapper (fail-fast)
├── docker-vllm-tpu.sh    # vllm/vllm-tpu image launcher
├── SETUP_MAXTEXT.md      # MaxText install + checkpoint 변환 guide
├── convert_workload.py   # dataset CSV → sim workload JSONL
├── compare.py            # 3-way 비교 + figures
└── README.md
```

`data/datasets` 와 `workloads` 는 inf2_baseline 의 그것을 symlink — 같은 50 batch ×
4 dataset 사용해서 cross-HW 일관성 유지.

## Sweep matrix

| 차원       | 값                                     |
| ---------- | -------------------------------------- |
| Dataset    | arxiv, cnn, sharegpt, writing_prompts  |
| Batch size | 1, 2, 4, 8, 16, 32                     |
| TP         | 1 (multi-chip pod 면 4, 8 추가)        |
| Source     | LENS-MaxText, LENS-vLLM, LLMServingSim |

각 (dataset, batch_size) 조합 = `batch × 50` requests, dataset CSV 위에서부터 순서대로.

## Workflow

### 0. Profile bundle 준비

`profiler/perf/TPU-v6e-1/<model>/<variant>/tp{N}/` 가 있어야 함
(TPU notebook 측정 → v2 변환 결과. 이 commit 에 포함).

```
profiler/perf/TPU-v6e-1/meta-llama/Llama-3.2-1B-Instruct/bf16/
├── meta.yaml
└── tp1/
    ├── dense.csv         (873 rows = 9 layers × 97 tokens)
    ├── attention.csv     (162 rows = 97 prefill + 65 decode)
    └── per_sequence.csv  (194 rows)
```

### 1. Workload 변환 (한 번만)

이미 inf2_baseline 의 workloads 를 symlink 하므로 추가 변환 불필요. 새 batch_size 면:

```bash
python studies/tpu_v6e_baseline/convert_workload.py --all
```

### 2.b. vLLM-TPU sweep (open production)

```bash
# Docker 통한 path 권장 — vllm/vllm-tpu image 사용
bash studies/tpu_v6e_baseline/docker-vllm-tpu.sh
# (container 안)
cd /app/LLMServingSim
bash studies/tpu_v6e_baseline/sweep_vllm.sh
```

Docker 안 쓰고 host 에 vllm 설치된 경우엔 그냥:

```bash
bash studies/tpu_v6e_baseline/sweep_vllm.sh
```

### 3. Simulator 실행 (CPU 인스턴스 + docker, multi-container 병렬)

시뮬레이션은 **결정적** + **CPU bound** (ASTRA-Sim 단일 process 가
core 1개 사용). 가속기 (TPU) 자원과 무관 → MaxText / vLLM-TPU sweep
와 동시 진행 가능. 별도 cheap CPU 인스턴스 권장: c6i.8xlarge
(32 vCPUs, $1.36/h) 또는 c5.4xlarge (16 vCPUs, $0.68/h). 또는 사용자
로컬 머신 (Mac docker).

> ⚠️ **단일 컨테이너 안 multi-process 병렬 (xargs -P) 금지** — 모든
> process 가 같은 `astra-sim/inputs/` (network.yml, system.json,
> trace, workload) 를 share 해서 race condition 으로 wallclock 1초
> 동안 simulated time 이 N×N 초 진행되는 garbage 결과 발생. 반드시
> **container 별 file system 격리** (multi-container 패턴) 사용.

#### 3.a. Built image 한 번 만들기

```bash
# 호스트에서
./scripts/docker-sim.sh         # 컨테이너 시작 + 진입
# 컨테이너 안에서
./scripts/compile.sh            # ASTRA-Sim + Chakra 빌드
exit                            # 컨테이너 빠져나옴

# 호스트에서 — 빌드된 container state 를 image 로 commit
docker ps -a | grep tutorial-micro2024     # container 이름 확인
docker commit <container_name> llmservingsim:built
```

이제 `llmservingsim:built` image 가 ASTRA-Sim/Chakra 빌드 완료 상태로
보존. 매 sim run 이 이 image 로 fresh container 띄움.

#### 3.b. 시뮬레이션 실행 (host 에서, multi-container 병렬)

TP=1 sweep matrix = 4 datasets × 6 batches × 1 TP = **24 runs**. 단일
container 시퀀셜 ~12시간, 8-way 병렬이면 ~1.5시간, 16-way 면 ~45분.

> **사전 조건** — host 에서 workload jsonl 생성 (inf2_baseline 의
> `workloads/` 와 symlink — 이미 만들어졌으면 skip):
>
> ```bash
> python studies/inf2_baseline/convert_workload.py --all
> # → studies/inf2_baseline/workloads/{ds}_bs{N}.jsonl 24개
> # studies/tpu_v6e_baseline/workloads → symlink 이므로 자동 공유
> ```

각 task 마다 `docker run --rm` 으로 fresh container 띄움. container 별
writable layer 가 격리되므로 `astra-sim/inputs/` race condition 없음.
`workloads/`, `configs/`, `profiler/perf/` 는 read-only mount (input),
`results/` 만 read-write mount (각 container 가 unique sub-path 에
write 라 race 없음).

```bash
# 호스트에서 (LLMServingSim 디렉토리 안). docker container 는 host
# 의 xargs -P 로 N개 동시 실행됨.

REPO=$(pwd)
PARALLEL=${PARALLEL:-8}             # 동시 container 수 (CPU vCPU 의 절반)

# (tp, ds, bs) 매트릭스 한 줄씩
for tp in 1; do
  for ds in arxiv cnn sharegpt writing_prompts; do
    for bs in 1 2 4 8 16 32; do
      echo "${tp} ${ds} ${bs}"
    done
  done
done > /tmp/sim_matrix.txt

cat /tmp/sim_matrix.txt | xargs -n3 -P${PARALLEL} bash -c '
  tp=$0; ds=$1; bs=$2
  rel_out=studies/tpu_v6e_baseline/results/sim/Llama-3.2-1B/tp${tp}/bs${bs}/${ds}.csv
  abs_out=/app/LLMServingSim/${rel_out}
  mkdir -p $(dirname '"${REPO}"'/${rel_out})

  docker run --rm \
    -v '"${REPO}"':/app/LLMServingSim \
    -v '"${REPO}"'/astra-sim/inputs:/tmp/inputs_template:ro \
    -v /app/LLMServingSim/astra-sim/inputs \
    -w /app/LLMServingSim \
    llmservingsim:built \
    bash -c "
      cp -r /tmp/inputs_template/. /app/LLMServingSim/astra-sim/inputs/
      python -m serving \
        --cluster-config configs/cluster/tpu_v6e_llama1b_tp${tp}.json \
        --dataset studies/tpu_v6e_baseline/workloads/${ds}_bs${bs}.jsonl \
        --output ${rel_out} \
        --max-num-seqs ${bs} \
        --no-enable-chunked-prefill \
        --no-enable-prefix-caching \
        --max-num-batched-tokens 8192 \
        --dtype bfloat16 \
        > ${abs_out%.csv}.log 2>&1
    "
  echo "[done] tp${tp} bs${bs} ${ds}"
'
```

Mount 정책:

| 디렉토리                                                  | mount 종류                                                              | 이유                                                                                                                                                                                                                                           |
| --------------------------------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `${REPO}` (호스트 repo 전체)                            | bind mount →`/app/LLMServingSim`                                     | image 의 mount point 가 빈 dir 라 host 의 serving/, configs/, profiler/perf/, astra-sim binary 등 다 같이 들어와야 함. write 도 허용 (results/ 외에는 simulator 가 write 안 함).                                                               |
| `astra-sim/inputs`                                      | **anonymous volume** (`-v /app/LLMServingSim/astra-sim/inputs`) | container 별 자동 unique volume. host 의 inputs/ 와 무관 — race 해결의 핵심.`--rm` 시 자동 cleanup.                                                                                                                                         |
| `${REPO}/astra-sim/inputs → /tmp/inputs_template` (ro) | bind ro                                                                 | host 의 base inputs/ template (system.json 등). config_builder.py 가 read-modify-write 라 시작 시점에 anonymous volume 안에 한 번 cp. anonymous volume 위치(`/app/LLMServingSim/astra-sim/inputs`) 와 다른 path 로 mount 해야 가려지지 않음. |

`cp -r /tmp/inputs_template/. .../astra-sim/inputs/` — host 의 base inputs (system.json template + sub-dir 구조) 를 container 별 anonymous volume 에 복사. 이후 simulator 의 read-modify-write 동작 OK + write 는 anonymous volume 안에서만 일어나므로 host 무관 + race 격리 유지.

**Container 시작 overhead** ~2-3 sec / task. 24 × 3 = 1.2 min 추가.
무시 가능.

**Sequential fallback** (안전한 baseline): `PARALLEL=1` 로 같은 명령
실행. ~12시간. 결과는 동일 — multi-container 와 sequential 의
같은 task 결과는 byte-by-byte 일치 (deterministic).

진행 상황 모니터:

```bash
tail -f studies/tpu_v6e_baseline/results/sim/Llama-3.2-1B/tp*/bs*/*.log
docker ps                       # 현재 실행 중 container 수 확인
```

#### 3.c. (CPU 인스턴스가 별도면) 결과 transfer 로 가져오기

CPU 인스턴스에서 sweep 끝났으면 로컬로 가져와서 비교 진행:

```bash
# 로컬 (Mac) 에서
rsync -av --progress \
    ubuntu@cpu-instance:~/LLMServingSim/studies/tpu_v6e_baseline/results/sim/ \
    studies/tpu_v6e_baseline/results/sim/
```

CSV + `.log` 둘 다 같은 트리에 들어옴.

`--max-num-batched-tokens 8192` 는 vLLM-TPU 의 max_model_len 와 일치
시키기 위함 (default 2048 면 큰 prompt 가 chunked prefill 처럼 여러
step 으로 나뉘어 schedule 차이 발생). 우리는 chunked prefill off
이므로 `max-num-batched-tokens` 가 충분히 커야 한 step 에 fit.

### 4. 비교

```bash
python studies/tpu_v6e_baseline/compare.py --tps 1 --batch-sizes 1,2,4,8,16,32
# 표 + figures/{e2e_grid.png, per_tp_bs/tp{N}_bs{B}.png}
```

3 bar / dataset: LLMServingSim2.0 / MaxText / vLLM.

## Inf2 와의 핵심 차이

- **No NKI knobs** — TPU 는 NeuronCC/NKI 가 아닌 XLA + Pallas. `attn_kernel_enabled`,
  `qkv_kernel_enabled` 같은 NeuronConfig override 없음. measure_vllm.py 의 LLM(...)
  init 도 그만큼 단순.
- **No `--compiled-dir`** — TPU 의 compile cache 는 JAX/XLA 가 자동 (`~/jax_cache_*`).
- **TPU single-tenancy** — `/dev/vfio/0` 가 한 process 만 점유. measure 전
  `sudo fuser -v /dev/vfio/0` 로 잡힌 process 정리 필수 (`pkill -9 -f ipykernel`
  자주 필요).
- **MaxText 의 checkpoint 변환 비용** — HF Llama → MaxText Orbax 변환이 별도 단계.
  inf2 의 NxDI 는 HF format 직접 load 가능했지만 MaxText 는 사전 변환 필요. 자세히
  `SETUP_MAXTEXT.md`.

## Troubleshooting

**`ModuleNotFoundError: No module named 'jax'`** — JAX TPU build 안 설치.
`pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`

**`TPU initialization failed: open(/dev/vfio/0): Device or resource busy`** —
다른 process 가 TPU 점유 중. `pkill -9 -f ipykernel` 또는 인스턴스 reboot.

**`ModuleNotFoundError: No module named 'vllm'` (container 안)** —
`vllm/vllm-tpu` image 가 editable install 이라 source 가 `/workspace/...` 에 있음.
우리 docker-vllm-tpu.sh 는 mount target 을 `/app/LLMServingSim` 로 두어 image 의
vllm source 를 가리지 않음. 만약 다른 mount path 사용 시 vllm 사라짐.

**`Cannot load the request to batch due to max_num_batched_tokens limitation`** —
simulator default 가 2048. arxiv 처럼 input 큰 dataset 은 `--max-num-batched-tokens 8192`
(또는 max_model_len 까지) 명시. vLLM 의 default 와 align 하는 게 fair.

## 측정 결과 해석 시 유의

LLMServingSim 의 perf bundle 은 TPU notebook (lazy XLA + per-iter sync + perf_counter)
의 host wallclock 측정. 그 mechanism 은 **per-layer launch overhead 가 누적**되어
production NEFF/HLO execution 보다 inflated.

- estimate (sim 의 layer-CSV 합산) ↔ measure (TPU notebook cell 5 의 host wallclock)
  → 같은 mechanism 의 inflation 이라 self-consistency 4% 이하 (paper 의 framing)
- estimate ↔ vLLM/MaxText production → **5x off 가능**

즉 simulator 의 4% accuracy 가 production-grade fidelity 를 보장하지 않음. 우리
3-way 비교의 의의는 그 gap 의 정량적 evidence 확보.
