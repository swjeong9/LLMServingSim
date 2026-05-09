# Inf2 baseline study

LLMServingSim 의 Inferentia 2 baseline 검증. 같은 (dataset, batch, TP)
조합에 대해 LENS-NxD / LENS-vLLM / LLMServingSim 세 source 의 latency 를
비교해 시뮬레이터의 baseline 가치 + framework overhead 를 동시 측정.

## Layout

```
studies/inf2_baseline/
├── data/
│   ├── profile_min14.csv        # LENS bucket 정의 (참고)
│   └── datasets/                # LENS bs32 dataset (4개 + meta)
│       └── {arxiv,cnn,sharegpt,writing_prompts}.csv
├── workloads/                   # 변환된 LLMServingSim JSONL (자동 생성)
│   └── <dataset>_bs<N>.jsonl
├── results/
│   ├── lens_nxd/<model>/tp<N>/bs<B>/<dataset>.csv     # LENS run_profiling.py 출력
│   ├── lens_vllm/<model>/tp<N>/bs<B>/<dataset>.csv    # LENS run_profiling_vllm.py 출력
│   └── sim/<model>/tp<N>/bs<B>/<dataset>.csv          # LLMServingSim 출력
├── comparison/                  # 3-way per-batch 결과 (자동 생성)
├── convert_workload.py          # dataset CSV + batch → JSONL
├── compare.py                   # 3-way 비교
└── README.md
```

## Sweep matrix

| 차원       | 값                                    |
| ---------- | ------------------------------------- |
| Dataset    | arxiv, cnn, sharegpt, writing_prompts |
| Batch size | 1, 2, 4, 8, 16, 32 (OOM 까지)         |
| TP         | 1, 2                                  |
| Source     | LENS-NxD, LENS-vLLM, LLMServingSim    |

각 (dataset, batch_size) 조합 = `batch × 50` requests, dataset CSV 위에서부터 순서대로.
같은 입력 → 결과 reproducibility. (dataset 의 row 수가 부족하면 자동 skip.)

## Workflow

### 0. Profile bundle 준비

`profiler/perf/Inferentia2/<model>/<variant>/tp{1,2}/` 가 있어야 함
(profile_neuron.py + profile_inf2.sh sweep 산출물).

### 1. Workload 변환

```bash
python studies/inf2_baseline/convert_workload.py --all
# → workloads/{arxiv,cnn,sharegpt,writing_prompts}_bs{1,2,4,8,16,32}.jsonl
```

### 2. LENS 측정 (인스턴스에서)

두 framework 측정 스크립트는 study folder 안에 self-contained
(LENS repo 의존성 없음, NxDI / vLLM-Neuron 패키지만 필요):

* `measure_nxd.py`  — LENS NxD-direct (continuous batching off,
  `max(output_len)` padding). LENS `run_eval.py` 의 포팅.
* `measure_vllm.py` — vLLM-Neuron (`max_num_seqs > 1` 이면 continuous
  batching 자동 ON). LENS `run_profiling_vllm.py` + `run_eval.py` 결합.

두 스크립트 인자 동일. 출력 위치도 자동으로
`results/lens_{nxd,vllm}/<model>/tp<N>/bs<B>/<dataset>_<ts>.csv` 로 저장
(stable symlink `<dataset>.csv` 가 항상 최신 결과 가리킴).

#### 2.a-pre. 모델 사전 다운로드 (`--model` 은 local path 권장)

NxDI 의 `model.load()` 가 weight 가져올 때 model_path 끝에 `/` 가
붙으면서 HF Hub repo id validator (`HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'`) 에 거부됨.
LENS 도 같은 이유로 local path (`/home/ubuntu/models/...`) 사용.

```bash
mkdir -p ~/models
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
    --local-dir ~/models/Llama-3.2-1B-Instruct
# (HF_TOKEN 필요 시: export HF_TOKEN=...)
```

이후 `--model ~/models/Llama-3.2-1B-Instruct` 형태로 호출.
vLLM 도 동일 local path 받음.

#### 2.a. venv 선택

AWS Neuron DLAMI 는 여러 venv 를 `/opt/aws_neuronx_venv_*` 에 미리 설치.
스크립트마다 필요한 패키지가 다르므로 venv 를 분리해 활성화.

```bash
ls /opt | grep aws_neuronx_venv

# 어느 venv 에 무엇이 있는지 직접 확인
for v in /opt/aws_neuronx_venv_*; do
  echo "=== $(basename $v) ==="
  source $v/bin/activate
  python -c "
import importlib.util as u
for pkg in ['neuronx_distributed_inference', 'vllm', 'torch_neuronx']:
    print(f'  {pkg}: ' + ('OK' if u.find_spec(pkg) else 'MISSING'))
" 2>&1 | grep -E "OK|MISSING"
  deactivate
done
```

표준 매핑 (DLAMI 버전에 따라 이름이 약간 다름):

| Script                        | 필요 패키지                       | venv (default name)                                                                                      |
| ----------------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `profile_neuron.py` (sweep) | `torch_neuronx`                 | `aws_neuronx_venv_pytorch_2_9`                                                                         |
| `measure_nxd.py`            | `neuronx_distributed_inference` | `aws_neuronx_venv_pytorch_2_9_nxd_inference`                                                           |
| `measure_vllm.py`           | `vllm` (vllm-neuronx)           | `aws_neuronx_venv_pytorch_2_9_nxd_inference` (NxDI venv 안에 vllm 같이 있는 경우 흔함) — 확인 후 결정 |

**두 framework 가 같은 venv 에 들어있다면** 한 sweep 으로 묶어도 OK,
**다르다면** 아래 NxD / vLLM 두 sweep 을 각각 해당 venv 활성화 상태에서.

#### 2.b. NxD-direct sweep — TP=2 먼저, TP=1 그 다음

NxDI 가 두 TP 모두 "CONVERT_TO_MHA" warning 출력하지만 misleading —
실제 runtime 은 GQA 그대로 동작 (issue #1289, 알려진 framework
artifact 섹션 참고). 두 TP 모두 fair 측정 가능. TP=2 가 inf2.xlarge
의 1 chip × 2 NeuronCore 모두 활용하는 natural setting 이라 먼저 진행.

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# 1) TP=2 — fair comparison
for ds in arxiv cnn sharegpt writing_prompts; do
  for bs in 1 2 4 8 16 32; do
    python studies/inf2_baseline/measure_nxd.py \
        --dataset ${ds} --batch-size ${bs} \
        --model ~/models/Llama-3.2-1B-Instruct \
        --tp-degree 2 --max-model-len 8192 \
        --compiled-dir /home/ubuntu/compiled_models_inf2_baseline_nxd
  done
done

# 2) TP=1 — warning 출력되지만 실제 GQA 동작 (위 issue #1289 참고)
for ds in arxiv cnn sharegpt writing_prompts; do
  for bs in 1 2 4 8 16 32; do
    python studies/inf2_baseline/measure_nxd.py \
        --dataset ${ds} --batch-size ${bs} \
        --model ~/models/Llama-3.2-1B-Instruct \
        --tp-degree 1 --max-model-len 8192 \
        --compiled-dir /home/ubuntu/compiled_models_inf2_baseline_nxd_tp1
  done
done

deactivate
```

#### 2.c. vLLM-Neuron sweep — **TP=2 먼저, 그 다음 TP=1**

NxD 와 같은 순서. vLLM-Neuron 도 NxDI 위에 올라가 있어 동일한
CONVERT_TO_MHA 가 TP=1 에서 발동될 가능성 — 측정 시 warning 확인.

```bash
source /opt/aws_neuronx_venv_<vllm-capable>/bin/activate

# 1) TP=2
for ds in arxiv cnn sharegpt writing_prompts; do
  for bs in 1 2 4 8 16 32; do
    python studies/inf2_baseline/measure_vllm.py \
        --dataset ${ds} --batch-size ${bs} \
        --model ~/models/Llama-3.2-1B-Instruct \
        --tp-degree 2 --max-model-len 8192 \
        --compiled-dir /home/ubuntu/compiled_models_inf2_baseline_vllm
  done
done

# 2) TP=1 (NxDI warning 무시 OK — issue #1289 참고)
for ds in arxiv cnn sharegpt writing_prompts; do
  for bs in 1 2 4 8 16 32; do
    python studies/inf2_baseline/measure_vllm.py \
        --dataset ${ds} --batch-size ${bs} \
        --model ~/models/Llama-3.2-1B-Instruct \
        --tp-degree 1 --max-model-len 8192 \
        --compiled-dir /home/ubuntu/compiled_models_inf2_baseline_vllm_tp1
  done
done

deactivate
```

OOM 또는 too-long 의 경우 자동 skip / ERROR 행으로 기록.
같은 `(model, tp, batch, max_model_len)` 의 두 번째 호출은 NEFF cache hit
(약 30 sec 의 model.load 만), `--skip-compile` 로 더 빠르게 가능.

#### 2.d. 빠른 sanity check

```bash
# venv 활성화 + 모델 local 다운로드된 상태에서.
python studies/inf2_baseline/measure_nxd.py --dataset cnn --batch-size 1 \
    --model ~/models/Llama-3.2-1B-Instruct --tp-degree 1 \
    --max-runs 3 --skip-warmup
```

NxDI compile 약 4분 걸림 (NKI bypass 한 native attention path).
이후 cache hit 으로 빠름.

### 3. Simulator 실행 (CPU 인스턴스 + docker, multi-container 병렬)

시뮬레이션은 **결정적** + **CPU bound** (ASTRA-Sim 단일 process 가
core 1개 사용). 인스턴스 자원 (가속기 시간) 와 무관 → inf2 측정 sweep
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

전체 sweep matrix = 4 datasets × 6 batches × 2 TPs = **48 runs**. 단일
container 시퀀셜 1-2 일, 8-way 병렬이면 ~6시간, 16-way 면 ~3시간.

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
for tp in 1 2; do
  for ds in arxiv cnn sharegpt writing_prompts; do
    for bs in 1 2 4 8 16 32; do
      echo "${tp} ${ds} ${bs}"
    done
  done
done > /tmp/sim_matrix.txt

cat /tmp/sim_matrix.txt | xargs -n3 -P${PARALLEL} bash -c '
  tp=$0; ds=$1; bs=$2
  rel_out=studies/inf2_baseline/results/sim/Llama-3.2-1B/tp${tp}/bs${bs}/${ds}.csv
  abs_out=/app/LLMServingSim/${rel_out}
  mkdir -p $(dirname '"${REPO}"'/${rel_out})

  docker run --rm \
    -v '"${REPO}"'/studies/inf2_baseline/workloads:/app/LLMServingSim/studies/inf2_baseline/workloads:ro \
    -v '"${REPO}"'/studies/inf2_baseline/results:/app/LLMServingSim/studies/inf2_baseline/results \
    -v '"${REPO}"'/configs:/app/LLMServingSim/configs:ro \
    -v '"${REPO}"'/profiler/perf:/app/LLMServingSim/profiler/perf:ro \
    -w /app/LLMServingSim/astra-sim \
    llmservingsim:built \
    bash -c "python -m serving \
      --cluster-config configs/cluster/inf2_xlarge_llama1b_tp${tp}.json \
      --dataset studies/inf2_baseline/workloads/${ds}_bs${bs}.jsonl \
      --output ${rel_out} \
      --max-num-seqs ${bs} \
      --no-enable-chunked-prefill \
      --no-enable-prefix-caching \
      --max-num-batched-tokens 8192 \
      --dtype bfloat16 \
      > ${abs_out%.csv}.log 2>&1"
  echo "[done] tp${tp} bs${bs} ${ds}"
'
```

| 디렉토리 mount | mode | 이유 |
|---|---|---|
| `studies/inf2_baseline/workloads` | ro | 입력 — 모든 container 가 read |
| `studies/inf2_baseline/results` | rw | 출력 — 각 container 가 unique path 에 write |
| `configs` | ro | cluster config — read |
| `profiler/perf` | ro | profile bundle — read |
| `astra-sim/inputs` | **mount 안 함** | race 해결의 핵심 — 각 container 자체 layer |

**Container 시작 overhead** ~2-3 sec / task. 48 × 3 = 2.4 min 추가.
무시 가능.

**Sequential fallback** (안전한 baseline): `PARALLEL=1` 로 같은 명령
실행. 1-2 일 걸림. 결과는 동일 — multi-container 와 sequential 의
같은 task 결과는 byte-by-byte 일치 (deterministic).

진행 상황 모니터:
```bash
tail -f studies/inf2_baseline/results/sim/Llama-3.2-1B/tp*/bs*/*.log
docker ps                       # 현재 실행 중 container 수 확인
```

#### 3.c. (CPU 인스턴스가 별도면) 결과 transfer 로 가져오기

CPU 인스턴스에서 sweep 끝났으면 로컬로 가져와서 비교 진행:

```bash
# 로컬 (Mac) 에서
scp -r ubuntu@cpu-instance:~/LLMServingSim/studies/inf2_baseline/results/sim/ \
       studies/inf2_baseline/results/sim/

# 또는 sync (이미 일부 있는 경우)
rsync -av --progress \
    ubuntu@cpu-instance:~/LLMServingSim/studies/inf2_baseline/results/sim/ \
    studies/inf2_baseline/results/sim/
```

CSV + `.log` 둘 다 같은 트리에 들어옴. 로그가 의미 있어 보이면
나중에 S3 archiving 으로:

```bash
# S3 bucket 에 backup (선택적, 추후)
aws s3 sync studies/inf2_baseline/results/ \
    s3://<bucket>/inf2_baseline/results/$(date +%Y%m%d)/
```

`--max-num-batched-tokens 8192` 는 LENS 의 max_model_len 와 일치
시키기 위함 (default 2048 면 큰 prompt 가 chunked prefill 처럼 여러
step 으로 나뉘어 schedule 차이 발생). 우리는 chunked prefill off
이므로 `max-num-batched-tokens` 가 충분히 커야 한 step 에 fit.

**다중 컨테이너 옵션 (더 강한 격리)**: xargs -P 대신 `docker compose`
또는 `docker run --rm` 으로 N 개 컨테이너 띄우는 것도 가능. 단 ASTRA-Sim
빌드를 N 번 또는 image 에 미리 포함해야 setup 비용 늘어남. xargs -P
로 한 컨테이너 안 multi-process 가 가장 단순.

### 4. 3-way 비교

```bash
python studies/inf2_baseline/compare.py --tp 1 --batch-sizes 1,2,4,8,16,32
python studies/inf2_baseline/compare.py --tp 2 --batch-sizes 1,2,4,8,16,32
# 콘솔 출력 + comparison/<model>_tp<N>_bs<B>_<dataset>.csv
```

## 해석 가이드

* **Sim vs LENS-NxD** — 우리 simulator 가 NxD-direct (vLLM 미경유) 동작
  모델링 잘하는지. LENS framework 의 진짜 baseline.
* **Sim vs LENS-vLLM** — vLLM-Neuron 의 continuous batching 동작 모델링.
  batch>1 에서만 차이 의미 있음.
* **LENS-NxD vs LENS-vLLM** — framework overhead 자체.
  batch=1 에서는 거의 동일, batch>1 에서 vLLM 의 continuous batching 효과.

평균 e2e abs error 기준:

* < 15% — baseline 으로 valid → 다른 모델/TP 로 확장
* 15-30% — "approximate baseline" 으로 paper framing
* \> 30% — bucketing gap 큰 것, framing pivot

### 알려진 framework artifact

* **NxDI warning "TP degree (X) and KV heads (8) are not divisible.
  Overriding attention sharding strategy to GQA.CONVERT_TO_MHA"**:
  label 만 mislabel (AWS 가 [aws-neuron-sdk #1289](https://github.com/aws-neuron/aws-neuron-sdk/issues/1289)
  에서 인정). 실제 runtime 의 attention 은 `get_shardable_head_counts()`
  의 looser 조건 (`kv < tp` or `kv % tp != 0`) 만 보고 head 수 결정하므로
  TP=1/2 + KV=8 모두 진짜 GQA (TP=1: Q=32, KV=8 / TP=2: Q=16, KV=4 per
  rank). 무시 OK.
* **NKI attention_cte kernel bypass — `attn_kernel_enabled=False`**:
  inf2.xlarge + Llama-3.2-1B + TP < KV (즉 TP=1, 2) + max_model_len
  ≥ 4096 조합에서 NxDI 의 NKI `attention_cte` kernel 이
  compiler verifier 의 `checkDMATranspose` 에 걸림 ("transpose only
  supported for HBM->SB"). 이 stack 의 public report 0건 (LENS 도
  TP≥8 / inf2.24xlarge+ 에서만 NxD-direct 검증). 우리는
  `measure_nxd.py` / `measure_vllm.py` 에 `attn_kernel_enabled=False`

  + `attn_block_cte_nki_kernel_enabled=False` + `qkv_kernel_enabled=False`
    강제로 NKI 우회 → native PyTorch attention path 사용. compile 통과,
    결과 numerical 동등, 속도만 NKI 보다 느림.

  **Trade-off**: LENS 의 NKI-on reference 측정 (TP=8, inf2.24xlarge+)
  과는 framework path 다름. 우리 inf2.xlarge 측정 셋 (NxD + vLLM) 은
  내부 self-consistent — 둘 다 NKI off 라 두 측정 직접 비교 + 시뮬레이터
  비교는 fair. LENS reference 는 framework note 로만 reuse.
