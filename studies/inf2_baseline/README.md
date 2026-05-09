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

| 차원 | 값 |
|---|---|
| Dataset | arxiv, cnn, sharegpt, writing_prompts |
| Batch size | 1, 2, 4, 8, 16, 32 (OOM 까지) |
| TP | 1, 2 |
| Source | LENS-NxD, LENS-vLLM, LLMServingSim |

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
붙으면서 HF Hub repo id validator (`HFValidationError: Repo id must
be in the form 'repo_name' or 'namespace/repo_name'`) 에 거부됨.
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

| Script | 필요 패키지 | venv (default name) |
|---|---|---|
| `profile_neuron.py` (sweep) | `torch_neuronx` | `aws_neuronx_venv_pytorch_2_9` |
| `measure_nxd.py` | `neuronx_distributed_inference` | `aws_neuronx_venv_pytorch_2_9_nxd_inference` |
| `measure_vllm.py` | `vllm` (vllm-neuronx) | `aws_neuronx_venv_pytorch_2_9_nxd_inference` (NxDI venv 안에 vllm 같이 있는 경우 흔함) — 확인 후 결정 |

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

### 3. Simulator 실행 (로컬 docker)

3.a. 컨테이너 진입 + 빌드 (한 번만)

```bash
# 호스트에서 — 시뮬레이터 컨테이너 시작 + 진입
./scripts/docker-sim.sh

# 컨테이너 안에서 — ASTRA-Sim + Chakra 빌드 (한 번만, 이미 빌드되어 있으면 skip)
./scripts/compile.sh
```

3.b. 시뮬레이션 실행 (컨테이너 안에서)

LENS 측정 두 path 모두 `enable_chunked_prefill=False`,
`enable_prefix_caching=False` 로 돌므로 시뮬레이터도 일치시킴 — 안 그러면
schedule 차이가 framework 차이로 잡혀버림.

```bash
# studies/inf2_baseline/results/sim/<model>/tp<N>/bs<B>/<dataset>.csv 로 저장
for tp in 1 2; do
  for ds in arxiv cnn sharegpt writing_prompts; do
    for bs in 1 2 4 8 16 32; do
      python -m serving \
        --cluster-config configs/cluster/inf2_xlarge_llama1b_tp${tp}.json \
        --dataset studies/inf2_baseline/workloads/${ds}_bs${bs}.jsonl \
        --output studies/inf2_baseline/results/sim/Llama-3.2-1B/tp${tp}/bs${bs}/${ds}.csv \
        --max-num-seqs ${bs} \
        --no-enable-chunked-prefill \
        --no-enable-prefix-caching \
        --dtype bfloat16
    done
  done
done
```

`--max-num-batched-tokens` 의 default 2048 도 LENS 의 max_model_len=8192 와
다르지만, 우리 워크로드의 단일 prefill 이 8192 를 안 넘으므로 보통 무관 —
큰 prompt 가 더 많은 step 에 나뉘어 처리될 뿐. 정확히 LENS 동작에 맞추려면
`--max-num-batched-tokens 8192` 추가.

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
