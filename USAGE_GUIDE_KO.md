# LLMServingSim 2.0 사용 가이드 — NPU LLM Inference Latency 예측

> **목적**: Inferentia 2, TPU v4 / v5e / v6 같은 단일 NPU 위에서 single inference 또는 Tensor Parallel(TP) 만 적용한 LLM inference latency 예측을 LLMServingSim 2.0 으로 수행하기 위한 실전 매뉴얼.
> **대상**: 본 리포지토리(`/Users/swjeong/Desktop/LLMServingSim`) 를 baseline 으로 사용하려는 연구자.
> **scope 제외**: PD(Prefill/Decode) disaggregation, MoE, multi-instance, prefix sharing, PIM, sub-batch interleaving — 사용자의 시나리오에서 빠짐.

---

## 0. TL;DR — 가장 짧은 사용 흐름

```bash
# 1) 시뮬레이터 컨테이너 띄우기 (호스트에서)
cd /Users/swjeong/Desktop/LLMServingSim
./scripts/docker-sim.sh                        # astrasim/tutorial-micro2024 컨테이너 시작

# 2) ASTRA-Sim + Chakra 빌드 (컨테이너 안에서)
./scripts/compile.sh

# 3) 프로파일 데이터(perf bundle) 가 있는지 확인
ls profiler/perf/                              # 현재는 RTXPRO6000 만 존재

# 4) 시뮬레이션 실행 (컨테이너 안에서)
python -m serving \
  --cluster-config 'configs/cluster/single_node_single_instance.json' \
  --dtype bfloat16 --block-size 16 \
  --dataset 'workloads/example_trace.jsonl' \
  --output 'outputs/quicktest.csv' \
  --num-reqs 10 --log-interval 1.0
```

위 4번 명령은 `meta-llama/Llama-3.1-8B` 를 `RTXPRO6000` perf 번들 위에서 시뮬레이션한다. **TPU/Inferentia 를 쓰려면 3번 단계에서 그 하드웨어용 perf 번들을 직접 만들어 넣어야 한다** — 자세한 절차는 §5 참조.

---

## 1. 본 리포지토리의 큰 그림

LLMServingSim 2.0 은 세 개의 Python 모듈 + 하나의 C++ 시뮬레이터로 구성된다.

| 모듈 | 진입점 | 역할 | 실행 환경 |
|---|---|---|---|
| **profiler** | `python -m profiler` | 실제 하드웨어에서 vLLM 의 `layerwise_profile` 훅으로 per-layer latency 를 수집해 CSV 로 저장 | vLLM 컨테이너 (NVIDIA GPU 전용) |
| **serving** | `python -m serving` | 시뮬레이션 메인. profiler CSV + cluster config 를 읽어 ASTRA-Sim 을 구동하고 per-request latency CSV 를 생성 | 시뮬레이터 컨테이너 |
| **bench** | `python -m bench` | 시뮬레이터 검증용. 실제 vLLM 으로 동일 workload 를 돌리고 시뮬레이션 결과와 비교 | vLLM 컨테이너 |
| **ASTRA-Sim** | C++ 바이너리 | 사이클 단위 네트워크/메모리 시뮬레이터. `serving` 이 자식 프로세스로 띄워서 stdin/stdout IPC 로 통신 | 시뮬레이터 컨테이너 (빌드 필요) |

논문 Fig. 1 의 *Serving Engine* 이 `serving/` 모듈에 해당하고, *System Simulator* 가 ASTRA-Sim 자식 프로세스다. *Profiles* 는 `profiler/perf/` 디렉토리의 CSV 들이다.

### 1.1 데이터 흐름 (사용자 입장에서 본)

```
사용자가 만들어야 하는 것
├── (A) configs/cluster/<my>.json     ← 클러스터/하드웨어 토폴로지
├── (B) configs/model/<org>/<m>.json  ← 모델 architecture (HF config 부분집합)
├── (C) workloads/<my>.jsonl          ← 요청 트레이스 (input_toks, output_toks, arrival_time_ns ...)
└── (D) profiler/perf/<HW>/<MODEL>/<variant>/tp<N>/{dense,per_sequence,attention,moe}.csv
                                        + meta.yaml
        ← 하드웨어별 per-operator latency 번들. 가장 손이 많이 가는 부분.

런타임에 자동 생성되는 것
├── astra-sim/inputs/network/network.yml     ← config_builder.py 가 매 실행마다 재생성
├── astra-sim/inputs/system/system.json
├── astra-sim/inputs/memory/memory_expansion.json
└── astra-sim/workload/*.et                  ← 매 iteration 마다 Chakra 가 만드는 protobuf trace
```

### 1.2 디렉토리 지도 (요약)

| 경로 | 용도 |
|---|---|
| `serving/__main__.py` | 시뮬레이션 진입점, main loop, CLI |
| `serving/core/scheduler.py` | vLLM 스타일 continuous batching |
| `serving/core/trace_generator.py` | profiler CSV 조회 → 텍스트 trace |
| `serving/core/graph_generator.py` | 텍스트 trace → Chakra `.et` |
| `serving/core/controller.py` | ASTRA-Sim 자식 프로세스 IPC |
| `serving/core/config_builder.py` | cluster JSON → ASTRA-Sim 입력 파일 |
| `serving/core/memory_model.py` | KV cache, weight 메모리 추적 |
| `configs/cluster/*.json` | 클러스터 토폴로지 (npu 수, tp_size, link bw 등) |
| `configs/model/<org>/<m>.json` | HF 모델 config 부분집합 |
| `profiler/perf/<HW>/<MODEL>/<variant>/tp<N>/` | per-hardware latency CSV |
| `profiler/models/<model_type>.yaml` | architecture catalog (vLLM 클래스명 매핑) |
| `workloads/*.jsonl` | 요청 트레이스 |
| `astra-sim/` | C++ 시뮬레이터 (git submodule) |
| `scripts/docker-sim.sh` | 시뮬레이터 컨테이너 |
| `scripts/docker-vllm.sh` | vLLM (profiler/bench) 컨테이너 |
| `scripts/compile.sh` | ASTRA-Sim + Chakra 빌드 |

---

## 2. 사용자의 시나리오에서 가장 중요한 한 가지

> **본 리포의 vLLM 기반 profiler 는 NVIDIA GPU 전용이다.**

사용자가 관심 있는 하드웨어 — Inferentia 2 (Neuron SDK), TPU v4 / v5e / v6 (XLA/TPU) — 는 **모두** `python -m profiler` 로 직접 프로파일할 수 없다. 이유는 다음과 같다.

1. `profiler/core/engine.py` 가 `vllm.LLM(...)` 으로 vLLM 엔진을 부팅하고 `tensor_parallel_size=1` 단일 GPU 전제 위에서 동작한다.
2. `profiler/core/hooks/extension.py` 가 vLLM 의 worker extension hook 을 통해 `layerwise_profile` 결과(CUDA kernel timing) 를 수집한다.
3. `profiler/core/writer.py::_gpu_name()` 이 `torch.cuda.get_device_name(0)` 로 GPU 식별. CUDA 가 없으면 동작 불능.
4. 코드베이스 전체에서 `tpu`, `inferentia`, `trainium`, `trillium` 키워드로 분기되는 코드 경로는 **존재하지 않는다** (검증 완료).
5. CHANGELOG line 304 에 "Hardware performance profiles for TPU-v6e-1" 항목이 있으나, 이는 IISWC 24 / ISPASS 26 artifact 브랜치(`origin/iiswc24-artifact`, `origin/ispass26-artifact`) 에만 들어있을 가능성이 높다. 현재 `main` 브랜치의 `profiler/perf/` 에는 `RTXPRO6000` 만 존재.

논문(Fig. 9) 에서 TPU-v6e-1 검증을 보여주지만, 그 케이스의 profiler 는 "extend the profiler to a TPU-v6e-1 instance ... via a TPU-specific profiler" 라는 한 줄로 처리되어 있고 본 리포에 공개된 코드 경로가 아니다.

### 2.1 그래서 어떻게?

신규 NPU 에 대해서는 **CSV 번들을 직접 합성(synthesize) 한다**. 이것이 공식 권장 경로이며 `docs/docs/profiler/adding-hardware.md` 의 "Adding non-GPU hardware" 섹션이 그것을 설명한다. 시뮬레이터 입장에서는 CSV 번들이 어떻게 만들어졌는지 모르며, 형식만 맞으면 그대로 동작한다.

CSV 번들의 데이터 출처로는 (정확도 순서대로):

1. **벤더 cycle-accurate / analytical 모델** — Inferentia 의 `neuron-profile`, TPU 의 XLA profiler / hardware perf model. 가장 정확.
2. **외부 시뮬레이터 또는 roofline 모델** — GEMM-perf, FlashAttention 분석 모델 등. 중간 정확도.
3. **datasheet + 공개 벤치마크 손계산** — 최후 수단. peak FLOPs 와 HBM bandwidth 만으로 roofline 계산. 낙관적 예측.

§5 에서 단계별로 다룬다.

---

## 3. 환경 준비 (호스트)

### 3.1 사전 요구사항

- **OS**: Linux (검증) 또는 macOS (Docker 만 있으면 됨, 현재 사용자 환경)
- **Docker**: 컨테이너 두 종류를 사용
  - `astrasim/tutorial-micro2024` — 시뮬레이터용. GPU 불필요.
  - `vllm/vllm-openai:v0.19.0` — profiler / bench 용. NVIDIA GPU 필요. 사용자가 NPU 만 가지고 있다면 이 컨테이너는 쓰지 않는다.
- **git submodule**: `astra-sim/` 은 submodule. 처음 클론 시 `--recurse-submodules` 필요.
- **HuggingFace 토큰** (선택): gated 모델(Llama 등) 의 `config.json` 자동 다운로드용. 없으면 `configs/model/` 에 수동으로 넣어주면 됨.

### 3.2 Submodule 확인

```bash
cd /Users/swjeong/Desktop/LLMServingSim
git submodule status                # astra-sim 이 fetched 상태인지 확인
git submodule update --init --recursive  # 비어있다면 초기화
```

### 3.3 시뮬레이터 컨테이너 띄우기

`scripts/docker-sim.sh` 의 핵심:

```bash
docker run --name servingsim_docker \
  -it \
  -v "$REPO_ROOT":/app/LLMServingSim \
  -w /app/LLMServingSim \
  astrasim/tutorial-micro2024 \
  bash -c "pip3 install pyyaml pyinstrument transformers datasets \
  msgspec scikit-learn xgboost==3.1.2 matplotlib==3.5.3 pandas==1.5.3 \
  numpy==1.23.5 && exec bash"
```

실행:

```bash
./scripts/docker-sim.sh             # 첫 실행 (의존성 설치 포함, 5~10분)
docker start -ai servingsim_docker  # 다음번부터 재접속
```

컨테이너 내부의 작업 디렉토리는 `/app/LLMServingSim`. 호스트의 `/Users/swjeong/Desktop/LLMServingSim` 와 동일.

### 3.4 ASTRA-Sim + Chakra 빌드 (컨테이너 안에서)

`scripts/compile.sh`:

```bash
# 1) Chakra (text trace -> protobuf .et 변환기) 설치
cd astra-sim/extern/graph_frontend/chakra && pip3 install .

# 2) ASTRA-Sim analytical 백엔드 컴파일
cd astra-sim && bash ./build/astra_analytical/build.sh
```

실행:

```bash
./scripts/compile.sh                # 2~5분
```

빌드 후 생성되는 바이너리 (시뮬레이터가 자식 프로세스로 spawn 함):

```
astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra
```

이 경로는 `serving/__main__.py` 안에 하드코드되어 있으니 변경 금지.

---

## 4. 처음 한 번 — 동봉된 RTXPRO6000 perf 로 sanity check

신규 하드웨어 작업 들어가기 전에, 이미 들어있는 `RTXPRO6000` 번들 위에서 시뮬레이션이 잘 도는지 먼저 확인한다.

### 4.1 가장 작은 cluster config

`configs/cluster/single_node_single_instance.json` (그대로):

```json
{
    "num_nodes": 1,
    "link_bw": 16,
    "link_latency": 20000,
    "nodes": [
        {
            "num_instances": 1,
            "cpu_mem": { "mem_size": 512, "mem_bw": 256, "mem_latency": 0 },
            "instances": [
                {
                    "model_name": "meta-llama/Llama-3.1-8B",
                    "hardware": "RTXPRO6000",
                    "npu_mem": { "mem_size": 96, "mem_bw": 1597, "mem_latency": 0 },
                    "num_npus": 1,
                    "tp_size": 1,
                    "pd_type": null
                }
            ]
        }
    ]
}
```

필드 의미는 §6.1 표 참조.

### 4.2 실행

컨테이너 안에서:

```bash
cd /app/LLMServingSim
python -m serving \
  --cluster-config 'configs/cluster/single_node_single_instance.json' \
  --dtype bfloat16 --block-size 16 \
  --dataset 'workloads/example_trace.jsonl' \
  --output 'outputs/sanity.csv' \
  --num-reqs 5 \
  --log-interval 1.0
```

> `serving/__main__.py` 는 main 함수 진입 직후 `os.chdir("astra-sim")` 으로 cwd 를 옮긴다. 따라서 `--cluster-config` 등은 **리포 루트 기준 상대 경로**여도 코드에서 자동으로 `../` 처리되어 정상 작동한다. 이 동작은 `serving/__main__.py:63-69` 에 코멘트와 함께 명시되어 있다.

### 4.3 기대 출력

표준출력에 다음과 같은 라인들이 흘러간다:

```
[INFO] step=12 batch=3 prompt_t=1.2k tok/s decode_t=420 tok/s npu_mem=14.3 GB
...
▶ Simulation results...
Total simulation time: 0h 0m 12.345s
─── Throughput Results ──
Total requests:                                   5
Total clocks (ns):                                12345678900
Total latency (s):                                12.346
...
```

그리고 `outputs/sanity.csv` 가 생성된다. 컬럼은 §7 참조.

이게 깔끔하게 돌면, 코드/빌드/Docker 환경은 OK. 이제 NPU 차례.

---

## 5. 신규 NPU(TPU/Inferentia) perf 번들 만들기 — 핵심 단계

`docs/docs/profiler/adding-hardware.md` 의 "Adding non-GPU hardware" 섹션이 공식 절차다. 본 §은 그것을 사용자 시나리오에 맞춰 풀어쓴 것이다.

### 5.1 결정적 사항: variant 라벨 정하기

번들 경로는 다음과 같이 구성된다.

```
profiler/perf/<HARDWARE>/<MODEL>/<variant>/tp<N>/*.csv
```

| 토큰 | 값 예시 | 결정 규칙 |
|---|---|---|
| `<HARDWARE>` | `Inferentia2`, `TPUv4`, `TPUv5e`, `TPUv6e` | 자유. cluster config 의 `instances[].hardware` 와 정확히 같은 문자열이어야 함. 폴더명 안전 문자만. |
| `<MODEL>` | `meta-llama/Llama-3.1-8B` | HF ID. cluster config 의 `model_name` 과 일치 + `configs/model/<MODEL>.json` 에 동일 ID 의 architecture config 존재. |
| `<variant>` | `bf16`, `fp16`, `bf16-kvfp8` | weight dtype + (선택) `-kv<…>` suffix. `--dtype bfloat16 --kv-cache-dtype auto` 면 `bf16`. `--kv-cache-dtype fp8` 이면 `bf16-kvfp8`. `serving/core/trace_generator.py::resolve_variant()` 가 이 규칙 자체. |
| `<N>` | `1`, `2`, `4` | TP 도수. **TP=1 폴더는 반드시 만들어야 함** (tp_stable layer 가 거기 들어감). |

사용자의 시나리오는 single inference 또는 TP only 이므로, 일단 `tp1/`, 필요하면 `tp2/`, `tp4/`, `tp8/` 정도면 충분.

### 5.2 만들어야 할 CSV 4종 + meta.yaml

| 파일 | 카테고리 | 컬럼 | 설명 |
|---|---|---|---|
| `dense.csv` | dense | `layer,tokens,time_us` | 토큰 선형(token-linear) layer 들. embedding, qkv_proj, o_proj, gate_up_proj, act_fn, down_proj, rotary_emb, layernorm, final_layernorm. |
| `per_sequence.csv` | per_sequence | `layer,sequences,time_us` | sequence 선형 layer. lm_head, sampler. |
| `attention.csv` | attention | `prefill_chunk,kv_prefill,n_decode,kv_decode,time_us` | 4D 그리드. FlashAttention 커널. |
| `moe.csv` | moe | `tokens,activated_experts,time_us` | MoE 모델일 때만. dense 모델은 생략 가능. |
| `meta.yaml` | (메타) | YAML | 변형 폴더 식별, 그리드 범위, skew alpha 기본값. **반드시 필요**. |

선택사항 (없어도 됨):

- `skew.csv`, `skew_fit.csv` — heterogeneous decode 보정용. NPU 데이터 합성에서는 보통 생략하고 `meta.yaml::skew_fit.per_tp.<TP>.alpha_default = 0.3` 정도 상수로 대체. 자세히는 `docs/docs/profiler/skew-alpha-fit.md`.

**시간 단위는 마이크로초(μs)** — 시뮬레이터가 로드 시 `×1000` 해서 ns 로 변환한다 (`serving/core/trace_generator.py::_load_perf_db()`).

### 5.3 어떤 layer 들이 들어가야 하나? — architecture YAML 과 catalog

`profiler/models/<model_type>.yaml` 이 그 모델이 사용하는 canonical layer 들을 선언한다. Llama 3.x 의 경우 (`profiler/models/llama.yaml`):

```yaml
sequence:
  prologue:  [embedding]
  pre_attn:  [layernorm, qkv_proj, rotary_emb, attention]
  post_attn: [o_proj, layernorm]
  mlp_dense: [gate_up_proj, act_fn, down_proj]
  mlp_moe:   []
  head:      [final_layernorm, lm_head, sampler]

catalog:
  dense:
    embedding:  { vllm: VocabParallelEmbedding }
    layernorm:  { vllm: RMSNorm, within: LlamaDecoderLayer, tp_stable: true }
    qkv_proj:   { vllm: QKVParallelLinear }
    rotary_emb: { vllm: Llama3RotaryEmbedding }
    o_proj:     { vllm: RowParallelLinear, within: LlamaAttention }
    gate_up_proj: { vllm: MergedColumnParallelLinear }
    act_fn:     { vllm: SiluAndMul }
    down_proj:  { vllm: RowParallelLinear, within: LlamaMLP }
    final_layernorm: { vllm: RMSNorm, within: LlamaForCausalLM, tp_stable: true }
  per_sequence:
    lm_head: { vllm: LogitsProcessor }
    sampler: { vllm: Sampler, tp_stable: true }
  attention:
    attention: { vllm: Attention }
```

해석:
- `catalog.dense` 의 모든 layer 가 `dense.csv` 의 `layer` 컬럼에 한 번 이상 등장해야 함.
- `catalog.per_sequence` 의 layer 는 `per_sequence.csv` 에.
- `catalog.attention` 의 layer 는 `attention.csv` 에 (이건 layer 이름 컬럼이 없음 — 4D 키가 컬럼).
- `tp_stable: true` 인 layer 는 TP 에 무관하게 동일하므로 `tp1/` 에만 넣으면 시뮬레이터가 다른 TP 에서도 그 값을 사용한다 (`profiler/core/writer.py::replicate_tp_stable()`). 합성 시에는 그냥 모든 `tp<N>/` 에 같은 값을 복제해 넣어도 무방.

### 5.4 CSV grid 를 어떤 점들로 채워야 하나?

GPU profiler 가 자동으로 sweep 하는 축들과 같은 축을 따라가야 한다. 너무 듬성듬성하면 시뮬레이터가 extrapolation 으로 보정하지만, 정확도 저하.

#### dense.csv

축: `tokens`. 시뮬레이터가 한 iteration 에서 보는 총 토큰 수.

권장 grid: `1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048` (이배수 시퀀스). 사용자의 `--max-num-batched-tokens` 상한까지 커버해야 함 (기본 2048).

각 layer 마다 그 grid 의 각 토큰 수에 대한 latency 한 줄씩.

예:
```
layer,tokens,time_us
embedding,1,3.2
embedding,2,3.4
...
qkv_proj,1,5.1
qkv_proj,2,5.5
...
```

#### per_sequence.csv

축: `sequences`. 동시에 디코딩 중인 시퀀스(요청) 수.

권장 grid: `1, 2, 4, 8, 16, 32, 64, 128, 256` — `--max-num-seqs` 상한까지.

```
layer,sequences,time_us
lm_head,1,8.5
lm_head,2,12.4
...
sampler,1,4.1
sampler,2,4.8
...
```

#### attention.csv (가장 복잡)

축 4개:

| 컬럼 | 의미 | 예시 grid |
|---|---|---|
| `prefill_chunk` | 이번 step 에 처리할 prefill 토큰 수 | `0, 16, 32, 64, ..., 2048` |
| `kv_prefill` | 그 prefill 의 KV history 길이 | `0, 512, 1024, 2048, ..., 16384` |
| `n_decode` | 동시 decode 요청 수 | `0, 1, 2, 4, 8, ..., 256` |
| `kv_decode` | 각 decode 의 KV 길이 | `16, 32, 64, ..., 16384` |

순수 prefill 케이스: `n_decode=0, kv_decode=0`. 순수 decode 케이스: `prefill_chunk=0, kv_prefill=0`. 혼합 케이스: 둘 다 양수.

grid 가 너무 크면 합성 비용 폭발. 우선 axis 별 8~10 점 정도로 시작하고, simulation 결과의 "extrapolation" 경고가 자주 뜨면 dense 화. 사용자가 `--enable-chunked-prefill` 끄면 prefill_chunk 축은 0 과 prompt 길이 두 점만 있어도 됨.

#### moe.csv (MoE 모델만)

```
tokens,activated_experts,time_us
1,8,15.3
2,8,22.5
...
```

Llama / Qwen3 dense 모델만 돌릴 거면 생략.

### 5.5 meta.yaml — 최소 형태

`docs/docs/profiler/adding-hardware.md` 가 제시하는 최소 합성 템플릿:

```yaml
profiler_version: "synthetic-v1"
vllm_version: "n/a"
gpu: "TPUv5e"                          # <HARDWARE> 와 동일하게
hardware: "TPUv5e"
profiled_at: "2026-05-07"
architecture: llama                    # configs/model/<model_name>.json 의 model_type 과 일치
model: "meta-llama/Llama-3.1-8B"
variant: "bf16"
tp_degrees: [1, 2, 4]

engine_effective:
  max_num_batched_tokens: 2048         # CSV 가 커버하는 상한과 일치
  max_num_seqs: 256
  dtype: bfloat16
  kv_cache_dtype: auto

attention_grid:
  max_kv: 16384
  chunks: "0, 16-2048 x2"              # 정보 표시용. 실제 시뮬레이터는 CSV 만 봄
  n_decode: "0, 1-256 x2"
  kv: "0, 16-16384 x2"

skew_fit:
  per_tp:
    1: { method: synthetic-constant, alpha_default: 0.3 }
    2: { method: synthetic-constant, alpha_default: 0.3 }
    4: { method: synthetic-constant, alpha_default: 0.3 }
```

`alpha_default` 는 heterogeneous-decode 보정 계수의 fallback 값. 0.3 은 무난한 시작값 (논문 Section IV.A 의 "skew alpha ∈ [0,1]" 정의 기준). 실측이 있으면 더 정확한 값 권장.

### 5.6 실전 합성 레시피 — 어디서 데이터를 가져올까

#### Inferentia 2 (Trn2 family)

- AWS Neuron SDK 의 `neuron-profile` 또는 `neuron-cc` 가 그래프 단위 timing 정보를 제공.
- 가장 정확한 길: `vllm-neuron` 으로 모델을 올린 후 batch shape 을 바꿔가며 e2e latency 를 측정 → operator-level 분해는 손으로.
- 두 번째 길: 각 layer 의 GEMM shape 을 추출 (`q_dim = num_heads * head_dim`, `kv_dim = num_kv_heads * head_dim`, `intermediate_size` 등은 `configs/model/<m>.json` 에서) → Inferentia 2 의 peak BF16 TFLOPs (~190 TFLOPS per chip) 와 HBM bandwidth (~820 GB/s) 로 roofline.
- attention 은 Neuron 의 `nki` 또는 미리 작성된 FlashAttention kernel 의 시간 모델 사용.

#### TPU v4 / v5e / v6e

- TPU 는 XLA Profiler (`tf.profiler` / `jax.profiler`) 로 op-level timing 추출 가능. 단, vLLM-TPU (`vllm/vllm-tpu` image) 는 v0.19.0 시점에서는 single-instance dense 만 안정.
- 가장 정확한 길: 하드웨어가 있다면 vLLM-TPU 로 단일 GPU 설정 모방하듯이 sweep 을 돌리고 timing 추출. 본 리포의 profiler 는 못 쓰지만, 사용자가 외부 스크립트로 같은 grid 를 sweep 하면 됨.
- 두 번째 길: TPU 백서 (예: TPU v6e 의 ~918 TFLOPS BF16, ~1.6 TB/s HBM bw) + Megatron-LM 또는 `jax.profile_function` 으로 모델 단위 측정 → 분해.

세부 측정을 하지 못하는 경우, **roofline 손계산** 도 첫 가설로 충분히 유의미하다. 단 §5.7 의 검증을 반드시 수행.

### 5.7 합성 후 sanity check

1. **smoke test**: §4.2 의 명령에서 `hardware` 만 새 라벨로 바꾼 cluster config 로 실행.
2. **시작 시 경고 확인**: 시뮬레이터는 시작 시점에 cluster config 의 `--max-num-batched-tokens`/`--max-num-seqs` 가 `meta.yaml::engine_effective` 의 sweep 범위를 넘으면 한 번씩 warning 을 찍는다. extrapolation 경고가 많으면 grid 를 더 촘촘히.
3. **매크로 sanity**:
   - `prompt_t (tok/s)` × 모델 크기(GFLOPs/token) ≈ 하드웨어 peak compute 의 50~80% 정도면 정상.
   - `decode_t` 는 메모리 bandwidth bound 이므로 모델 weight 크기 / mem_bw 와 부합해야.
4. **공개 벤치 비교**: 가능하면 같은 (hardware, model, batch) 조합의 공개 TTFT/TPOT 와 ±20% 이내인지.

---

## 6. cluster config / model config — 사용자 시나리오에서 만질 부분

### 6.1 cluster config 필드 매뉴얼

`configs/cluster/single_node_single_instance.json` 의 각 필드:

| 위치 | 필드 | 의미 | 사용자 시나리오에서 |
|---|---|---|---|
| top | `num_nodes` | 노드(서버) 수 | **항상 1** |
| top | `link_bw` | 인터-노드 링크 GB/s | 단일 노드면 무관 (하지만 값 있어야 함) |
| top | `link_latency` | 링크 latency ns | 단일 노드면 무관 |
| node | `num_instances` | 노드 안 인스턴스(독립 LLM serving 엔진) 수 | **항상 1** |
| node.cpu_mem | `mem_size` | CPU(host) 메모리 GB | LLM offloading 안 하면 의미 작음. 256~512 GB 권장. |
| node.cpu_mem | `mem_bw` | CPU mem bw GB/s | 위와 같음. 256 GB/s 정도. |
| node.cpu_mem | `mem_latency` | CPU mem latency ns | 0 그대로. |
| inst | `model_name` | HF ID. **`configs/model/<model_name>.json` 와 정확히 일치해야** | `meta-llama/Llama-3.1-8B` 등 |
| inst | `hardware` | **`profiler/perf/<hardware>/` 와 정확히 일치해야** | `Inferentia2`, `TPUv5e` 등 |
| inst.npu_mem | `mem_size` | NPU 메모리 GB | TPU v6e: 32, TPU v5e: 16, TPU v4: 32, Inferentia2 (per chip): 32 |
| inst.npu_mem | `mem_bw` | NPU 메모리 bw GB/s | TPU v6e: ~1640, TPU v5e: ~819, TPU v4: ~1228, Inf2: ~820 |
| inst.npu_mem | `mem_latency` | NPU mem latency ns | 0 그대로. |
| inst | `num_npus` | 인스턴스 NPU 수 (= `tp_size × pp_size`) | TP=1: 1. TP=2: 2. |
| inst | `tp_size` | TP 도수 | 1 (single inf) 또는 2/4/8 (TP only) |
| inst | `pp_size` | PP 도수. 생략 시 1 | **항상 1** (사용자 시나리오) |
| inst | `ep_size` | EP 도수. dense 모델에서는 1 | dense 모델이면 무관 |
| inst | `pd_type` | `null`, `"prefill"`, `"decode"` | **`null`** (PD disagg 안 함) |
| inst | `dp_group` | DP 그룹 ID. 다른 인스턴스와 공유 시 같은 문자열 | 생략 또는 `null` |

### 6.2 사용자용 cluster config 템플릿 4가지

#### (a) Inferentia 2, single inference (TP=1)

`configs/cluster/inf2_single.json`:

```json
{
    "num_nodes": 1,
    "link_bw": 16,
    "link_latency": 20000,
    "nodes": [
        {
            "num_instances": 1,
            "cpu_mem": { "mem_size": 256, "mem_bw": 256, "mem_latency": 0 },
            "instances": [
                {
                    "model_name": "meta-llama/Llama-3.1-8B",
                    "hardware": "Inferentia2",
                    "npu_mem": { "mem_size": 32, "mem_bw": 820, "mem_latency": 0 },
                    "num_npus": 1,
                    "tp_size": 1,
                    "pd_type": null
                }
            ]
        }
    ]
}
```

#### (b) TPU v5e, single inference (TP=1)

```json
{
    "num_nodes": 1, "link_bw": 16, "link_latency": 20000,
    "nodes": [{
        "num_instances": 1,
        "cpu_mem": { "mem_size": 256, "mem_bw": 256, "mem_latency": 0 },
        "instances": [{
            "model_name": "meta-llama/Llama-3.1-8B",
            "hardware": "TPUv5e",
            "npu_mem": { "mem_size": 16, "mem_bw": 819, "mem_latency": 0 },
            "num_npus": 1, "tp_size": 1, "pd_type": null
        }]
    }]
}
```

#### (c) TPU v4, TP=4

```json
{
    "num_nodes": 1, "link_bw": 50, "link_latency": 1000,
    "nodes": [{
        "num_instances": 1,
        "cpu_mem": { "mem_size": 512, "mem_bw": 256, "mem_latency": 0 },
        "instances": [{
            "model_name": "meta-llama/Llama-3.1-70B",
            "hardware": "TPUv4",
            "npu_mem": { "mem_size": 32, "mem_bw": 1228, "mem_latency": 0 },
            "num_npus": 4, "tp_size": 4, "pd_type": null
        }]
    }]
}
```

→ TP=4 이면 `profiler/perf/TPUv4/meta-llama/Llama-3.1-70B/<variant>/tp4/` 가 반드시 있어야 함.

#### (d) TPU v6e, TP=2

```json
{
    "num_nodes": 1, "link_bw": 100, "link_latency": 500,
    "nodes": [{
        "num_instances": 1,
        "cpu_mem": { "mem_size": 256, "mem_bw": 256, "mem_latency": 0 },
        "instances": [{
            "model_name": "meta-llama/Llama-3.1-8B",
            "hardware": "TPUv6e",
            "npu_mem": { "mem_size": 32, "mem_bw": 1640, "mem_latency": 0 },
            "num_npus": 2, "tp_size": 2, "pd_type": null
        }]
    }]
}
```

### 6.3 model config

`configs/model/<org>/<name>.json` 은 HF `config.json` 의 부분집합. 시뮬레이터가 다음 필드를 읽는다 (`serving/core/utils.py::get_config()`):

- `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `vocab_size`
- `head_dim` (있으면; Qwen3 등은 명시; 없으면 `hidden_size // num_attention_heads` 로 fallback)
- `model_type` — `profiler/models/<model_type>.yaml` 매칭에 사용
- `torch_dtype` — `--dtype` 미지정 시 기본
- MoE 의 경우 `num_local_experts`, `num_experts_per_tok`

이미 들어있는 모델: `meta-llama/Llama-3.1-8B.json`, `meta-llama/Llama-3.1-70B.json`, `Qwen/Qwen3-32B.json`, `Qwen/Qwen3-30B-A3B-Instruct-2507.json`, `mistralai/Mixtral-8x7B-v0.1.json`, `microsoft/Phi-mini-MoE-instruct.json`.

새 모델을 쓰고 싶으면 `configs/model/<org>/<name>.json` 에 HF `config.json` 을 그대로 떨어뜨리면 됨 (HF Hub 에서 받아와도 OK).

---

## 7. 시뮬레이션 실행 — CLI 레퍼런스 (사용자 시나리오 기준 간추림)

### 7.1 핵심 플래그

```bash
python -m serving \
  --cluster-config <path>          # 위 §6 의 클러스터 JSON
  --dataset <path.jsonl>           # 요청 트레이스 (없으면 코드 내 수동 추가만 가능)
  --output <path.csv>              # per-request 결과 CSV
  --dtype {bfloat16,float16,float32,fp8,int8}
                                   # 미지정 시 model config 의 torch_dtype, 그것도 없으면 bf16
  --kv-cache-dtype {auto,fp8}      # auto 면 weight dtype 따라감
  --max-num-batched-tokens 2048    # iteration 당 토큰 예산
  --max-num-seqs 128               # iteration 당 동시 시퀀스 상한
  --block-size 16                  # KV cache block 크기 (토큰)
  --num-reqs 0                     # dataset 에서 로드할 요청 수 (0 = 전체)
  --log-interval 1.0               # throughput 로그 출력 간격 (초)
  --log-level WARNING              # WARNING | INFO | DEBUG
  --network-backend analytical     # analytical(빠름) | ns3(상세, WIP)
```

### 7.2 사용자 시나리오에서 보통 켜고/끄는 플래그

| 플래그 | 기본 | 사용자 시나리오 권장 |
|---|---|---|
| `--enable-prefix-caching` | True | 기본 그대로. 단일 인스턴스에서도 재사용 측정 가능. |
| `--enable-chunked-prefill` | True | 기본 그대로. vLLM v1 동작. |
| `--enable-block-copy` | True | 기본 그대로. layer 만큼 trace 반복 → 매우 빠름. |
| `--enable-prefix-sharing` | False | 끔 (multi-instance 기능). |
| `--enable-local-offloading` | False | 끔 (offloading 시나리오 아님). |
| `--enable-attn-offloading` | False | 끔 (PIM 시나리오). |
| `--enable-sub-batch-interleaving` | False | 끔. |
| `--prioritize-prefill` | False | 끔. vLLM 기본은 안 함. |
| `--skip-prefill` | False | 끔. 보통 prefill+decode 같이 측정. decode-only 측정하려면 켬. |

### 7.3 실행 한 줄 예시들

#### 예시 1: TPU v5e, Llama-3.1-8B, single inference, 100 req

```bash
python -m serving \
  --cluster-config 'configs/cluster/tpuv5e_single.json' \
  --dtype bfloat16 \
  --dataset 'workloads/example_trace.jsonl' \
  --output 'outputs/tpuv5e_llama8b.csv' \
  --num-reqs 100 \
  --log-interval 1.0
```

#### 예시 2: Inferentia 2, Llama-3.1-8B, decode-only (prefill 결과는 별도)

```bash
python -m serving \
  --cluster-config 'configs/cluster/inf2_single.json' \
  --dtype bfloat16 \
  --skip-prefill \
  --dataset 'workloads/example_trace.jsonl' \
  --output 'outputs/inf2_decodeonly.csv' \
  --num-reqs 50
```

#### 예시 3: TPU v4, Llama-3.1-70B, TP=4, ShareGPT 300 요청

```bash
python -m serving \
  --cluster-config 'configs/cluster/tpuv4_tp4_70b.json' \
  --dtype bfloat16 \
  --dataset 'workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl' \
  --output 'outputs/tpuv4_70b_sharegpt.csv' \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 256 \
  --log-interval 1.0
```

(원래 ShareGPT 트레이스의 token id 는 8B 토크나이저 기준이지만, 시뮬레이터는 token id 의 의미는 모르고 길이와 hash 만 보므로 다른 모델에서도 그대로 사용 가능.)

---

## 8. 결과 해석

### 8.1 stdout 로그 (실시간)

매 `--log-interval` 초마다:

```
[INFO] step=42 batch=8 prompt_t=1.2k tok/s decode_t=420 tok/s npu_mem=88.4 GB
```

해석:
- `step` — 시뮬레이션 iteration 번호
- `batch` — 그 iteration 의 배치 크기
- `prompt_t` — 평균 prompt(=prefill) throughput (prefix cache 히트 토큰 포함)
- `decode_t` — 평균 decode throughput
- `npu_mem` — NPU 메모리 사용량 (KV cache + weight)

종료 시 `▶ Simulation results...` 블록에 총합:

```
Total requests:                                  100
Total clocks (ns):                               12345678900
Total latency (s):                               12.346
Total input tokens:                              5000
Total generated tokens:                          20000
Request throughput (req/s):                      8.10
Average prompt throughput (tok/s):               405.0
Average generation throughput (tok/s):           1620.0
Total token throughput (tok/s):                  2025.0
```

### 8.2 per-request CSV (`--output` 으로 지정한 파일)

컬럼:

| 컬럼 | 단위 | 의미 |
|---|---|---|
| `instance id` | int | 처리한 인스턴스 (single 이면 0) |
| `request id` | int | 라우터가 부여한 단조증가 ID |
| `model` | str | 모델 이름 |
| `input` | int | 프롬프트 토큰 (prefix cache 히트 포함) |
| `output` | int | 생성된 토큰 |
| `arrival` | ns | 요청 도착 시각 |
| `end_time` | ns | 마지막 토큰 생성 완료 시각 |
| `latency` | ns | end-to-end = `end_time - arrival` |
| `queuing_delay` | ns | 도착 ~ 첫 스케줄 사이 대기 |
| `TTFT` | ns | Time To First Token |
| `TPOT` | ns | Time Per Output Token (평균) |
| `ITL` | str | 토큰간 latency 리스트 (Python literal) |

**모든 시간은 ns**. 변환:
- `ms = ns / 1e6`, `s = ns / 1e9`

분석 코드 예시 (Python):

```python
import pandas as pd, ast
df = pd.read_csv("outputs/tpuv5e_llama8b.csv")
df["TTFT_ms"] = df["TTFT"] / 1e6
df["TPOT_ms"] = df["TPOT"] / 1e6
df["latency_s"] = df["latency"] / 1e9

print(f"평균 TTFT (ms): {df['TTFT_ms'].mean():.2f}")
print(f"p50/p99 TTFT (ms): {df['TTFT_ms'].quantile(0.5):.2f} / {df['TTFT_ms'].quantile(0.99):.2f}")
print(f"평균 TPOT (ms): {df['TPOT_ms'].mean():.2f}")

# ITL 리스트 풀기
df["ITL_list"] = df["ITL"].apply(ast.literal_eval)
df["ITL_p50_ms"] = df["ITL_list"].apply(lambda xs: pd.Series(xs).quantile(0.5) / 1e6)
```

---

## 9. 자주 빠지는 함정 (CLAUDE.md "Common Pitfalls" 발췌 + 사용자 시나리오 보강)

1. **`hardware` 라벨 불일치** — cluster config 의 `hardware` 와 `profiler/perf/<HW>/` 의 폴더명이 글자 단위로 같아야 한다. `TPUv5e` ≠ `TPU-v5e`.
2. **`model_name` 불일치** — cluster config, `configs/model/<...>.json`, `profiler/perf/<HW>/<MODEL>/...` 세 곳 모두 정확히 동일해야 함.
3. **TP=1 폴더 누락** — `tp_size=1` 안 쓰더라도 `tp1/` 은 반드시 있어야 함 (tp_stable layer 들이 거기서 복제됨).
4. **시간 단위 혼동** — profiler CSV 는 **마이크로초(μs)**, 시뮬레이터 내부 및 결과 CSV 는 **나노초(ns)**.
5. **`hidden_size == num_heads × head_dim` 가정** — Qwen3 처럼 `head_dim` 이 따로 명시되는 모델 있음. 항상 `head_dim` 을 explicit 으로 쓰라.
6. **canonical layer 이름** — `dense.csv` 의 `layer` 컬럼은 반드시 `profiler/models/<model_type>.yaml::catalog` 의 키 중 하나. 새 layer 이름 만들지 말 것.
7. **`astra-sim/inputs/*` 직접 편집** — `serving/core/config_builder.py` 가 매 실행마다 덮어씀. cluster config 만 고쳐라.
8. **submodule 미초기화** — `astra-sim/` 폴더가 비어있으면 `git submodule update --init --recursive` 필요.
9. **컨테이너 안에서 빌드** — `compile.sh` 는 `astrasim/tutorial-micro2024` 컨테이너 안에서 실행해야 함. 호스트에서 돌리면 의존성 미스.
10. **macOS 호스트** — Docker Desktop 으로 시뮬레이터 컨테이너는 잘 돌지만, vLLM 컨테이너는 NVIDIA GPU 필요라서 사실상 불가. 사용자가 NPU 만 가지고 있다면 vLLM 컨테이너 자체가 필요 없으므로 문제 없음.
11. **Working directory 헷갈림** — `serving/__main__.py` 가 `os.chdir("astra-sim")` 호출. 코드 안에서 보는 상대 경로는 모두 `astra-sim/` 기준. 호스트에서 `python -m serving` 실행하는 cwd 는 리포 루트여야 함.

---

## 10. (선택) Bench 모듈 — 실측 vLLM 대비 검증

본 사용자처럼 NPU 만 가진 경우, bench 모듈은 **검증 자체가 불가능** 하다 (실측이 NVIDIA GPU 의 vLLM 으로만 동작). 그럼에도 다음 두 시나리오에서 유용:

- 사용자가 NVIDIA GPU 도 함께 쓰고 있어서 GPU 케이스로 simulator → vLLM 매칭을 보고 싶을 때.
- 합성한 NPU CSV 의 신뢰도를 GPU 패턴과 교차 검증할 때 (모델 동작 패턴 확인 용도).

기본 사용:

```bash
# 1) 실측 (vLLM 컨테이너 안)
./bench/bench.sh                  # 환경변수 MODEL, TP 등 편집

# 2) 시뮬 (시뮬레이터 컨테이너 안)
python -m serving ... --output outputs/sim.csv 2>&1 | tee outputs/sim.log

# 3) 비교
./bench/validate.sh bench/results/<run_id> outputs/sim.csv outputs/sim.log eval
```

비교 결과는 PDF 플롯 + summary.txt 로 `bench/results/<run_id>/<output_subdir>/` 에 저장.

---

## 11. workload 트레이스 만들기

`workloads/example_trace.jsonl` 의 한 줄:

```json
{"input_toks":10,"output_toks":70,"arrival_time_ns":46926808,"input_tok_ids":[1,...,10],"output_tok_ids":[11,...,80]}
```

필드:
- `input_toks` — prompt 토큰 수
- `output_toks` — 생성 토큰 수 (실제 vLLM bench 와 매칭하려고 미리 fix)
- `arrival_time_ns` — 요청 도착 시각 (ns). `0` 이면 시뮬 시작 시점.
- `input_tok_ids`, `output_tok_ids` — prefix cache hash 용. 길이만 맞으면 됨.

ShareGPT 같은 실데이터에서 만들려면:

```bash
# vLLM 컨테이너 안에서 (token id 만 뽑는 모드)
python -m workloads.generators sharegpt \
    --model meta-llama/Llama-3.1-8B \
    --num-reqs 300 --sps 10 --seed 42 \
    --output workloads/my-sharegpt-300-sps10.jsonl
```

`sps` = sessions per second (Poisson). 사용자가 NVIDIA GPU 가 없으면 `--use-vllm` 모드는 못 씀. 위처럼 토크나이저-only 모드로 충분.

---

## 12. 작업 순서 체크리스트 (사용자용)

- [ ] 호스트에서 `git clone --recurse-submodules` 또는 기존 클론에서 `git submodule update --init --recursive`
- [ ] `./scripts/docker-sim.sh` 로 시뮬레이터 컨테이너 띄우기
- [ ] 컨테이너 안에서 `./scripts/compile.sh` 로 ASTRA-Sim + Chakra 빌드
- [ ] `python -m serving --cluster-config configs/cluster/single_node_single_instance.json --dataset workloads/example_trace.jsonl --output outputs/sanity.csv --num-reqs 5` 로 sanity check (RTXPRO6000 perf 사용)
- [ ] 사용자의 모델(`meta-llama/Llama-3.1-8B` 등) 의 model config 가 `configs/model/` 에 있는지 확인. 없으면 HF Hub 에서 `config.json` 을 받아 그대로 떨어뜨리기
- [ ] **신규 NPU 의 perf 번들 합성** (가장 큰 작업, §5 참조)
   - [ ] `<HARDWARE>` 라벨 정함 (예: `TPUv5e`)
   - [ ] `dense.csv` 합성 — token grid 8~12 점, layer 별 latency
   - [ ] `per_sequence.csv` 합성 — sequence grid 8 점
   - [ ] `attention.csv` 합성 — 4D grid (가장 손이 많이 감)
   - [ ] (MoE 모델이면) `moe.csv` 합성
   - [ ] `meta.yaml` 작성, `alpha_default: 0.3`
   - [ ] `tp1/`, 필요시 `tp2/`, `tp4/` 폴더 분리 (tp_stable layer 는 동일 값 복제 OK)
- [ ] 새 cluster config 작성 (§6.2 템플릿 변형)
- [ ] 시뮬레이션 실행, `outputs/<…>.csv` 생성
- [ ] CSV 분석 — TTFT, TPOT, end-to-end latency 추출
- [ ] (선택) 공개 벤치마크와 ±20% 이내 정합성 확인. 안 맞으면 grid 촘촘히 / roofline → 측정 기반 데이터로 교체

---

## 13. 빠른 참조 — 명령 모음

```bash
# 컨테이너 / 빌드
./scripts/docker-sim.sh                                  # 컨테이너 시작 (초회)
docker start -ai servingsim_docker                       # 재접속
./scripts/compile.sh                                     # ASTRA-Sim + Chakra 빌드

# 시뮬레이션 (예시)
python -m serving \
  --cluster-config 'configs/cluster/<my>.json' \
  --dtype bfloat16 \
  --dataset 'workloads/example_trace.jsonl' \
  --output 'outputs/result.csv' \
  --num-reqs 100 --log-interval 1.0

# workload 생성 (vLLM 컨테이너, NVIDIA GPU 환경)
python -m workloads.generators sharegpt \
  --model meta-llama/Llama-3.1-8B --num-reqs 300 --sps 10 \
  --output workloads/sharegpt-300.jsonl

# 결과 빠른 분석
python -c "
import pandas as pd
df = pd.read_csv('outputs/result.csv')
print(df[['TTFT','TPOT','latency']].describe() / 1e6)
"
```

---

## 14. 참고 문서 위치

리포 안에 이미 있는 공식 문서들 (한글 이 문서와 함께 읽으면 좋음):

- `README.md` — 짧은 about + getting started
- `CLAUDE.md` (= `AGENTS.md`) — 코드베이스 전체 architecture 설명. 가장 정보량 많음.
- `CHANGELOG.md` — 버전별 변화. TPU v6e profile 자취도 여기서 확인 (line 304).
- `docs/docs/getting-started/quickstart.md` — 1분 quickstart
- `docs/docs/getting-started/installation/simulator.mdx` — 컨테이너 설치 절차
- `docs/docs/profiler/adding-hardware.md` — **신규 하드웨어 추가 공식 가이드 (사용자 시나리오 핵심)**
- `docs/docs/profiler/output-bundle.md` — CSV 번들 정확한 schema
- `docs/docs/profiler/skew-alpha-fit.md` — heterogeneous decode 보정
- `docs/docs/simulator/reading-output.md` — 결과 해석 상세
- `docs/docs/reference/cli-flags.md` — 전체 CLI 플래그
- `docs/docs/reference/cluster-config.md` — cluster JSON 스키마
- `docs/docs/reference/model-config.md` — model JSON 스키마
- `docs/docs/reference/trace-format.md` — workload JSONL 스키마
- 외부: <https://llmservingsim.ai> (Docusaurus 사이트, 같은 docs/ 의 호스팅판)
- 논문: ISPASS 2026 "LLMServingSim 2.0: A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure", Jaehong Cho et al. (arXiv:2602.23036)

---

## 15. 한줄 요약

1. 시뮬레이터 자체는 NPU/GPU 무관 — **CSV 번들 + cluster config + workload** 만 있으면 어디서든 동일하게 돈다.
2. 본 리포의 vLLM-기반 profiler 는 NVIDIA GPU 전용. **사용자의 NPU(Inf2/TPU) 는 §5 의 합성 경로**를 따라 CSV 번들을 직접 만들어야 한다.
3. 사용자 시나리오 (single inf 또는 TP only) 는 cluster config 의 `tp_size`/`num_npus` 두 필드만 바꾸면 끝. PD/MoE/multi-instance 관련 옵션은 모두 끄거나 default 유지.
4. 결과는 `outputs/*.csv` 의 TTFT, TPOT, latency 컬럼 (모두 ns) 으로 추출. pandas 한 줄로 통계 가능.

---

> **이 plan 파일에 대해**: 사용자가 요청한 "한국어 사용 가이드" 자체를 plan 파일에 작성했다. plan mode 가 종료되어 ExitPlanMode 를 거치면 (사용자가 OK 하면), 이 내용을 리포 안의 적절한 위치 — 예: `docs/llmservingsim2-npu-guide-ko.md` 또는 사용자가 지정한 경로 — 로 옮길 수 있다. 별도의 위치/파일명을 원하면 알려줄 것.
