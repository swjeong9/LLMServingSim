# Inferentia 2 프로파일링 단계별 가이드 (Llama 3.2 1B / Mistral 7B v0.3 / Qwen3 14B, TP=1~8)

> **목적**: AWS Inferentia 2 위에서 위 3개 모델을 TP=1, 2, 4, 8 각각에 대해 프로파일링하여 LLMServingSim 2.0 의 `profiler/perf/Inferentia2/<MODEL>/<variant>/tp<N>/` 번들을 채우는 절차서.
> **선행 조건**: `USAGE_GUIDE_KO.md` §1~§4 까지 (시뮬레이터 컨테이너 띄우고 sanity check 끝낸 상태) 통과.
> **방법론**: 논문(ISPASS 2026) 의 TPU-v6e-1 Colab 노트북 (`references/ispass26-artifact/llm_profile/perf_models/TPU-v6e-1/llm_profiler_tpu.ipynb`) 을 Inferentia 2 로 옮긴 dual-SDK 전략. 이 가이드의 모든 합리적 결정은 그 노트북의 의사결정을 따른다.

---

## 0. 전체 그림 — 논문 노트북 그대로

본 리포의 메인 profiler (`python -m profiler`) 는 NVIDIA CUDA 전용이라 Inferentia 2 에선 못 쓴다. 대신 논문 (ISPASS 2026) 의 TPU-v6e-1 노트북 흐름을 Inferentia 2 로 이식한 두 단계 파이프라인.

```
┌──────────────────────────────────┐
│ 1) 프로파일 (NUM_LAYERS=1)       │
│ transformers + torch_neuronx     │     scripts/profile_neuron.py
│ 모듈 단위 직접 호출 + perf_counter│
└────────────┬─────────────────────┘
             │
             ↓
profiler/perf/Inferentia2/<MODEL>/bf16/tp<N>/
{dense,per_sequence,attention}.csv + meta.yaml   (raw eager 값)
             │
             ↓
┌──────────────────────────────────┐
│ 2) 검증 + 스칼라 보정            │
│  NUM_LAYERS=full 같은 SDK        │     scripts/validate_eager.py
│  e2e generate() 측정 ↔ CSV 합산  │
│  predict 비교 → median 비율 fit   │
│  → 모든 CSV time_us × s          │
└────────────┬─────────────────────┘
             │
             ↓
   LLMServingSim 시뮬레이션
```

**왜 보정이 필요한가?**
- profile 단계는 layer 하나하나 isolation 으로 잼. 실제 N-layer 모델 forward 는 fusion / cache 재사용 / scheduling 오버헤드 등으로 **per-layer × N + head ≠ 풀 N-layer 통째 forward**.
- 같은 SDK (eager) 안에서 N-layer 모델 e2e 한 번 측정 → CSV 의 추정치와 비교 → **글로벌 스칼라 한 개 (예: 1.10)** 로 보정. 논문 노트북 cell 5·6·11 의 `measure_generation_latency` + `validate_and_scale` 그대로.
- NxDI 는 본 파이프라인에 **등장 안 함**. 논문도 안 씀. 프로덕션 NxDI 와 시뮬 정렬이 필요하면 별도 작업이지만 베이스라인 연구엔 불필요.

**공식 도구**:
- `scripts/profile_neuron.py` — eager-mode layer-wise profiler (논문 cell 4·9 이식)
- `scripts/validate_eager.py` — full-model e2e 검증 + 스칼라 보정 (논문 cell 5·6·11 이식)
- `scripts/synth_perf_bundle.py` (§7) — roofline 합성 백업

---

## 1. 환경 사전 준비 (한 번만, 로컬에서)

### 1.1 작업 디렉토리 라벨

| 라벨 | 위치 | 용도 |
|---|---|---|
| **[로컬]** | macOS 호스트 `/Users/swjeong/Desktop/LLMServingSim` | git, 파일 편집, sanity 시뮬레이션 |
| **[inf2]** | AWS Inferentia 2 인스턴스 (DLAMI 또는 Neuron DLC) | 프로파일 sweep + 검증 측정 |
| **[sim-docker]** | 시뮬레이터 컨테이너 안 (`servingsim_docker`) | LLMServingSim 시뮬레이션 |

### 1.2 model config 3개 받아오기 — [로컬]

```bash
cd /Users/swjeong/Desktop/LLMServingSim
mkdir -p configs/model/meta-llama configs/model/mistralai configs/model/Qwen

HF_TOKEN="hf_xxx_your_token_here"

# Llama 3.2 1B (gated)
curl -L -H "Authorization: Bearer $HF_TOKEN" \
  -o configs/model/meta-llama/Llama-3.2-1B.json \
  https://huggingface.co/meta-llama/Llama-3.2-1B/resolve/main/config.json

# Mistral 7B v0.3 (gated)
curl -L -H "Authorization: Bearer $HF_TOKEN" \
  -o configs/model/mistralai/Mistral-7B-v0.3.json \
  https://huggingface.co/mistralai/Mistral-7B-v0.3/resolve/main/config.json

# Qwen3 14B (public)
curl -L -o configs/model/Qwen/Qwen3-14B.json \
  https://huggingface.co/Qwen/Qwen3-14B/resolve/main/config.json
```

### 1.3 검증 — [로컬]

```bash
python3 -c "
import json
for p in [
    'configs/model/meta-llama/Llama-3.2-1B.json',
    'configs/model/mistralai/Mistral-7B-v0.3.json',
    'configs/model/Qwen/Qwen3-14B.json',
]:
    c = json.load(open(p))
    print(f'{p}:')
    for k in ('model_type','hidden_size','num_hidden_layers',
              'num_attention_heads','num_key_value_heads',
              'head_dim','intermediate_size','vocab_size','torch_dtype'):
        if k in c: print(f'  {k:25s} = {c[k]}')
    print()
"
```

기대 `model_type`: `llama`, `mistral`, `qwen3`. → `profiler/models/<model_type>.yaml` 매칭에 사용.

### 1.4 architecture YAML — [로컬]

| 모델 | 필요한 yaml | 상태 |
|---|---|---|
| Llama 3.2 1B | `profiler/models/llama.yaml` | ✅ 이미 존재 |
| Qwen3 14B | `profiler/models/qwen3.yaml` | ✅ 이미 존재 |
| Mistral 7B v0.3 | `profiler/models/mistral.yaml` | ❌ **만들어야 함** |

`profiler/models/mistral.yaml` 을 다음 내용으로 신규 작성:

```yaml
# Mistral family — class catalog (dense decoder-only LLMs).
# Covers Mistral-7B-v0.1 / v0.2 / v0.3. Structure mirrors Llama
# (RMSNorm + GQA + SwiGLU); only vLLM class names differ.

sequence:
  prologue:  [embedding]
  pre_attn:  [layernorm, qkv_proj, rotary_emb, attention]
  post_attn: [o_proj, layernorm]
  mlp_dense: [gate_up_proj, act_fn, down_proj]
  mlp_moe:   []
  head:      [final_layernorm, lm_head, sampler]

catalog:
  dense:
    embedding:
      vllm: VocabParallelEmbedding
    layernorm:
      vllm: RMSNorm
      within: MistralDecoderLayer
      tp_stable: true
    qkv_proj:
      vllm: QKVParallelLinear
    rotary_emb:
      vllm: RotaryEmbedding
    o_proj:
      vllm: RowParallelLinear
      within: MistralAttention
    gate_up_proj:
      vllm: MergedColumnParallelLinear
    act_fn:
      vllm: SiluAndMul
    down_proj:
      vllm: RowParallelLinear
      within: MistralMLP
    final_layernorm:
      vllm: RMSNorm
      within: MistralForCausalLM
      tp_stable: true
  per_sequence:
    lm_head:
      vllm: LogitsProcessor
    sampler:
      vllm: Sampler
      tp_stable: true
  attention:
    attention:
      vllm: Attention
```

> 합성/eager 경로만 쓰면 `vllm:`/`within:` 필드는 시뮬레이터가 안 본다. layer **이름** (왼쪽 키들) 만 정확하면 OK.

### 1.5 Inferentia 2 인스턴스 / 코어 토폴로지

| inf2 인스턴스 | NeuronCore-v2 수 | chip 수 | HBM 합 | 가능 TP |
|---|---|---|---|---|
| inf2.xlarge | 2 | 1 | 32 GB | 1, 2 |
| inf2.8xlarge | 2 | 1 | 32 GB | 1, 2 |
| inf2.24xlarge | 12 | 6 | 192 GB | 1, 2, 4, 8 (12 가능) |
| inf2.48xlarge | 24 | 12 | 384 GB | 1, 2, 4, 8, 12, 24 |

→ 본 가이드는 **`inf2.xlarge` (또는 inf2.8xlarge) 하나면 충분**.

> **중요**: 본 가이드의 핵심은 "TP=1~8 각각을 단일 NeuronCore-v2 에서 1-layer 모델로 프로파일" 한다는 것. **TP=8 이라고 해서 실제로 8 코어가 필요한 게 아니다** — `hf_overrides` 로 head/intermediate 차원만 1/8 로 잘라서 단일 코어에서 측정한다 (본 리포 GPU profiler 와 동일 트릭). 검증(§4)도 NUM_LAYERS=full 모델을 단일 코어에 올려서 e2e 측정하므로 multi-core 필요 없음. 단 메모리가 빠듯한 모델 (Qwen3 14B 등) 은 검증 시 NUM_LAYERS 를 낮춰야 함.

cluster config 매핑 (시뮬레이터 단계에서):
- 1 NPU = 1 NeuronCore-v2
- `npu_mem.mem_size = 16` GB (chip 32 GB 를 2 코어가 공유)
- `npu_mem.mem_bw = 820` GB/s

### 1.6 모델 weight 크기 sanity check — [로컬]

```bash
python3 -c "
import json
for name, path in [
    ('Llama-3.2-1B',    'configs/model/meta-llama/Llama-3.2-1B.json'),
    ('Mistral-7B-v0.3', 'configs/model/mistralai/Mistral-7B-v0.3.json'),
    ('Qwen3-14B',       'configs/model/Qwen/Qwen3-14B.json'),
]:
    c = json.load(open(path))
    h = c['hidden_size']; L = c['num_hidden_layers']
    inter = c['intermediate_size']; V = c['vocab_size']
    head_dim = c.get('head_dim', h // c['num_attention_heads'])
    qkv = c['num_attention_heads']*head_dim + 2*c['num_key_value_heads']*head_dim
    per_layer = h*qkv + h*h + 3*h*inter
    total = V*h*2 + L*per_layer + V*h
    print(f'{name:20s}: full {total*2/1e9:5.1f} GB, 1-layer trick: '
          f'{(V*h*2 + per_layer + V*h)*2/1e9:5.2f} GB')
"
```

실측 결과 (BF16, 2026-05-08, 사용자 환경):
```
Llama-3.2-1B        : full   3.5 GB, 1-layer trick:  1.70 GB
Mistral-7B-v0.3     : full  14.8 GB, 1-layer trick:  1.24 GB
Qwen3-14B           : full  31.1 GB, 1-layer trick:  5.33 GB
```

(위 한-줄 계산은 weight 만 잡고 activation/임시 버퍼는 무시. 실제 런타임은 +30~50% 잡아두는 게 안전.)

→ **1-layer 트릭으로 모두 단일 NeuronCore-v2 (16 GB) 에 여유롭게 들어감** (Qwen3 14B 도 5.33 GB → activation 2~3× 잡아도 16 GB 미만). 이게 가능해야 TP=1 부터 8 까지 다 같은 코어에서 sweep 할 수 있음.

> **검증(§4) 단계는 다름** — 거기서는 NUM_LAYERS=full 로 다시 로드하므로 Qwen3 14B 의 31 GB 는 단일 코어 OOM. `--num-layers 4` 같은 식으로 줄여서 검증.

---

## 2. inf2 인스턴스 셋업 — [inf2]

### 2.1 인스턴스 부팅

```
인스턴스: inf2.xlarge (단일 코어 1-layer 트릭으로 충분)
        또는 inf2.24xlarge (이 가이드 사용자의 실제 환경; 코어 12개)
이미지: Deep Learning AMI Neuron PyTorch 2.x (Ubuntu 22.04)
```

부팅 후:
```bash
neuron-ls           # NeuronCore-v2 들 인식 확인
neuron-top          # 자원 상태 (Ctrl+C 로 빠져나옴)
```

**참고 — `neuron-ls` 실제 출력 (inf2.24xlarge, 2026-05-08)**:

```
instance-type: inf2.24xlarge
instance-id: i-00a9f3ecb6d998968
+--------+--------+----------+--------+-----------+--------------+-------------+------+
| NEURON | NEURON |  NEURON  | NEURON | CONNECTED |     PCI      |     CPU     | NUMA |
| DEVICE | CORES  | CORE IDS | MEMORY |  DEVICES  |     BDF      |  AFFINITY   | NODE |
+--------+--------+----------+--------+-----------+--------------+-------------+------+
| 0      | 2      | 0-1      | 32 GB  | 1         | 0000:20:1e.0 | 24-47,72-95 | 1    |
| 1      | 2      | 2-3      | 32 GB  | 0, 2      | 0000:20:1f.0 | 24-47,72-95 | 1    |
| 2      | 2      | 4-5      | 32 GB  | 1, 3      | 0000:10:1e.0 | 0-23,48-71  | 0    |
| 3      | 2      | 6-7      | 32 GB  | 2, 4      | 0000:10:1f.0 | 0-23,48-71  | 0    |
| 4      | 2      | 8-9      | 32 GB  | 3, 5      | 0000:10:1d.0 | 0-23,48-71  | 0    |
| 5      | 2      | 10-11    | 32 GB  | 4         | 0000:20:1d.0 | 24-47,72-95 | 1    |
+--------+--------+----------+--------+-----------+--------------+-------------+------+
```

읽는 법:
- **NEURON DEVICE**: 물리 chip (Inferentia 2 칩). 6개 = inf2.24xlarge.
- **NEURON CORES / CORE IDS**: 칩당 NeuronCore-v2 2개. 코어 ID 0~11 까지 12개.
- **NEURON MEMORY**: chip 단위 HBM (32 GB). 한 chip 의 2 코어가 공유. 본 가이드 cluster config 의 `npu_mem.mem_size = 16 GB` 는 코어당 환산값.
- **CONNECTED DEVICES**: chip 간 NeuronLink 토폴로지 (ring). 칩 0↔1↔2↔3↔4↔5↔0.
- **CPU AFFINITY / NUMA**: 칩 0,1,5 는 NUMA node 1 / 칩 2,3,4 는 NUMA node 0. 본 가이드 sweep 은 단일 칩만 쓰니까 무시해도 됨.

본 가이드의 `profile_neuron.py` 와 `validate_eager.py` 는 모두 **단일 NeuronCore (예: 코어 0)** 만 사용. 다른 코어들은 idle. TP=8 같은 멀티코어 토폴로지도 1-layer 트릭 + `hf_overrides` 로 단일 코어에서 emulate.

### 2.2 의존성

DLAMI 의 Neuron 가상환경 활성화 + 추가 패키지:

```bash
source /opt/aws_neuronx_venv_pytorch_2_5/bin/activate
# 핵심 의존성 (DLAMI 에 이미 있을 가능성 높음)
pip install -U transformers accelerate sentencepiece pyyaml

# 동작 확인
python -c "import torch; import torch_xla; import torch_neuronx; print('eager OK')"
```

> NxDI (`neuronx-distributed-inference`) 는 본 가이드에 **불필요**. 사용자가 production 으로 NxDI 를 따로 쓰는 것과 무관 — 프로파일링/검증은 eager 안에서 다 끝남.

### 2.3 리포 + 모델 가져오기 — [inf2]

```bash
# (옵션 A) 리포 그대로 쓰려면 inf2 에 clone — sweep CSV 가 inf2 안에 떨어짐
git clone https://github.com/swjeong9/LLMServingSim.git ~/LLMServingSim
cd ~/LLMServingSim

# (옵션 B) 호스트에서 rsync 로 동기화 후 sweep — 결과 CSV 만 다시 가져오기
# rsync -av --exclude='.git' /Users/swjeong/Desktop/LLMServingSim/ ubuntu@<inf2-ip>:~/LLMServingSim/

# 모델 다운로드 (HF login 후)
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.2-1B   --local-dir ~/models/Llama-3.2-1B
huggingface-cli download mistralai/Mistral-7B-v0.3 --local-dir ~/models/Mistral-7B-v0.3
huggingface-cli download Qwen/Qwen3-14B            --local-dir ~/models/Qwen3-14B
```

> `profile_neuron.py` 는 모델 weight 를 안 다운받음 — random init 으로 timing 만 측정. `validate_eager.py` 도 random weight 로 충분 (timing 만 잼). 따라서 위 다운로드는 **본 측정 단계 (§6, ShareGPT workload 사용 시) 에서만** 필요. 시간 아끼려면 sweep 끝낸 후로 미뤄도 OK.

---

## 3. eager 프로파일링 sweep — [inf2]

핵심 도구: `scripts/profile_neuron.py` (논문 노트북의 Python 이식판). 이미 리포에 있음.

### 3.1 스크립트 동작 요약

```
입력 (CLI):
  --model meta-llama/Llama-3.2-1B
  --tp 1,2,4,8
  --output-root profiler/perf
  (+ sweep grid, warmup, repeat, dtype 등)

내부 동작 (TP 별 반복):
  1. AutoConfig 로 HF config 로드
  2. num_hidden_layers = 1 강제 (1-layer 트릭)
  3. num_attention_heads, num_key_value_heads, intermediate_size 를 1/tp 로 분할
  4. AutoModelForCausalLM.from_config 로 random-weight 모델 생성
  5. .to(xm.xla_device()) 로 NeuronCore 위로 올림
  6. layer 단위 module 직접 호출 + mark_step + wait + perf_counter_ns
  7. warmup N 회 후 repeat N 회 측정, median 기록
  8. (dense, per_sequence, attention) 3개 CSV 저장

출력:
  profiler/perf/Inferentia2/<model>/bf16/
    meta.yaml
    tp1/{dense,per_sequence,attention}.csv
    tp2/...
    tp4/...
    tp8/...
```

### 3.2 sweep grid 기본값 (lean)

| 축 | 기본값 | 의미 |
|---|---|---|
| `--tokens-grid` | `1,16,64,256,1024,2048` | dense.csv 의 token 수 |
| `--sequences-grid` | `1,8,32,128` | per_sequence.csv 의 batch |
| `--prefill-grid` | `16,64,256,1024,2048` | attention pure-prefill chunk |
| `--kv-prefill-grid` | `0,1024,4096,8192` | attention prefill 의 KV history |
| `--decode-n-grid` | `1,4,16,64` | attention pure-decode batch |
| `--kv-decode-grid` | `64,256,1024,4096,8192,16384` | decode 의 KV 길이 |
| `--warmup` | `10` | warmup forward 수 |
| `--repeat` | `30` | timed forward 수 (median) |

이 기본값으로 **모델당 4 TP × ~80 측정점 = 약 15~30분/모델** (Neuron 컴파일 캐시 잡힌 후). 첫 컴파일에서 모델당 10~20분 추가.

> 정확도가 더 필요하면 grid 를 촘촘히. `--tokens-grid 1,2,4,8,16,32,64,128,256,512,1024,2048` 처럼.

### 3.3 sweep 실행 — [inf2]

```bash
cd ~/LLMServingSim
source /opt/aws_neuronx_venv_pytorch_2_5/bin/activate
export HF_TOKEN="hf_xxx_your_token_here"

# Llama 3.2 1B (TP=1,2,4,8)
python scripts/profile_neuron.py \
  --model meta-llama/Llama-3.2-1B \
  --tp 1,2,4,8 \
  --output-root profiler/perf

# Mistral 7B v0.3 (TP=1,2,4,8)
python scripts/profile_neuron.py \
  --model mistralai/Mistral-7B-v0.3 \
  --tp 1,2,4,8 \
  --output-root profiler/perf

# Qwen3 14B (TP=2,4,8 — TP=1 도 1-layer 면 들어가지만, 본 시나리오 의미 작음)
python scripts/profile_neuron.py \
  --model Qwen/Qwen3-14B \
  --tp 2,4,8 \
  --output-root profiler/perf
```

> **head 수가 TP 로 안 나눠지면 실패**. Llama 3.2 1B (heads=32, kv=8) 는 TP ∈ {1,2,4,8} 모두 OK. Mistral 7B (heads=32, kv=8) 도 OK. Qwen3 14B (heads=40, kv=8) 는 TP=8 에서 heads=5, kv=1 — kv 가 1 까지 줄어드는데 OK. TP=12, 16 등은 안 나눠짐.

각 sweep 의 stdout 예시:
```
========== TP=2 ==========
  loaded model with sharded dims: heads=16, kv=4, inter=4096
  -- dense sweep --
    dense  embedding          tokens=    1  ->     2.143 us
    dense  embedding          tokens=   16  ->     2.345 us
    ...
    dense  qkv_proj           tokens= 1024  ->   142.567 us
    ...
  [✓] dense.csv (96 rows)
  -- per_sequence sweep --
    per_s  lm_head            seqs=    1    ->    18.432 us
    ...
  [✓] per_sequence.csv (8 rows)
  -- attention sweep --
    attn   prefill pc=   16 kv_p=    0                  ->     6.234 - 0.567 = 5.667 us
    ...
  [✓] attention.csv (44 rows)
```

### 3.4 결과 확인 — [inf2 또는 로컬]

```bash
find profiler/perf/Inferentia2 -type f | sort
head -3 profiler/perf/Inferentia2/meta-llama/Llama-3.2-1B/bf16/tp2/dense.csv
cat profiler/perf/Inferentia2/meta-llama/Llama-3.2-1B/bf16/meta.yaml
```

기대 트리:
```
profiler/perf/Inferentia2/
├── meta-llama/Llama-3.2-1B/bf16/
│   ├── meta.yaml
│   ├── tp1/{dense,per_sequence,attention}.csv
│   ├── tp2/...
│   ├── tp4/...
│   └── tp8/...
├── mistralai/Mistral-7B-v0.3/bf16/    (동일 구조)
└── Qwen/Qwen3-14B/bf16/                (tp2/4/8)
```

이 단계 끝나면 `meta.yaml::calibration::scaling_factor: 1.0` (raw eager 출력). 다음 §4 에서 eager 검증 + 보정 적용.

### 3.5 (선택) 호스트로 가져오기 — [inf2 → 로컬]

호스트에서 시뮬레이션 돌릴 거면:

```bash
# inf2 에서
cd ~/LLMServingSim
tar czf /tmp/inf2-perf.tgz profiler/perf/Inferentia2/

# 로컬에서
scp -i <key.pem> ubuntu@<inf2-ip>:/tmp/inf2-perf.tgz /tmp/
tar xzf /tmp/inf2-perf.tgz -C /Users/swjeong/Desktop/LLMServingSim/
```

---

## 4. eager 검증 + 글로벌 보정 — [inf2]

논문 노트북 cell 5·6·11 의 `validate_and_scale` 그대로. NxDI 안 씀.

### 4.1 왜 보정이 필요한가

`profile_neuron.py` 가 만든 CSV 는 layer 한 개를 isolation 으로 잰 값. 시뮬레이터가 이걸 `× num_hidden_layers` 해서 풀 모델 latency 를 계산하는데, **per-layer 합산 ≠ 풀 모델 통째 forward**:
- 매 layer 호출이 별도의 mark_step 으로 끊겨서 fusion 이 안 됨
- 동기화 오버헤드가 layer 마다 한 번씩 누적
- 1-layer 모델은 inter-layer KV/메모리 효과를 못 봄

→ 같은 SDK (eager transformers + torch_neuronx) 안에서 NUM_LAYERS=full 모델을 한 번 e2e 돌려서 측정 → CSV 합산 추정치와 비교 → **글로벌 스칼라 한 개 (예: s=1.10)** 로 모든 CSV 의 `time_us` 를 보정.

### 4.2 도구

`scripts/validate_eager.py` 가 다음을 자동 수행:

1. NUM_LAYERS=full (또는 `--num-layers` 오버라이드) 로 모델 재로드 → Neuron core 위에 올림
2. 각 (input_len, output_len) shape 에 대해 prefill + (output_len-1) decode 의 wall time 측정
3. 같은 shape 에 대해 CSV 번들 lookup 으로 추정치 계산 (per-layer × num_layers + 어텐션 + lm_head + sampler)
4. per-shape `measured / estimated` 비율의 median 으로 스칼라 fit
5. 모든 TP 폴더의 모든 `time_us` × s, 원본은 `*.pre_calib.csv` 백업
6. `meta.yaml::calibration` 갱신

### 4.3 한 모델당 검증 명령 — [inf2]

```bash
cd ~/LLMServingSim
source /opt/aws_neuronx_venv_pytorch_2_5/bin/activate
export HF_TOKEN="hf_xxx_your_token_here"

# Llama 3.2 1B (full=16 layers, 단일 코어 OK)
python scripts/validate_eager.py \
  --model meta-llama/Llama-3.2-1B \
  --variant-root profiler/perf/Inferentia2/meta-llama/Llama-3.2-1B/bf16 \
  --shapes 128:32,512:32,1024:64,2048:128

# Mistral 7B v0.3 (full=32 layers, 단일 코어 빠듯하지만 OK)
python scripts/validate_eager.py \
  --model mistralai/Mistral-7B-v0.3 \
  --variant-root profiler/perf/Inferentia2/mistralai/Mistral-7B-v0.3/bf16 \
  --shapes 128:32,512:32,1024:64,2048:128 \
  --max-position-embeddings 4096

# Qwen3 14B (full=40 layers, 단일 코어 OOM → NUM_LAYERS=4 로 줄임)
python scripts/validate_eager.py \
  --model Qwen/Qwen3-14B \
  --variant-root profiler/perf/Inferentia2/Qwen/Qwen3-14B/bf16 \
  --validate-tp 2 \
  --shapes 128:32,512:32,1024:64 \
  --num-layers 4
```

> **`--validate-tp`** 는 어느 `tp<N>/` 폴더의 CSV 와 비교할지. 보통 메모리 가장 작은 TP=1 사용. Qwen3 14B 는 TP=1 폴더가 없으니 `--validate-tp 2`. 단 검증 모델은 **단일 코어 emulation** 이므로 sharded shape 위에서 하는 측정. 스칼라는 shape-agnostic 이므로 TP 무관하게 모든 폴더에 적용됨.
>
> **`--num-layers 4`** (Qwen3 14B 만): full 40-layer 가 단일 코어 16 GB 에 안 들어가서 4 layer 로 검증. inter-layer bias 는 여전히 잡힘 (1 보다 크면 됨). 논문도 이런 케이스에선 같은 트릭 가능.

### 4.4 stdout 예시

```
[*] reading CSV bundle from profiler/perf/Inferentia2/.../bf16/tp1
[*] loading model with num_hidden_layers=16 on Neuron core
[*] validating on 4 shape(s)
       128:32   measured=  41832.5 us   estimated=  37520.1 us   ratio=1.1149
       512:32   measured=  84115.0 us   estimated=  76830.4 us   ratio=1.0948
      1024:64   measured= 142567.2 us   estimated= 128945.7 us   ratio=1.1056
     2048:128   measured= 295123.5 us   estimated= 268541.8 us   ratio=1.0989

[*] median scaling factor s = 1.1023
[✓] wrote validation_data.json + validation_fit.json
  scaled profiler/perf/.../tp1/dense.csv  (backup: dense.csv.pre_calib.csv)
  scaled .../tp1/per_sequence.csv
  scaled .../tp1/attention.csv
  scaled .../tp2/dense.csv
  ...
  updated profiler/perf/.../meta.yaml
[✓] calibration applied
```

### 4.5 효과 + 멱등

- 모든 TP 의 모든 `time_us` 가 같은 비율로 곱해짐. 원본은 `*.pre_calib.csv` 로 백업.
- `meta.yaml::calibration` 에 scaling factor + per-shape ratios + timestamp 기록.
- 재실행 시 자동으로 이전 scaling 을 1/s 로 되돌린 뒤 새 s 로 다시 곱함 (멱등).
- `--dry-run` 으로 스칼라만 보고 CSV 안 건드리는 것도 가능.

### 4.6 보정 안 할 권리도 있다

검증 단계는 paper-faithful 이지만 **반드시 필요한 건 아님**. 베이스라인 비교 (LLMServingSim vs 다른 시뮬레이터) 처럼 **상대값** 만 보면 모든 시뮬레이터에 같은 bias 가 들어가므로 결론이 안 바뀜. 절대 latency 예측 정확도 (논문 Fig 9 의 0.97% 처럼) 가 필요한 경우에만 §4 단계 진행.

이 단계 skip 시: §3 의 raw eager profile 그대로 사용, `meta.yaml::calibration::scaling_factor: 1.0` 유지. ±20~30% 정도 절대값 오차.

---

## 5. 시뮬레이터 동작 검증 — [sim-docker]

번들이 만들어졌으면 시뮬레이터에서 실제로 도는지 확인.

### 5.1 cluster config 일괄 생성 — [로컬]

```bash
cd /Users/swjeong/Desktop/LLMServingSim
mkdir -p configs/cluster

for model in "meta-llama/Llama-3.2-1B" "mistralai/Mistral-7B-v0.3" "Qwen/Qwen3-14B"; do
  short=$(echo $model | awk -F/ '{print $2}' | tr A-Z a-z | tr . _ | tr - _)
  for tp in 1 2 4 8; do
    [[ "$model" == "Qwen/Qwen3-14B" && "$tp" == "1" ]] && continue
    cat > configs/cluster/inf2_${short}_tp${tp}.json <<EOF
{
    "num_nodes": 1,
    "link_bw": 100,
    "link_latency": 500,
    "nodes": [{
        "num_instances": 1,
        "cpu_mem": { "mem_size": 256, "mem_bw": 256, "mem_latency": 0 },
        "instances": [{
            "model_name": "${model}",
            "hardware": "Inferentia2",
            "npu_mem": { "mem_size": 16, "mem_bw": 820, "mem_latency": 0 },
            "num_npus": ${tp},
            "tp_size": ${tp},
            "pd_type": null
        }]
    }]
}
EOF
  done
done
ls configs/cluster/inf2_*.json
```

### 5.2 sanity 시뮬레이션 — [sim-docker]

```bash
docker start -ai servingsim_docker

# 컨테이너 안에서
cd /app/LLMServingSim

for cfg in configs/cluster/inf2_*.json; do
  name=$(basename $cfg .json)
  python -m serving \
    --cluster-config "$cfg" \
    --dtype bfloat16 \
    --dataset 'workloads/example_trace.jsonl' \
    --output "outputs/${name}_smoke.csv" \
    --num-reqs 5 --log-interval 1.0 \
    2>&1 | tee "outputs/${name}_smoke.log"
done
```

흔한 실패:
- `FileNotFoundError: profiler/perf/Inferentia2/.../tp<N>/dense.csv` → sweep 못 한 TP. cluster config 의 TP 줄이거나 다시 sweep.
- `KeyError: '<layer>'` → architecture YAML 의 catalog layer 가 dense.csv 에 없음. profile_neuron.py 가 모든 layer 를 채우는지 확인.

---

## 6. 본 측정 — 실험 [sim-docker]

```bash
for cfg in configs/cluster/inf2_*.json; do
  name=$(basename $cfg .json)
  python -m serving \
    --cluster-config "$cfg" \
    --dtype bfloat16 \
    --dataset 'workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl' \
    --output "outputs/${name}_sharegpt.csv" \
    --num-reqs 100 --max-num-batched-tokens 2048 --max-num-seqs 128 \
    --log-interval 1.0 \
    2>&1 | tee "outputs/${name}_sharegpt.log"
done
```

분석 (Python):
```python
import pandas as pd, glob
frames = []
for path in glob.glob("outputs/inf2_*_sharegpt.csv"):
    name = path.split("/")[-1].replace("_sharegpt.csv","")
    df = pd.read_csv(path); df["scenario"] = name
    frames.append(df)
big = pd.concat(frames, ignore_index=True)
print(big.groupby("scenario").agg(
    n=("request id","count"),
    p50_TTFT_ms=("TTFT", lambda x: x.quantile(0.5)/1e6),
    p99_TTFT_ms=("TTFT", lambda x: x.quantile(0.99)/1e6),
    p50_TPOT_ms=("TPOT", lambda x: x.quantile(0.5)/1e6),
    e2e_p50_s=("latency", lambda x: x.quantile(0.5)/1e9),
).round(2))
```

---

## 7. (백업) Roofline 합성 — inf2 접근 어려울 때 [로컬]

inf2 인스턴스가 막히거나 sweep 디버깅 필요할 때. Spec sheet (190 TFLOPS BF16, 820 GB/s HBM) 만으로 5분 만에 1차 데이터 만들기.

`scripts/synth_perf_bundle.py`:

```python
"""
Pure-roofline synthetic perf bundle for Inferentia 2.
±20-30% accuracy. Replace with profile_neuron.py output when available.
"""
import argparse, json, csv, math
from pathlib import Path

INF2_TFLOPS_BF16 = 190.0
INF2_HBM_GBPS    = 820.0
PER_CORE_TFLOPS = INF2_TFLOPS_BF16 / 2
PER_CORE_HBM    = INF2_HBM_GBPS / 2

ARCH_LAYERS = {
    "llama":   ["embedding","layernorm","qkv_proj","rotary_emb",
                "o_proj","gate_up_proj","act_fn","down_proj","final_layernorm"],
    "mistral": ["embedding","layernorm","qkv_proj","rotary_emb",
                "o_proj","gate_up_proj","act_fn","down_proj","final_layernorm"],
    "qwen3":   ["embedding","layernorm","qkv_proj","qk_norm","rotary_emb",
                "o_proj","gate_up_proj","act_fn","down_proj","final_layernorm"],
}

def gemm_us(M, N, K, tflops):
    flops = 2 * M * N * K
    t_compute = flops / (tflops * 1e12) * 1e6
    bytes_io = (M*K + K*N + M*N) * 2
    t_mem = bytes_io / (PER_CORE_HBM * 1e9) * 1e6
    return max(t_compute, t_mem)

def layer_us(layer, n, cfg, tp):
    H = cfg["hidden_size"]
    head_dim = cfg.get("head_dim", H // cfg["num_attention_heads"])
    nh = cfg["num_attention_heads"]; nkv = cfg["num_key_value_heads"]
    inter = cfg["intermediate_size"]; V = cfg["vocab_size"]
    qkv_out = (nh + 2*nkv) * head_dim // tp
    inter_per = inter // tp
    nh_per = nh // tp
    tflops_eff = PER_CORE_TFLOPS * tp
    if layer == "embedding":      return 0.5 + 0.0001*n
    if layer in ("layernorm","final_layernorm"): return 0.3 + 0.001*n
    if layer == "qk_norm":        return 0.3 + 0.001*n
    if layer == "rotary_emb":     return 0.5 + 0.002*n
    if layer == "qkv_proj":       return gemm_us(n, qkv_out, H, tflops_eff)
    if layer == "o_proj":         return gemm_us(n, H, nh_per*head_dim, tflops_eff)
    if layer == "gate_up_proj":   return gemm_us(n, 2*inter_per, H, tflops_eff)
    if layer == "act_fn":         return 0.4 + 0.0005*n
    if layer == "down_proj":      return gemm_us(n, H, inter_per, tflops_eff)
    return 0.0

def per_seq_us(layer, s, cfg, tp):
    H = cfg["hidden_size"]; V = cfg["vocab_size"]
    if layer == "lm_head": return gemm_us(s, V, H, PER_CORE_TFLOPS*tp)
    if layer == "sampler": return 1.0 + 0.05*s
    return 0.0

def attention_us(pc, kv_p, n, kv_d, cfg, tp):
    head_dim = cfg.get("head_dim", cfg["hidden_size"]//cfg["num_attention_heads"])
    nh_per = max(cfg["num_attention_heads"]//tp, 1)
    tflops_eff = PER_CORE_TFLOPS * tp; hbm_eff = PER_CORE_HBM * tp
    t_pre = (pc*(pc + 2*kv_p)*head_dim*nh_per*2) / (tflops_eff*1e12) * 1e6 if pc>0 else 0
    t_dec = (n*kv_d*head_dim*nh_per*2) / (hbm_eff*1e9) * 1e6 if n>0 and kv_d>0 else 0
    return max(0.5, t_pre + t_dec)

def build_for_tp(cfg, tp, out_dir):
    arch = cfg["model_type"]; layers = ARCH_LAYERS[arch]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [(l,n,layer_us(l,n,cfg,tp))
            for l in layers
            for n in [1,2,4,8,16,32,64,128,256,512,1024,2048]]
    with open(out_dir/"dense.csv","w",newline="") as f:
        w = csv.writer(f); w.writerow(["layer","tokens","time_us"])
        for r in rows: w.writerow([r[0], r[1], f"{r[2]:.6g}"])
    rows = [(l,s,per_seq_us(l,s,cfg,tp))
            for l in ["lm_head","sampler"]
            for s in [1,2,4,8,16,32,64,128]]
    with open(out_dir/"per_sequence.csv","w",newline="") as f:
        w = csv.writer(f); w.writerow(["layer","sequences","time_us"])
        for r in rows: w.writerow([r[0], r[1], f"{r[2]:.6g}"])
    rows = []
    for pc in [16,32,64,128,256,512,1024,2048]:
        for kv_p in [0,512,1024,2048,4096,8192]:
            rows.append((pc,kv_p,0,0, attention_us(pc,kv_p,0,0,cfg,tp)))
    for n in [1,2,4,8,16,32,64,128]:
        for kv_d in [16,64,256,1024,4096,8192]:
            rows.append((0,0,n,kv_d, attention_us(0,0,n,kv_d,cfg,tp)))
    with open(out_dir/"attention.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["prefill_chunk","kv_prefill","n_decode","kv_decode","time_us"])
        for r in rows: w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.6g}"])

def main():
    import yaml, time as _t
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tp-list", required=True)
    ap.add_argument("--variant", default="bf16"); ap.add_argument("--hardware", default="Inferentia2")
    ap.add_argument("--repo-root", required=True)
    args = ap.parse_args()
    org, name = args.model.split("/", 1)
    cfg = json.loads((Path(args.repo_root)/"configs/model"/org/f"{name}.json").read_text())
    tps = [int(x) for x in args.tp_list.split(",")]
    out_root = Path(args.repo_root)/"profiler/perf"/args.hardware/args.model/args.variant
    out_root.mkdir(parents=True, exist_ok=True)
    for tp in tps:
        build_for_tp(cfg, tp, out_root/f"tp{tp}")
        print(f"[✓] synthesized tp{tp}")
    meta = {
        "profiler_version": "synthetic-roofline-v1",
        "vllm_version": "n/a", "hardware": args.hardware, "gpu": args.hardware,
        "profiled_at": _t.strftime("%Y-%m-%dT%H:%M:%S+00:00", _t.gmtime()),
        "architecture": cfg["model_type"], "model": args.model,
        "variant": args.variant, "tp_degrees": tps,
        "engine_effective": {"max_num_batched_tokens":2048,"max_num_seqs":256,
                             "dtype":"bfloat16","kv_cache_dtype":"auto"},
        "attention_grid": {"max_kv":16384, "chunks":"16-2048","n_decode":"0,1-128","kv":"0,16-8192"},
        "skew_fit": {"per_tp": {tp: {"method":"synthetic-constant","alpha_default":0.3} for tp in tps}},
        "calibration": {"scaling_factor": 1.0, "scaled_by": "raw roofline (no eager validation)"},
    }
    (out_root/"meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"[✓] meta.yaml at {out_root}")

if __name__ == "__main__": main()
```

호스트 실행:
```bash
python3 scripts/synth_perf_bundle.py \
  --model meta-llama/Llama-3.2-1B --tp-list 1,2,4,8 --repo-root .
python3 scripts/synth_perf_bundle.py \
  --model mistralai/Mistral-7B-v0.3 --tp-list 1,2,4,8 --repo-root .
python3 scripts/synth_perf_bundle.py \
  --model Qwen/Qwen3-14B --tp-list 2,4,8 --repo-root .
```

이후 §4 의 eager 검증 (`validate_eager.py`) 을 그대로 적용 가능 → ±10% 수준까지 끌어올림.

---

## 8. 작업 체크리스트 (인쇄용)

- [ ] **§1.2** 모델 config 3종 받기 (HF Hub)
- [ ] **§1.3** model_type 확인
- [ ] **§1.4** `profiler/models/mistral.yaml` 신규 작성
- [ ] **§1.5** inf2 인스턴스 결정 (sweep + 검증 모두 inf2.xlarge 한 대로 충분)
- [ ] **§2.1, 2.2** inf2 부팅, neuronx 가상환경 + transformers 설치
- [ ] **§2.3** 리포 sync 또는 clone, HF 모델 다운로드 (선택, 검증/본 측정용)
- [ ] **§3.3** `profile_neuron.py` 로 모델 3개 sweep (15~30분/모델, 첫 컴파일 포함 1~2시간)
- [ ] **§3.4** 결과 트리 확인
- [ ] **§4.3** `validate_eager.py` 로 모델 3개 검증·보정 (10~30분/모델). 또는 §4.6 처럼 skip 결정.
- [ ] **§5.1** cluster config 11개 일괄 생성
- [ ] **§5.2** sanity 시뮬레이션 11개 통과
- [ ] **§6** 본 워크로드로 시뮬, 결과 분석
- [ ] (선택) **§7** roofline 합성으로 cross-check

---

## 9. 빠른 참조 — 명령어 모음

```bash
# [inf2] eager sweep (모델당 한 번)
python scripts/profile_neuron.py \
  --model meta-llama/Llama-3.2-1B --tp 1,2,4,8 --output-root profiler/perf

# [inf2] eager 검증 (모델당 한 번; skip 가능)
python scripts/validate_eager.py \
  --model meta-llama/Llama-3.2-1B \
  --variant-root profiler/perf/Inferentia2/meta-llama/Llama-3.2-1B/bf16 \
  --shapes 128:32,512:32,1024:64,2048:128

# [로컬] roofline 백업
python3 scripts/synth_perf_bundle.py \
  --model meta-llama/Llama-3.2-1B --tp-list 1,2,4,8 --repo-root .

# [sim-docker] sanity 시뮬레이션
python -m serving \
  --cluster-config configs/cluster/inf2_llama_3_2_1b_tp1.json \
  --dtype bfloat16 --dataset workloads/example_trace.jsonl \
  --output outputs/sanity.csv --num-reqs 5
```

---

## 10. 트러블슈팅

| 증상 | 원인 | 처방 |
|---|---|---|
| `ValueError: TP=8 doesn't evenly divide ...` | head 수가 TP 배수 아님 | TP 를 `num_attention_heads`/`num_key_value_heads` 의 약수로 (Qwen3 14B 는 TP=10 안 됨) |
| `RuntimeError: out of memory` (1-layer 인데도) | vocab 너무 큼 (Qwen3: 152k × 5120) + Neuron 컴파일 임시 메모리 | `--max-position-embeddings 8192` 등으로 컨텍스트 줄임 |
| `TypeError: forward() got unexpected keyword 'position_embeddings'` | HF transformers 버전 | profile_neuron.py 의 `call_self_attn` 가 두 시그니처 모두 try; 그래도 실패하면 transformers 버전 명시: `pip install transformers==4.45.0` |
| Neuron 첫 컴파일이 너무 오래 걸림 | 매 (tp, shape) 쌍마다 컴파일 캐시 miss | 캐시 디렉토리 확인 (`ls /var/tmp/neuron-compile-cache/`). 첫 모델만 길고 두 번째부터 빨라짐. 정 늦으면 `--repeat 5 --warmup 2` 로 임시 단축 |
| `KeyError: 'qk_norm'` (Qwen3) | dense.csv 에 qk_norm 누락 | profile_neuron.py 의 ARCH_DESC.qwen3 확인. HF Qwen3 이 q_norm/k_norm 어트리뷰트 노출하는지 (HF >= 4.42) |
| 시뮬 결과 TPOT 가 비정상적으로 큼 | scaling factor 가 잘못 fit | `meta.yaml::calibration::scaling_factor` 확인. 1.0 (보정 안 된 raw) 인지 / 너무 큰 값인지 |
| `FileNotFoundError: profiler/perf/Inferentia2/<MODEL>/bf16/tp1/dense.csv` | 그 TP 못 sweep 했거나 파일 동기화 실패 | inf2 sweep 출력 확인 + scp/rsync |
| 검증 시 OOM (`RuntimeError: out of memory`) | full N-layer 가 단일 코어에 안 들어감 (Qwen3 14B) | `--num-layers 4` 또는 `--num-layers 2` 로 줄임. inter-layer bias 는 여전히 잡힘 |
| `validate_eager.py` 가 너무 느림 | 첫 컴파일에 시간 걸림 | `--repeat 3 --warmup 2` 로 줄여서 1차 확인. 캐시 잡히면 재실행 빠름 |

---

## 11. 한 줄 요약

1. **eager 프로파일** (`scripts/profile_neuron.py`): transformers + torch_neuronx, NUM_LAYERS=1, hf_overrides 로 TP 분할 흉내, 모듈 직접 호출 + mark_step. 모델당 15~30분.
2. **eager 검증** (`scripts/validate_eager.py`): NUM_LAYERS=full 모델로 e2e 측정 → CSV 합산 추정과 비교 → 글로벌 스칼라 한 개로 모든 CSV 보정. 모델당 10~30분. **선택 사항** — 베이스라인 비교만 하면 skip 가능.
3. **roofline 백업** (`scripts/synth_perf_bundle.py`): inf2 막히면 5분 만에 ±25% 정확도 1차 데이터.
4. 시뮬레이터 호출은 USAGE_GUIDE_KO.md §7 그대로. cluster config 의 `hardware: "Inferentia2"` + `tp_size` 만 바꾸면 됨.

**핵심 함정**:
- `profiler/models/mistral.yaml` 직접 작성해야 함
- TP 가 head 수의 약수가 아니면 sweep 실패 (Qwen3-14B 의 heads=40 은 TP ∈ {1,2,4,5,8,10}; TP=16 안 됨)
- eager 검증 (§4) 안 거치면 per-layer-합산 bias 때문에 ±20~30% 절대 오차. 상대 비교엔 무관

---

## 12. 참고 자료

리포 안의 관련 파일:
- `references/ispass26-artifact/llm_profile/perf_models/TPU-v6e-1/llm_profiler_tpu.ipynb` — 논문이 실제로 쓴 TPU profiler 노트북. profile_neuron.py 의 원본.
- `references/README.md` — v0/v1 schema 차이
- `scripts/profile_neuron.py` — 본 가이드 §3 의 도구 (논문 cell 4·9 이식)
- `scripts/validate_eager.py` — 본 가이드 §4 의 도구 (논문 cell 5·6·11 이식)
- `scripts/synth_perf_bundle.py` — 본 가이드 §7 의 roofline 백업 (인라인)
- `USAGE_GUIDE_KO.md` — 시뮬레이터 사용 전반
- `docs/docs/profiler/adding-hardware.md` — non-GPU 하드웨어 추가 공식 가이드 (영문)

외부:
- AWS Neuron SDK 문서: <https://awsdocs-neuron.readthedocs-hosted.com/>
- (선택) NxDI 가 production 인 경우: <https://github.com/aws-neuron/neuronx-distributed-inference> — 본 가이드 워크플로엔 불필요
- transformers + torch_neuronx eager: <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/programming-guide.html>
