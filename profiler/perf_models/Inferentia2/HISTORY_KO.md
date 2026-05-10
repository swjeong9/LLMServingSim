# Inferentia 2 layer-wise profiling — 시간 순 history & lessons

이 문서는 LLMServingSim 2.0 의 Inferentia 2 layer-wise profiler 를
구축하면서 우리가 거친 모든 시도, 막힌 지점, 잘못된 판단, 결정적 발견을
시간 순으로 자세히 기록한다.

목적은 (1) future me / contributor 가 같은 미궁을 다시 돌지 않도록 + (2)
"왜 ipynb 가 이 위치에 있는가, 왜 두 종류의 profiler 를 유지하는가" 라는
질문에 reference 답을 제공하는 것.

**상태**: 우리는 결국 두 path 를 cross-validation 으로 같이 사용하는
결정에 도달했다. profile_neuron.py 는 host wallclock approximation 으로
큰 sweep, llm_profiler_inf2.ipynb (AOT NEFF + neuron-profile capture) 는
launch-overhead-free baseline 으로 작은 sweep. 둘의 차이가 launch overhead
inflation 의 정량적 evidence 가 된다.

---

## 1. 시작 — profile_neuron.py 의 결과가 의심스럽다 (대화 초기)

### 상태

- main branch 에 이미 partial Inferentia 2 perf bundle 이 있다
  (`commit d5ccaa8`):
  ```
  profiler/perf/Inferentia2/meta-llama/Llama-3.2-1B/bf16/
  ├── meta.yaml         (profiler_version: neuron-eager-v1)
  └── tp{1,2}/
      ├── dense.csv         (8.9 KB)
      ├── per_sequence.csv  (731 B)
      ├── attention.csv     (2.2 KB)
      └── profile_timing_*.json
  ```
- 위 bundle 은 사용자가 이전에 작성한 `scripts/profile_neuron.py`
  (eager mode + lazy XLA + module forward 직접 호출 + `perf_counter`)
  로 측정한 결과.
- 사용자가 위 bundle 을 simulator 에 입력해서 시뮬레이션 돌렸는데
  결과가 over-prediction.

### 사용자의 가설 (정확함, 단 부분만 검증)

`profile_neuron.py` 의 `time_callable` 함수의 measurement loop:

```python
for _ in range(repeat):
    sync()                              # ← per-iter sync
    t0 = time.perf_counter_ns()
    out = fn()
    sync()
    t1 = time.perf_counter_ns()
    timed_samples.append((t1 - t0) / 1000.0)
```

매 iteration 의 wallclock 안에 다음이 다 들어감:
1. forward dispatch (host overhead)
2. kernel launch (host overhead)
3. 실제 device 실행
4. sync 까지 wait

작은 kernel (예: layernorm 4 tokens, 실제 device time ~1us) 에선:
- device time: 1us
- per-iter host overhead (dispatch + launch): 수백 us
- 측정값 = 수백 us → **실제 device time 의 100× inflation**

큰 kernel (예: matmul 1024 tokens, 실제 device 50us) 에선 launch overhead
비율이 작아서 inflation 도 작음.

직접 본 evidence: `layernorm tokens=4` 가 RTX-PRO-6000 = 2.4us vs
Inferentia 2 = 410us (약 170× 차이). RTX 와 Inferentia 의 hardware FLOPS
차이 가 이 정도일 리 없음 → measurement 의 launch overhead inflation 이
원인이라고 판단.

### 잘못된 결정 (retrospective)

판단 자체는 맞았다 — per-iter sync 가 launch overhead 를 포함시켜서
작은 kernel 의 measurement 가 inflated 되는 게 사실. 단 **fix 방향이
잘못 결정됨**:

- **올바른 fix**: per-iter sync 를 for 문 밖으로 빼서 N forward 의 total
  wallclock / N 으로 launch overhead amortize (같은 graph 가 N 번
  cache hit 되면).
- **잘못된 fix (우리가 한 것)**: profile_neuron.py 자체를 invalid
  판정하고 TPU notebook 의 측정 mechanism (xp.start_trace + xp.Trace tag
  + Chrome trace JSON) 으로 reroute.

이 잘못된 결정의 시간 cost 가 그 후 모든 작업의 root cause.

이 시점에 "사용자 의도한 amortize fix 를 먼저 시도하고, 그게 안 되면 TPU
mechanism 으로 reroute" 가 합리적 순서였다. 우리는 fix 시도 없이 reroute
부터 했다.

---

## 2. TPU notebook port 시작 — `llm_profiler_inf2.ipynb` 작성

### 한 일

- ispass26-artifact branch 의 paper artifact 인
  `llm_profile/perf_models/TPU-v6e-1/llm_profiler_tpu.ipynb` 를
  `profiler/perf_models/Inferentia2/llm_profiler_inf2.ipynb` 로 verbatim
  copy (`commit 333785b`).
- 13 cell 구조:
  - Cell 0-3: env setup (TPU XLA install + Google Drive mount)
  - Cell 4: 메인 logic (~27KB) — `xla_timed_wrapper`, `patch_model`,
    `xla_timed_wrapper`, `_load_events`, `_exclusive_total`, `run_profile`
  - Cell 5: validation utilities (`measure_generation_latency`,
    `validate_latency_estimation`, `scale_latency_csv`)
  - Cell 6: `validate_and_scale` (scaling factor fit)
  - Cell 8: profiling parameters
  - Cell 9: run_profile
  - Cell 10-12: validation parameters / run / retest

### TPU notebook 의 측정 mechanism

```python
# Cell 4 의 핵심
model.to(torch_xla.device())                  # lazy XLA
patch_model(model, ...)                       # sub-layer .forward 들에
                                              # xp.Trace(tag) wrapper monkey-patch
xp.start_trace(log_dir)
for _ in range(repeat):
    model(input_ids, ...)
torch_xla.sync()
xp.stop_trace()

events = _load_chrome_trace_json(log_dir)
exclusive = _exclusive_total(events)          # tag 별 device time
```

이게 동작하려면 다음 3 개가 다 만족돼야:

* (P1) PyTorch 의 profiler 가 NeuronCore (또는 TPU) 에서 측정한 op-별
  timestamp 를 받아온다.
* (P2) lazy XLA mode 의 graph cache 가 같은 shape 의 두 번째 forward 를
  recompile 안 한다.
* (P3) `xp.Trace("tag")` string 이 XLA → HLO → 디바이스 binary 까지
  보존돼서, 측정된 op 들이 어느 sub-layer 소속인지 attribution 가능.

TPU 에선 셋 다 됨. Inferentia 2 에서는?

---

## 3. sanity1 — `xp.start_trace` 단독 시도 (TPU 와 동일하게)

### Script

`/tmp/sanity1.py` (한 번 만들고 commit 안 함):

```python
import torch, torch_xla
import torch_xla.debug.profiler as xp
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM

device = torch_xla.device()
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
).eval()
model.model.layers = model.model.layers[:1]
model.config.num_hidden_layers = 1
model.to(device)
input_ids = torch.randint(0, model.config.vocab_size, (1, 128), device=device)

# warmup
for _ in range(3):
    model(input_ids, use_cache=True)
    torch_xla.sync(wait=True)

# trace
log_dir = "/tmp/xla_trace_inf2"
xp.start_trace(log_dir=log_dir)
for _ in range(10):
    model(input_ids, use_cache=True)
torch_xla.sync(wait=True)
xp.stop_trace()
```

### 직접 본 결과 (stdout quote)

```
device_type: NEURON
...
warmup done
trace done

=== Trace files ===
       964 /tmp/xla_trace_inf2/plugins/profile/2026_05_10_04_58_54/ip-...trace.json.gz
      1411 /tmp/xla_trace_inf2/plugins/profile/2026_05_10_04_58_54/ip-...xplane.pb
```

trace JSON 파일은 만들어졌다 (964 byte). 풀어서 안 봄:

```
total events: 42
phases: {'M': 6, 'X': 35, None: 1}

first 20 X (timed) events:
  ts=1142.445 dur=     4.841 name=PjRtComputationClient::TransferToDevice ... pid=701
  ts=1198.769 dur=      2.59 name=PjRtComputationClient::TransferToDevice ... pid=701
  ts=3339.581 dur=      5.46 name=PjRtComputationClient::TransferToDevice ... pid=701
  ...
unique pids: [701]
```

35 개 X-phase event 모두 **`PjRtComputationClient::TransferToDevice`
(host-side input transfer)** 이고 pid 1개. NeuronCore 가 실제로 op 를
실행한 device-side timestamp 는 **0개**.

또 추가로 stdout 후반에서:

```
warmup done
..
Compiler status PASS
... Compilation Successfully Completed for model.MODULE_9702...
..
Compiler status PASS
... Compilation Successfully Completed for model.MODULE_1075...
... (이후 8회 더)
```

warmup 후의 trace window 안 10 forward → **`neuronx-cc` compile 도 ~10 회
발생** (각 ~30s). 즉 graph cache 가 **매 iteration miss**.

### 의미

- (P1) 부분 — `xp.start_trace` 가 trace 파일은 생성하지만 device-side
  timing 0 개. PyTorch/XLA profiler 가 받는 input 자체가 비어있다.
- (P2) 부분 — lazy XLA cache 가 **이 환경에서** miss 폭발. 매 forward 가
  새 compile.
- (P3) 부분 — 검증 못 함 (trace 가 device event 자체가 없으니).

### 정확한 root cause 가설 (직접 source 못 봄)

PyTorch/XLA 의 profiler (`xp.start_trace`) 는 PJRT plugin 이 push 해주는
event 들을 모은다.

> **PJRT plugin 이란?** PyTorch 가 가속기 (TPU/GPU/Neuron) 와 통신하는
> driver layer. TPU plugin 은 device-side op timestamp 를 push,
> GPU plugin 도 push. **Neuron PJRT plugin 은 push 안 하는 것으로 보임**
> — 우리가 본 host-only 결과의 가장 자연스러운 설명.

> **XPlane 이란?** TensorFlow/JAX/PyTorch-XLA 의 trace event 표현
> format. plane 별로 host events / device events 분리. TPU plugin 은
> device plane 을 채우고, Neuron plugin 은 host plane 만 채움 (것으로
> 보임).

> **"wire 한다" 의 의미**: device 가 측정한 op timing 을 XPlane 의 device
> plane 에 채워넣어서 PyTorch profile 이 읽을 수 있게 하는 것.

AWS 의 Neuron docs 가 `xp.Trace` 사용을 항상
`torch_neuronx.experimental.profiler.profile(...)` *안* 에서만 보여주고,
standalone `xp.start_trace`/`stop_trace` 예제는 없는 것도 우리가 본 결과
와 일관됨.

단 직접 Neuron PJRT plugin 의 source code 본 게 아니다. "stream 안 한다"
가 가장 가능성 높은 설명일 뿐이고, 다른 설명 (예: 우리 환경의 어떤 환경
변수가 disable 시켰다) 도 100% 배제는 못 한다.

### Retrospective — sanity1 의 cache miss 가 어디서 왔나

이 시점에 "warmup 후에도 매 forward 마다 새 compile" 이라는 evidence 를
보고 우리는 **"Inf2 lazy XLA 는 cache miss 폭발 하는 것" 으로 일반화**
했다. 이 일반화가 그 후 결정의 한 root cause.

근데 sanity1 의 환경은 다음이 다 set 된 상태:
- `XLA_HLO_DEBUG=1` (xp.Trace tag 보존을 위해 필수)
- `xp.start_trace` context (profile mode 진입)
- monkey-patch 한 sub-layer wrappers (Python frame 차이)

위 중 어느 하나가 graph hash perturb 의 원인일 수 있다. **Plain lazy XLA
환경 (`profile_neuron.py` 가 사용하는 환경) 에선 cache hit 할지 안 할지
sanity1 만으론 알 수 없다.** 우리가 그걸 검증하지 않고 "Inf2 lazy XLA 는
cache miss 가 되니 amortize 도 무리" 라는 단정으로 갔다.

---

## 4. sanity2 — `torch_neuronx.experimental.profiler.profile` wrapper 시도

### Script (요약)

`/tmp/sanity2.py`:

```python
import torch_neuronx.experimental.profiler as nprof

with nprof.profile(port=9012, profile_type='operator',
                   ms_duration=10000,
                   neuron_tensorboard_plugin_dir=log_dir):
    for _ in range(10):
        with xp.Trace("forward_full"):
            model(input_ids)
        torch_xla.sync(wait=False)
    torch_xla.sync(wait=True)
```

### 직접 본 stdout

```
WARNING:Neuron:Profiler stop: waiting 5s for async profile_stop in profile_id_p4216
INFO[0001] Successfully initialized OpenAPI server configuration.  basePath=/api/v2 version=v2
neuron-profile exited with an error.
neuron-profile 2.29.22.0%kaena-tools/2.29@b486b0a built on 2026-04-29T00:46:08Z
Unknown command `analyze'. Please specify one command of: capture, inspect, show-session or view
```

이 메시지가 ~10회 반복. 그 후:

```
INFO:Neuron:Profiling completed. Output directories:
trace done

=== Files ===
(빈 출력)
```

### 직접 본 것

- `nprof.profile` context 자체는 진입/종료된다.
- 내부적으로 `neuron-profile analyze` 라는 sub-command 호출.
- 우리 SDK (`neuron-profile 2.29.22.0`) 에선 이 명령이 **없음** —
  `neuron-profile --help` 가 `capture, inspect, show-session, view` 만
  보여줌.
- 결과 파일 0 개.

직접 source grep 으로 확인:
- `torch_neuronx.experimental.profiler.v2_x.profiling.py:407-408` 에
  정확히 `["..../neuron-profile", "analyze", ...]` subprocess 호출.

### 외부 자료 (직접 페이지 본 게 아님 — agent cite)

- AWS 가 Neuron 2.29 release notes 에서 Profiler 2.0 (`neuron-profile` +
  `torch_neuronx.experimental.profiler`) 의 EOL 을 announce 했고 Neuron
  Explorer 라는 새 toolchain 으로 대체 진행 중이라고 한다. 우리가 본 SDK
  호환성 깨짐과 일관됨.
- aws-neuron-sdk #1065 (Profiler 2.0 broken on inf2) 같은 issue 가
  존재한다고 agent 가 cite. 단 우리가 직접 issue 페이지 본 건 아니라
  정확한 quote 미확보.

### 결론

이 path 는 우리 SDK version 에서 사용 불가능. patch 시도해도 곧 사라질
toolchain 이라 ROI 0.

---

## 5. Web search agent — Inf2 의 device-side profiling 가능성 광범위 조사

### Agent 결과 (요약)

- TPU 의 lazy XLA + xp.start_trace 패턴이 Inf2 에선 architecturally
  동작 안 함:
  - Neuron PJRT plugin 이 device timing 을 XPlane 에 publish 안 함
  - lazy XLA cache 가 매 iteration miss (위 sanity1 일관)
- `torch_neuronx.experimental.profiler.profile` 도 SDK 2.29 에서 broken
  (Profiler 2.0 EOL)
- 권장 path:
  1. `torch_neuronx.trace` 로 NEFF compile (한 번)
  2. inference loop 안에서 호출 → host 측 timing 또는
  3. `neuron-profile capture/view` 로 NTFF + JSON 추출 → device-side
     timing
- 권장 env vars: `XLA_HLO_DEBUG=1` + `NEURON_FRAMEWORK_DEBUG=1` 면
  `xp.Trace` tag 가 NEFF metadata 까지 살아남음

### 결정

이 결과로 **TPU mechanism 의 1:1 reproduce 가 fundamentally 불가능**
이라 판단하고, AOT NEFF + `neuron-profile capture/view` path (B 안) 로
전환.

### Retrospective

- agent 결과의 일부 (특히 PJRT plugin gap) 는 우리가 직접 source 본 게
  아니다. 단 sanity1 의 직접 결과와 일관되니 가설로 받아들였다.
- 이 시점에 "lazy XLA 의 cache miss 가 그 환경 (xp.start_trace +
  XLA_HLO_DEBUG=1 + monkey-patch) 의 결과인지, 아니면 plain lazy XLA
  에서도 나타나는지" 별도 검증을 안 했다 — 결정에 필요했는데도.

---

## 6. sanity_c — AOT NEFF + neuron-profile capture (B 안 동작 확인)

### Script

```python
import torch_neuronx
import os
os.environ["XLA_HLO_DEBUG"] = "1"
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"

# 1. monkey-patch with xp.Trace tags
def wrap(tag, orig):
    def fn(*a, **kw):
        with xp.Trace(tag): return orig(*a, **kw)
    return fn

layer = model.model.layers[0]
layer.self_attn.q_proj.forward = wrap("self_attn/q_proj", layer.self_attn.q_proj.forward)
layer.self_attn.k_proj.forward = wrap("self_attn/k_proj", layer.self_attn.k_proj.forward)
layer.self_attn.v_proj.forward = wrap("self_attn/v_proj", layer.self_attn.v_proj.forward)
layer.self_attn.o_proj.forward = wrap("self_attn/o_proj", layer.self_attn.o_proj.forward)

# 2. AOT trace (NOT lazy XLA) — model 을 한 번 NEFF 로 compile
example_ids = torch.randint(0, vocab, (1, 128), dtype=torch.long)
class Wrap(torch.nn.Module):
    def forward(self, ids): return self.m(ids, use_cache=False).logits
traced = torch_neuronx.trace(Wrap(model), (example_ids,),
                              compiler_workdir="/tmp/sanity_c_workdir")

# 3. neuron-profile capture + view
subprocess.run(["neuron-profile", "capture", "-n", "/tmp/sanity_c_workdir/graph.neff",
                "-s", "/tmp/sanity_c.ntff"])
subprocess.run(["neuron-profile", "view", "--output-format", "json",
                "--output-file", "/tmp/sanity_c.json",
                "-n", "/tmp/sanity_c_workdir/graph.neff",
                "-s", "/tmp/sanity_c.ntff"])
```

### 직접 본 결과

```
trace OK
inference OK, output shape: torch.Size([1, 128, 128256])
NEFF: /tmp/sanity_c_workdir/graph.neff (940 MB)
```

JSON 안:
```
top-level keys: dma, layer_summary, instruction, summary, model_info, ...
layer_summary: 75 entries, 각각 {name, start, end, duration,
                                  tensor_engine_active_time,
                                  scalar_engine_active_time,
                                  vector_engine_active_time,
                                  sync_engine_active_time, ...}
instruction: 21567 entries
```

`xp.Trace` tag 가 살아남음:
```
string "q_proj" found 1084× in JSON
string "k_proj" found 273× in JSON
string "self_attn" found 2515× in JSON

/sg00/self_attn/q_proj.9                                     start=    3830  end= 2192540  tensor_active=38781
/sg00/self_attn/k_proj.10/aten__mm_dot.137                   start=    4317  end= 2204799  tensor_active=10451
/sg00/self_attn/v_proj.11                                    start=   15594  end= 2203379  tensor_active= 5196
/sg00/self_attn/o_proj.21                                    start= 2173545  end= 2528433  tensor_active=38646
```

### 결정적 검증 — GQA flops ratio

- `tensor_engine_flop_count`: q : k : v : o = 2147483648 : 536870912 :
  536870912 : 2147483648 = **4 : 1 : 1 : 4**
- 이게 정확히 Llama-3.2-1B 의 GQA (Q heads = 32, KV heads = 8) FLOPs 비율
- `tensor_engine_active_time`: 38781 : 10451 : 5196 : 38646 ≈ 같은 비율
- 즉 우리가 받는 device timing 이 hardware 의 실제 사용 패턴을 잡고
  있음

### 단위 확인

```
total_time (ns)             = 5992592       <- summary.total_time × 1e9
total_active_time (ns)      = 3886979
tensor_engine_active (ns)   = 1643395

layer_summary stats:
  count = 75
  max(end)        = 5992588    <- 거의 = total_time → ns 단위 확정
  sum(duration)   = 40551269   <- total 의 6.7×, layer window 끼리 overlap 가능 (parallel scheduling)
```

→ `layer_summary.duration / start / end` 와 `instruction.duration / timestamp`
는 **nanoseconds**. `summary.*_time` 은 seconds. 두 단위가 한 JSON 안에
섞여있음.

### 결정

이 path 가 동작한다. tag 보존됐고, GQA 구조가 timing 에 그대로 보임.
sub-layer 단위 device time 측정 가능. **사용 가능한 유일한 device-side
timing path** 라고 판단.

---

## 7. sanity_b — AOT NEFF path 의 cost 분석 (cache + inline=False)

### Phase 1: cold compile

```
NEURON_CC_FLAGS=--cache_dir=/tmp/sanity_b_cc_cache
=== Phase 1: COLD compile (cache empty) ===
  trace1: 63.4s  (workdir=/tmp/sanity_b_work1)
```

### Phase 2: 같은 shape 두 번째 trace (cache hit 기대)

```
=== Phase 2: SAME shape, NEW workdir — does NEURON_CC_FLAGS cache hit? ===
  trace2: 63.5s  (workdir=/tmp/sanity_b_work2)
  speedup: 1.0x  (cache MISS — recompiled cold)
```

→ **`NEURON_CC_FLAGS=--cache_dir=…` 가 silently broken**. 같은 shape 두 번째
trace 가 속도 향상 0, 명시한 cache 디렉토리는 비어있음.

### Phase 3: cache_dir 검증

```
=== Phase 3: cache_dir contents after 2 compiles ===
  /tmp/sanity_b_cc_cache: (0 files)
  /tmp/sanity_b_work1: 15 files, 1587.6 MB
            87 .../command.txt
     940493824 .../graph.neff
  /tmp/sanity_b_work2: 15 files, 1587.6 MB
     940493824 .../graph.neff
```

cache 가 진짜 사용 안 됨. 우리 cell 4 의 workdir disk reuse (`if not
os.path.exists(neff_path)`) 가 사실상 우회책.

### Phase 4: `inline_weights_to_neff=False` 로 변경

```
=== Phase 4: inline_weights_to_neff=False — does NEFF shrink? ===
  trace3: 10.3s  (inline_weights_to_neff=False)
  /tmp/sanity_b_work3_inline_false: 3 files, 1.3 MB
            91 .../command.txt
       1250304 .../graph.neff
         71077 .../model/graph.hlo
  NEFF size: inlined=940.5MB  not-inlined=1.3MB  shrink=752x
```

극적인 단축:
- NEFF 940 MB → **1.3 MB** (752× 감소).
- trace 시간 60s → **10s** (6× 감소).

### Phase 5: 작은 NEFF capture 시도

```
=== Phase 5: capture small NEFF — does capture get faster? ===
  test capture on .../graph.neff (1.3MB)
  capture: 7.2s  rc=0
  NTFF size: 17.2MB
```

- capture 30s+ → **7s** (4-5× 감소). NTFF 도 정상 17 MB 생성.

### 직접 본 것 — 비용 break-down (`sanity_a` 의 inference time 으로)

```
inf 0: 13323.86 ms   (NEFF first device load)
inf 1:    16.28 ms
inf 2:    16.54 ms
inf 3:    11.87 ms
inf 4:    15.87 ms
```

NEFF 한 번 device 에 올라간 뒤의 inference 는 ~16 ms (device time 6 ms +
Python wrapper 10 ms 정도). 즉 **device 가 느린 게 아니라 매 capture
process 마다 NEFF 를 다시 device 에 올리는 게 비용**.

### 직접 검증 못 한 부분 (open)

- TPU 의 같은 모델 / 같은 shape compile 시간 직접 측정 안 함. 60s 가
  abnormal 인지 normal 인지 비교 baseline 없음.
- `--cache_dir` 가 왜 broken 인지 격리 못 함. 다른 cache env var
  (`NEURONX_CACHE`, `NEURONX_PERSISTENT_CACHE_DIR` 등) 시도 안 함.
- `inline_weights_to_neff=False` 의 latency 정확성. NEFF 안에 weight 대신
  placeholder 있을 때 capture 가 어떤 weight 로 inference 하는지 (zero
  init? random? 별도 load?) 직접 확인 안 함. layer_summary 의 device
  timing 이 inline=True 와 동일한지 비교 측정 — TODO.

---

## 8. sanity_a — `NEURON_RT_INSPECT_ENABLE` 시도 (in-process NTFF auto-emit)

### 첫 시도 (NeuronCore 점유 conflict)

```
2026-May-10 06:59:24.358064 16842:16842 ERROR   NRT:nrt_allocate_neuron_cores
  Logical Neuron Core(s) not available - Requested:2 Available:0 Logical Core size 1
```

원인: 다른 process (아마 jupyter kernel) 가 NeuronCore 점유 중. kernel
restart 후 두 번째 시도.

### 두 번째 — phase 별 결과

```
=== Phase 2: inference 5x — measure host wallclock per call ===
  inf 0: 13323.86 ms                  ← NEFF first device load
  inf 1:    16.28 ms                  ← 그 다음부터는 16 ms
  inf 2:    16.54 ms
  inf 3:    11.87 ms
  inf 4:    15.87 ms

=== Phase 3: search for emitted NTFF ===
  /tmp/**/*.ntff:
        11830942 /tmp/sanity_c.ntff   ← 이전 sanity_c 의 잔존물
        11830942 ./sanity_c.ntff
  total NTFF files found: 2           ← 즉 NEURON_RT_INSPECT_ENABLE=1 으로 새로 emit 된 NTFF 0

=== Phase 4: try in-process profile context (alternative path) ===
  trying nprof.profile(...) context — output_dir=/tmp/sanity_a_inproc
  context done in 1.2s
  nprof.profile failed: AssertionError: No NEFFs were emitted - error during profiling
```

### 결정적 발견

- traced model 의 inference 는 한 번 device load 후 ~16 ms (sanity_c
  와 일관). 즉 **inference 자체는 빠르다**. capture 의 host overhead 가
  cost 의 dominant.
- `NEURON_RT_INSPECT_ENABLE=1` 환경변수만으론 NTFF 자동 emit **안 됨**.
- nprof.profile 도 broken (위 sanity2 와 같은 reason).
- 즉 NTFF 받는 유일한 path 는 `neuron-profile capture` subprocess.

### Retrospective — 우리가 빠뜨린 env var

이후 NxDI agent 조사에서 발굴: AWS vLLM-on-Trainium tutorial 의 정확한 env
var 조합은 다음 3 개:

* `NEURON_RT_INSPECT_ENABLE`
* `NEURON_RT_INSPECT_DEVICE_PROFILE`     ← 우리가 빠뜨림
* `NEURON_RT_INSPECT_OUTPUT_DIR`

만약 이 3 개를 다 set 하면 NTFF auto-emit 가능할 수도 있다 — sanity_a 를
아직 재시도 안 함. 동작하면 한 Python process 안에서 한 번 NEFF compile
+ N inference 호출로 N NTFF 자동 emit → batch view → per-cfg ~16 ms 까지
단축 잠재.

---

## 9. ipynb 작성 + 갱신 반복 — `llm_profiler_inf2.ipynb` 다듬기

### 첫 번째 build

cell 별 swap:
- Cell 0-3: env setup (TPU XLA install → no-op + Neuron venv 가정)
- Cell 4: capture mechanism 만 swap (xp.start_trace → torch_neuronx.trace
  + subprocess(neuron-profile capture/view)). `_load_events` /
  `_exclusive_total` → `_aggregate_layer_summary`. patch_model /
  xla_timed_wrapper / KV cache builder / _map_tags_to_results 다 그대로
- Cell 5/6/10/11/12: validation cells stub 처리 (이후 implement)
- Cell 8: parameters Inferentia2 / Llama-3.2-1B-Instruct
- Cell 9: run_profile

### 사용자 catch — `compile time 도 측정 가능?`

CSV fieldnames 에 `compile(ns)` column 추가 + `_trace_neff` 호출 wallclock
측정. 3 regime 자동 분류:
- 0 (workdir 의 graph.neff disk reuse → trace 호출 자체 안 함)
- 1-5 s (NEURON_CC_FLAGS cache hit, host trace overhead 만)
- 30 s+ (true cold compile)

### 사용자 catch — `cache 위치가 어디지?`

직접 본 결과로 정정:
- `/var/tmp/neuron-compile-cache/` — 8 KB, NEFF 0 개. **NEURON_CC_FLAGS
  cache 가 진짜 사용 안 됨**.
- `/tmp/inf2_neff/<model>/<cfg>/` — 4.5 GB, 진짜 cache.

### 사용자 catch — `compile 병렬?`

paper baseline 의 framing 으로 best-effort compile time 이 fair. 단 본
사이즈 (6 cfg) 에선 의미 작아 진행 안 함.

### 사용자 catch — `capture 가 60s 씩? 말이 되나?`

직접 measurement 으로 확정:
- 1 cfg 240 s (compile 60 s + capture 30 s × 3 = 90 s + view 30 s × 3)
- repeat=3 으로 N forward 평균 의도였는데 mechanism 잘못 — N subprocess
  capture 가 아니라 한 capture 안에 N forward 가 맞음.

→ `--profile-nth-exec=N` 발견. 적용하면 1 capture 안 N inference, N 번째
만 dump. host overhead 1 회 + device N inference. fix:
- `_capture_and_view` 에 `--profile-nth-exec=N` (default 20)
- main loop 의 `for it in range(repeat):` 제거
- 1 cfg 240 s → 60 s (4× 단축)

### 사용자 catch — `여전히 1분이라니 미치겠다`

Web agent 추가 조사 결과:
- 60 s/capture 가 expected (NEFF DMA load + runtime init + OpenAPI server boot)
- Optimization 옵션:
  - `--profile-nth-exec=N` (적용 완료)
  - `inline_weights_to_neff=False` (NEFF 940 MB → 1.3 MB → capture 30 s → 7 s)
  - persistent view server (cost 작음)

### 결정적 — silent failure

CSV 가 header 만 있고 row 0 개. 직접 evidence:
- `/tmp/inf2_ntff/` 에 잔존 file 발견 (성공 시 즉시 삭제하니, 잔존 = fail)
- 즉 **모든 config 가 silent capture fail**. 위 sanity_a 의 NeuronCore
  점유 conflict 와 같은 원인일 것

→ verbose error logging 추가:
- `_capture_and_view` 가 non-zero exit 시 stderr 600 자 + 정확한 cmd 출력
- fail 시 NTFF/JSON 보존 (debug)
- agg empty / `_map_tags_to_results` empty → 명시적 warning

### Retrospective

이 단계가 가장 시간 cost 큼. 단 결과적으로 working ipynb 와 최적화
옵션들 정리. 단 이 모든 게 "TPU mechanism reroute" 라는 잘못된 root
decision 위에 쌓임.

---

## 10. NxDI agent 조사

### 질문

NxDI (`neuronx-distributed-inference`) 가 layer-wise device timing hook
제공해서 우리 path 보다 better mechanism 인지 확인.

### Agent 결과 요약

- **Verdict: PARTIAL** — NxDI 가 약간 cleaner / faster compile loop
  이지만 per-sub-layer device-side timing 을 free 로 주지 않음. 결국
  `NEURON_RT_INSPECT_ENABLE` + `neuron-profile capture` 같은 path 를
  거치게 됨. 우리가 이미 채택한 path 보다 net 이득 없음.

- 우리 가설 검증 결과:
  - NxDI 가 HF sub-module 을 자기 구현으로 swap (`NeuronAttentionBase`,
    `RowParallelLinear`/`ColumnParallelLinear`, `NeuronLlamaAttention`/
    `NeuronLlamaMLP`) — 우리 monkey-patch 가 fire 안 함 ✅ 확인
  - `xp.Trace` tag 가 NEFF 에 안 들어감 ✅ 확인 (XLA HLO-debug
    propagation 이 Inf2 PJRT plugin 에서 broken — 동일 원인)
  - per-layer breakdown 부재 (LENS 의 NxDI 측정도 batch_e2e_ms 만)
    ✅ 확인
  - NxDI compile 시간 비슷 (Persistent Cache 가 advantage 지만 raw
    `torch_neuronx.trace` 도 같은 cache)

- 추가 발견 — `fused_qkv` 의 함정:
  - NxDI 가 production 에서 `fused_qkv=True` default-ish optimization
  - q/k/v projection 이 한 op 으로 collapse → sub-layer 별 timing 받을 수
    없음

- 단 — 새 evidence:
  - vLLM-on-Trainium tutorial 의 정확한 env var 조합:
    `NEURON_RT_INSPECT_ENABLE` + `NEURON_RT_INSPECT_DEVICE_PROFILE` +
    `NEURON_RT_INSPECT_OUTPUT_DIR`
  - 우리 sanity_a 가 가운데 var 빠뜨림 → 재시도 가치 있음
  - `XLA_IR_DEBUG=1` 도 추가로 켜면 instruction-level metadata 풍부해질
    가능성

### 결론

NxDI 채택 안 함. 단 NxDI 조사로 발굴된 위 두 env var 발견은 향후 시도 가치
있음.

---

## 11. LLMServingSim 정식 path 분석 — 우리 작업이 schema mismatch 임을 발견

### 발견한 문서

| 문서 | 역할 |
|---|---|
| `docs/docs/profiler/adding-hardware.md` | vLLM 지원 GPU vs 비-GPU(NPU/TPU) 두 path |
| `docs/docs/profiler/output-bundle.md` | CSV bundle schema (dense/per_sequence/attention/moe/skew) |
| `PROFILING_GUIDE_INF2_KO.md` | **Inferentia 2 한국어 가이드 — 완전 워크플로우** |
| `USAGE_GUIDE_KO.md` §5 | NPU 추가 전략 (이론) |

### CSV bundle schema (정식)

| File | Schema | Lookup |
|---|---|---|
| `dense.csv` | `layer, tokens, time_us` | 1D linear over tokens |
| `per_sequence.csv` | `layer, sequences, time_us` | 1D linear over sequences |
| `attention.csv` | `prefill_chunk, kv_prefill, n_decode, kv_decode, time_us` | Nearest-neighbour on (pc, n_decode) + bilinear on (kv_pre, kv_dec) |
| `moe.csv` | `tokens, activated_experts, time_us` | optional |
| `skew.csv` / `skew_fit.csv` | bucketed alpha | optional — `alpha_default` 만 두면 skip 가능 |
| `meta.yaml` | profiler_version, engine_effective, skew_fit.per_tp 등 | 필수 |

**Times = microseconds** (시뮬레이터가 ns 변환).

### 우리 작업이 schema mismatch

- 우리: TPU notebook 의 single CSV `hardware,model,layer_name,input,kv_cache,latency(ns)`
- 정식: per-category bundle (dense + per_sequence + attention + meta.yaml)

→ 우리가 그동안 만든 single CSV 는 simulator 가 못 읽음. 정식 schema 로
변환 작업이 추가 필요.

### main 의 partial Inferentia2 bundle 이미 존재 발견

`commit d5ccaa8` 의 `profiler/perf/Inferentia2/meta-llama/Llama-3.2-1B/bf16/`:
- meta.yaml: `profiler_version: neuron-eager-v1` (사용자가 만든 eager
  profiler. 우리 path 아님)
- attention.csv / dense.csv / per_sequence.csv: **정식 schema 따름**
- `calibration: scaled_by: raw eager profiler output (no NxDI calibration applied yet)`
  → 사용자가 이전에 어떤 eager profiler 작성 + bundle 만들었음

### 결정적 retrospective

이 시점에야 비로소 `scripts/profile_neuron.py` 가 사용자가 이전에 작성한
정식 path 임을 발견. PROFILING_GUIDE_INF2_KO §0 의 정의:

> 본 리포의 메인 profiler (`python -m profiler`) 는 NVIDIA CUDA 전용이라
> Inferentia 2 에선 못 쓴다. 대신 논문 (ISPASS 2026) 의 TPU-v6e-1 노트북
> 흐름을 Inferentia 2 로 이식한 두 단계 파이프라인.
>
> 1. **profile (NUM_LAYERS=1)** — `transformers + torch_neuronx`, 모듈
>    단위 직접 호출 + `perf_counter` → `scripts/profile_neuron.py`
> 2. **validate + 스칼라 보정** — NUM_LAYERS=full 같은 SDK, e2e generate()
>    측정 ↔ CSV 합산 → `scripts/validate_eager.py`
>
> NxDI 는 본 파이프라인에 **등장 안 함**. 논문도 안 씀.

즉 **우리가 그동안 한 모든 시도가 사실 같은 mechanism 의 다른 reroute** 
였음. 사용자가 처음에 invalid 판단한 시점에 `wait=True` fix (commit
99add22) 가 이미 적용된 상태였는데 그 후 sanity 다시 안 돌리고 reroute
했다.

---

## 12. vLLM-Neuron agent 조사 — 정식 GPU path 가 Neuron 에서 가능한지

### 질문 5 가지

LLMServingSim 의 정식 vLLM-based profiler:
```python
from vllm import LLM
llm = LLM(model=..., tensor_parallel_size=1, worker_extension_cls="...",
          hf_overrides={"num_hidden_layers": 1, ...}, ...)
# Worker 안에서:
self.model_runner.execute_model(_fresh_batch())
with layerwise_profile() as hook:
    for _ in range(iterations):
        self.model_runner.execute_model(_fresh_batch())
```

이게 vLLM-Neuron 에서 가능한지 5 question.

### Agent 결과 (요약)

| Q | 답 | 근거 |
|---|---|---|
| Q1. `vllm.LLM` API | **PARTIAL** — `block_size`, `num_gpu_blocks_override` 강제. `hf_overrides` single-rank shape emulation 이 NxDI 통과 안 함 | NxDI vLLM user guide |
| Q2. `worker_extension_cls` | **NO** — `NeuronWorker.__init__` 에 plumbing 자체 없음 + V1 `collective_rpc` broken | upstreaming-to-vllm `neuron_worker.py` + vLLM PR #15324 |
| Q3. `layerwise_profile` | **NO** ← **결정적** | `vllm/profiler/layerwise_profile.py` 가 `ProfilerActivity.CUDA` hard-coded, `cuda_time_us` 만 추출. Neuron 에선 모든 layer 가 0 us |
| Q4. catalog matching (class names) | **NO** — vLLM-Neuron 이 model 을 `NeuronCausalLM` 으로 swap. opaque NEFF wrapper. `LlamaAttention`/`QKVParallelLinear` tree 자체가 사라짐 | upstreaming-to-vllm `model_loader/neuron.py` |
| Q5. `model_runner.execute_model` | **PARTIAL** — `ModelInputForNeuron` (V0 dataclass) 받음, V1 `SchedulerOutput` 아님 | `neuron_model_runner.py` |

### 결정

**LLMServingSim 의 정식 vLLM-based path 가 Inferentia 2 에서 동작
불가능**. Q3 (layerwise_profile CUDA hard-coded) + Q4 (model swap 으로
nn.Module tree 사라짐) 가 결정적 두 blocker. patch 해도 둘 다 broken 이라
우회 불가능.

agent 결론 quote:

> Profiling Inferentia 2 latencies would require a separate path built on
> `torch_neuronx.experimental.profiler` plus operator-trace post-processing
> — essentially a new profiler module, not the existing one.

→ 즉 `profile_neuron.py` 가 그 separate path. 사용자가 이미 작성.

---

## 13. 사용자 핵심 의문 — per-iter sync 가 launch overhead 누적

### 사용자 질문

> 지금 결과는 이미 wait=True 상태에서 돌린 것이다. 하지만 문제가, 지금
> sync 를 for 문 안에서 하고 있어. 나는 이렇게 하면 kernel launch 까지
> 누적이 된다고 생각해. 이러면 매우 작은 operation 에 대해서 제대로
> 측정이 안 될 거라는 생각이 들었고 실제로 결과가 이상하게 나왔다고
> 생각한 거야.
>
> 그래서 처음에는 sync 를 for 문 밖으로 빼서 repeat 을 거의 1000 까지
> 늘린 다음에 1/1000 정도를 해야 제대로 된 측정이 나오는 것이 아닐까
> 라는 생각을 했어. 하지만 이것을 진행하기 전에, 애초에 이 측정 방식이
> valid 한가? LLMServingSim 의 저자들이 했던 방식을 그대로 채택해야 하는
> 것 아닐까? 라는 의문점이 들어서 작업을 시작했던 건데 너무 이상해져
> 버린 것 같다.

### 분석 (직접 본 것)

`profile_neuron.py:415-490` 의 `time_callable`:

```python
for _ in range(repeat):
    sync()
    t0 = time.perf_counter_ns()
    out = fn()
    sync()
    t1 = time.perf_counter_ns()
    timed_samples.append((t1 - t0) / 1000.0)
```

매 iter 가 host wallclock + sync. 사용자 분석 정확:
- 작은 kernel (layernorm 4 tokens, 실제 device 1us): launch overhead
  ~수백 us 가 dominant → 측정값 100× inflation
- 큰 kernel (matmul 1024 tokens, 실제 device 50us): launch overhead
  비율 작음 → inflation 작음

### `time_callable` 의 comment (직접 quote, line 447-451)

> Mirror the timed phase exactly: each fn() call is bracketed by sync()
> (= mark_step + wait_device_ops) so the compiled graph in the cache
> matches what the timed phase will look up. Without per-call sync, the
> warmup would build one big N-fn-calls graph and the timed phase's
> single-fn-call graph would be a fresh cache miss.

즉 이전 작성자도 이 issue 인지함. per-call sync 면 1-fn-call graph cache
hit, 빼면 N-fn-calls graph 로 build → fix 시 warmup 도 같은 방식 필요.

### TPU 의 mechanism 과 비교

TPU 는:
- `xp.start_trace` 안에서 N forward (sync 무관)
- Chrome trace JSON 에 device-side timestamp 박혀있음
- per-tag exclusive time = launch overhead 와 무관한 진짜 device time

Inferentia 2 에서 이 path 가 안 되는 이유:
- Neuron PJRT plugin 이 device timestamp publish 안 함 (sanity1 + agent
  결과)
- 즉 **Inf2 에서 device-side trace 가능한 유일한 path 가
  `neuron-profile capture/view`** ← 우리 그동안 한 작업

---

## 14. 최종 결정 — A 안 (cross-validation), profile_neuron.py amortize fix

### 두 path 의 비교

| Path | Measurement 종류 | Inf2 실현 가능 |
|---|---|---|
| TPU notebook 의 device-side trace | 진짜 device time, launch overhead 무관 | ❌ PJRT plugin |
| profile_neuron.py (host wallclock + per-iter sync) | host wallclock = device + launch overhead. 큰 kernel OK, 작은 kernel inflated | ✅ 작동 |
| **AOT NEFF + neuron-profile capture** ← 우리 ipynb | **진짜 device time** (instruction.duration in NTFF). launch overhead 무관 | ✅ 작동, 단 wallclock 비쌈 |

### 결정 — A: 두 path 다 유지

1. `profile_neuron.py` — 큰 sweep (전체 grid) 측정. 빠름. 단 작은 kernel
   inflated. 사용자가 의도한 amortize fix 적용 (sync 를 for 문 밖, N=1000)
2. `llm_profiler_inf2.ipynb` — 일부 representative point (5-10 cfg) 만
   측정. cost 크지만 진짜 device time. profile_neuron.py 의 inflation
   보정 baseline

두 결과 비교해서:
- 큰 kernel 에서 두 측정 일치 → profile_neuron.py 가 valid 임을 증명
- 작은 kernel 에서 inflation magnitude 정량 → §4 validate_eager.py 의
  scaling factor 가 보정 가능한지 검증

### Retrospective — 우리가 한 일이 사실 가치 있음

우리가 그동안 만든 ipynb (`llm_profiler_inf2.ipynb`) 가 cross-validation
용 도구. profile_neuron.py 단독은 inflation 보정의 layer 별 정확성을
증명할 baseline 이 없는데, 우리 ipynb 가 그 baseline 제공.

단 **"우리가 한 일이 가치 있다"는 것과 "이 길로 가는 게 합리적이었다" 는
다른 얘기**. 합리적 순서는:
1. 사용자 amortize fix 먼저 시도
2. 결과 valid 하면 (큰 kernel 에서 valid 면 OK) 거기서 stop
3. inflation 의심되면 cross-validation 도구로 ipynb 작성
4. 최후 수단 — TPU mechanism reroute

우리는 1, 2, 3 다 skip 하고 4 부터 갔다.

---

## 15. profile_neuron.py amortize fix (다음 작업)

사용자 의도:
- sync 를 for 문 밖으로 빼서 N forward dispatch + 1 sync + total/N
- N=1000 — 실행 시간 작아서 N 큰 거 OK
- 6 시간 걸려도 OK (이전 sweep 도 그 정도)

설계:
- `time_callable` 의 measurement loop 변경 옵션 추가
- 두 mode 다 지원: per-iter (현재 default) + loop-amortize (사용자 idea)
- warmup 도 같은 mode 로 (graph cache match — 위 comment 참고)
- N 너무 크면 graph compile size 폭발 risk → 일단 N=100 으로 시작, 결과
  보고 확장

상세 변경은 별도 commit 으로 추적.

---

## Lessons Learned

1. **Measurement validity 의심 → 검증 절차 (작은 case 에 대해 두 mechanism
   비교) 가 진행보다 우선.** 우리는 validity 의심 후 reroute 부터 했다.
   비교 절차 없이.
2. **새 path 시도 전에 기존 path 의 fix 가능성 재검토.** 사용자의 amortize
   fix 가 fundamental 이 아니라 trivial 한 변경인데 시도조차 안 했다.
3. **SDK 환경 의존성이 결과에 영향 — 한 환경의 결과가 다른 환경에 일반화
   X.** sanity1 의 cache miss (xp.start_trace + XLA_HLO_DEBUG + monkey-patch
   환경) 를 plain lazy XLA 에 일반화한 게 root cause 일부.
4. **agent 결과 cite 시 직접 source 본 게 아닌 부분 명시.** 우리 처음
   PROFILING_DESIGN.md 가 agent 가 추론한 것 과 우리가 직접 본 것 을 섞어
   서 단정조로 적었다.
5. **큰 결정 (path 폐기) 전에 작은 sanity (mechanism 검증) 가 비용 절약.**
   이 모든 시도의 시간 cost 가 amortize fix 시도 1 회보다 컸다.
6. **사용자가 처음 내린 판단의 정확도를 의심하기 전에 그 판단의 evidence
   를 다시 검토.** 사용자가 invalid 판단 시점에 wait=True fix 가 이미
   적용됐는지 우리는 commit log 안 봤다.
