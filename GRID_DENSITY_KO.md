# Profile Grid 밀도 — 논문 데이터와 우리 lean default 비교

> 사용자가 `profiler/perf/RTXPRO6000/meta-llama/Llama-3.1-8B/bf16/tp1` (논문이 만든 GPU 프로파일) 을 보고 "왜 우리 sweep 양이 적은지" 물어본 것에 대한 답.

---

## 0. 각 CSV 파일이 무엇을 담는가

모델당 한 variant (예: `bf16`) / TP 폴더 안에 들어있는 파일들:

| 파일 | 무엇 | Schema (컬럼) | 역할 |
|---|---|---|---|
| `dense.csv` | **토큰-선형 layer 들의 latency**. embedding, layernorm, qkv_proj, rotary_emb, o_proj, gate_up_proj, act_fn, down_proj, final_layernorm 등. token 수에 비례하는 layer. | `layer, tokens, time_us` | 시뮬레이터가 한 iteration 의 dense 부분을 합성할 때 lookup |
| `per_sequence.csv` | **시퀀스-선형 layer**. lm_head, sampler. token 수가 아니라 **요청(시퀀스) 수** 에 비례. | `layer, sequences, time_us` | iteration 끝부분 (다음 토큰 sampling) 비용 |
| `attention.csv` | **attention kernel** (Q×K^T + softmax + softmax×V). 4D 키. qkv_proj 와 o_proj 는 여기 안 들어감 (그건 dense.csv) | `prefill_chunk, kv_prefill, n_decode, kv_decode, time_us` | (pc, kv_p, n, kv_d) 조합에 따른 attention cost |
| `moe.csv` | **MoE expert block** (Mixtral / Qwen3-MoE / Phi-MoE 같은 모델만). dense 모델에는 없음. | `tokens, activated_experts, time_us` | sparse expert routing 의 cost |
| `skew.csv` | **heterogeneous-decode raw 측정**. 한 batch 안 요청들의 KV history 길이가 다를 때 (예: 4 요청의 kv_d=[128,1024,4096,8192]) FlashAttention padding overhead 측정 | 14 컬럼 (regime, n, nb, ratio, skew, pc, kp, kvs, kv_big, kv_mean, t_mean_us, t_max_us, t_skew_us, alpha) | skew 보정 데이터 raw |
| `skew_fit.csv` | `skew.csv` 를 5축으로 bucketing + weighted least-squares fit 한 결과. lookup table. | `pc, n_label, skew_rate_label, kv_big_label, kp_label, alpha, n_samples` | 시뮬레이터가 attention 측정값에 alpha (∈ [0,1]) 곱해서 heterogeneous batch 보정 |
| `meta.yaml` | 메타: 어떤 grid 로 sweep 했는지, dtype / KV cache dtype / hardware 정보, calibration scaling factor, skew_fit 요약 | YAML | variant 식별 + 시뮬레이터의 grid 검증 + scaling factor 적용 |

본 가이드 (`profile_neuron.py`) 가 만드는 것:
- ✅ `dense.csv` — 모든 layer × token grid
- ✅ `per_sequence.csv` — lm_head 측정 + sampler 합성
- ✅ `attention.csv` — pure prefill + pure decode (mixed regime 미지원)
- ❌ `moe.csv` — dense 모델만 다루므로 미생성
- ❌ `skew.csv` / `skew_fit.csv` — 미생성, `meta.yaml::skew_fit::alpha_default=0.3` 상수 fallback

추가로:
- `profile_timing.json` — 프로파일링 자체의 wall time / per-shot compile vs measure 비용 (commit `aa800f7` 이후)
- `validation_timing.json` — 검증 단계 wall time (validate_eager.py 돌렸을 때)

---

## 1. 실측한 차이

| 항목 | 논문 bundle (Llama-3.1-8B / RTXPRO6000 / TP=1) | 우리 lean default | 비율 |
|---|---|---|---|
| `dense.csv` 토큰 grid 점 | **152 점** | 6 점 | 25× |
| `dense.csv` 총 행 | 1,368 | ~50 | 27× |
| `per_sequence.csv` seq 점 | 40 점 | 4 점 | 10× |
| `attention.csv` 총 행 | **19,364** | ~44 | **440×** |
| ↳ pure prefill | 233 | ~17 | 14× |
| ↳ pure decode | 171 | ~24 | 7× |
| ↳ **mixed prefill+decode 동시** | **18,960** | **0** | ∞ |
| `skew.csv` heterogeneous-decode 보정 | 13,010 행 | 없음 | — |

핵심: **attention 의 mixed 케이스 (한 iteration 안에서 prefill chunk 와 다른 요청의 decode 가 같이 섞이는 상황) 가 논문은 19k 점, 우리는 0 점**. 그리고 모든 축이 우리보다 5~25× 더 촘촘함.

**구체적으로 논문의 dense token grid**:
```
1, 2, 3, 4, ..., 16,    (1 단위)
20, 24, ..., 64,         (4 단위)  
80, 96, ..., 256,        (16 단위)
272, 288, ..., 2048      (16 단위)
```
→ small batch 영역 (1~16) 을 매우 촘촘히, 큰 영역은 16 단위 step.

**우리 lean default**:
```
1, 16, 64, 256, 1024, 2048
```
→ 6 점, 그 사이는 simulator 가 linear interpolation 으로 보간.

---

## 2. 왜 이만큼 차이가 나는가

논문 profiler 의 grid 는 **vLLM continuous batching 의 모든 동적 시나리오** 를 정확히 시뮬하기 위해 설계됨. 특히:

### (a) Mixed regime 의 존재
실제 vLLM 스케줄러는 한 iteration 에서 다음과 같은 조합을 허용:
- 요청 A 의 prefill 256 토큰 청크 + 요청 B,C,D 의 decode 1 토큰씩 (동시)
- 요청 A 가 길면 chunked prefill 로 256 씩 N step 에 걸쳐 처리, 그 동안 다른 요청 decode 는 계속 진행

이 mixed 케이스에서 attention 커널의 cost 는 단순히 `pure_prefill + pure_decode` 의 합이 아니다. FlashAttention 의 padding / SM 활용도 / GPU memory pattern 이 다르기 때문.

논문은 이걸 잡으려고 (prefill_chunk, kv_prefill, n_decode, kv_decode) 의 4D 격자를 18,960 점 측정함.

### (b) 작은 batch 영역의 촘촘한 sampling
decode latency 는 batch=1, 2, 3, ..., 16 영역에서 GPU 활용도가 빠르게 변함 (under-utilized → saturating). interpolation 만으로는 못 잡는 비선형성. → 논문은 1 단위로 촘촘하게.

### (c) Skew 보정
같은 batch 안에서도 요청마다 KV history 길이가 다르면 (예: 4개 요청의 kv_d 가 [128, 1024, 4096, 8192]), FlashAttention 의 padding overhead 때문에 실제 cost ≠ avg(kv_d) 에서 측정한 cost. 이 보정을 위한 별도 sweep (`skew.csv` 13k 행).

---

## 3. 그래서 우리 시나리오에 정말 필요한가?

**우리는 동적 도착 안 함, static offline batch 만**. 그래서 논문 시나리오와 prior 가 다름:

| 논문 시나리오 | 우리 시나리오 |
|---|---|
| Poisson 도착 (10 req/s) | t=0 에 200개 한꺼번에 |
| ShareGPT 길이 분산 큼 | uniform 또는 fixed 길이 (사용자 선택) |
| continuous batching 의 동적 효과 | **mode A**: 일부 발생 / **mode B**: 안 발생 |
| Chunked prefill | 우리는 `--no-enable-chunked-prefill` 로 끔 |
| Prefix caching | `--no-enable-prefix-caching` 로 끔 |

→ 결론: **우리 시나리오에선 논문만큼 dense grid 가 필요 없다**. 그러나 모드별로 좀 다름:

### Mode A (continuous batching, max_num_seqs=B)
- 200개 요청을 t=0 에 enqueue → 처음에는 B 개 prefill 동시 → 끝나는 순서대로 다음 요청 prefill
- 짧은 요청이 먼저 끝나고 빈 자리에 다음 요청 admit → **mixed regime 발생 가능** (한 iter 에 prefill + decode 섞임)
- 중간~말기엔 모두 decode (pure decode)
- → mixed 데이터 어느 정도 필요. 단 논문의 18,960 점 까진 과함.

### Mode B (strict static, batch i 끝나야 batch i+1 시작)
- 한 batch 안에서는 모든 4 요청이 같이 prefill → 같이 decode
- 한 batch 끝나면 다음 batch
- **mixed regime 거의 없음** (batch 경계에서만 잠깐)
- → mixed 데이터 거의 불필요. pure prefill / pure decode 만으로 충분.

### batch_size sweep 의 부담
batch_size B 별로 200 요청 = 50 batch 의 latency 를 측정 → batch 안 길이 분포에 따라:
- fixed-length 데이터: 모든 batch 가 동일 shape → 하나의 (pc, kv_d) 점만 매칭 → grid 1 점이면 충분
- uniform/sampled 데이터: batch 마다 길이 다름 → grid 가 그 분포를 커버해야 함
  - 예: input ∈ [256, 1024], output ∈ [32, 128] 면 prefill_chunk 는 [256, 1024] 사이, kv_decode 는 prefill 끝난 후 0~1152 사이를 sweep 해야 함

→ **데이터 분포가 좁으면 lean grid 도 충분, 분포가 넓으면 grid 확장 필요**.

---

## 4. lean → 정밀 grid 로 늘리는 법

`profile_neuron.py` 의 CLI 플래그 모두 grid 직접 지정 가능. 논문 수준으로 가려면:

```bash
# (예) 논문 RTXPRO6000 grid 와 동일 수준의 dense / attention sweep
python scripts/profile_neuron.py \
  --model meta-llama/Llama-3.2-1B \
  --tp 1,2,4,8 \
  --output-root profiler/perf \
  \
  --tokens-grid "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,28,32,36,40,44,48,52,56,60,64,80,96,112,128,144,160,192,224,256,320,384,448,512,640,768,896,1024,1280,1536,1792,2048" \
  --sequences-grid "1,2,3,4,5,6,7,8,12,16,24,32,48,64,96,128,192,256" \
  \
  --prefill-grid "0,16,32,64,128,256,512,1024,2048" \
  --kv-prefill-grid "0,512,1024,2048,4096,8192,16384" \
  --decode-n-grid "0,1,2,4,8,16,32,64,128,256" \
  --kv-decode-grid "0,16,32,64,128,256,512,1024,2048,4096,8192,16384" \
  \
  --warmup 10 --repeat 30
```

이러면 본 가이드의 sweep 시간이 약 **5~10×** 늘어남. 대략 모델당 첫 sweep 5~10 시간.

### 단, mixed regime 은 현재 profile_neuron.py 가 sweep 안 함

직전 commit (`17afe3e`) 에서 attention sweep 을 SDPA 직접 호출로 바꾸면서 pure prefill / pure decode 만 측정. **mixed (pc>0, kv_p>=0, n_decode>0, kv_d>0 동시) 케이스는 측정 안 함**. 만약 mode A 의 mixed regime 까지 정확히 잡고 싶으면 `sweep_attention()` 에 mixed loop 를 추가해야 함 (중간 정도 작업).

논문의 18,960 점 mixed 데이터를 따라가려면 sweep 에 **수십 시간** 추가. static offline 시나리오에선 보통 권장 안 함.

### Skew profile 도 현재 미지원

`profile_neuron.py` 는 skew sweep 안 함. simulator 가 `meta.yaml::skew_fit::alpha_default = 0.3` 를 상수로 사용 (3장 §5.2 참조). 정밀이 더 필요하면 별도 구현 필요 — 현재로선 미구현.

---

## 5. 권장 단계적 접근

처음부터 dense grid 를 돌리지 말고:

### Step 1 — Lean default 로 시작
현재 default. 모델당 ~30분.
```bash
python scripts/profile_neuron.py --model meta-llama/Llama-3.2-1B --tp 1,2,4,8 \
  --output-root profiler/perf
```

### Step 2 — 시뮬레이터 ↔ 실측 비교 (§6.2)
`compare_static.py` 의 `summary.txt` 를 봄. 핵심 메트릭은 `|pct err|` 의 median / p90.

### Step 3 — 에러 진단
- `|pct err|` median < 10% → lean grid 충분. 더 안 늘려도 됨.
- `|pct err|` median 10~25% → grid 살짝 densify (Step 4).
- `|pct err|` median > 25% → grid 문제일 가능성 큼 (Step 4 + 5).

`per_request_*.csv` 에서 어느 input/output 길이에서 에러 큰지 확인.

### Step 4 — 데이터 분포 영역만 densify
사용자 dataset 의 input_toks / output_toks 분포를 보고, 그 영역에 맞는 grid 만 늘림. 예:
- 데이터 input ∈ [128, 768] 이면 `--tokens-grid "1,32,64,128,256,512,768,1024"` (768 근처 추가)
- 데이터 output ∈ [16, 128] 이면 `--decode-n-grid "1,2,4,8,16,32,64,128"` 까지 늘림

부분적으로 늘려도 sweep 시간은 새 점 수에 비례해서만 증가.

### Step 5 — Mode A 인데 에러 큰 경우
Mixed regime 누락이 원인일 가능성. 현재 profile_neuron.py 미지원이므로:
- 옵션 A: `compare_static.py` 결과로 mode B 만 baseline 으로 사용
- 옵션 B: `sweep_attention()` 에 mixed loop 추가 구현 (~50줄). 필요하면 별도 요청

---

## 6. 한 줄 요약

논문이 19k 점 attention sweep 을 한 이유 = vLLM continuous batching 의 mixed regime 까지 잡으려고. 우리 사용자는 static offline 만 하므로 그 데이터의 99% 는 불필요. **lean default 로 시작 → 비교 결과 보고 필요한 영역만 densify** 하는 게 효율적. 처음부터 논문 따라 19k 안 가도 된다.
