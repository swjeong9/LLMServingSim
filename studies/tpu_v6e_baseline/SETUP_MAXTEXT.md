# MaxText setup — TPU baseline 의 두 번째 production path

`measure_tpu.py` 는 LENS `run_eval_tpu.py` 의 LENS-free port — MaxText
(Google JAX-based LLM framework) 의 `MaxEngine` 으로 inference 측정.
vLLM-TPU 가 빠를 가능성 (Pallas attention 의 aggressive optimization)
의 cross-check 용 baseline.

같은 (dataset, batch, TP) 에 대해 **MaxText 와 vLLM 둘 다 측정** → 두 production framework 의 latency 가 일치하면 vLLM 측정이 신뢰 가능, 차이 크면 framework overhead 의 evidence.

---

## 1. JAX (TPU build) 설치

이미 설치했으면 skip. sanity:
```bash
python -c "import jax; print(jax.devices())"
# 기대: [TpuDevice(...)]
```

설치 안 했으면:
```bash
pip install "jax[tpu]" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

---

## 2. MaxText 설치

pip package 없음 — git clone + editable install.

```bash
git clone https://github.com/AI-Hypercomputer/maxtext.git ~/maxtext
cd ~/maxtext
pip install -r requirements.txt
pip install -e .
```

requirements.txt 가 크다 (jax, flax, optax, orbax-checkpoint, grain, sentencepiece, tiktoken 등). 5-15분.

검증:
```bash
python -c "from maxtext.configs import pyconfig; print('maxtext OK')"
python -c "from maxtext.inference.maxengine.maxengine import MaxEngine; print('MaxEngine OK')"
```

---

## 3. HF Llama checkpoint 다운로드

Llama-3.2-1B-Instruct 의 PyTorch weights.

```bash
# HF token 필요 (gated model)
export HF_TOKEN="hf_xxxxxxx"

# CLI 로
pip install huggingface_hub
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
    --local-dir ~/hf_models/Llama-3.2-1B-Instruct \
    --include "*.safetensors" "*.json" "tokenizer*"

# 또는 transformers 의 from_pretrained 가 알아서 download
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')
t = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')
print(m.config.architectures, '\\nlocal cache at:', m.config._name_or_path)
"
# → cache 위치: ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/<hash>/
```

`~/.cache/huggingface/...` 의 snapshot dir 경로 기억해두기 (다음 단계 input).

---

## 4. HF → MaxText Orbax checkpoint 변환

MaxText 의 `llama_or_mistral_ckpt.py` 가 변환 script.

```bash
# HF cache 의 snapshot path 를 BASE_PATH 로
BASE_PATH="$HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/<hash>"
# 또는 별도 다운받은 path 면 거기 사용
# BASE_PATH="$HOME/hf_models/Llama-3.2-1B-Instruct"

MAXTEXT_CKPT="$HOME/maxtext_ckpts/llama3.2-1b"
mkdir -p "$MAXTEXT_CKPT"

cd ~/maxtext
JAX_PLATFORMS='' python -m MaxText.llama_or_mistral_ckpt \
    --base-model-path "$BASE_PATH" \
    --maxtext-model-path "$MAXTEXT_CKPT" \
    --model-size llama3.2-1b
```

* `JAX_PLATFORMS=''` — 변환 자체는 CPU 면 충분. TPU 점유 안 함 (다른 측정과 동시 실행 가능)
* 1B 모델: ~10분, ~3GB disk
* 결과: `$HOME/maxtext_ckpts/llama3.2-1b/0/items/` (Orbax checkpoint format)

검증:
```bash
ls $HOME/maxtext_ckpts/llama3.2-1b/0/items/
# 기대: manifest.ocdbt, metadata, _0.parquet 등 Orbax 파일들
```

---

## 5. `measure_tpu.py` 실행 (sanity)

```bash
cd ~/LLMServingSim

# TPU 점유 확인 (다른 process 가 잡고 있지 않은지)
sudo fuser -v /dev/vfio/0
# 빈 출력이면 OK. 잡혔으면 pkill -9 -f ipykernel 등 정리

# 1 dataset × 1 batch_size sanity
BATCH_SIZES="1" DATASETS="cnn" \
LOAD_PARAMETERS_PATH="$HOME/maxtext_ckpts/llama3.2-1b/0/items" \
TOKENIZER_PATH="meta-llama/Llama-3.2-1B-Instruct" \
MAXTEXT_MODEL_NAME="llama3.2-1b" \
    bash studies/tpu_v6e_baseline/sweep_tpu.sh
```

기대 출력:
```
=== [tp1 bs1 cnn] start ... ===
[init_engines_multi] decode_buckets=[128, 256, 512, 1024, 2048, 4096, 8192]
[init_engines_multi] devices=[TpuDevice(...)]
  engine[bucket=128] ready ...
  engine[bucket=256] ready ...
  ...
  warmup (il=..., ol=...): ...
  [run  0] max_il=... max_ol=... OK e2e=  XXX ms
  ...
[done] tp1 bs1 cnn
```

첫 init 시 7 buckets × N 분 (Llama-3.2-1B 작음 → 각 bucket ~30s = 약 5분 total init). 두 번째부터 JAX cache (`~/jax_cache_lens_profiling`) 히트.

---

## 6. Full sweep + 회수

sanity 통과 후 full sweep:

```bash
LOAD_PARAMETERS_PATH="$HOME/maxtext_ckpts/llama3.2-1b/0/items" \
TOKENIZER_PATH="meta-llama/Llama-3.2-1B-Instruct" \
MAXTEXT_MODEL_NAME="llama3.2-1b" \
    bash studies/tpu_v6e_baseline/sweep_tpu.sh
```

기본 sweep: TP=1 × {bs=1,2,4,8,16,32} × {arxiv,cnn,sharegpt,writing_prompts}.

결과 위치: `studies/tpu_v6e_baseline/results/lens_tpu/Llama-3.2-1B-Instruct/tp1/bs<B>/<dataset>.csv`

호스트 (Mac) 로 회수:
```bash
# 로컬에서
rsync -avL v6e-1:~/LLMServingSim/studies/tpu_v6e_baseline/results/lens_tpu/ \
    studies/tpu_v6e_baseline/results/lens_tpu/
```

---

## 7. 비교 (sim + vLLM + MaxText 3-way)

`compare.py` 가 이미 3-way 지원 (sim / lens_tpu / lens_vllm):

```bash
python studies/tpu_v6e_baseline/compare.py --tps 1 --batch-sizes 1
```

출력 표 + figure (`studies/tpu_v6e_baseline/figures/`):
- 3 bar per dataset: LLMServingSim2.0 / MaxText / vLLM
- vLLM ≈ MaxText 면 → vLLM 측정이 fair (두 production path 일치)
- vLLM ≪ MaxText 면 → vLLM 의 Pallas attention 이 aggressive 하게 빠른 것 (또는 MaxText 가 overhead 큰 것)

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'maxtext'`** — `pip install -e .` 안 됐거나, 다른 venv 에서 실행 중. `pip show maxtext` 로 install 위치 확인.

**`TPU initialization failed: open(/dev/vfio/0): Device or resource busy`** — 다른 process (jupyter kernel, 다른 measure script) 가 TPU 점유. `sudo fuser -v /dev/vfio/0` + `pkill -9 -f ipykernel` 로 free.

**`Failed to find a target node ... model_name=llama3.2-1b`** — MaxText 의 yaml registry 에 그 name 없음. `cat ~/maxtext/MaxText/configs/models/` 로 actual model name 확인 (예: `llama3.1-8b`, `llama3-8b` 같은 식. `llama3.2-1b` 가 LENS run_eval_tpu.py 에서 사용한 그대로면 그대로).

**Checkpoint 변환 시 OOM** — 1B 모델은 CPU 32GB 면 충분. 더 큰 모델 (8B) 면 별도 host 또는 sharded conversion 필요.

**Orbax checkpoint format 미스매치** — MaxText 의 minor version 따라 manifest schema 변경 가능. 변환 시 사용한 MaxText commit 과 measure 시 사용한 commit 일치시켜야.

---

## 비용 예상

| 단계 | 시간 |
|---|---|
| MaxText install | 5-15분 |
| HF checkpoint 다운로드 | 5-10분 (network 따라) |
| Checkpoint 변환 (1B) | ~10분 (CPU) |
| 첫 init (7 buckets) | ~5분 (TPU + JAX compile) |
| Full sweep 1B (24 (dataset, bs) 조합) | ~30분-1시간 |
| **total (1B)** | **약 1시간** |

`jax_cache_lens_profiling` 이 채워지면 두 번째 sweep 부터 init 빨라짐 (compile 재사용).
