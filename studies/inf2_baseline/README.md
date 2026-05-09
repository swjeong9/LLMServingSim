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

LENS repo 의 두 script 사용. profile_csv 인자로 우리 dataset CSV 직접 참조.

```bash
# LENS root: ~/npu_chip_project/LENS

# NxD-direct
python LENS/inference_profiling/inf2/run_profiling.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --tp-degree 1 --batch-size 4 --max-model-len 8192 \
    --n-runs 3 \
    --profile-csv studies/inf2_baseline/data/datasets/arxiv.csv \
    --output-dir studies/inf2_baseline/results/lens_nxd/Llama-3.2-1B/tp1/bs4

# vLLM-Neuron
python LENS/inference_profiling/inf2/run_profiling_vllm.py ... \
    --output-dir studies/inf2_baseline/results/lens_vllm/Llama-3.2-1B/tp1/bs4
```

### 3. Simulator 실행 (로컬 docker)

```bash
# studies/inf2_baseline/results/sim/<model>/tp<N>/bs<B>/<dataset>.csv 로 정리해 저장
for ds in arxiv cnn sharegpt writing_prompts; do
  for bs in 1 2 4 8 16 32; do
    python -m serving \
      --cluster-config configs/cluster/inf2_xlarge_llama1b_tp1.json \
      --dataset studies/inf2_baseline/workloads/${ds}_bs${bs}.jsonl \
      --output studies/inf2_baseline/results/sim/Llama-3.2-1B/tp1/bs${bs}/${ds}.csv \
      --max-num-seqs ${bs} \
      --dtype bfloat16
  done
done
# tp2 도 동일 (--cluster-config tp2 + 디렉토리 tp2)
```

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
