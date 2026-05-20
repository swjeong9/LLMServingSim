"""profile_extras.py — fill in rows missing from profile_chain_v2.py output.

profile_chain_v2.py only measures LlamaDecoderLayer (1 layer) prefill. Missing:
  - embedding (model.model.embed_tokens)            → dense.csv
  - final_layernorm (model.model.norm)              → dense.csv
  - lm_head (model.lm_head)                         → per_sequence.csv
  - sampler (argmax / softmax)                      → per_sequence.csv
  - attention decode case (n=1, kvd>0)              → attention.csv

This script appends the missing rows to the existing CSVs in:
    profiler/perf_chain_v2/<hw>/<model>/<variant>/tp<N>/

Per-call sync + perf_counter wrap, same mechanism as profile_chain_v2.py.
"""
import argparse, csv, statistics, time
from collections import defaultdict
from pathlib import Path

import torch
try:
    import torch_neuronx
except ImportError:
    pass
import torch_xla
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding, apply_rotary_pos_emb, repeat_kv,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",   default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--hw",      default="Inferentia2")
    p.add_argument("--variant", default="bf16")
    p.add_argument("--tp",      type=int, default=1)
    p.add_argument("--mode",    required=True, choices=("dense_extras", "per_sequence", "attn_decode"),
                   help="dense_extras=embedding+final_layernorm; per_sequence=lm_head+sampler; "
                        "attn_decode=batched-decode attention sweep (batch_size × kv_cache_size)")
    # token / kv sweep (used by dense_extras + per_sequence)
    p.add_argument("--N-min",  type=int, default=64)
    p.add_argument("--N-max",  type=int, default=4736)
    p.add_argument("--N-step", type=int, default=64)
    p.add_argument("--include-N1", action="store_true",
                   help="prepend N=1 to the sweep")
    # attn_decode sweep grids (mirrors v0 batch_sampling.py)
    p.add_argument("--bs-list",  default="1,2,4,8,16,32",
                   help="comma-separated batch sizes for attn_decode")
    p.add_argument("--kv-list",  default="32,64,128,256,512,1024,2048,4096",
                   help="comma-separated kv_cache_size values for attn_decode")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=5)
    args = p.parse_args()

    device = torch_xla.device()
    cfg = AutoConfig.from_pretrained(args.model)
    cfg.torch_dtype = torch.bfloat16
    cfg._attn_implementation = "sdpa"

    # === TP-N emulation: shard per-rank dims (paper SHARD_FIELDS) ===
    # vocab_size sharded too — embedding (VocabParallelEmbedding) and
    # lm_head (ColumnParallelLinear) split vocab across ranks in vLLM-style
    # tensor parallel. Per-rank weights = full // tp.
    if args.tp > 1:
        for field in ("intermediate_size", "num_attention_heads",
                      "num_key_value_heads", "vocab_size"):
            val = getattr(cfg, field)
            if val % args.tp != 0:
                raise ValueError(
                    f"{field}={val} not divisible by tp={args.tp}")
            setattr(cfg, field, val // args.tp)
        print(f"[tp{args.tp}] sharded  nh={cfg.num_attention_heads} "
              f"nkv={cfg.num_key_value_heads} interm={cfg.intermediate_size} "
              f"vocab={cfg.vocab_size}")

    H = cfg.hidden_size
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    n_rep = nh // nkv
    print(f"[load] {args.model}  hidden={H} heads={nh} kv={nkv} hd={hd}  mode={args.mode}")

    bag = defaultdict(int)
    def wrap(tag, fn):
        def wrapped(*a, **k):
            torch_xla.sync(wait=True)
            t0 = time.perf_counter_ns()
            out = fn(*a, **k)
            torch_xla.sync(wait=True)
            bag[tag] += time.perf_counter_ns() - t0
            return out
        return wrapped

    out_dir = Path("profiler/perf_chain_v2") / args.hw / args.model / args.variant / f"tp{args.tp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Build N sweep list ===
    N_LIST = list(range(args.N_min, args.N_max + 1, args.N_step))
    if args.include_N1 and 1 not in N_LIST:
        N_LIST = [1] + N_LIST

    # ====================================================================
    # MODE: dense_extras — embedding + final_layernorm
    # ====================================================================
    if args.mode == "dense_extras":
        # tp=1: real weights (matches existing tp=1 dataset).
        # tp>1: sharded cfg → from_config (random weights; latency depends
        # on shape, not values).
        if args.tp == 1:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).eval()
        else:
            model = AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).eval()
        model.to(device)
        embed = model.model.embed_tokens
        norm = model.model.norm

        f_dense = open(out_dir / "dense.csv", "a", newline="")
        w_dense = csv.writer(f_dense)
        print(f"sweep N = {N_LIST[0]}..{N_LIST[-1]}  step={args.N_step}  ({len(N_LIST)} cfgs)")
        for N in N_LIST:
            input_ids = torch.randint(0, cfg.vocab_size, (1, N)).to(device)
            hidden = torch.randn(1, N, H, dtype=torch.bfloat16).to(device)
            for _ in range(args.warmup):
                bag.clear()
                wrap("embedding", embed.forward)(input_ids)
                wrap("final_layernorm", norm.forward)(hidden)
            shots = []
            for _ in range(args.repeat):
                bag.clear()
                wrap("embedding", embed.forward)(input_ids)
                wrap("final_layernorm", norm.forward)(hidden)
                shots.append(dict(bag))
            per = {tag: statistics.median(s.get(tag, 0) for s in shots) / 1000.0 for tag in shots[0]}
            for layer_name, val in [
                ("embedding",       per["embedding"]),
                ("final_layernorm", per["final_layernorm"]),
            ]:
                w_dense.writerow((layer_name, N, val))
            f_dense.flush()
            print(f"  N={N:>5}  embedding={per['embedding']:>7.2f}us  "
                  f"final_layernorm={per['final_layernorm']:>7.2f}us", flush=True)
        f_dense.close()

    # ====================================================================
    # MODE: per_sequence — lm_head + sampler
    # ====================================================================
    elif args.mode == "per_sequence":
        if args.tp == 1:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).eval()
        else:
            model = AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).eval()
        model.to(device)
        lm_head = model.lm_head

        per_seq_path = out_dir / "per_sequence.csv"
        write_header = not per_seq_path.exists()
        f_pseq = open(per_seq_path, "a", newline="")
        w_pseq = csv.writer(f_pseq)
        if write_header:
            w_pseq.writerow(["layer", "sequences", "time_us"])

        # per_sequence: number of sequences (request-level). For lm_head we feed
        # (B=S, hidden) — one final hidden per sequence.
        S_LIST = N_LIST  # reuse same sweep, but interpret as # sequences
        print(f"sweep S = {S_LIST[0]}..{S_LIST[-1]}  ({len(S_LIST)} cfgs)")
        for S in S_LIST:
            hidden = torch.randn(S, H, dtype=torch.bfloat16).to(device)  # (S, H)
            for _ in range(args.warmup):
                bag.clear()
                logits = wrap("lm_head", lm_head.forward)(hidden)
                wrap("sampler", lambda l: torch.argmax(l, dim=-1))(logits)
            shots = []
            for _ in range(args.repeat):
                bag.clear()
                logits = wrap("lm_head", lm_head.forward)(hidden)
                wrap("sampler", lambda l: torch.argmax(l, dim=-1))(logits)
                shots.append(dict(bag))
            per = {tag: statistics.median(s.get(tag, 0) for s in shots) / 1000.0 for tag in shots[0]}
            for layer_name, val in [
                ("lm_head", per["lm_head"]),
                ("sampler", per["sampler"]),
            ]:
                w_pseq.writerow((layer_name, S, val))
            f_pseq.flush()
            print(f"  S={S:>5}  lm_head={per['lm_head']:>9.2f}us  "
                  f"sampler={per['sampler']:>7.2f}us", flush=True)
        f_pseq.close()

    # ====================================================================
    # MODE: attn_decode — batched-decode SDPA, sweep (batch_size × kv_cache_size)
    # Mirrors v0 batch_sampling.py: get_attention_batch_sizes_to_profile +
    # get_seq_lengths_to_profile, but as a tractable cross-product (not full
    # v0 dense grid which is ~10k cfgs).
    # ====================================================================
    elif args.mode == "attn_decode":
        bs_list  = [int(x) for x in args.bs_list.split(",")]
        kv_list  = [int(x) for x in args.kv_list.split(",")]

        f_attn = open(out_dir / "attention.csv", "a", newline="")
        w_attn = csv.writer(f_attn)
        print(f"sweep (batch_size × kv_cache_size) = {len(bs_list)} × {len(kv_list)} = "
              f"{len(bs_list)*len(kv_list)} cfgs")
        print(f"  bs_list={bs_list}")
        print(f"  kv_list={kv_list}")
        for B in bs_list:
            for kvd in kv_list:
                # batched decode shapes:
                #   q : (B, nh,  1,    hd)        ← 1 token per sequence in batch
                #   k : (B, nkv, kvd+1, hd) → repeat_kv → (B, nh, kvd+1, hd)
                #   v : (B, nkv, kvd+1, hd) → repeat_kv → (B, nh, kvd+1, hd)
                q = torch.randn(B, nh,  1,       hd, dtype=torch.bfloat16).to(device)
                k_un = torch.randn(B, nkv, kvd + 1, hd, dtype=torch.bfloat16).to(device)
                v_un = torch.randn(B, nkv, kvd + 1, hd, dtype=torch.bfloat16).to(device)
                torch_xla.sync(wait=True)
                k = k_un.repeat_interleave(n_rep, dim=1); torch_xla.sync(wait=True)
                v = v_un.repeat_interleave(n_rep, dim=1); torch_xla.sync(wait=True)

                def _sdpa(q, k, v):
                    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)

                try:
                    for _ in range(args.warmup):
                        bag.clear()
                        wrap("sdpa_decode", _sdpa)(q, k, v)
                    shots = []
                    for _ in range(args.repeat):
                        bag.clear()
                        wrap("sdpa_decode", _sdpa)(q, k, v)
                        shots.append(dict(bag))
                    t = statistics.median(s.get("sdpa_decode", 0) for s in shots) / 1000.0
                except RuntimeError as e:
                    print(f"  B={B:>4} kvd={kvd:>5}  FAILED: {type(e).__name__}: {str(e)[:80]}",
                          flush=True)
                    continue
                # attention.csv row: prefill_chunk=0, kv_prefill=0, n_decode=B, kv_decode=kvd
                w_attn.writerow((0, 0, B, kvd, t))
                f_attn.flush()
                print(f"  B={B:>4} kvd={kvd:>5}  sdpa_decode={t:>7.2f}us", flush=True)
        f_attn.close()

    print(f"\n[done] mode={args.mode}  output: {out_dir}")


if __name__ == "__main__":
    main()
