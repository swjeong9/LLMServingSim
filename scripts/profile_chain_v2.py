"""profile_chain_v2.py — LlamaDecoderLayer.forward 를 우리 코드로 직접 chain.

profile_chain.py 의 issue: untagged ops (apply_rotary_pos_emb, repeat_kv,
F.scaled_dot_product_attention) 가 lazy XLA graph 에 쌓이면 다음 wrap 의
sync 시점에 그 cost 가 흡수됨 → o_proj 의 측정값이 attention 의 cost 까지
포함하는 mis-attribution.

이 script 는 LlamaDecoderLayer.forward 를 직접 chain 으로 호출해서 **모든
op 가 wrap 안에서 호출** 되도록 함. untagged op = 0.

Tags:
    input_layernorm, post_layernorm, qkv_proj (q+k+v 합),
    rope, repeat_kv, sdpa, o_proj,
    gate_up_proj (gate+up 합), silu_mul, down_proj,
    residual_add (×2)
"""
import argparse, csv, statistics, time
from collections import defaultdict
from pathlib import Path

import torch
try:
    import torch_neuronx   # registers libneuronpjrt on Inf2
except ImportError:
    pass
import torch_xla
import torch.nn.functional as F

from transformers import AutoConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, LlamaRotaryEmbedding,
    apply_rotary_pos_emb, repeat_kv,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",   default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--hw",      default="Inferentia2")
    p.add_argument("--variant", default="bf16")
    p.add_argument("--tp",      type=int, default=1)
    p.add_argument("--L-max",   type=int, default=8192)
    p.add_argument("--L-min",   type=int, default=None,
                   help="sweep start (inclusive); defaults to --L-step")
    p.add_argument("--L-step",  type=int, default=64)
    p.add_argument("--warmup",  type=int, default=3)
    p.add_argument("--repeat",  type=int, default=5)
    p.add_argument("--append",  action="store_true")
    args = p.parse_args()

    device = torch_xla.device()
    cfg = AutoConfig.from_pretrained(args.model)
    cfg.torch_dtype = torch.bfloat16
    cfg._attn_implementation = "sdpa"

    # === TP-N emulation: shard per-rank dims (paper SHARD_FIELDS) ===
    # Run a single NeuronCore with the per-rank shapes — actual TP
    # collectives are modelled by ASTRA-Sim. vocab_size omitted because
    # LlamaDecoderLayer does not touch the embedding / lm_head.
    if args.tp > 1:
        for field in ("intermediate_size", "num_attention_heads",
                      "num_key_value_heads"):
            val = getattr(cfg, field)
            if val % args.tp != 0:
                raise ValueError(
                    f"{field}={val} not divisible by tp={args.tp}")
            setattr(cfg, field, val // args.tp)
        print(f"[tp{args.tp}] sharded  nh={cfg.num_attention_heads} "
              f"nkv={cfg.num_key_value_heads} interm={cfg.intermediate_size}")

    H, nh, nkv, hd = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    n_rep = nh // nkv
    print(f"[load] {args.model}   hidden={H} heads={nh} kv={nkv} hd={hd}  n_rep={n_rep}")

    layer = LlamaDecoderLayer(cfg, layer_idx=0).to(torch.bfloat16).eval().to(device)
    rope_cpu = LlamaRotaryEmbedding(config=cfg)

    bag = defaultdict(int)
    def wrap(tag, fn):
        """sync + perf_counter wrap. Tagging all ops including transcendental/SDPA."""
        def wrapped(*a, **k):
            torch_xla.sync(wait=True)
            t0 = time.perf_counter_ns()
            out = fn(*a, **k)
            torch_xla.sync(wait=True)
            bag[tag] += time.perf_counter_ns() - t0
            return out
        return wrapped

    def my_forward(hidden, cos, sin):
        """Manual chain — every op wrapped, no untagged work between syncs."""
        sa, mlp = layer.self_attn, layer.mlp
        B, L, _ = hidden.shape

        # === Attention ===
        residual = hidden
        h = wrap("input_layernorm", layer.input_layernorm.forward)(hidden)

        q = wrap("q_proj", sa.q_proj.forward)(h)
        k = wrap("k_proj", sa.k_proj.forward)(h)
        v = wrap("v_proj", sa.v_proj.forward)(h)

        q = q.view(B, L, nh,  hd).transpose(1, 2)
        k = k.view(B, L, nkv, hd).transpose(1, 2)
        v = v.view(B, L, nkv, hd).transpose(1, 2)

        def _rope(q, k):
            return apply_rotary_pos_emb(q, k, cos, sin)
        q, k = wrap("rope", _rope)(q, k)

        def _repeat_kv(k, v):
            return repeat_kv(k, n_rep), repeat_kv(v, n_rep)
        k, v = wrap("repeat_kv", _repeat_kv)(k, v)

        def _sdpa(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = wrap("sdpa", _sdpa)(q, k, v)

        attn_out = attn_out.transpose(1, 2).reshape(B, L, nh * hd)
        o = wrap("o_proj", sa.o_proj.forward)(attn_out)
        hidden = wrap("residual_add", lambda r, o: r + o)(residual, o)

        # === MLP ===
        residual = hidden
        h = wrap("post_layernorm", layer.post_attention_layernorm.forward)(hidden)
        gate = wrap("gate_proj", mlp.gate_proj.forward)(h)
        up   = wrap("up_proj",   mlp.up_proj.forward)(h)
        def _silu_mul(g, u):
            return F.silu(g) * u
        activated = wrap("silu_mul", _silu_mul)(gate, up)
        down = wrap("down_proj", mlp.down_proj.forward)(activated)
        hidden = wrap("residual_add", lambda r, d: r + d)(residual, down)

        return hidden

    L_min = args.L_min if args.L_min is not None else args.L_step
    L_LIST = list(range(L_min, args.L_max + 1, args.L_step))
    print(f"sweep L = {L_LIST[0]}..{L_LIST[-1]}  step={args.L_step}  ({len(L_LIST)} cfgs)  "
          f"{'(append)' if args.append else '(fresh)'}")
    print()

    out_dir = Path("profiler/perf_chain_v2") / args.hw / args.model / args.variant / f"tp{args.tp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    f_dense = open(out_dir / "dense.csv", mode, newline="")
    w_dense = csv.writer(f_dense)
    if not args.append: w_dense.writerow(["layer", "tokens", "time_us"])
    f_attn = open(out_dir / "attention.csv", mode, newline="")
    w_attn = csv.writer(f_attn)
    if not args.append: w_attn.writerow(["prefill_chunk","kv_prefill","n_decode","kv_decode","time_us"])

    for L in L_LIST:
        hidden = torch.randn(1, L, H, dtype=torch.bfloat16).to(device)
        pos_ids_cpu = torch.arange(L).unsqueeze(0)
        cos_cpu, sin_cpu = rope_cpu(torch.randn(1, L, hd, dtype=torch.bfloat16), pos_ids_cpu)
        cos = cos_cpu.to(device); sin = sin_cpu.to(device)

        for _ in range(args.warmup):
            bag.clear()
            my_forward(hidden, cos, sin)

        shots = []
        for _ in range(args.repeat):
            bag.clear()
            my_forward(hidden, cos, sin)
            shots.append(dict(bag))

        per = {tag: statistics.median(s.get(tag, 0) for s in shots) / 1000.0
               for tag in shots[0]}

        # === Map fine-grained tags → simulator-expected layer names ===
        qkv = per["q_proj"] + per["k_proj"] + per["v_proj"]
        gu  = per["gate_proj"] + per["up_proj"]
        # simulator's `layernorm` row is per-invocation; estimator multiplies ×2
        layernorm_per_call = (per["input_layernorm"] + per["post_layernorm"]) / 2.0

        # Write ONLY rows the simulator understands:
        for layer_name, val in [
            ("layernorm",    layernorm_per_call),   # input+post avg, simulator ×2
            ("qkv_proj",     qkv),
            ("rotary_emb",   per["rope"]),
            ("o_proj",       per["o_proj"]),
            ("gate_up_proj", gu),
            ("act_fn",       per["silu_mul"]),
            ("down_proj",    per["down_proj"]),
        ]:
            w_dense.writerow((layer_name, L, val))
        # attention.csv = sdpa only (rope/repeat_kv are in dense or absorbed)
        w_attn.writerow((L, 0, 0, 0, per["sdpa"]))
        f_dense.flush(); f_attn.flush()

        # repeat_kv, residual_add are measured for debug but NOT in CSV
        print(f"  L={L:>5}  sdpa={per['sdpa']:>9.2f}  rope={per['rope']:>6.2f}  "
              f"qkv={qkv:>7.2f}  o={per['o_proj']:>7.2f}  gu={gu:>7.2f}  "
              f"silu_mul={per['silu_mul']:>7.2f}  down={per['down_proj']:>6.2f}  "
              f"[debug rk={per['repeat_kv']:.1f} ra={per['residual_add']:.1f}]",
              flush=True)

    f_dense.close(); f_attn.close()
    print(f"\n[wrote] {out_dir}")


if __name__ == "__main__":
    main()
