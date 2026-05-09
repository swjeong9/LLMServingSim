"""Verify whether torch_xla.sync() truly waits for device completion.

Improved design (per user feedback): the previous version compared
single-matmul vs 2-matmul cases, but those are *different NEFF graphs*,
so the 2-matmul measurement was polluted by NEFF compile / device-load
time (worsened on EBS-backed instances). This version uses the SAME
NEFF throughout, with enough warmup that the NEFF is hot in the
device (no further disk I/O), so measurements isolate per-call work.

Three measurement modes, all on the same NEFF:

  [A] per-iter sync:        sync(); t0; fn(); sync(); t1   (N iters)
                            -> mimics profile_neuron.py
  [B] single sync at end:   t0; for _ in N: fn(); sync(); t1
                            -> wall/N gives the true per-call device
                               time, regardless of sync behavior, since
                               the final sync drains the entire queue
  [C] per-iter readback:    sync(); t0; fn(); out.item(); t1  (N iters)
                            -> .item() forces device->host transfer per
                               call, so this is "forced wait" per call

Decision:
  if [A] ≈ [B]/N ≈ [C]  : sync() truly waits per call. Profile data OK.
  if [A] << [B]/N ≈ [C] : sync() does not wait per call. Profile broken.
  if all three small    : the matmul itself is genuinely fast (Inf2
                          fast path); not a sync issue.
"""
import time
import torch
import torch_xla
import torch_xla.core.xla_model as xm


def sync():
    if hasattr(torch_xla, "sync"):
        torch_xla.sync()
    else:
        xm.mark_step()
        xm.wait_device_ops()


def main():
    device = torch_xla.device()

    # 8192^3 BF16 matmul = 1.1 TFLOPs.
    # At Inf2 95 TFLOPS/core BF16 dense: ~11.6 ms expected
    # At sparse 380 TFLOPS: ~2.9 ms
    # If we measure < 1 ms with sync(), sync is broken.
    N_LARGE = 8192
    A = torch.randn(N_LARGE, N_LARGE, dtype=torch.bfloat16).to(device)
    B = torch.randn(N_LARGE, N_LARGE, dtype=torch.bfloat16).to(device)

    def matmul():
        return A @ B

    # === Compile + warmup ===
    # Warm up enough to ensure NEFF is loaded into device HBM and the
    # XLA cache is hot. After this, all calls use the SAME NEFF.
    print("Compiling NEFF + warmup (10 iters)...")
    sync()
    t0 = time.perf_counter()
    for _ in range(10):
        out = matmul()
        sync()
        del out
    print(f"  warmup wall: {(time.perf_counter()-t0)*1000:.1f} ms total")

    N = 50

    # === [A] per-iter sync (profile_neuron pattern) ===
    per_iter_samples = []
    for _ in range(N):
        sync()
        t0 = time.perf_counter()
        out = matmul()
        sync()
        t1 = time.perf_counter()
        per_iter_samples.append((t1 - t0) * 1000)
        del out
    A_mean = sum(per_iter_samples) / N
    print(f"\n[A] per-iter sync, N={N}:")
    print(f"    mean per-call: {A_mean:8.3f} ms")
    print(f"    total wall:    {sum(per_iter_samples):8.1f} ms")

    # === [B] single sync at end (ground truth: wall/N) ===
    sync()
    t0 = time.perf_counter()
    outs = []
    for _ in range(N):
        outs.append(matmul())
    sync()
    t1 = time.perf_counter()
    B_total = (t1 - t0) * 1000
    B_per_call = B_total / N
    del outs
    print(f"\n[B] single sync at end, N={N}:")
    print(f"    total wall:    {B_total:8.1f} ms")
    print(f"    per-call:      {B_per_call:8.3f} ms  (= wall / N, ground truth)")

    # === [C] per-iter forced readback ===
    # Use sum() to reduce to scalar, then item() for transfer (avoids
    # the .flatten()[0].item() pattern that previously failed to compile).
    readback_samples = []
    # Re-warmup with the new graph (sum-reduce) so NEFF for it exists.
    for _ in range(3):
        sync()
        out = matmul()
        _ = out.sum().item()
    for _ in range(N):
        sync()
        t0 = time.perf_counter()
        out = matmul()
        _ = out.sum().item()
        t1 = time.perf_counter()
        readback_samples.append((t1 - t0) * 1000)
    C_mean = sum(readback_samples) / N
    print(f"\n[C] per-iter readback (sum().item()), N={N}:")
    print(f"    mean per-call: {C_mean:8.3f} ms")
    print(f"    (note: NEFF includes the sum reduction, so slightly more work")
    print(f"     than [A]/[B]; if sync works, [A] ≈ [B]/N anyway.)")

    # === Verdict ===
    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  [A] per-iter sync     : {A_mean:8.3f} ms / call")
    print(f"  [B] single-sync wall/N: {B_per_call:8.3f} ms / call  (ground truth)")
    print(f"  [C] forced readback   : {C_mean:8.3f} ms / call")
    print(f"{'=' * 60}")
    ratio_A_to_B = A_mean / max(B_per_call, 1e-6)
    if ratio_A_to_B > 0.7:
        print(f"VERDICT: [A] / [B/N] = {ratio_A_to_B:.2f}")
        print(f"         sync() truly waits per call. profile_neuron is OK.")
        print(f"         (Roofline anomalies come from elsewhere.)")
    else:
        print(f"VERDICT: [A] / [B/N] = {ratio_A_to_B:.2f}  ← per-iter sync"
              f" {1/ratio_A_to_B:.0f}x faster than ground truth")
        print(f"         sync() does NOT wait per call.")
        print(f"         profile_neuron measurements are dispatch overhead,")
        print(f"         not real device time. Need NTFF-based timing.")


if __name__ == "__main__":
    main()
