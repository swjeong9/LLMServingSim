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
    # MUST pass wait=True — torch_xla.sync()'s default is wait=False,
    # which only dispatches the lazy graph and returns immediately
    # (this is the bug we're verifying / fixed in profile_neuron.py).
    if hasattr(torch_xla, "sync"):
        torch_xla.sync(wait=True)
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

    N = 1000

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
    A_total = sum(per_iter_samples)
    print(f"\n[A] per-iter sync, N={N}:")
    print(f"    mean per-call: {A_mean:8.3f} ms")
    print(f"    total wall:    {A_total:8.1f} ms")

    # === [B] mark_step per iter, wait_device_ops only at end ===
    # Previous "outs.append() then sync at end" caused lazy XLA to fuse
    # all N matmuls into ONE huge graph (1000 × 134 MB outputs OOMed,
    # 10 × 134 MB graph also tripped a cached failed-compile NEFF).
    #
    # Use the same single-matmul NEFF (cache-hot from [A]) and dispatch
    # it N times via mark_step(), which breaks the graph per call but
    # does NOT wait. Only call wait_device_ops() once at the end.
    #   - if wait_device_ops truly waits: wall = N × real_per_call
    #   - if it doesn't either: wall = N × dispatch_overhead (≈ [A])
    sync()                            # clean start
    t0 = time.perf_counter()
    for _ in range(N):
        out = matmul()
        xm.mark_step()                # break graph, dispatch, NO wait
        del out
    xm.wait_device_ops()              # the single real wait
    t1 = time.perf_counter()
    B_total = (t1 - t0) * 1000
    B_per_call = B_total / N
    print(f"\n[B] mark_step per iter + wait_device_ops at end, N={N}:")
    print(f"    total wall:    {B_total:8.1f} ms")
    print(f"    per-call:      {B_per_call:8.3f} ms  (= wall / N, ground truth)")

    # === Verdict ===
    print(f"\n{'=' * 60}")
    print(f"Summary (N={N}):")
    print(f"  [A] per-iter sync     : {A_mean:8.3f} ms / call  "
          f"(total wall {A_total/1000:.2f} s)")
    print(f"  [B] single-sync wall/N: {B_per_call:8.3f} ms / call  "
          f"(total wall {B_total/1000:.2f} s)  ← ground truth")
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
