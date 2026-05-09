"""Verify torch_xla.sync() / xm.wait_device_ops() actually wait for
NeuronCore execution to complete (vs only flushing dispatch).

If sync truly waits:
  * 1 matmul ≈ 1×T
  * N matmuls in a single sync window ≈ N×T  (N times longer)
  * .cpu() readback ≈ 1×T  (forces real device→host completion)

If sync only dispatches (the queue keeps draining async):
  * 1 matmul ≈ tiny (dispatch only)
  * N matmuls ≈ tiny (just N dispatches)
  * .cpu() readback ≈ N×T  (ground truth, since CPU read forces wait)

Run on inf2.xlarge:
    python scripts/verify_sync.py
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

    # 8192^3 × 2 ≈ 1.1 TFLOPs per matmul.
    # At Inf2 95 TFLOPS/core: 11.6 ms/matmul (single core)
    # At Inf2 190 TFLOPS dense: 5.8 ms/matmul (both cores)
    # If sync is broken we'd see microsecond-level numbers.
    N_LARGE = 8192
    A = torch.randn(N_LARGE, N_LARGE, dtype=torch.bfloat16).to(device)
    B = torch.randn(N_LARGE, N_LARGE, dtype=torch.bfloat16).to(device)

    def matmul():
        return A @ B

    # Warm up to compile NEFF
    sync()
    print("Compiling NEFF (first call)...")
    t0 = time.perf_counter()
    out = matmul()
    sync()
    print(f"  first call (compile): {(time.perf_counter()-t0)*1000:.1f} ms")
    del out

    # 2 more warmup
    for _ in range(2):
        sync()
        out = matmul()
        sync()
        del out

    # === Test 0: NO sync — pure dispatch latency ===
    # If our sync()-bracketed measurement equals this, sync is broken
    # (we'd just be measuring dispatch time both ways).
    sync()
    t0 = time.perf_counter()
    out = matmul()
    t1 = time.perf_counter()
    nosync_ms = (t1 - t0) * 1000
    sync()                          # drain so it doesn't pollute next test
    del out
    print(f"\n[0] NO-sync single matmul:        {nosync_ms:8.3f} ms"
          f"  (pure dispatch baseline)")

    # === Test 1: single matmul, sync time ===
    # This mirrors the exact pattern profile_neuron.py uses (l447-461).
    sync()
    t0 = time.perf_counter()
    out = matmul()
    sync()
    t1 = time.perf_counter()
    single_ms = (t1 - t0) * 1000
    del out
    print(f"[1] sync()  single matmul:        {single_ms:8.3f} ms"
          f"  (this is what profile_neuron.py records)")

    # === Test 2: N matmuls in one sync window ===
    for N in (2, 5, 10):
        sync()
        t0 = time.perf_counter()
        outs = []
        for _ in range(N):
            outs.append(matmul())
        sync()
        t1 = time.perf_counter()
        batch_ms = (t1 - t0) * 1000
        del outs
        print(f"[2] sync()  {N:>2} matmuls:       "
              f"{batch_ms:8.3f} ms total / {batch_ms/N:6.3f} ms each")

    # === Test 3: single matmul + 1-element readback (ground truth) ===
    # Reading just out[0,0] forces the full matmul to complete (the
    # element can't exist before the kernel is done) but transfers
    # only 2 bytes — DMA overhead is negligible (vs ~5 ms for a full
    # 134 MB .cpu() of an 8192² bf16 tensor). So this isolates
    # device compute time without transfer pollution.
    sync()
    t0 = time.perf_counter()
    out = matmul()
    val = out.flatten()[0].item()   # 2 B transfer, forced wait
    t1 = time.perf_counter()
    cpu_ms = (t1 - t0) * 1000
    print(f"\n[3] item() single matmul:        {cpu_ms:8.3f} ms"
          f"  (out[0]={val:+.4f}, 2-byte readback)")

    # === Verdict ===
    # 3-way decision tree:
    #   [0] no-sync  ≈ pure dispatch / kernel-launch overhead
    #   [1] sync()   ≈ what profile_neuron.py records
    #   [3] .cpu()   ≈ ground truth (forced device→host = real wait)
    print(f"\n{'=' * 60}")
    print(f"  [0] NO-sync : {nosync_ms:8.3f} ms  (dispatch only)")
    print(f"  [1] sync()  : {single_ms:8.3f} ms  (our profiler pattern)")
    print(f"  [3] item()  : {cpu_ms:8.3f} ms  (ground truth, no DMA pollution)")
    print(f"{'=' * 60}")
    if abs(single_ms - cpu_ms) / max(cpu_ms, 0.01) < 0.30:
        print(f"VERDICT: sync() ≈ .cpu() → sync() truly waits.")
        print(f"         profile_neuron.py timings ARE real device latency.")
        print(f"         Roofline anomalies have a different root cause.")
    elif abs(single_ms - nosync_ms) / max(nosync_ms, 0.01) < 0.30:
        print(f"VERDICT: sync() ≈ NO-sync → sync() does NOT wait.")
        print(f"         All profile_neuron.py timings are dispatch overhead,")
        print(f"         not real device time. Sweep results are garbage.")
        print(f"         Fix: use out.cpu() / out.sum().item() to force wait.")
    else:
        print(f"VERDICT: ambiguous. sync() partially waits.")
        print(f"         Need to investigate further — possibly NEFF-specific.")


if __name__ == "__main__":
    main()
