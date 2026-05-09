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

    # === Test 1: single matmul, sync time ===
    sync()
    t0 = time.perf_counter()
    out = matmul()
    sync()
    t1 = time.perf_counter()
    single_ms = (t1 - t0) * 1000
    del out
    print(f"\n[1] sync()  single matmul:        {single_ms:8.3f} ms")

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

    # === Test 3: single matmul + .cpu() readback (ground truth) ===
    # .cpu() forces a real device→host transfer, which absolutely
    # cannot complete before the matmul is done.
    sync()
    t0 = time.perf_counter()
    out = matmul()
    result = out.cpu()
    t1 = time.perf_counter()
    cpu_ms = (t1 - t0) * 1000
    print(f"\n[3] .cpu()  single matmul:        {cpu_ms:8.3f} ms"
          f"  (result[0,0]={result[0,0].item():+.4f})")

    # === Verdict ===
    print(f"\n{'=' * 50}")
    if abs(single_ms - cpu_ms) / max(cpu_ms, 0.01) < 0.30:
        print(f"VERDICT: sync() ≈ .cpu() (within 30%) → sync() truly waits")
        print(f"         Profile timings ARE real device latency.")
    else:
        print(f"VERDICT: sync()={single_ms:.2f}ms vs .cpu()={cpu_ms:.2f}ms"
              f"  ({cpu_ms/max(single_ms,0.01):.1f}x diff)")
        print(f"         sync() does NOT actually wait for device.")
        print(f"         All previous profile timings are dispatch-only,"
              f" not real device time.")


if __name__ == "__main__":
    main()
