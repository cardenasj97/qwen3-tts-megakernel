# Improvements Found During Integration

## Blackwell Scheduler Bug

This was the biggest issue we hit. On RTX 5090 (sm_120), launching the same persistent-barrier kernel 3+ consecutive times causes a block scheduling failure — some of the 128 thread blocks never get dispatched, the `AtomicGridSync` barrier deadlocks, and the GPU spins at 100%.

In the original text-LLM use case this doesn't come up — `generate()` runs one long decode loop. But in the TTS pipeline, each decode step has other GPU work (code predictor, embedding construction) between megakernel launches, creating rapid same-kernel launches that trigger the bug.

### How we narrowed it down

We went through 8 hypotheses before finding the real cause:

1. Stream mismatch — ruled out, `torch.cuda.synchronize()` was already between all ops
2. Shared memory exhaustion — ruled out, 34KB used vs 48KB limit
3. `__syncthreads` divergence at tile boundaries — ruled out, works at all positions in isolation
4. Missing `cudaFuncSetAttribute` for the `from_embeds` kernel — ruled out
5. Barrier reset race (`cudaMemsetAsync`) — ruled out, synchronous memset didn't help
6. Cooperative launch needed — ruled out, `cudaLaunchCooperativeKernel` didn't fix it
7. Bug in the `_decode_from_embeds` C++ binding — ruled out, swapping to use `_decode` binary still hangs
8. **Repeated same-kernel launches** — confirmed, the scheduler fails on the 3rd launch of any kernel function

The breakthrough was using `cudaMallocManaged` to observe from the CPU that the kernel's first instruction never executes on the 3rd launch. That pointed directly at a scheduler/dispatch issue rather than a kernel code bug.

### Evidence

| Test | Result |
|------|--------|
| `step()` (kernel A) at position 17 | Works |
| `step_from_embeds()` (kernel B) at position 17 | Hangs on 3rd call |
| Swap kernel B to use kernel A's binary | Still hangs on 3rd call |
| Use kernel A for ALL steps | Hangs on 3rd call |
| **Alternate A/B on each step** | **Works** |

The pattern is clear: same kernel 3x = hang, alternating kernels = fine.

### The fix

```python
# model_talker.py
self._step_toggle = not self._step_toggle
if self._step_toggle:
    _decode_from_embeds(...)                      # kernel binary A
else:
    self._embed_weight[0].copy_(input_embeds)     # copy embedding to row 0
    _decode(... token_id=0 ...)                   # kernel binary B
```

Both paths run the same transformer logic and produce identical output. The only difference is the kernel binary identity, which keeps the scheduler from entering whatever stale state causes the hang.

Overhead is one `tensor.copy_()` (4 KB) every other step — around 0.001ms, negligible vs the 2ms kernel.

### Impact

| | Before | After |
|---|---|---|
| Decode throughput | 0 tok/s (infinite hang) | ~495 tok/s |
| Pipeline status | Broken on 3rd decode step | Stable across hundreds of turns |
| Workaround overhead | — | ~0.001ms per alternate step (4 KB copy) |

### Environment

- RTX 5090 (sm_120, 170 SMs), CUDA 13.0, Driver 580.82
- 128 persistent blocks x 512 threads, `AtomicGridSync` requiring all blocks co-resident
- Does not reproduce on Ampere (sm_80) or Hopper (sm_90)
- Not yet reported to NVIDIA — may be related to persistent kernel residency tracking in the Blackwell architecture

## Parametric Dimensions and `decode_from_embeds`

These are covered in detail in [kernel-adaptation.md](kernel-adaptation.md) — the short version:

- **Parametric dimensions:** Added `#ifndef` guards so the kernel compiles for any Qwen3-family model size via `-D` build flags. No source edits needed to retarget.
- **`decode_from_embeds`:** New kernel entry point accepting a pre-computed bf16 embedding vector. The TTS talker builds composite embeddings from multiple codebooks + text injection, which can't be expressed as a single table lookup. This entry point also enables the Blackwell workaround — we need two distinct kernel functions to alternate between.
