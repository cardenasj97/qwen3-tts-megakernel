# Kernel Adaptation: 0.6B → 1.7B Talker

The dimension differences are in the main [README](../README.md#kernel-adaptation-06b--17b-talker). This doc covers the implementation details.

## Compile-Time Parametric Dimensions

The original kernel hardcodes dimensions:

```c
constexpr int HIDDEN_SIZE = 1024;
constexpr int INTERMEDIATE_SIZE = 3072;
constexpr int VOCAB_SIZE = 2048;
```

We wrapped these in `#ifndef` guards:

```c
#ifndef LDG_HIDDEN_SIZE
#define LDG_HIDDEN_SIZE 1024
#endif
constexpr int HIDDEN_SIZE = LDG_HIDDEN_SIZE;
```

`build_talker.py` passes the talker values at compile time:

```python
KERNEL_FLAGS = [
    "-DLDG_HIDDEN_SIZE=2048",
    "-DLDG_INTERMEDIATE_SIZE=6144",
    "-DLDG_VOCAB_SIZE=3072",
    "-DLDG_NUM_BLOCKS=128",
    "-DLDG_BLOCK_SIZE=512",
    "-DLDG_LM_NUM_BLOCKS=32",
    "-DLDG_LM_BLOCK_SIZE=256",
    "-DLDG_LM_ROWS_PER_WARP=2",
    "-DLDG_ATTN_BLOCKS=8",
    # ... optimization flags unchanged
]
```

All loop bounds, matvec widths, and reductions are driven by `constexpr` values, so the kernel logic didn't need any changes.

## `decode_from_embeds` Entry Point

The original kernel takes a token ID and looks it up:

```c
__global__ void ldg_decode_kernel_direct(
    int input_token_id,
    const __nv_bfloat16 *embed_weight,
    ...
) {
    const __nv_bfloat16 *embed_row = embed_weight + input_token_id * HIDDEN_SIZE;
    // ... transformer layers
}
```

The TTS talker builds a composite embedding on the host side:

```python
embedding = sum(codec_embed(token_i) for i in range(16)) + text_hidden[step]
```

This can't go through a table lookup, so we added a second entry point:

```c
__global__ void ldg_decode_kernel_from_embeds(
    const __nv_bfloat16 *input_embeds,  // pre-computed (HIDDEN_SIZE,) bf16
    const LDGLayerWeights *layer_weights,
    ...
) {
    // input_embeds IS the embedding — everything after this is identical
}
```

## Torch Binding

New C++ wrapper in `torch_bindings.cpp`:

```cpp
extern "C" void launch_ldg_decode_from_embeds(
    const void *input_embeds, int *output_token_id,
    const LDGLayerWeights *layer_weights, ...
    cudaStream_t stream);

void decode_from_embeds(torch::Tensor output_token, torch::Tensor input_embeds, ...) {
    launch_ldg_decode_from_embeds(
        input_embeds.data_ptr(),
        output_token.data_ptr<int>(),
        ...
        c10::cuda::getCurrentCUDAStream().stream());
}
```

Registered under `qwen_talker_megakernel_C` (separate namespace from the original `qwen_megakernel_C`).

## LM Head Block Count

The LM head argmax kernel covers `VOCAB_SIZE` rows. With `LDG_LM_BLOCK_SIZE=256` and `LDG_LM_ROWS_PER_WARP=2`, each block handles 16 rows. For vocab=3072 that's 192 candidate blocks — we use 32 for a reasonable balance.

## Shared Memory

| | Original (hidden=1024) | Talker (hidden=2048) |
|---|---|---|
| Peak smem | ~24 KB | ~40 KB |
| RTX 5090 limit | 96+ KB/block | 96+ KB/block |

The increase comes from RMSNorm and MLP buffers that scale with `HIDDEN_SIZE`. Plenty of headroom.

## Performance

2x `hidden_size` roughly doubles compute per layer, plus 4 extra layers add ~14%. Expected ~2.3x slowdown.

Measured: 1.0 ms/step (0.6B) → 2.03 ms/step (1.7B) = 2.0x slowdown. The kernel stays memory-bandwidth bound on the 5090's 1792 GB/s GDDR7.
