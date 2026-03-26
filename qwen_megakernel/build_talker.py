"""JIT compilation of the megakernel CUDA extension for the Qwen3-TTS talker model.

Real talker dimensions from Qwen/Qwen3-TTS-12Hz-1.7B-Base:
  hidden_size=2048, intermediate_size=6144, vocab_size=3072,
  28 layers, 16 Q heads, 8 KV heads, head_dim=128.
"""

import os
from torch.utils.cpp_extension import load

_module = None
_DIR = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_DIR, "../csrc")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


# Real talker dimensions
TALKER_HIDDEN_SIZE = 2048
TALKER_INTERMEDIATE_SIZE = 6144
TALKER_VOCAB_SIZE = 3072

# LM head tuning for vocab=3072:
# With LDG_LM_BLOCK_SIZE=256 (8 warps) and LDG_LM_ROWS_PER_WARP=2,
# each block covers 16 rows. ceil(3072/16) = 192 blocks.
# Use 32 blocks for a reasonable balance.
TALKER_LM_NUM_BLOCKS = _env_int("LDG_TALKER_LM_NUM_BLOCKS", 32)
TALKER_LM_BLOCK_SIZE = _env_int("LDG_TALKER_LM_BLOCK_SIZE", 256)
TALKER_LM_ROWS_PER_WARP = _env_int("LDG_TALKER_LM_ROWS_PER_WARP", 2)

KERNEL_FLAGS = [
    f"-DLDG_HIDDEN_SIZE={TALKER_HIDDEN_SIZE}",
    f"-DLDG_INTERMEDIATE_SIZE={TALKER_INTERMEDIATE_SIZE}",
    f"-DLDG_NUM_BLOCKS={_env_int('LDG_NUM_BLOCKS', 128)}",
    f"-DLDG_BLOCK_SIZE={_env_int('LDG_BLOCK_SIZE', 512)}",
    f"-DLDG_LM_NUM_BLOCKS={TALKER_LM_NUM_BLOCKS}",
    f"-DLDG_LM_BLOCK_SIZE={TALKER_LM_BLOCK_SIZE}",
    f"-DLDG_LM_ROWS_PER_WARP={TALKER_LM_ROWS_PER_WARP}",
    f"-DLDG_VOCAB_SIZE={TALKER_VOCAB_SIZE}",
    f"-DLDG_ATTN_BLOCKS={_env_int('LDG_ATTN_BLOCKS', 8)}",
    f"-DLDG_PREFETCH_QK={_env_int('LDG_PREFETCH_QK', 0)}",
    f"-DLDG_PREFETCH_THREAD_STRIDE={_env_int('LDG_PREFETCH_THREAD_STRIDE', 10)}",
    f"-DLDG_PREFETCH_DOWN={_env_int('LDG_PREFETCH_DOWN', 1)}",
    f"-DLDG_PREFETCH_ELEM_STRIDE={_env_int('LDG_PREFETCH_ELEM_STRIDE', 1)}",
    f"-DLDG_PREFETCH_BLOCK_STRIDE={_env_int('LDG_PREFETCH_BLOCK_STRIDE', 1)}",
    f"-DLDG_PREFETCH_GATE={_env_int('LDG_PREFETCH_GATE', 1)}",
    f"-DLDG_PREFETCH_UP={_env_int('LDG_PREFETCH_UP', 1)}",
    "-DLDG_USE_UINT4",
    "-DLDG_ATTENTION_VEC4",
    "-DLDG_WEIGHT_LDCS",
    "-DLDG_MLP_SMEM",
]

CUDA_FLAGS = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "-arch=sm_120a",
    f"-I{_CSRC}",
] + KERNEL_FLAGS


def get_extension():
    """Build (or return cached) the talker megakernel extension.

    Registers as torch.ops.qwen_talker_megakernel_C.* (separate namespace
    from the original qwen_megakernel_C).
    """
    global _module
    if _module is not None:
        return _module

    _module = load(
        name="qwen_talker_megakernel_C",
        sources=[
            os.path.join(_CSRC, "torch_bindings.cpp"),
            os.path.join(_CSRC, "kernel.cu"),
        ],
        extra_cuda_cflags=CUDA_FLAGS,
        extra_cflags=[f"-I{_CSRC}"],
        verbose=False,
    )
    return _module
