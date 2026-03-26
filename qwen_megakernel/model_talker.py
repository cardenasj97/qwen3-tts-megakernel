"""Weight loading and high-level decode API for the Qwen3-TTS talker backbone.

Real talker dimensions from Qwen/Qwen3-TTS-12Hz-1.7B-Base:
  - 28 transformer layers
  - hidden_size 2048
  - intermediate_size 6144
  - vocab_size 3072 (codec tokens)
  - 16 Q heads, 8 KV heads, head_dim 128
  - rope_theta 1,000,000

The megakernel handles the transformer backbone only. The codec_head linear
projection and code predictor are handled externally by the adapter.
"""

import math
import struct

import torch

# --- Real talker model constants ---
NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 6144
Q_SIZE = 16 * HEAD_DIM   # 2048
KV_SIZE = 8 * HEAD_DIM   # 1024
MAX_SEQ_LEN = 2048
VOCAB_SIZE = 3072
ROPE_THETA = 1_000_000.0

# Will be populated by build_talker.get_extension()
_decode = None
_decode_from_embeds = None


def _ensure_ops():
    """Lazily load the talker megakernel extension."""
    global _decode, _decode_from_embeds
    if _decode is not None:
        return
    from qwen_megakernel.build_talker import get_extension
    get_extension()
    _decode = torch.ops.qwen_talker_megakernel_C.decode
    _decode_from_embeds = torch.ops.qwen_talker_megakernel_C.decode_from_embeds


def _pack_layer_weights(layer_weights: list[torch.Tensor]) -> torch.Tensor:
    """Pack 11-tensor-per-layer flat list into a device blob of LDGLayerWeights structs."""
    ptr_size = 8   # 64-bit pointers
    n_ptrs = 11
    struct_bytes = n_ptrs * ptr_size
    buf = bytearray(NUM_LAYERS * struct_bytes)
    for i in range(NUM_LAYERS):
        for j in range(n_ptrs):
            ptr = layer_weights[i * n_ptrs + j].data_ptr()
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
    t = torch.frombuffer(buf, dtype=torch.uint8).cuda()
    return t


class TalkerDecoder:
    """Stateful decoder wrapping the megakernel for the Qwen3-TTS talker backbone.

    Accepts pre-extracted weight tensors matching the real talker architecture.
    """

    def __init__(
        self,
        embed_weight: torch.Tensor,
        layer_weights: list[torch.Tensor],
        final_norm_weight: torch.Tensor,
        lm_head_weight: torch.Tensor,
        cos_table: torch.Tensor | None = None,
        sin_table: torch.Tensor | None = None,
    ):
        """
        Args:
            embed_weight: (VOCAB_SIZE, HIDDEN_SIZE) = (3072, 2048) bf16
            layer_weights: flat list of 28*11 = 308 tensors, same order as Qwen3-0.6B:
                per layer: input_layernorm, q_proj, k_proj, v_proj, q_norm, k_norm,
                           o_proj, post_attn_layernorm, gate_proj, up_proj, down_proj
            final_norm_weight: (HIDDEN_SIZE,) = (2048,) bf16
            lm_head_weight: (VOCAB_SIZE, HIDDEN_SIZE) = (3072, 2048) bf16 — the codec_head weight
            cos_table: optional precomputed RoPE cos table
            sin_table: optional precomputed RoPE sin table
        """
        _ensure_ops()
        self._position = 0

        # Store weight references to prevent GC
        self._embed_weight = embed_weight
        self._layer_weights_list = layer_weights
        self._final_norm_weight = final_norm_weight
        self._lm_head_weight = lm_head_weight

        # RoPE tables — using rope_theta=1,000,000
        if cos_table is not None and sin_table is not None:
            self._cos_table = cos_table
            self._sin_table = sin_table
        else:
            inv_freq = 1.0 / (
                ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
            )
            positions = torch.arange(MAX_SEQ_LEN, dtype=torch.float32)
            freqs = torch.outer(positions, inv_freq)
            self._cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
            self._sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()

        self._layer_weights_packed = _pack_layer_weights(layer_weights)
        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)

        # KV cache — 28 layers
        self._k_cache = torch.zeros(
            NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda",
        )
        self._v_cache = torch.zeros_like(self._k_cache)

        # Scratch buffers (single-token decode) — real talker dimensions
        f32 = dict(dtype=torch.float32, device="cuda")
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        self._hidden = torch.empty(HIDDEN_SIZE, **bf16)
        self._act = torch.empty(HIDDEN_SIZE, **f32)
        self._res = torch.empty(HIDDEN_SIZE, **f32)
        self._q = torch.empty(Q_SIZE, **f32)
        self._k = torch.empty(KV_SIZE, **f32)
        self._v = torch.empty(KV_SIZE, **f32)
        self._attn_out = torch.empty(Q_SIZE, **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._norm_out = torch.empty(HIDDEN_SIZE, **f32)
        # LM head scratch — sized for the talker LM block count
        from qwen_megakernel.build_talker import TALKER_LM_NUM_BLOCKS
        self._bmax_vals = torch.empty(TALKER_LM_NUM_BLOCKS, **f32)
        self._bmax_idxs = torch.empty(TALKER_LM_NUM_BLOCKS, dtype=torch.int32, device="cuda")
        self._out_token = torch.empty(1, dtype=torch.int32, device="cuda")

    def step(self, token_id: int) -> int:
        """Decode one token through the talker backbone + codec_head argmax.
        Returns the next token id (first-codebook token).
        """
        _decode(
            self._out_token,
            token_id,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            NUM_LAYERS,
            self._position,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += 1
        return self._out_token.item()

    def step_from_embeds(self, input_embeds: torch.Tensor) -> int:
        """Decode one step from a pre-computed embedding vector.

        Accepts a pre-constructed bf16 embedding of shape (HIDDEN_SIZE,) = (2048,).
        Used by the TTS talker adapter where the input is a composite of
        codec embeddings + text injection.

        Returns the next token id (first-codebook token via argmax).
        """
        assert input_embeds.shape == (HIDDEN_SIZE,), \
            f"Expected ({HIDDEN_SIZE},), got {input_embeds.shape}"
        assert input_embeds.dtype == torch.bfloat16, \
            f"Expected bfloat16, got {input_embeds.dtype}"

        # WORKAROUND: alternate between _decode_from_embeds and _decode
        # to avoid Blackwell driver bug with 3+ consecutive same-kernel launches.
        if not hasattr(self, "_step_toggle"):
            self._step_toggle = False
        self._step_toggle = not self._step_toggle
        
        if self._step_toggle:
            # Use _decode_from_embeds (original path)
            _decode_from_embeds(
                self._out_token,
                input_embeds.contiguous(),
                self._layer_weights_packed,
                self._final_norm_weight,
                self._lm_head_weight,
                self._cos_table,
                self._sin_table,
                self._k_cache,
                self._v_cache,
                self._hidden,
                self._act,
                self._res,
                self._q,
                self._k,
                self._v,
                self._attn_out,
                self._mlp_inter,
                self._norm_out,
                self._bmax_vals,
                self._bmax_idxs,
                NUM_LAYERS,
                self._position,
                MAX_SEQ_LEN,
                self._attn_scale,
            )
        else:
            # Use _decode with embedding copied to row 0
            self._embed_weight[0].copy_(input_embeds)
            _decode(
                self._out_token,
                0,
                self._embed_weight,
                self._layer_weights_packed,
                self._final_norm_weight,
                self._lm_head_weight,
                self._cos_table,
                self._sin_table,
                self._k_cache,
                self._v_cache,
                self._hidden,
                self._act,
                self._res,
                self._q,
                self._k,
                self._v,
                self._attn_out,
                self._mlp_inter,
                self._norm_out,
                self._bmax_vals,
                self._bmax_idxs,
                NUM_LAYERS,
                self._position,
                MAX_SEQ_LEN,
                self._attn_scale,
            )
        self._position += 1
        return self._out_token.item()

    def get_hidden_state(self) -> torch.Tensor:
        """Return the current normalized hidden state (after final RMSNorm).

        Shape: (HIDDEN_SIZE,) = (2048,) float32.
        """
        return self._norm_out.clone()

    def reset(self):
        """Reset KV cache and position counter."""
        import torch
        torch.cuda.synchronize()  # wait for any in-flight kernel
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()
        self._step_toggle = False  # reset alternating kernel toggle

    @property
    def position(self) -> int:
        return self._position
