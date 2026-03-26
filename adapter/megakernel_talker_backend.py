"""MegakernelTalkerBackend — adapter that replaces HF generate() for the talker decode loop.

Prefill: uses the original HF talker model (PyTorch path).
Decode:  uses the CUDA megakernel for the transformer backbone via decode_from_embeds.
         Embedding construction and code predictor still use HF model components.
"""

import logging
import time
from dataclasses import dataclass
from typing import Generator

import torch

_log = logging.getLogger(__name__)


@dataclass
class PrefillResult:
    past_hidden: torch.Tensor       # (batch, 1, hidden_dim)
    first_token_id: int             # first-codebook token from prefill logits
    hf_past_key_values: object      # HF DynamicCache


@dataclass
class DecodeStepResult:
    next_token_id: int              # first-codebook token
    codec_frame: torch.Tensor       # (num_code_groups,) full codec frame
    past_hidden: torch.Tensor       # (batch, 1, hidden_dim)
    generation_step: int            # incremented


@dataclass
class CodecFrame:
    step_index: int
    first_codebook_token: int
    codec_frame: torch.Tensor       # (num_code_groups,)
    timestamp: float
    latency_ms: float


class MegakernelTalkerBackend:
    """Adapter that keeps Qwen3-TTS's outer orchestration but replaces the
    talker decode backend with the CUDA megakernel.

    Embedding construction and code predictor use HF model components.
    The transformer backbone + codec_head argmax use the megakernel.
    """

    def __init__(self, talker_model, talker_config):
        self.talker_model = talker_model
        self.config = talker_config
        self.device = next(talker_model.parameters()).device

        self._talker_decoder = self._build_talker_decoder()

    def _build_talker_decoder(self):
        """Extract weights from the HF talker model and create a TalkerDecoder."""
        from qwen_megakernel.model_talker import TalkerDecoder

        model = self.talker_model.model  # Qwen3TTSTalkerModel

        embed_weight = model.codec_embedding.weight.data.contiguous()
        final_norm_weight = model.norm.weight.data.contiguous()
        lm_head_weight = self.talker_model.codec_head.weight.data.contiguous()

        layer_weights = []
        for layer in model.layers:
            attn = layer.self_attn
            mlp = layer.mlp
            layer_weights.extend([
                layer.input_layernorm.weight.data.contiguous(),
                attn.q_proj.weight.data.contiguous(),
                attn.k_proj.weight.data.contiguous(),
                attn.v_proj.weight.data.contiguous(),
                attn.q_norm.weight.data.contiguous(),
                attn.k_norm.weight.data.contiguous(),
                attn.o_proj.weight.data.contiguous(),
                layer.post_attention_layernorm.weight.data.contiguous(),
                mlp.gate_proj.weight.data.contiguous(),
                mlp.up_proj.weight.data.contiguous(),
                mlp.down_proj.weight.data.contiguous(),
            ])

        return TalkerDecoder(
            embed_weight=embed_weight,
            layer_weights=layer_weights,
            final_norm_weight=final_norm_weight,
            lm_head_weight=lm_head_weight,
        )

    @torch.no_grad()
    def prefill(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        trailing_text_hidden: torch.Tensor,
        tts_pad_embed: torch.Tensor,
    ) -> PrefillResult:
        """Run prefill using the HF talker model (PyTorch path)."""
        result = self.talker_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            generation_step=-1,
        )

        past_hidden = result.past_hidden
        logits = result.logits
        first_token_id = logits[:, -1, :].argmax(dim=-1).item()
        hf_past_kv = result.past_key_values

        self._transfer_kv_cache(hf_past_kv, inputs_embeds.shape[1])

        return PrefillResult(
            past_hidden=past_hidden,
            first_token_id=first_token_id,
            hf_past_key_values=hf_past_kv,
        )

    def _transfer_kv_cache(self, hf_past_kv, seq_len: int):
        """Transfer HF DynamicCache into the megakernel's flat KV cache tensors."""
        from qwen_megakernel.model_talker import NUM_LAYERS

        self._talker_decoder.reset()

        for layer_idx in range(NUM_LAYERS):
            k, v = hf_past_kv[layer_idx]
            self._talker_decoder._k_cache[layer_idx, :, :seq_len, :] = \
                k[0, :, :seq_len, :].to(torch.bfloat16)
            self._talker_decoder._v_cache[layer_idx, :, :seq_len, :] = \
                v[0, :, :seq_len, :].to(torch.bfloat16)

        self._talker_decoder._position = seq_len

    @torch.no_grad()
    def decode_step(
        self,
        last_token_id: int,
        generation_step: int,
        trailing_text_hidden: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        past_hidden: torch.Tensor,
        subtalker_dosample: bool = True,
        subtalker_top_p: float = 1.0,
        subtalker_top_k: int = 50,
        subtalker_temperature: float = 0.9,
    ) -> DecodeStepResult:
        """Run one decode step: HF embedding construction → megakernel backbone.

        Mimics Qwen3TTSTalkerForConditionalGeneration.forward() decode branch:
        1. Embed last first-codebook token via codec_embedding
        2. Call code predictor for codebooks 2..N
        3. Sum all codec embeddings
        4. Inject trailing_text_hidden or tts_pad_embed
        5. Run megakernel backbone (decode_from_embeds)
        6. Get hidden state → codec_head → argmax → next token
        """
        from qwen_megakernel.model_talker import HIDDEN_SIZE

        t0 = time.time()
        _log.debug(f"decode_step gs={generation_step} token={last_token_id}: embedding...")

        # --- Step 1: Embed last first-codebook token ---
        last_id_hidden = self.talker_model.get_input_embeddings()(
            torch.tensor([[last_token_id]], device=self.device)
        )  # (1, 1, hidden)
        torch.cuda.synchronize()
        _log.debug(f"decode_step gs={generation_step}: embed done ({time.time()-t0:.3f}s), running code_predictor...")

        # --- Step 2: Code predictor for codebooks 2..N ---
        predictor_result = self.talker_model.code_predictor.generate(
            inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1),
            max_new_tokens=self.config.num_code_groups - 1,
            do_sample=subtalker_dosample,
            top_p=subtalker_top_p,
            top_k=subtalker_top_k,
            temperature=subtalker_temperature,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        torch.cuda.synchronize()
        _log.debug(f"decode_step gs={generation_step}: code_predictor done ({time.time()-t0:.3f}s), building embeds...")

        # Build full codec frame: first codebook token + predicted tokens
        input_ids = torch.tensor([[last_token_id]], device=self.device)
        codec_ids = torch.cat((input_ids, predictor_result.sequences), dim=-1)

        # --- Step 3: Sum all codec embeddings ---
        codec_hiddens = torch.cat(
            [last_id_hidden]
            + [
                self.talker_model.code_predictor.get_input_embeddings()[i](
                    predictor_result.sequences[..., i:i+1]
                )
                for i in range(self.config.num_code_groups - 1)
            ],
            dim=1,
        )
        inputs_embeds = codec_hiddens.sum(1, keepdim=True)  # (1, 1, hidden)

        # --- Step 4: Inject streaming text ---
        if generation_step < trailing_text_hidden.shape[1]:
            inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step].unsqueeze(1)
        else:
            inputs_embeds = inputs_embeds + tts_pad_embed

        _log.debug(f"decode_step gs={generation_step}: running megakernel ({time.time()-t0:.3f}s), position={self._talker_decoder.position}...")

        # --- Step 5: Megakernel backbone via decode_from_embeds ---
        # Flatten (1, 1, hidden) → (hidden,) bf16
        embed_flat = inputs_embeds.squeeze(0).squeeze(0).to(torch.bfloat16).contiguous()
        has_nan = embed_flat.isnan().any().item()
        has_inf = embed_flat.isinf().any().item()
        _log.debug(f"decode_step gs={generation_step}: embed stats: nan={has_nan} inf={has_inf} norm={embed_flat.float().norm().item():.4f}")
        next_token_id = self._talker_decoder.step_from_embeds(embed_flat)
        torch.cuda.synchronize()
        _log.debug(f"decode_step gs={generation_step}: megakernel done, next_token={next_token_id} ({time.time()-t0:.3f}s)")

        # --- Step 6: Get hidden state for next step's code predictor ---
        # The megakernel's _norm_out is float32 (HIDDEN_SIZE,) — same as
        # hidden_states[:, -1:, :] which feeds past_hidden in the HF path.
        norm_out = self._talker_decoder.get_hidden_state()  # (HIDDEN_SIZE,) f32
        new_past_hidden = norm_out.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)  # (1, 1, hidden) bf16

        return DecodeStepResult(
            next_token_id=next_token_id,
            codec_frame=codec_ids.squeeze(0),
            past_hidden=new_past_hidden,
            generation_step=generation_step + 1,
        )

    @torch.no_grad()
    def stream_decode(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        trailing_text_hidden: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        max_new_tokens: int = 4096,
        eos_token_id: int | None = None,
        **kwargs,
    ) -> Generator[CodecFrame, None, None]:
        """Python generator: prefill once, then yield one codec frame per step."""
        if eos_token_id is None:
            eos_token_id = self.config.codec_eos_token_id

        prefill_result = self.prefill(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
        )

        last_token_id = prefill_result.first_token_id
        past_hidden = prefill_result.past_hidden
        generation_step = 0

        for step in range(max_new_tokens):
            t0 = time.time()

            result = self.decode_step(
                last_token_id=last_token_id,
                generation_step=generation_step,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                past_hidden=past_hidden,
                **kwargs,
            )

            t1 = time.time()

            yield CodecFrame(
                step_index=step,
                first_codebook_token=result.next_token_id,
                codec_frame=result.codec_frame,
                timestamp=t1,
                latency_ms=(t1 - t0) * 1000,
            )

            if result.next_token_id == eos_token_id:
                break

            last_token_id = result.next_token_id
            past_hidden = result.past_hidden
            generation_step = result.generation_step

    def reset(self):
        """Reset megakernel KV cache and position."""
        self._talker_decoder.reset()
