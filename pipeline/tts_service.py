"""Pipecat TTSService adapter for the megakernel talker backend.

Wraps MegakernelTalkerBackend so Pipecat can call it as a standard TTS service.
Yields audio chunks incrementally (no full-utterance buffering).
"""

import asyncio
import queue as queue_module
import threading
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import numpy as np
import torch
from loguru import logger

from pipecat.frames.frames import Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService


class CUDAWorkerThread:
    """Dedicated thread for ALL CUDA work.

    Created before the asyncio event loop starts so it is completely
    isolated from asyncio's default thread-pool executor and any
    concurrent Pipecat pipeline activity.

    Usage:
        worker = CUDAWorkerThread()           # before asyncio.run(...)
        result = await worker.run(callable)   # inside async context
    """

    def __init__(self):
        self._queue: queue_module.Queue = queue_module.Queue()
        self._thread = threading.Thread(target=self._loop, name="cuda-worker", daemon=True)
        self._thread.start()
        logger.info("CUDAWorkerThread: started")

    def _loop(self):
        # Pin this thread to cuda:0 — CUDA context is per-process but
        # set_device ensures correct device affinity for all kernels.
        import torch
        torch.cuda.set_device(0)
        while True:
            item = self._queue.get()
            if item is None:
                break
            fn, loop, future = item
            try:
                result = fn()
                loop.call_soon_threadsafe(future.set_result, result)
            except Exception as exc:
                loop.call_soon_threadsafe(future.set_exception, exc)

    async def run(self, fn):
        """Submit fn to the CUDA worker thread and await its result."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._queue.put((fn, loop, future))
        return await future

    def stop(self):
        self._queue.put(None)


@dataclass
class MegakernelTTSSettings(TTSSettings):
    pass


class MegakernelTTSService(TTSService):
    """Pipecat TTS service backed by the CUDA megakernel talker.

    Generates codec frames via MegakernelTalkerBackend, decodes them
    incrementally via the Qwen3-TTS tokenizer decoder, and yields
    raw PCM audio chunks to the Pipecat pipeline.

    All GPU work runs on a dedicated CUDAWorkerThread (created before
    asyncio starts) to avoid CUDA context conflicts with Pipecat's
    internal thread pool.
    """

    Settings = MegakernelTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        tts_model,          # Qwen3TTSModel instance (already loaded)
        backend,            # MegakernelTalkerBackend instance
        cuda_worker: CUDAWorkerThread,  # dedicated pre-asyncio CUDA thread
        chunk_size: int = 12,
        overlap: int = 5,
        max_new_tokens: int = 256,
        sample_rate: Optional[int] = None,
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        default_settings = MegakernelTTSSettings()
        if settings:
            default_settings.update_settings(settings)

        super().__init__(sample_rate=sample_rate or 24000, settings=default_settings, **kwargs)

        self._tts_model = tts_model
        self._backend = backend
        self._cuda_worker = cuda_worker
        self._talker = tts_model.model.talker
        self._tc = self._talker.config
        self._model = tts_model.model
        self._device = self._talker.device
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._max_new_tokens = max_new_tokens

        # Get decoder components
        tokenizer_model = self._model.speech_tokenizer.model
        self._decoder = tokenizer_model.decoder
        self._upsample_rate = tokenizer_model.config.decode_upsample_rate
        self._sr = tokenizer_model.config.output_sample_rate

    def _prepare_inputs(self, text: str):
        """Prepare talker input embeddings from text (voice_design non_streaming path)."""
        talker = self._talker
        tc = self._tc
        model = self._model
        device = self._device

        # Tokenize
        formatted = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_id = self._tts_model.processor(text=formatted, return_tensors="pt")["input_ids"].to(device)
        if input_id.dim() == 1:
            input_id = input_id.unsqueeze(0)

        # TTS special embeds
        tts_special = torch.tensor(
            [[model.config.tts_bos_token_id, model.config.tts_eos_token_id, model.config.tts_pad_token_id]],
            device=device, dtype=input_id.dtype,
        )
        tts_bos, tts_eos, tts_pad = talker.text_projection(
            talker.get_text_embeddings()(tts_special)
        ).chunk(3, dim=1)

        # Codec prefill (language=Auto, no speaker)
        codec_0 = talker.get_input_embeddings()(torch.tensor(
            [[tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id]],
            device=device, dtype=input_id.dtype,
        ))
        codec_1 = talker.get_input_embeddings()(torch.tensor(
            [[tc.codec_pad_id, tc.codec_bos_id]],
            device=device, dtype=input_id.dtype,
        ))
        codec_embed = torch.cat([codec_0, codec_1], dim=1)

        # Role embed
        role = talker.text_projection(talker.get_text_embeddings()(input_id[:, :3]))

        # Non-streaming: full text at once
        prefill_pad = tts_pad.expand(-1, codec_embed.shape[1] - 2, -1)
        prefill_codec = torch.cat([prefill_pad, tts_bos], dim=1) + codec_embed[:, :-1]

        text_tokens = input_id[:, 3:-5]
        text_len = text_tokens.shape[1]
        text_emb = talker.text_projection(talker.get_text_embeddings()(text_tokens))
        text_with_eos = torch.cat([text_emb, tts_eos], dim=1)
        pad_codec = talker.get_input_embeddings()(torch.tensor(
            [[tc.codec_pad_id] * (text_len + 1)], device=device, dtype=input_id.dtype,
        ))
        text_part = text_with_eos + pad_codec
        final_pad = tts_pad + talker.get_input_embeddings()(torch.tensor(
            [[tc.codec_bos_id]], device=device, dtype=input_id.dtype,
        ))

        embeds = torch.cat([role, prefill_codec, text_part, final_pad], dim=1)
        mask = torch.ones(1, embeds.shape[1], dtype=torch.long, device=device)
        trailing = tts_pad  # non-streaming

        return embeds, mask, trailing, tts_pad

    def _decode_codes_chunk(self, codes_t, start, end, overlap_frames):
        """Decode a chunk of codec frames to PCM bytes."""
        ctx_start = max(0, start - overlap_frames)
        chunk = codes_t[..., ctx_start:end]
        actual_overlap = start - ctx_start
        with torch.no_grad():
            wav = self._decoder(chunk)
        trim = actual_overlap * self._upsample_rate
        pcm_float = wav[0, 0, trim:].cpu().float().numpy()
        # Convert float32 to int16 PCM bytes
        pcm_int16 = np.clip(pcm_float * 32767, -32768, 32767).astype(np.int16)
        return pcm_int16.tobytes()

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text via megakernel.

        ALL GPU work runs on the dedicated CUDAWorkerThread (a single long-lived
        thread created before asyncio starts), completely isolated from Pipecat's
        internal thread pool and any other pipeline CUDA activity.
        """
        t_start = time.time()
        logger.info(f"MegakernelTTS: request start text={text!r}")

        def _generate_all():
            """Run entire TTS pipeline: inputs → decode → audio."""
            self._backend.reset()
            embeds, mask, trailing, tts_pad = self._prepare_inputs(text)

            codec_frames = []
            for frame in self._backend.stream_decode(
                inputs_embeds=embeds,
                attention_mask=mask,
                trailing_text_hidden=trailing,
                tts_pad_embed=tts_pad,
                max_new_tokens=self._max_new_tokens,
            ):
                logger.debug(f"MegakernelTTS: decode step {frame.step_index} took {frame.latency_ms:.0f}ms")
                if frame.step_index == 0:
                    logger.info(f"MegakernelTTS: first codec frame at {time.time()-t_start:.3f}s")
                codec_frames.append(frame.codec_frame)

            logger.info(f"MegakernelTTS: decode finished, {len(codec_frames)} frames")

            # Decode all codec frames to PCM in one shot
            if not codec_frames:
                return b""
            codes_t = torch.stack(codec_frames, dim=0).unsqueeze(0).transpose(1, 2).to(torch.long).to(self._device)
            pcm_bytes = self._decode_codes_chunk(codes_t, 0, len(codec_frames), self._overlap)
            logger.info(f"MegakernelTTS: audio ready at {time.time()-t_start:.3f}s")
            return pcm_bytes

        pcm_bytes = await self._cuda_worker.run(_generate_all)

        if pcm_bytes:
            logger.info(f"MegakernelTTS: first audio chunk at {time.time()-t_start:.3f}s")
            yield TTSAudioRawFrame(pcm_bytes, self._sr, 1, context_id=context_id)

        logger.info(f"MegakernelTTS: done at {time.time()-t_start:.3f}s")
