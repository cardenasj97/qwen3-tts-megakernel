"""Pipecat voice agent server.

Full pipeline: WebSocket transport → OpenAI STT → OpenAI LLM → Megakernel TTS → WebSocket output.
Runs on the remote GPU machine. Connects via SSH tunnel from the local client.

Usage (from repo root on GPU machine):
    PYTHONPATH=/path/to/Qwen3-TTS:. \
        python -m pipeline.voice_agent_server
"""

import asyncio
import logging
import os
import sys
import time

import torch
from loguru import logger

# Enable Python logging for megakernel_talker_backend messages
logging.basicConfig(level=logging.INFO, format="%(name)s:%(levelname)s:%(message)s")

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    TextFrame,
    TTSAudioRawFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.openai.llm import OpenAILLMContext, OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.transports.websocket.server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)

# ── Import TTS adapter ────────────────────────────────────────────────
from pipeline.tts_service import CUDAWorkerThread, MegakernelTTSService
from adapter.megakernel_talker_backend import MegakernelTalkerBackend

WS_PORT = 9000
LLM_MODEL = "gpt-4.1-mini"
SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep responses short — one or two sentences. "
    "Be conversational and concise."
)


# ── Metrics processor ─────────────────────────────────────────────────

class MetricsProcessor(FrameProcessor):
    """Logs per-turn metrics: TTFC, E2E latency, token count."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._turn_start: float | None = None
        self._first_audio: float | None = None
        self._last_audio: float | None = None
        self._audio_bytes = 0
        self._text_tokens = 0

    def _reset(self):
        self._turn_start = None
        self._first_audio = None
        self._last_audio = None
        self._audio_bytes = 0
        self._text_tokens = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text:
            # Log metrics for the *previous* turn before resetting
            if self._turn_start is not None and self._first_audio is not None:
                self._log_metrics()
            self._reset()
            self._turn_start = time.time()
            logger.info(f"[METRICS] User said: \"{frame.text}\"")

        elif isinstance(frame, TextFrame) and frame.text:
            self._text_tokens += 1  # approximate: one TextFrame per token chunk

        elif isinstance(frame, TTSAudioRawFrame):
            now = time.time()
            if self._first_audio is None:
                self._first_audio = now
            self._last_audio = now
            self._audio_bytes += len(frame.audio)

        elif isinstance(frame, EndFrame):
            self._log_metrics()

        await self.push_frame(frame, direction)

    def _log_metrics(self):
        if self._turn_start is None:
            return

        now = time.time()
        e2e = now - self._turn_start

        # Time to first audio chunk (TTFC)
        ttfc = (self._first_audio - self._turn_start) if self._first_audio else None

        # Audio duration (24kHz, 16-bit mono = 48000 bytes/sec)
        audio_duration = self._audio_bytes / 48000.0 if self._audio_bytes else 0

        # Real-time factor (RTF) = wall time / audio duration
        wall_time = (self._last_audio - self._first_audio) if (self._first_audio and self._last_audio) else 0
        rtf = wall_time / audio_duration if audio_duration > 0 else 0

        # Tokens per second
        toks = self._text_tokens / e2e if e2e > 0 else 0

        ttfc_str = f"{ttfc:.2f}s" if ttfc else "N/A"
        logger.info(
            f"[METRICS] Turn complete: "
            f"TTFC={ttfc_str}, "
            f"E2E={e2e:.2f}s, "
            f"RTF={rtf:.2f}, "
            f"tok/s={toks:.1f}, "
            f"audio={audio_duration:.1f}s"
        )
        self._reset()


# ── Transcript logger (taps the pipeline) ─────────────────────────────

class TranscriptLogger(FrameProcessor):
    """Logs transcription and LLM response text."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._llm_response_chunks: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame) and frame.text:
            self._llm_response_chunks.append(frame.text)

        elif isinstance(frame, TTSAudioRawFrame) and self._llm_response_chunks:
            # Log the accumulated LLM response when TTS starts
            full_text = "".join(self._llm_response_chunks)
            logger.info(f"[LLM] Response: \"{full_text}\"")
            self._llm_response_chunks.clear()

        await self.push_frame(frame, direction)


# ── Main ───────────────────────────────────────────────────────────────

async def run_session(tts_model, backend, cuda_worker):
    """Build and run one pipeline session. Returns when the client disconnects."""
    # ── Build Pipecat services ─────────────────────────────────────────
    transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
            add_wav_header=False,
            serializer=ProtobufFrameSerializer(),
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(stop_secs=0.7),
            ),
        ),
        host="0.0.0.0",
        port=WS_PORT,
    )

    stt = OpenAISTTService(
        api_key=os.environ["OPENAI_API_KEY"],
        model="whisper-1",
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        model=LLM_MODEL,
    )

    tts = MegakernelTTSService(
        tts_model=tts_model,
        backend=backend,
        cuda_worker=cuda_worker,
        chunk_size=12,
        overlap=5,
        max_new_tokens=512,
    )

    # ── LLM context ───────────────────────────────────────────────────
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = OpenAILLMContext(messages=messages)
    context_aggregator = llm.create_context_aggregator(context)

    # ── Custom processors ─────────────────────────────────────────────
    metrics = MetricsProcessor()
    transcript_logger = TranscriptLogger()

    # ── Pipeline ──────────────────────────────────────────────────────
    pipeline = Pipeline(
        [
            transport.input(),              # WS audio in → InputAudioRawFrame
            stt,                            # → TranscriptionFrame
            context_aggregator.user(),      # → LLMMessagesFrame
            llm,                            # → TextFrame chunks
            transcript_logger,              # logs LLM response
            tts,                            # → TTSAudioRawFrame chunks
            metrics,                        # logs metrics
            transport.output(),             # → WS audio out
            context_aggregator.assistant(), # accumulates assistant context
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )

    # ── Event handlers ────────────────────────────────────────────────
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected: {client}")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected: {client}")
        await task.queue_frames([EndFrame()])

    # ── Run ───────────────────────────────────────────────────────────
    runner = PipelineRunner()
    await runner.run(task)


async def main(cuda_worker: CUDAWorkerThread):
    # ── Load TTS model once (expensive) ───────────────────────────────
    logger.info("Loading Qwen3-TTS model...")
    from qwen_tts import Qwen3TTSModel

    tts_model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    talker = tts_model.model.talker
    tc = talker.config

    backend = MegakernelTalkerBackend(talker, tc)
    logger.info("Model loaded.")

    # ── Session loop: restart pipeline for each client ────────────────
    while True:
        logger.info(f"WebSocket server ready on :{WS_PORT} — waiting for client...")
        try:
            await run_session(tts_model, backend, cuda_worker)
        except Exception as e:
            logger.error(f"Session error: {e}")
        logger.info("Session ended. Resetting for next client...")
        backend.reset()


if __name__ == "__main__":
    # CUDAWorkerThread must be created BEFORE asyncio.run() so it is fully
    # isolated from the event loop's default thread-pool executor.
    cuda_worker = CUDAWorkerThread()
    asyncio.run(main(cuda_worker))
