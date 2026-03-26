"""WebSocket voice client for the Pipecat voice agent server.

Captures mic audio, sends it over WebSocket (protobuf frames),
receives audio/text frames, and plays back audio on the speaker.

Usage:
    python3 ws_voice_client.py [--server ws://localhost:9000]

Requires SSH tunnel: ssh -L 9000:localhost:9000 <remote>
"""

import argparse
import asyncio
import queue
import sys
import threading

import numpy as np
import sounddevice as sd

from pipecat.frames.frames import (
    InputAudioRawFrame,
    OutputAudioRawFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.serializers.protobuf import ProtobufFrameSerializer

MIC_SAMPLE_RATE = 16000
MIC_CHANNELS = 1
MIC_BLOCKSIZE = 1600  # 100ms at 16kHz

SPEAKER_SAMPLE_RATE = 24000
SPEAKER_CHANNELS = 1
SPEAKER_BLOCKSIZE = 2400  # 100ms at 24kHz

DEFAULT_SERVER = "ws://localhost:9000"


class VoiceClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.serializer = ProtobufFrameSerializer()
        self._ws = None
        self._running = False

        # Audio playback queue: holds int16 PCM bytes
        self._playback_queue: queue.Queue[bytes] = queue.Queue(maxsize=200)

        # Accumulated text for display
        self._current_response: list[str] = []

    async def run(self):
        import websockets

        self._running = True
        print(f"[CLIENT] Connecting to {self.server_url}...")

        async with websockets.connect(
            self.server_url,
            ping_interval=30,
            ping_timeout=120,
            close_timeout=10,
        ) as ws:
            self._ws = ws
            print("[CLIENT] Connected. Speak into your microphone. Ctrl+C to quit.")

            # Start mic capture in background thread
            mic_task = asyncio.create_task(self._mic_sender())
            recv_task = asyncio.create_task(self._receiver())
            playback_thread = threading.Thread(
                target=self._playback_loop, daemon=True
            )
            playback_thread.start()

            try:
                await asyncio.gather(mic_task, recv_task)
            except asyncio.CancelledError:
                pass
            finally:
                self._running = False

    async def _mic_sender(self):
        """Capture mic audio and send as protobuf frames over WebSocket."""
        loop = asyncio.get_event_loop()
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

        def mic_callback(indata, frames, time_info, status):
            if status:
                print(f"[MIC] {status}", file=sys.stderr)
            # Convert float32 to int16 PCM bytes
            pcm_int16 = (indata[:, 0] * 32767).astype(np.int16)
            loop.call_soon_threadsafe(audio_queue.put_nowait, pcm_int16.tobytes())

        stream = sd.InputStream(
            samplerate=MIC_SAMPLE_RATE,
            channels=MIC_CHANNELS,
            blocksize=MIC_BLOCKSIZE,
            dtype="float32",
            callback=mic_callback,
        )

        with stream:
            print("[MIC] Recording started (16kHz mono)")
            while self._running:
                try:
                    audio_bytes = await asyncio.wait_for(
                        audio_queue.get(), timeout=0.5
                    )
                except asyncio.TimeoutError:
                    continue

                # Serialize as OutputAudioRawFrame (server deserializes as InputAudioRawFrame)
                frame = OutputAudioRawFrame(
                    audio=audio_bytes,
                    sample_rate=MIC_SAMPLE_RATE,
                    num_channels=MIC_CHANNELS,
                )
                data = await self.serializer.serialize(frame)
                if data and self._ws:
                    await self._ws.send(data)

    async def _receiver(self):
        """Receive frames from WebSocket and dispatch to playback/display."""
        while self._running and self._ws:
            try:
                data = await self._ws.recv()
            except Exception:
                break

            frame = await self.serializer.deserialize(data)
            if frame is None:
                print(f"[DEBUG] Received non-deserializable data, len={len(data)}")
                continue

            if isinstance(frame, InputAudioRawFrame):
                # Audio from server → queue for playback
                audio_len = len(frame.audio)
                if not hasattr(self, '_audio_frame_count'):
                    self._audio_frame_count = 0
                self._audio_frame_count += 1
                if self._audio_frame_count <= 3 or self._audio_frame_count % 20 == 0:
                    print(f"[AUDIO] Frame #{self._audio_frame_count}: {audio_len} bytes, sr={frame.sample_rate}, ch={frame.num_channels}, queue={self._playback_queue.qsize()}")
                self._playback_queue.put_nowait(frame.audio)

            elif isinstance(frame, TranscriptionFrame) and frame.text:
                # User transcript from STT
                print(f"\n[YOU] {frame.text}")
                self._current_response.clear()

            elif isinstance(frame, TextFrame) and frame.text:
                # LLM response tokens streaming in
                self._current_response.append(frame.text)
                # Print inline
                sys.stdout.write(frame.text)
                sys.stdout.flush()

    def _playback_loop(self):
        """Blocking thread that plays audio from the queue via sounddevice."""
        self._playback_chunks_played = 0

        def speaker_callback(outdata, frames, time_info, status):
            if status:
                print(f"[SPEAKER] status: {status}", file=sys.stderr)
            bytes_needed = frames * 2  # int16 = 2 bytes per sample
            collected = b""
            while len(collected) < bytes_needed:
                try:
                    chunk = self._playback_queue.get_nowait()
                    collected += chunk
                except queue.Empty:
                    break

            if collected:
                # Trim or pad to exact frame size
                collected = collected[:bytes_needed]
                if len(collected) < bytes_needed:
                    collected += b"\x00" * (bytes_needed - len(collected))
                arr = np.frombuffer(collected, dtype=np.int16).astype(np.float32) / 32767.0
                outdata[:len(arr), 0] = arr
                outdata[len(arr):, 0] = 0.0
            else:
                outdata[:, 0] = 0.0

        with sd.OutputStream(
            samplerate=SPEAKER_SAMPLE_RATE,
            channels=SPEAKER_CHANNELS,
            blocksize=SPEAKER_BLOCKSIZE,
            dtype="float32",
            callback=speaker_callback,
        ):
            while self._running:
                sd.sleep(100)


async def amain(server_url: str):
    client = VoiceClient(server_url)
    try:
        await client.run()
    except KeyboardInterrupt:
        pass
    finally:
        client._running = False


def main():
    parser = argparse.ArgumentParser(description="Voice agent WebSocket client")
    parser.add_argument(
        "--server", default=DEFAULT_SERVER, help=f"WebSocket server URL (default: {DEFAULT_SERVER})"
    )
    args = parser.parse_args()
    asyncio.run(amain(args.server))


if __name__ == "__main__":
    main()
