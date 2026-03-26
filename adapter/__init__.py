"""Megakernel talker backend adapter for Qwen3-TTS."""

from .megakernel_talker_backend import (
    MegakernelTalkerBackend,
    PrefillResult,
    DecodeStepResult,
    CodecFrame,
)

__all__ = [
    "MegakernelTalkerBackend",
    "PrefillResult",
    "DecodeStepResult",
    "CodecFrame",
]
