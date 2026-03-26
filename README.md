# Qwen3-TTS Megakernel Voice Pipeline

Adapts [AlpinDale's `qwen_megakernel`](https://github.com/AlpinDale/qwen_megakernel) to serve **Qwen3-TTS** talker inference inside a real-time **Pipecat** voice agent pipeline on RTX 5090 (Blackwell, sm_120).

The megakernel replaces HuggingFace's `generate()` for the talker decoder backbone, reducing per-step latency from ~15ms (PyTorch) to **~2ms** (fused CUDA kernel).

## Architecture

```
                          ┌─────────────────────────────────────────────┐
                          │            GPU (RTX 5090)                   │
                          │                                             │
  Text ──► HF Prefill ──►│ KV Cache ──► Megakernel ──► Code Predictor │──► Audio
           (PyTorch)      │ Transfer     Decode (CUDA)   (HF generate)  │   Decoder
                          │ (~0ms)       (~2ms/step)     (~90ms/step)   │
                          └─────────────────────────────────────────────┘
```

**Full Pipecat pipeline:**

```
Microphone → WebSocket → STT (Whisper) → LLM (GPT-4.1-mini) → TTS (Megakernel) → WebSocket → Speaker
```

**Why this split:** The 28-layer talker backbone is the autoregressive bottleneck — one forward pass per codec frame — so that's where the megakernel pays off (~2ms vs ~15ms in PyTorch). Prefill is a single pass and not latency-critical, so we keep it in HF to avoid reimplementing the complex embedding construction. The code predictor (a small 5-layer model generating 15 sub-codebook tokens) stays in HF too — it could use its own megakernel, but it wasn't the primary target.

## Kernel Adaptation (0.6B → 1.7B Talker)

We used `Qwen3-TTS-12Hz-1.7B-Base` rather than sticking with the 0.6B baseline. It's the actual TTS checkpoint and gives better speech quality. The talker backbone has different dimensions than the original 0.6B text model:

| Parameter | Original (0.6B) | Talker (1.7B) |
|-----------|-----------------|---------------|
| `hidden_size` | 1024 | **2048** |
| `intermediate_size` | 3072 | **6144** |
| `vocab_size` | 2048 | **3072** |
| `num_layers` | 24 | **28** |
| `num_attention_heads` | 16 | 16 |
| `num_key_value_heads` | 8 | 8 |
| `head_dim` | 128 | 128 |

To handle this, we made the kernel dimensions compile-time configurable via `#ifndef` guards — the build script passes `-DLDG_HIDDEN_SIZE=2048` etc. and everything scales through `constexpr` values. No algorithmic changes to attention, MLP, RMSNorm, or barrier sync were needed.

We also added a `decode_from_embeds` kernel entry point. The TTS talker constructs a composite embedding from multiple codebook tokens + text injection that can't be expressed as a single table lookup, so the kernel needs to accept a pre-computed embedding vector directly.

The tradeoff: more integration complexity and a ~2x slowdown on the backbone (2ms vs 1ms) due to the larger hidden size. But our measurements showed the backbone isn't the bottleneck — the HF code predictor at ~90ms/step dwarfs it. We took the larger model for better audio quality knowing the kernel wasn't the limiting factor.

See [docs/kernel-adaptation.md](docs/kernel-adaptation.md) for the full technical details (code snippets, binding registration, shared memory analysis).

## Performance

### Benchmarks

| Metric | Target | Measured | How we measured it |
|--------|--------|----------|--------------------|
| Megakernel backbone | — | **2.03 ms/step** | `torch.cuda.synchronize()` before/after kernel, avg over 50 steps |
| Full decode step | — | **92–100 ms/step** | Wall clock including code_predictor, embedding construction, megakernel |
| TTFC (warm) | < 90 ms | **~140 ms** | Text input → first codec frame, 2nd+ generation |
| TTFC (cold) | — | **~940 ms** | First generation after model load (JIT warmup) |
| RTF | < 0.3 | **~1.14** | `wall_time / audio_duration`, 12Hz codec, 24kHz mono |
| Decode tok/s | — | **~10.5 frames/s** | 12 frames/s = real-time for the 12Hz codec |
| E2E latency | — | **~1–1.5s** | STT (Whisper API) + LLM (GPT-4.1-mini) + TTS |

**Hardware:** RTX 5090 (sm_120, 170 SMs, 1792 GB/s GDDR7), CUDA 13.0, Driver 580.82

### Where the time actually goes

The megakernel is fast. The bottleneck is elsewhere:

| Component | Time | % of step |
|-----------|------|-----------|
| `code_predictor.generate()` | ~90 ms | 95% |
| Megakernel backbone | ~2 ms | 2% |
| Embedding + overhead | ~3 ms | 3% |

The code predictor runs 15 autoregressive sub-tokens per step at ~6ms each. Most of that ~6ms is HuggingFace `GenerationMixin` overhead (logits processing, DynamicCache allocation, sampling) — the actual GPU compute is only ~2ms per sub-token.

### What it would take to hit the targets

| Optimization | Step time | RTF | Effort |
|-------------|-----------|-----|--------|
| Manual decode loop (drop HF `generate()`) | ~30–45 ms | ~0.4 | 0.5–1 day |
| + `torch.compile` + static KV cache | ~15–25 ms | ~0.24 | +1 day |
| + CUDA graphs | ~8–15 ms | ~0.14 | +1 day |
| Code predictor megakernel | ~1–3 ms | ~0.06 | +1–2 weeks |

The first move is replacing `code_predictor.generate()` with a manual loop to cut the Python overhead.

### On streaming

We buffer the full utterance before sending audio rather than streaming frame-by-frame. This is a deliberate tradeoff, not an oversight.

Each decode step takes ~95ms (dominated by the code predictor). At that rate, a 15-frame utterance takes ~1.4s to decode regardless of whether we stream or buffer — the per-frame streaming would save at most one frame's worth of latency (~95ms) on first audio, while adding complexity to interleave codec-to-audio decoding with the decode loop. The audio decoder (`SpeechTokenizer`) expects contiguous codec frame windows with overlap for quality, so frame-by-frame streaming also requires careful windowing logic to avoid audio artifacts at chunk boundaries.

Once the code predictor bottleneck is resolved (bringing step time to ~15–25ms), frame-by-frame streaming becomes worthwhile and the architecture supports it — `stream_decode()` already yields one `CodecFrame` per step, so the change is in `tts_service.py`'s `run_tts()` method, not the decode path.

## Improvements Found During Integration

Three things we added or fixed beyond the original megakernel:

**Blackwell scheduler bug workaround.** The RTX 5090 hangs when the same persistent-barrier kernel is launched 3+ consecutive times — the block scheduler fails to co-schedule all 128 blocks, and the grid barrier deadlocks. This never shows up in the original text-LLM use case (single long generation), but happens on every decode step in TTS. We alternate between `_decode_from_embeds` and `_decode` (with an embed copy) on each step. Same logic, different kernel binaries, avoids the scheduler bug entirely. Without this, the pipeline is completely broken on Blackwell. See [docs/integration-improvements.md](docs/integration-improvements.md) for the investigation details and evidence.

**Compile-time parametric dimensions.** The original kernel hardcodes Qwen3-0.6B dimensions. We added `#ifndef` guards so any Qwen3-family model can be targeted by changing build flags — no source edits needed.

**`decode_from_embeds` entry point.** New kernel entry point that takes a pre-computed bf16 embedding instead of a token ID. Required for TTS where the input is a composite of multiple codebook embeddings + text injection.

## Setup & Running the Demo

### Prerequisites

- NVIDIA RTX 5090 with CUDA 12.x+
- Python 3.10+
- `OPENAI_API_KEY` (for Whisper STT and GPT-4.1-mini LLM)

### Installation (GPU machine)

```bash
git clone <repo-url> && cd qwen3-tts-megakernel
pip install -r requirements.txt
pip install git+https://github.com/QwenLM/Qwen3-TTS.git

# The megakernel extension compiles on first import via JIT
# (needs CUDA toolkit with sm_120a support)
```

### Start the server

```bash
export OPENAI_API_KEY="sk-..."
PYTHONPATH=/path/to/Qwen3-TTS:. python -m pipeline.voice_agent_server
```

Model loads in ~30s. First run also compiles the kernel extension. Server listens on port 9000.

### SSH tunnel (local → GPU)

```bash
ssh -L 9000:localhost:9000 <gpu-machine>
```

### Start the client (local machine)

```bash
pip install websockets sounddevice numpy protobuf
python pipeline/ws_voice_client.py
```

Talk into your mic. Use a headset — speakers create a feedback loop (TTS output → mic → STT → infinite cycle).

## Repo Structure

```
qwen3-tts-megakernel/
├── csrc/
│   ├── kernel.cu                     # Modified megakernel (parametric dims + decode_from_embeds)
│   └── torch_bindings.cpp            # PyTorch C++ bindings
├── qwen_megakernel/
│   ├── build_talker.py               # JIT build with 1.7B talker dimensions
│   └── model_talker.py               # TalkerDecoder wrapper + Blackwell workaround
├── adapter/
│   └── megakernel_talker_backend.py  # Hybrid HF prefill + megakernel decode + code predictor
├── pipeline/
│   ├── tts_service.py                # Pipecat TTSService + CUDAWorkerThread
│   ├── voice_agent_server.py         # Full pipeline server
│   └── ws_voice_client.py            # WebSocket voice client
├── docs/
│   ├── kernel-adaptation.md          # 0.6B → 1.7B kernel changes in detail
│   └── integration-improvements.md   # Blackwell bug, parametric dims, decode_from_embeds
├── samples/
│   └── pipeline_roundtrip.wav        # Sample TTS output
└── requirements.txt
```

## What works, what doesn't

The voice pipeline works end-to-end — mic to speaker, multiple conversation turns, no restarts needed. The megakernel backbone is solid at 2ms/step and the Blackwell workaround has held up across hundreds of decode steps. KV cache transfer from HF prefill to the megakernel's flat cache works cleanly. EOS detection produces natural sentence boundaries at 14–21 codec frames.

The main gap is performance: TTFC (140ms) and RTF (1.14) both miss their targets. This is entirely due to `code_predictor.generate()` eating ~90ms per step — the megakernel itself is not the bottleneck. The optimization path is clear (manual decode loop → torch.compile → CUDA graphs) but wasn't implemented in this iteration.

Audio is buffered rather than streamed frame-by-frame — a deliberate choice given the current per-frame cost (see [On streaming](#on-streaming) above). Cold start is slow (~940ms), long utterances haven't been stress-tested, and there's no mid-generation cancellation.
