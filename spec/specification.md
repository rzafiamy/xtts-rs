# Specification

## Goal

Run Coqui XTTS-v2 as a single native binary and a single GGUF model file,
with exact parity to the Python reference, low VRAM and low time to first
audio on CUDA, and serve it over an OpenAI-compatible HTTP API so that
[zallama](https://github.com/rzafiamy/zallama) can host it.

## Context and users

XTTS-v2 is a ~470M-parameter multilingual TTS (GPT-2 over audio codes +
HiFi-GAN) released by Coqui with a PyTorch implementation (Python, PyTorch,
a 1.9 GB checkpoint).

Users:
- **zallama integrator**: declares the server as a TTS backend; needs
  `/health`, `/v1/audio/speech`, `/stream`, a known VRAM footprint and one
  model file.
- **Application developer**: sends French or English text, gets WAV or
  streamed PCM within tens of milliseconds of first audio.
- **Command-line user**: synthesizes a sentence to a WAV file.
- **Maintainer**: converts checkpoints, checks parity and intelligibility.

Main path: `xtts convert <dir> --gpt-dtype q4k --no-cloning` →
`xtts serve -m xtts-v2-q4k.gguf` (zallama) or `xtts speak -m ...`.

## Priorities

- **Must (core, MVP)**: French and English, parity with coqui-tts, single
  GGUF, quantized GPT without intelligibility loss, CUDA, streaming, HTTP API.
- **Should**: CPU and Metal builds, the other 12 XTTS languages (cleaners
  ported, not validated by ear or ASR), speed control.
- **Could / later**: voice cloning from reference audio, zh/ja/ko (need
  their own tokenizer front ends).

## Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| REQ-TXT-001 | XTTS' `multilingual_cleaners` (lowercase, abbreviations, symbols, quotes) are reproduced for its 14 Latin/Cyrillic/Arabic/Devanagari languages; an unsupported language is an error. | Must |
| REQ-TXT-002 | Numbers, times, amounts and Markdown are spelled out in French and English before synthesis (tn-rs, on by default, `--no-normalize`); plain text is unchanged. | Must |
| REQ-TXT-003 | Text longer than the language's character limit is split at sentence ends (packed while they fit, wrapped at spaces otherwise). | Must |
| REQ-TXT-004 | The final period of each chunk is dropped so the model stops after the last word (no "point", no tail babble); `?`, `!` and ellipses stay. | Must |
| REQ-SMP-001 | Sampling follows HF `generate`: repetition penalty (prompt ids included), temperature, top-k, top-p; greedy and seeded modes. | Must |
| REQ-INF-001 | From the F32 checkpoint, greedy audio codes equal coqui-tts' exactly; latents within 1e-4, decoder ≥ 70 dB SNR. | Must |
| REQ-INF-002 | Generation ends on the stop code (or `max_codes`); the latent that produced it is decoded too, as in Coqui. | Must |
| REQ-INF-003 | `speed` resamples the latents before the decoder (0.25-4). | Should |
| REQ-STR-001 | Streaming decodes windows on the one-pass frame grid with ≥ 12 frames of context: same samples as one-pass decoding (> 40 dB), first audio after 8 codes. | Must |
| REQ-VOI-001 | The 58 built-in voices are stored in the GGUF and selected by name (case, `_` and `-` ignored); an unknown name is an error. | Must |
| REQ-GGF-001 | `xtts convert` writes one self-contained GGUF (config, tokenizer, voices, weights; weight norm folded). | Must |
| REQ-GGF-002 | The GPT can be quantized (q4k … q8_0); `--no-cloning` drops the unused cloning encoders: q4k ≤ 300 MB. | Must |
| REQ-QUA-001 | q4k is as intelligible as q8_0: Parakeet WER ≤ 1 % on `scripts/wer.py` (fr, en). | Must |
| REQ-GPU-001 | CUDA (RTX 4090, q4k): ≤ 1.1 GB VRAM peak, ≥ 10x real time, first streamed audio ≤ 60 ms. | Must |
| REQ-CLI-001 | `xtts speak` writes a 24 kHz WAV; `xtts voices` lists the voices. | Must |
| REQ-SRV-001 | `POST /v1/audio/speech` (OpenAI shape + `language`; WAV/PCM; `stream`), `GET /v1/voices`, `/v1/models`, `/health` after warmup; bad input → 400 with a JSON error. | Must |
| REQ-SRV-002 | `POST /stream` (`{text, voice, language}`) returns raw 24 kHz PCM16 as it is decoded; generation stops when the client disconnects. | Must |

## Non-functional requirements

- Platforms: Linux x86_64 and aarch64, Windows x86_64 (CPU), macOS (Metal);
  CUDA on Linux with the CUDA Toolkit ≥ 12.
- Toolchain pinned by `rust-toolchain.toml`; builds with `--locked`.
- No network access at run time; no secret needed.
- Weights stay under the Coqui Public Model License (non-commercial).

## Known limitations

- Voice cloning from reference audio is not implemented (the encoders are
  kept in the GGUF unless `--no-cloning`).
- Only French and English are validated; German babbles after sentences in
  the original model too. zh-cn, ja and ko are not supported.
- CPU synthesis is ~0.4x real time: usable offline, too slow to stream.
- One synthesis at a time per server (the model is behind a mutex).
