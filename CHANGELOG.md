# Changelog

All notable changes to **xtts-rs** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-03

First release: Coqui XTTS-v2 in Rust/Candle with GGUF weights.

### Added
- **Model**: GPT-2 (30 layers) with KV cache, learned per-segment position
  tables, HF `generate` sampling order (repetition penalty, temperature,
  top-k, top-p); HiFi-GAN decoder conditioned on the speaker embedding;
  the 58 built-in voices. Greedy codes identical to coqui-tts 0.27 from the
  F32 checkpoint.
- **GGUF**: `xtts convert` writes one file (config, tokenizer, voices,
  weights) with the GPT in q4k / q5k / q6k / q8_0 / f16 and the decoder in
  f16 / f32; weight norm folded at conversion. `--no-cloning` leaves out the
  voice-cloning encoders (401 → 276 MB in q4k).
- **Streaming**: `StreamDecoder` decodes windows of the one-pass frame grid
  (≥ 12 frames of context): first audio 35-45 ms on an RTX 4090.
- **Text**: tn-rs normalization (French, English), XTTS cleaners for the 14
  languages, sentence splitting at the language's character limit.
- **CLI**: `convert`, `speak`, `voices`, `serve`; `XTTS_*` environment
  variables for the model, device, host, port, voice and language.
- **HTTP server**: `POST /v1/audio/speech` (OpenAI shape plus `language`,
  WAV or PCM, `stream`), `POST /stream` (raw PCM16, pocket-tts contract),
  `GET /v1/voices`, `/v1/models`, `/health` (after a warmup sentence).
- **CUDA memory**: the stream-ordered pool is trimmed after each request
  (776 MiB idle, 1032 MiB peak for a 20 s utterance, q4k).
- Tests: unit tests (text, sampling), model tests (`XTTS_MODEL`),
  `tests/e2e.sh`; `scripts/wer.py` (intelligibility), `scripts/vram.sh`,
  `scripts/ref.py` + `examples/parity.rs`.

### Changed (from Coqui)
- The final period of each text chunk is dropped: XTTS often reads it out
  loud ("point") or babbles after it. Parakeet WER 1.1 → 0.6 % (French),
  0.5 → 0.0 % (English); pauses between sentences are unchanged.

[Unreleased]: https://github.com/rzafiamy/xtts-rs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/rzafiamy/xtts-rs/releases/tag/v0.1.0
