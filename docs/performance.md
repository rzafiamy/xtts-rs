# Performance and quality

Measured 2026-10-03 on an RTX 4090 (CUDA 12, driver 580), Ryzen CPU,
`models/xtts-v2-q4k.gguf` written with `--gpt-dtype q4k --no-cloning`
unless stated.

## Speed and latency (`examples/bench.rs`, CUDA)

| Text | First audio | Real-time factor |
|---|---|---|
| "Bonjour à tous, voici un test de latence pour la synthèse vocale." (3.95 s) | 30-36 ms | 13.3-13.6x |
| Short reply, through `xtts serve` `/stream` (3.4-4.5 s) | 34-44 ms | ~13x |

Windowed decoding (context 16, lookahead 12) vs one decoder pass: 91 dB SNR.
Model load: 0.1-1 s (warm / cold page cache). CPU (`--device cpu`): ~0.4x
real time.

## VRAM (`scripts/vram.sh`, one ~20 s utterance)

| GGUF | File | Idle | Peak |
|---|---|---|---|
| q4k, `--no-cloning` | 276 MB | 776 MiB | 1032 MiB |
| q4k | 401 MB | 776 MiB | 1032 MiB |
| q8_0 | 590 MB | 936 MiB | 1224 MiB |
| bare CUDA context (`examples/vram_floor.rs`) | — | 428 MiB | — |

The cloning encoders are F16/F32 and stay on the CPU, so `--no-cloning`
saves file size and ~30 MB of RAM, not VRAM. The F16 decoder would save
~35 MB but drops to 27 dB SNR (audible), so it stays F32 on the device.

## Intelligibility (`scripts/wer.py`, Parakeet TDT v3, 2 voices × 2-3 seeds)

| GGUF | French | English |
|---|---|---|
| q4k (with the final period) | 1.2 % | 0.5 % |
| q6k (with the final period) | 0.8 % | 0.0 % |
| q8_0 (with the final period) | 3.1 % | 0.2 % |
| **q4k, final period dropped (current)** | **0.6 %** | **0.0 %** |

Differences between quantizations are within sampling noise. Most errors
with the final period were an extra word after the sentence ("… vocale
point", "… Pinto"). Sampling settings barely change that: temperature
0.3-0.75 × top_p 0.5-0.85 all gave fr 1.1-1.7 %, en 0.0-0.5 %. The
probability of the stop code stays below 1e-3 until it is drawn, so a
stop-probability threshold (`--stop-prob`) cannot cut the babble either.

Trailing silence (~170 ms) and pauses between sentences of one chunk
(500-800 ms) are the same with and without the final period.
