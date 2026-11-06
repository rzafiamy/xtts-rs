# Voice cloning and voices from files

`xtts clone` computes a voice from WAV files, with the same numbers as Coqui's
`get_conditioning_latents` and without PyTorch.

## Usage

```bash
xtts convert <checkpoint> -o clone.gguf --only-cloning --speaker-dtype f16
xtts clone -m clone.gguf --device cpu --threads 4 --out voice.voice a.wav b.wav
xtts clone -m clone.gguf --manifest voices.json   # many voices, one model load
xtts convert <checkpoint> -o model.gguf --gpt-dtype q4k --no-cloning --no-voices
xtts serve -m model.gguf --voices-dir voices/
```

- **Manifest:** a JSON array of `{"name", "out", "files"}`. Each voice prints one
  JSON line with its timings.
- **`.voice` format:** `XTTSVOIC`, version, name, shapes, then both tensors as
  F32, about 133 KB. Details in `crates/xtts/src/voicefile.rs`.
- **Input:** WAV files only.

## Accuracy

Latents peak at about 14. Largest absolute difference over all the values of a
voice, 30 voices.

| Comparison | Latents 1-cos, max / median | Latents diff, max / median | Embedding 1-cos, max | Embedding diff, max |
|---|---|---|---|---|
| Rust F32, CPU, vs PyTorch CPU | 2.5e-10 / 1.5e-11 | 7.8e-4 / 1.3e-4 | 5.5e-9 | 1.5e-5 |
| Rust F16, CPU, vs PyTorch CPU | 2.2e-7 / 8.0e-8 | 0.019 / 0.0074 | 6.4e-7 | 1.8e-4 |
| Rust F16, GPU, vs PyTorch CPU | 2.2e-7 / 8.0e-8 | 0.019 / 0.0075 | 6.3e-7 | 1.8e-4 |
| PyTorch CPU vs PyTorch GPU, one voice | 2.5e-7 | 0.020 | 2.6e-7 | 1.1e-4 |

Rust matches PyTorch on the CPU more closely than PyTorch on the CPU matches
PyTorch on the GPU. That difference starts in the conditioning encoder, with
identical mel input.

## The resampler matters

PyTorch's chain with only the resampling changed, 12 voices, against
`torchaudio.functional.resample`, which the port reproduces exactly.

| Resampler | Latents 1-cos, max | Embedding 1-cos, median / max |
|---|---|---|
| polyphase | 1.7e-4 | 5.6e-4 / 2.4e-3 |
| soxr_hq, the librosa default | 5.0e-5 | 2.7e-3 / 9.4e-3 |
| torchaudio with a wide Kaiser window | 4.6e-5 | 2.8e-3 / 1.0e-2 |
| soxr_lq | 5.5e-5 | 3.6e-2 / 9.2e-2 |
| linear | 1.7e-2 | 2.5e-2 / 4.4e-2 |

soxr moves the embedding thousands of times more than F16 weights do.

## Weights and sizes

Against PyTorch on the CPU, 30 voices.

| File | Size | Latents 1-cos, max | Embedding 1-cos, max | Latents diff, median |
|---|---|---|---|---|
| F32 | 217 MB | 2.5e-10 | 5.5e-9 | 1.3e-4 |
| F16, speaker F32 | 125 MB | 2.2e-7 | 5.5e-9 | 0.0074 |
| **F16, speaker F16** | **109 MB** | 2.2e-7 | 6.4e-7 | 0.0074 |
| Q8, speaker F16 | 71 MB | 4.1e-4 | 6.4e-7 | 0.31 |
| Q4K, speaker F16 | 51 MB | 2.5e-3 | 6.4e-7 | 0.74 |

## Time per voice

30 voices of 16 s of audio, 109 MB weights, an 8-core desktop CPU and a recent
RTX GPU. Median per voice, without the first three.

| Where | Time per voice | RAM | VRAM of the process |
|---|---|---|---|
| CPU, 2 threads | 1.73-1.82 s | 525 MB | |
| CPU, 4 threads | 1.30-1.52 s | 526 MB | |
| CPU, 8 threads | 1.14-1.15 s | 529 MB | |
| GPU | 0.21-0.25 s | 518 MB | 685 MB |

## Effect on synthesis

Six sentences with two seeds, plus one `/stream` request, comparing the bytes of
the HTTP responses: a GGUF with voices and a GGUF without voices plus `.voice`
files gave the same bytes in 13 of 13 cases, on the GPU and on the CPU.

Byte-for-byte reproducibility only holds for the same voice file. Recomputing a
voice on another device gives a different take of the same speaker, because the
GPT is sensitive to differences as small as PyTorch's own between CPU and GPU.

## Reproducing the measures

- **Stage by stage:** `scripts/clone_ref.py` and `crates/xtts/examples/clone_parity.rs`.
- **Profile:** `crates/xtts/examples/clone_prof.rs`.
- **From a Coqui `latents.pt` to `.voice`:** `scripts/latents_to_voice.py`.
