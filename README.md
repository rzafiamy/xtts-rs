# xtts-rs

Rust/[Candle](https://github.com/huggingface/candle) port of
[Coqui XTTS-v2](https://huggingface.co/coqui/XTTS-v2), a ~470M multilingual
text-to-speech model (58 built-in voices, 24 kHz). One binary, one GGUF file,
CUDA or CPU; streams audio as it is generated.

Pipeline: text → BPE → GPT-2 (30 layers) over `[conditioning latents | text |
audio codes]` → one audio code per 1024 samples → HiFi-GAN conditioned on a
speaker embedding → 24 kHz audio.

## Results

- **Exact.** From the F32 checkpoint, greedy audio codes are identical to
  coqui-tts 0.27 (`scripts/ref.py` + `examples/parity.rs`); GPT latents within
  1e-4, decoder ~75-79 dB SNR.
- **Fast** (RTX 4090, q4k): ~13x real time, first streamed audio 35-45 ms,
  model load ~1 s. CPU: ~0.4x real time.
- **Small**: 276 MB GGUF (q4k GPT, `--no-cloning`), 776 MiB VRAM idle and
  1032 MiB peak for a 20 s utterance, 428 MiB of which is the CUDA context.
- **Intelligible**: Parakeet ASR WER 0.6 % French, 0.0 % English
  (`scripts/wer.py`, 2 voices × 3 seeds); q6k and q8_0 are no better.

One deliberate change from Coqui: the final period of each text chunk is
dropped. XTTS often reads it out loud ("point") or babbles a syllable after
it; without it the model stops right after the last word (WER 1.1 → 0.6 %
French, 0.5 → 0.0 % English). Pauses between sentences are unchanged.

## Features

Core (must work for a release):

- Text to speech in **French and English** with **58 built-in voices**, 24 kHz
  (`crates/xtts/src/model.rs`, `gguf.rs`).
- **Exact port**: greedy codes identical to coqui-tts 0.27
  (`gpt.rs`, `hifigan.rs`, `sampling.rs`).
- **One GGUF file** (`xtts convert`), GPT quantized to q4k … q8_0
  (`gguf.rs`, `weights.rs`).
- **Streaming**: audio as it is decoded, first audio after 8 codes
  (`hifigan.rs` `StreamDecoder`, `model.rs` `synthesize_stream`).
- **HTTP server**: OpenAI-compatible `/v1/audio/speech`, raw-PCM `/stream`
  (`crates/xtts-cli/src/server.rs`).
- **Text normalization** (tn-rs): numbers, dates, amounts, Markdown → words
  (`text.rs`).

Secondary:

- 12 more XTTS languages (es, de, it, pt, pl, tr, ru, nl, cs, ar, hu, hi):
  cleaners ported, not validated (`text.rs`, `text_tables.rs`).
- `speed`, sampling controls, seeds (`model.rs`, `sampling.rs`); CPU and
  Metal builds (`build.sh`).

Each feature maps to code and tests in [spec/matrix.md](spec/matrix.md).

### Known limitations

- No voice cloning from reference audio yet.
- German babbles after sentences (the original model too); zh-cn, ja, ko
  are not supported.
- CPU synthesis is ~0.4x real time: fine offline, too slow to stream.
- One synthesis at a time per server process.

## Installation

### Prerequisites

- Rust (version pinned by `rust-toolchain.toml`; `./prereq.sh` installs it
  with rustup), a C/C++ compiler, git, curl.
- NVIDIA GPU: the CUDA Toolkit ≥ 12 (`nvcc`) and a driver that supports it.
  Apple GPU: Xcode Command Line Tools.
- Optional: Python 3 for the parity and WER scripts in `scripts/`.

Platforms: Linux x86_64 and aarch64 (CPU, CUDA), macOS (Metal), Windows
x86_64 (CPU). Tested on Linux x86_64 with an RTX 4090.

### Build

```bash
./prereq.sh            # check / install the toolchain (CHECK_ONLY=1: check only)
./setup.sh --cuda      # fetch dependencies and compile-check (idempotent)
./build.sh --cuda      # release binary → build/xtts-<os>-<arch>-cuda-<version>
                       # ./build.sh for CPU, ./build.sh --metal for Apple GPUs
```

### Model

Download the Coqui checkpoint (`model.pth`, `config.json`, `vocab.json`,
`speakers_xtts.pth` from [coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2))
and convert it:

```bash
xtts convert ./XTTS-v2 -o models/xtts-v2-q4k.gguf --gpt-dtype q4k --no-cloning
xtts voices -m models/xtts-v2-q4k.gguf      # checks the file: lists 58 voices
```

`--no-cloning` leaves out the voice-cloning encoders (125 MB, unused for
now). The file is under the Coqui Public Model License, like the weights.

## Usage

```bash
xtts speak -m models/xtts-v2-q4k.gguf -l fr -v "Ana Florence" "Bonjour à tous." -o out.wav
xtts serve -m models/xtts-v2-q4k.gguf --lang fr --port 8091
curl -s localhost:8091/v1/audio/speech -H 'content-type: application/json' \
  -d '{"input":"Bonjour à tous.","voice":"Ana Florence","language":"fr"}' -o out.wav
```

API: [docs/serve.md](docs/serve.md).

## Configuration

**Location**: there is no configuration file. The only file on disk is the
model you point to, wherever you keep it — suggested:
`~/.local/share/xtts/xtts-v2-q4k.gguf` on Linux,
`~/Library/Application Support/xtts/` on macOS,
`%APPDATA%\xtts\` on Windows, or `models/` in a checkout. `.env` files are
read by your shell, not by the binary. Everything is set by command-line
options (`xtts <command> --help`) or environment variables — template:
[`xtts.example.env`](xtts.example.env), copy to `.env` and load it
(`set -a; . ./.env; set +a`) or export the variables. To change a setting,
change the option or variable and restart the process; options take
precedence over variables. Model settings (layers, tokenizer, voices) are
inside the GGUF. Under zallama, the registry entry's `params` set these
options.

| Option / variable | Role | Default |
|---|---|---|
| `-m, --model`, `XTTS_MODEL` | GGUF file or Coqui checkpoint directory | — |
| `--device`, `XTTS_DEVICE` | `auto` (CUDA if available), `cpu`, `cuda`, `cuda:N`, `metal` | `auto` |
| `--threads`, `XTTS_THREADS` | CPU threads | all cores |
| `--voice`, `XTTS_VOICE` (serve) | Default voice | `Claribel Dervla` |
| `--lang`, `XTTS_LANG` (serve) | Default language | `en` |
| `--host`, `--port`, `XTTS_HOST/PORT` | Server address | `127.0.0.1:8091` |
| `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty` | Sampling | 0.75, 50, 0.85, 5.0 |
| `--greedy`, `--seed` | Deterministic output | sampling, random seed |
| `--no-normalize`, `--decode-chunk`, `--no-warmup` | Text normalization, decoder window, warmup | on, 48, on |
| `RUST_LOG` | Log level | `info` |

No secret or token is needed: everything runs from the local file.

**Verify** a configuration with `xtts voices` (same model options as
`serve`): it loads the model on the chosen device and lists the voices; a
running server answers `GET /health` with `{"status":"ok"}` once ready.

## Tests

```bash
cargo test --release --workspace                                         # unit tests
XTTS_MODEL=$PWD/models/xtts-v2-q4k.gguf cargo test --release -p xtts --test model
XTTS_MODEL=models/xtts-v2-q4k.gguf tests/e2e.sh build/xtts-*             # binary + server
```

Manual tests (parity with coqui-tts, WER, VRAM): [spec/manual-tests.md](spec/manual-tests.md).

## Documentation

- [docs/architecture.md](docs/architecture.md) — pipeline and modules
- [docs/performance.md](docs/performance.md) — speed, VRAM, WER
- [docs/serve.md](docs/serve.md) — HTTP API
- [spec/](spec/) — specification, traceability matrix, manual tests
- [portfolio/](portfolio/) — overview, demos, diagram
- [CHANGELOG.md](CHANGELOG.md), [CREDITS.md](CREDITS.md), [CONTRIBUTING.md](CONTRIBUTING.md)
- Issues: https://github.com/rzafiamy/xtts-rs/issues · releases: https://github.com/rzafiamy/xtts-rs/releases

Used by [zallama](https://github.com/rzafiamy/zallama) as the `xtts-server`
backend.

## License

Code: MPL-2.0. The XTTS-v2 weights are under the
[Coqui Public Model License](https://coqui.ai/cpml) (non-commercial use);
GGUF files converted from them inherit it.
