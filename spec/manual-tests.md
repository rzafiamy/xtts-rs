# Manual tests

Tests that need the Coqui reference, a GPU or an ASR model. Results of the
last run: [docs/performance.md](../docs/performance.md).

## MT-01 — Parity with coqui-tts (REQ-INF-001, REQ-INF-002)

Setup: coqui-tts 0.27.5 in a venv (see `scripts/ref.py`), the
`coqui/XTTS-v2` checkpoint in `<dir>`.

```bash
python scripts/ref.py <dir> /tmp/ref.safetensors --text "Bonjour à tous." --lang fr --voice "Claribel Dervla"
cargo run --release --example parity -- <dir> /tmp/ref.safetensors fr "Bonjour à tous." --cpu
```

Expected: `text ids: match`, `codes: match`, latents max |diff| ≤ 1e-4,
decoder SNR ≥ 70 dB. Last run (2026-10-03, also `de` "Die Katze schläft auf
dem Sofa."): codes match (106/106), 1.03e-4, 74.2 dB.

## MT-02 — Intelligibility (REQ-QUA-001, REQ-TXT-002, REQ-TXT-004)

Setup: `xtts serve` on port 18091, an OpenAI-compatible ASR endpoint with
Parakeet TDT v3 (zallama: `parakeet-tdt-v3-cpu`).

```bash
python3 scripts/wer.py --tts http://127.0.0.1:18091 --out /tmp/wer --seeds 3
```

Expected: WER ≤ 1 % in French and English, no extra word after the last
one. Last run (q4k, `--no-cloning`): fr 0.6 %, en 0.0 %.

## MT-03 — CUDA footprint and latency (REQ-GPU-001, REQ-SRV-002)

```bash
scripts/vram.sh models/xtts-v2-q4k.gguf                 # idle and peak VRAM
cargo run --release -p xtts --features cuda --example bench -- models/xtts-v2-q4k.gguf fr "Bonjour à tous, voici un test de latence."
```

and the `stream:` log lines of `xtts serve` ("first audio N ms").
Expected (RTX 4090): peak ≤ 1.1 GB, ≥ 10x real time, first audio ≤ 60 ms.
Last run: 776 / 1032 MiB, ~13x, 35-45 ms.
