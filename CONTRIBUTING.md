# Contributing

1. `./setup.sh` (or `./setup.sh --cuda`) checks the toolchain and compiles.
2. Make the change; keep coqui-tts 0.27 (`TTS/tts/models/xtts.py`,
   `TTS/tts/layers/xtts/`) as the spec — see [AGENTS.md](AGENTS.md).
3. Before a pull request:
   ```bash
   cargo fmt --all
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --release --workspace
   XTTS_MODEL=$PWD/models/xtts-v2-q4k.gguf cargo test --release -p xtts --test model
   tests/e2e.sh target/release/xtts      # with XTTS_MODEL or XTTS_CKPT set
   ```
4. Model changes: greedy parity must stay exact
   (`scripts/ref.py` + `cargo run --release --example parity`), and
   `scripts/wer.py` must not get worse (see [docs/performance.md](docs/performance.md)).
5. New requirement → a `REQ-…` line in [spec/specification.md](spec/specification.md),
   a row in [spec/matrix.md](spec/matrix.md) and a test carrying `covers: REQ-…`.
6. Note user-visible changes under `[Unreleased]` in [CHANGELOG.md](CHANGELOG.md).

Commits follow Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:` …).
