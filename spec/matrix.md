# Traceability matrix

Requirement ([specification.md](specification.md)) → feature → module → test.
Automated tests carry the ID in a `covers: REQ-…` comment; manual tests are
described in [manual-tests.md](manual-tests.md). Model tests
(`crates/xtts/tests/model.rs`) run with `XTTS_MODEL` set.

| ID | Requirement | Feature | Module | Test | Status | Notes |
|---|---|---|---|---|---|---|
| REQ-TXT-001 | XTTS cleaners, languages | Text | `crates/xtts/src/text.rs`, `text_tables.rs` | `tables_compile`, `cleaners`, `unsupported_language_is_an_error` | ✅ | tables generated from coqui-tts |
| REQ-TXT-002 | Number / Markdown normalization (fr, en) | Text | `text.rs` (`normalize`), `tn` crate ([tn-rs](https://github.com/rzafiamy/tn-rs), own tests) | `numbers_are_spelled_out`, manual MT-02 | ✅ | |
| REQ-TXT-003 | Sentence splitting | Text | `text.rs` (`split_sentences`) | `splitting` | ✅ | |
| REQ-TXT-004 | Final period dropped | Text | `text.rs` (`drop_final_period`), `model.rs` (`prepare_text`) | `final_period`, `short_sentence_stops_after_the_last_word`, manual MT-02 | ✅ | WER fr 1.1 → 0.6 %, en 0.5 → 0.0 % |
| REQ-SMP-001 | HF sampling order | Sampling | `crates/xtts/src/sampling.rs` | `penalty_and_greedy`, `top_k_one_is_greedy` | ✅ | |
| REQ-INF-001 | Parity with coqui-tts | Inference | `gpt.rs`, `hifigan.rs`, `model.rs` | manual MT-01 | ✅ | codes exact, 74-79 dB |
| REQ-INF-002 | Stop code | Inference | `model.rs` (`generate_codes`) | `short_sentence_stops_after_the_last_word`, manual MT-01 | ✅ | |
| REQ-INF-003 | Speed | Inference | `hifigan.rs` (`frames`), `model.rs` | `speed_changes_duration` | ✅ | |
| REQ-STR-001 | Streaming decoder | Streaming | `hifigan.rs` (`StreamDecoder`), `model.rs` (`synthesize_stream`) | `streamed_audio_matches_one_pass` | ✅ | |
| REQ-VOI-001 | Built-in voices | Voices | `gguf.rs` (`read_voices`), `model.rs` (`voice`) | `gguf_has_builtin_voices`, `tests/e2e.sh` | ✅ | 58 voices |
| REQ-GGF-001 | Single GGUF | GGUF | `crates/xtts/src/gguf.rs`, `crates/xtts-cli/src/main.rs` | `tests/e2e.sh`, `gguf_has_builtin_voices` | ✅ | |
| REQ-GGF-002 | Quantized, `--no-cloning` | GGUF | `gguf.rs` (`dtype_for`, `is_cloning`) | `tests/e2e.sh` | ✅ | 276 MB q4k |
| REQ-QUA-001 | No intelligibility loss | Quantization | `weights.rs` | manual MT-02 | ✅ | [docs/performance.md](../docs/performance.md) |
| REQ-GPU-001 | CUDA VRAM / speed / latency | GPU | `lib.rs` (`release_cached_memory`), `model.rs` | manual MT-03 | ✅ | 1032 MiB, 13x, 35-45 ms |
| REQ-CLI-001 | CLI speak / voices | CLI | `crates/xtts-cli/src/main.rs` | `tests/e2e.sh` | ✅ | |
| REQ-SRV-001 | HTTP API | Server | `crates/xtts-cli/src/server.rs` | `tests/e2e.sh` | ✅ | unknown voice → 400 |
| REQ-SRV-002 | `/stream` | Server | `server.rs` (`stream_response`) | `tests/e2e.sh`, manual MT-03 | ✅ | used by zallama `/v1/realtime` |

Updated: 2026-10-03 (v0.1.0).
