# HTTP server

```bash
xtts serve -m models/xtts-v2-q4k.gguf --lang fr --port 8091
```

The listener binds after the model is loaded and a warmup sentence was
spoken (`--no-warmup` skips it), so `GET /health` answering means ready. One
synthesis runs at a time; cached CUDA memory is returned to the driver after
each request.

## `POST /v1/audio/speech`

OpenAI's shape plus `language`:

| Field | Default | Notes |
|---|---|---|
| `input` (or `text`) | — | text to speak |
| `voice` | `--voice` | one of `GET /v1/voices`; case, `_` and `-` ignored |
| `language` | `--lang` | en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, hu, hi (`fr-FR` → `fr`) |
| `speed` | 1.0 | 0.25-4 |
| `response_format` | `wav` | `wav` or `pcm` (16-bit LE mono, 24 kHz) |
| `stream` | false | send audio as it is decoded (chunked; WAV header with unknown size) |
| `temperature`, `top_k`, `top_p`, `repetition_penalty`, `seed` | server's | sampling |

Errors: `400` with `{"error": {"message": …}}` (empty input, unknown voice
or language, bad format).

```bash
curl -s localhost:8091/v1/audio/speech -H 'content-type: application/json' \
  -d '{"input":"Bonjour à tous.","voice":"Ana Florence","language":"fr"}' -o out.wav
```

## `POST /stream`

pocket-tts-server's streaming contract, used by zallama's `/v1/realtime`:
`{text, voice, language}` in, raw PCM16 at 24 kHz out as it is decoded.
Generation stops when the client disconnects.

## Other routes

- `GET /v1/voices` → `{"voices": [...], "default": "..."}`
- `GET /v1/models` → `{"data": [{"id": <--model-id>}]}`
- `GET /health` → `{"status": "ok"}`
