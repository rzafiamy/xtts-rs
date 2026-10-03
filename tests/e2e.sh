#!/usr/bin/env bash
# tests/e2e.sh — end-to-end check of the release binary: GGUF conversion,
# synthesis from the GGUF, and the HTTP server (/health, /v1/audio/speech,
# /stream, /v1/voices).
#
#   XTTS_CKPT=<coqui XTTS-v2 dir> tests/e2e.sh [binary]   converts, then tests the new GGUF
#   XTTS_MODEL=<model.gguf>       tests/e2e.sh [binary]   tests an existing GGUF
#
# binary: default target/release/xtts (build with ./build.sh or cargo build --release).
#
# covers: REQ-GGF-001, REQ-GGF-002, REQ-CLI-001, REQ-SRV-001, REQ-SRV-002
set -euo pipefail
cd "$(dirname "$0")/.."
BIN="${1:-target/release/xtts}"
[ -x "$BIN" ] || { echo "binary not found: $BIN (run ./build.sh or cargo build --release)" >&2; exit 1; }
WORK=$(mktemp -d)
PORT=${PORT:-18977}
SERVER=""
cleanup() { [ -n "$SERVER" ] && kill "$SERVER" 2>/dev/null; rm -rf "$WORK"; }
trap cleanup EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }
wav_seconds() { python3 -c "import wave,sys; w=wave.open(sys.argv[1]); print(w.getnframes()/w.getframerate())" "$1"; }
between() { python3 -c "import sys; s=float(sys.argv[1]); sys.exit(0 if float(sys.argv[2]) < s < float(sys.argv[3]) else 1)" "$@"; }

if [ -n "${XTTS_CKPT:-}" ]; then
  echo "== convert q4k --no-cloning"
  "$BIN" convert "$XTTS_CKPT" -o "$WORK/m.gguf" --gpt-dtype q4k --no-cloning >/dev/null || fail "convert"
  size=$(stat -c %s "$WORK/m.gguf")
  [ "$size" -lt 300000000 ] || fail "q4k file too large: $size bytes"
  MODEL="$WORK/m.gguf"
else
  MODEL="${XTTS_MODEL:?set XTTS_CKPT (checkpoint dir) or XTTS_MODEL (GGUF)}"
fi

echo "== speak"
"$BIN" speak -m "$MODEL" -l fr -v "Ana Florence" --seed 0 "Bonjour, ceci est un test de bout en bout." \
  -o "$WORK/a.wav" >/dev/null 2>&1 || fail "speak"
between "$(wav_seconds "$WORK/a.wav")" 1.0 6.0 || fail "speak: unexpected duration"

echo "== serve"
"$BIN" serve -m "$MODEL" --port "$PORT" --lang fr >"$WORK/server.log" 2>&1 &
SERVER=$!
for _ in $(seq 120); do curl -sf "localhost:$PORT/health" >/dev/null && break; sleep 0.5; done
curl -sf "localhost:$PORT/health" >/dev/null || { cat "$WORK/server.log"; fail "server did not start"; }

code=$(curl -s -o "$WORK/b.wav" -w '%{http_code}' "localhost:$PORT/v1/audio/speech" \
  -H 'content-type: application/json' \
  -d '{"input":"Le serveur répond.","voice":"claribel_dervla","language":"fr","seed":1}')
[ "$code" = 200 ] || fail "/v1/audio/speech returned $code"
between "$(wav_seconds "$WORK/b.wav")" 0.5 5.0 || fail "/v1/audio/speech: unexpected duration"

code=$(curl -s -o /dev/null -w '%{http_code}' "localhost:$PORT/v1/audio/speech" \
  -H 'content-type: application/json' -d '{"input":"Hello.","voice":"alloy"}')
[ "$code" = 400 ] || fail "unknown voice returned $code, expected 400"

bytes=$(curl -s "localhost:$PORT/stream" -H 'content-type: application/json' \
  -d '{"text":"Hello, streaming works.","language":"en"}' | wc -c)
[ "$bytes" -gt 24000 ] || fail "/stream returned $bytes bytes"

n=$(curl -s "localhost:$PORT/v1/voices" | python3 -c "import json,sys; print(len(json.load(sys.stdin)['voices']))")
[ "$n" = 58 ] || fail "/v1/voices lists $n voices"

echo "e2e OK"
