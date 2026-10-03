#!/usr/bin/env bash
# VRAM of `xtts serve` (idle after warmup, peak during one ~20 s request)
# and the request time. Usage: scripts/vram.sh <model.gguf> [extra serve args]
set -euo pipefail
BIN=${BIN:-target-cuda/release/xtts}
PORT=${PORT:-18091}
MODEL=$1; shift
LOG=$(mktemp)
"$BIN" serve -m "$MODEL" --port "$PORT" --lang fr "$@" >"$LOG" 2>&1 &
PID=$!
for _ in $(seq 120); do
  curl -sf "localhost:$PORT/health" >/dev/null && break
  kill -0 "$PID" 2>/dev/null || { cat "$LOG"; exit 1; }
  sleep 0.5
done
mem() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p="$PID" '$1==p{print $2}'; }
echo "idle: $(mem) MiB"
TEXT=$(python3 -c "print('Bonjour à tous. Aujourd hui, nous allons parler de la synthèse vocale en temps réel, et de la manière de réduire la latence sur une carte graphique. '*3)")
PEAKF=$(mktemp)
( while kill -0 "$PID" 2>/dev/null; do mem >>"$PEAKF"; sleep 0.1; done ) &
SAMPLER=$!
curl -s -o /tmp/xtts-vram.wav -w "request: %{http_code} %{time_total}s\n" "localhost:$PORT/v1/audio/speech" \
  -H 'content-type: application/json' -d "{\"input\":\"$TEXT\",\"voice\":\"Damien Black\"}"
kill "$PID"; wait "$PID" 2>/dev/null || true; kill "$SAMPLER" 2>/dev/null || true
echo "peak: $(sort -n "$PEAKF" | tail -1) MiB"
rm -f "$LOG" "$PEAKF"
