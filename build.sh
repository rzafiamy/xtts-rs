#!/usr/bin/env bash
# build.sh — builds the release `xtts` binary into
# build/xtts-<os>-<arch>-<backend>-<version>[.exe]
# (Linux, macOS, Windows through Git Bash/MSYS2; x86_64 or aarch64).
#
#   ./build.sh            CPU
#   ./build.sh --cuda     NVIDIA GPU (needs nvcc; CUDA_COMPUTE_CAP=89 for an RTX 40xx, 80 A100, 90 H100)
#   ./build.sh --metal    Apple GPU
#
# Non-interactive (CI friendly); writes only to target/ (target-cuda/ for
# --cuda) and build/.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.cargo/bin:$PATH"
FEATURES=(); SUFFIX=cpu; TARGET_DIR=target
case "${1:-}" in
  --cuda)
    [ -d /usr/local/cuda/bin ] && export PATH="/usr/local/cuda/bin:$PATH"
    command -v nvcc >/dev/null || { echo "nvcc not found: install the CUDA Toolkit" >&2; exit 1; }
    if [ -z "${CUDA_COMPUTE_CAP:-}" ] && command -v nvidia-smi >/dev/null; then
      CUDA_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
      export CUDA_COMPUTE_CAP
    fi
    FEATURES=(--features cuda); SUFFIX=cuda; TARGET_DIR=target-cuda ;;
  --metal) FEATURES=(--features metal); SUFFIX=metal ;;
  "") ;;
  *) echo "unknown option: $1 (--cuda, --metal)" >&2; exit 1 ;;
esac
VERSION=$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)
cargo build --release --locked -p xtts-cli --target-dir "$TARGET_DIR" "${FEATURES[@]}"
case "$(uname -s)" in
  Linux) OS=linux; EXE= ;;
  Darwin) OS=macos; EXE= ;;
  MINGW*|MSYS*|CYGWIN*) OS=windows; EXE=.exe ;;
  *) OS=$(uname -s | tr '[:upper:]' '[:lower:]'); EXE= ;;
esac
ARCH=$(uname -m); [ "$ARCH" = arm64 ] && ARCH=aarch64
mkdir -p build
OUT="build/xtts-$OS-$ARCH-$SUFFIX-$VERSION$EXE"
cp "$TARGET_DIR/release/xtts$EXE" "$OUT"
"$OUT" --version
echo "artifact: $OUT"
