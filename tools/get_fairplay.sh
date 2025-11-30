#!/usr/bin/env bash
set -euo pipefail

# Downloads FAIR-Play archives with resume and extracts into data/fairplay.
# Requires ~106 GB free space for videos and ~3.2 GB for audios.

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
DATA_DIR="$ROOT_DIR/data/fairplay"
ARCH_DIR="$DATA_DIR/_archives"
mkdir -p "$ARCH_DIR" "$DATA_DIR"

VID_URL="http://dl.fbaipublicfiles.com/FAIR-Play/videos.tar.gz"
AUD_URL="http://dl.fbaipublicfiles.com/FAIR-Play/audios.tar.gz"

echo "[FAIR-Play] downloading (resumable) to $ARCH_DIR"
cd "$ARCH_DIR"
wget -c "$VID_URL"
wget -c "$AUD_URL"

echo "[FAIR-Play] extracting videos..."
mkdir -p "$DATA_DIR/videos" "$DATA_DIR/audios"
tar -xzf videos.tar.gz -C "$DATA_DIR"
tar -xzf audios.tar.gz -C "$DATA_DIR"

echo "[FAIR-Play] done. Layout: $DATA_DIR/{videos,audios}"

