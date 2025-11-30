#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
MANIFEST="$ROOT_DIR/mmhoa/vid2spatial/data/fairplay/split1_train.jsonl"
ROOT_DATA="$ROOT_DIR/data" # contains FAIR-Play/{audios,videos}
OUT_DIR="$ROOT_DIR/mmhoa/vid2spatial/out/fairplay"

python3 -m mmhoa.vid2spatial.tools.dataset_report \
  --manifest "$MANIFEST" \
  --out "$OUT_DIR/_availability.json" \
  --sample 50

python3 -m mmhoa.vid2spatial.tools.auto_fairplay \
  --manifest "$MANIFEST" \
  --root "$ROOT_DATA" \
  --out_dir "$OUT_DIR" \
  --limit 10 \
  --method yolo \
  --stride 8

echo "[FAIR-Play] batch finished. See $OUT_DIR/report.json"

