#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
LOG_DIR="$ROOT_DIR/mmhoa/vid2spatial/out/logs"
mkdir -p "$LOG_DIR"

echo "[1/5] FAIR-Play download+extract" | tee -a "$LOG_DIR/run_all.log"
bash "$ROOT_DIR/mmhoa/vid2spatial/tools/get_fairplay.sh" 2>&1 | tee "$LOG_DIR/01_get_fairplay.log"

echo "[2/5] FAIR-Play batch (availability + 10 samples)" | tee -a "$LOG_DIR/run_all.log"
bash "$ROOT_DIR/mmhoa/vid2spatial/tools/run_fairplay_batch.sh" 2>&1 | tee "$LOG_DIR/02_fairplay_batch.log"

echo "[3/5] TAU 2021 metadata download" | tee -a "$LOG_DIR/run_all.log"
bash "$ROOT_DIR/mmhoa/vid2spatial/tools/get_tau.sh" 2>&1 | tee "$LOG_DIR/03_get_tau.log"

echo "[4/5] TAU 2021 manifest from metadata (dev-train)" | tee -a "$LOG_DIR/run_all.log"
python3 -m mmhoa.vid2spatial.tools.tau_manifest_from_meta \
  --meta_dir "$ROOT_DIR/data/tau2021/metadata_dev" \
  --split dev-train \
  --out "$ROOT_DIR/mmhoa/vid2spatial/data/tau/dev_train.jsonl" \
  --audio_prefix "data/tau2021/foa_dev" 2>&1 | tee "$LOG_DIR/04_tau_manifest.log"

echo "[5/5] SAM2 install (editable) + benchmark" | tee -a "$LOG_DIR/run_all.log"
(
  set -e
  cd "$ROOT_DIR"
  # Install SAM2 if not present
  python3 - << 'PY'
import importlib, sys
try:
    importlib.import_module('sam2')
    print('sam2 already installed')
except Exception:
    print('sam2 not found')
PY
  if ! python3 - << 'PY'
import importlib
import sys
try:
    importlib.import_module('sam2'); sys.exit(0)
except Exception:
    sys.exit(1)
PY
  then
    echo "Installing SAM2 from GitHub..."
    python3 -m pip install --break-system-packages 'git+https://github.com/facebookresearch/sam2.git'
  fi
  # Run benchmark
  python3 -m mmhoa.vid2spatial.tools.benchmark \
    --video "$ROOT_DIR/mmhoa/vid/synth/move_rb.mp4" \
    --methods kcf yolo sam2 \
    --stride 8 \
    --cls none
) 2>&1 | tee "$LOG_DIR/05_sam2_benchmark.log"

echo "[DONE] See logs under $LOG_DIR"

