#!/usr/bin/env bash
set -euo pipefail

MANIFEST=${1:?manifest jsonl}
ROOT=${2:?dataset root}
OUT_DIR=${3:?output dir}
METHOD=${4:?yolo|sam2|kcf}
LIMIT=${5:-50}
STRIDE=${6:-8}
SAM2_ID=${7:-facebook/sam2.1-hiera-base-plus}

EXTRA_ARGS=()
if [ "$METHOD" = "sam2" ]; then
  EXTRA_ARGS+=(--sam2_model_id "$SAM2_ID")
fi

CMD=(python3 -m mmhoa.vid2spatial.tools.auto_fairplay \
  --manifest "$MANIFEST" --root "$ROOT" --out_dir "$OUT_DIR" \
  --limit "$LIMIT" --method "$METHOD" --stride "$STRIDE")
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi
"${CMD[@]}"

python3 -m mmhoa.vid2spatial.tools.eval_update_report --report "$OUT_DIR/report.json" --root "$ROOT" --overwrite

python3 -m mmhoa.vid2spatial.tools.aggregate_report --report "$OUT_DIR/report.json" --out_base "$OUT_DIR/report"

echo "[DONE] $METHOD limit=$LIMIT → $OUT_DIR"
