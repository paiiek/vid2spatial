#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PROJ_ROOT="$REPO_ROOT/mmhoa/vid2spatial"
MANI_TAU="$PROJ_ROOT/data/tau/dev_train.jsonl"

OUT_DIR="$PROJ_ROOT/out"
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/doa_ckpt"

: "${WANDB_PROJECT:=vid2spatial}"
: "${WANDB_ENTITY:=paik402}"
: "${WANDB_NAME_DOA:=doa_crnn_main}"
: "${WANDB_NAME_MAP:=mapper_main}"
DOA_LIMIT=${DOA_LIMIT:-1000000}
MAP_LIMIT=${MAP_LIMIT:-500000}

echo "[run] DOA train -> $OUT_DIR/logs/train_doa.log"
WANDB_ENTITY="$WANDB_ENTITY" WANDB_PROJECT="$WANDB_PROJECT" nohup python3 -m mmhoa.vid2spatial.train_doa \
  --manifest "$MANI_TAU" --root "$REPO_ROOT/data" \
  --limit "$DOA_LIMIT" --epochs 10 --batch 16 --lr 3e-4 \
  --project "$WANDB_PROJECT" --name "$WANDB_NAME_DOA" \
  --save_dir "$OUT_DIR/doa_ckpt" \
  --tau_manifest "$MANI_TAU" --tau_root "$REPO_ROOT/data" --tau_limit 200 \
  > "$OUT_DIR/logs/train_doa.log" 2>&1 &

echo "[run] Mapper train -> $OUT_DIR/logs/train_mapper.log"
WANDB_ENTITY="$WANDB_ENTITY" WANDB_PROJECT="$WANDB_PROJECT" nohup python3 -m mmhoa.vid2spatial.train_mapper \
  --manifest "$MANI_TAU" --root "$REPO_ROOT/data" \
  --limit "$MAP_LIMIT" --epochs 5 --batch 16 --lr 3e-4 \
  --project "$WANDB_PROJECT" --name "$WANDB_NAME_MAP" \
  > "$OUT_DIR/logs/train_mapper.log" 2>&1 &

echo "Launched. Tail logs:"
echo "  tail -f $OUT_DIR/logs/train_doa.log"
echo "  tail -f $OUT_DIR/logs/train_mapper.log"
