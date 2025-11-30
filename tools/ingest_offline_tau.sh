#!/usr/bin/env bash
set -euo pipefail

SRC_DIR=${1:?"Usage: $0 /path/to/offline_tau_dir (must contain foa_dev.z01 and foa_dev.zip)"}
ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
ARCH="$ROOT_DIR/data/tau2021/_archives"
OUTDIR="$ROOT_DIR/data/tau2021/foa_dev"

mkdir -p "$ARCH" "$OUTDIR"

echo "[ingest_offline_tau] source: $SRC_DIR"
if [ ! -f "$SRC_DIR/foa_dev.z01" ] || [ ! -f "$SRC_DIR/foa_dev.zip" ]; then
  echo "Error: foa_dev.z01 or foa_dev.zip not found in $SRC_DIR" >&2
  exit 1
fi

cp -f "$SRC_DIR/foa_dev.z01" "$ARCH/foa_dev.z01"
cp -f "$SRC_DIR/foa_dev.zip" "$ARCH/foa_dev.zip"

# Prefer 7-Zip for true multi-part Zip64 support
SEVENZ_BIN="${ROOT_DIR}/external/bin/7zz"
if [ ! -x "$SEVENZ_BIN" ] && command -v 7zz >/dev/null 2>&1; then
  SEVENZ_BIN="$(command -v 7zz)"
fi

echo "[ingest_offline_tau] extracting foa_dev (using: ${SEVENZ_BIN##*/:-unzip})"
if [ -x "$SEVENZ_BIN" ]; then
  "$SEVENZ_BIN" x -y "$ARCH/foa_dev.zip" -o"$OUTDIR"
else
  # unzip can't handle Zip64 multi-part properly; will likely fail
  unzip -o "$ARCH/foa_dev.zip" -d "$OUTDIR" || {
    echo "[ingest_offline_tau] unzip failed; please install 7-Zip (7zz) and re-run." >&2
    exit 2
  }
fi

# Flatten dev-train/dev-*/ layout by linking WAVs into OUTDIR for legacy manifests
NESTED_ROOT="$OUTDIR/foa_dev"
if [ -d "$NESTED_ROOT/dev-train" ]; then
  echo "[ingest_offline_tau] flattening dev-train -> $OUTDIR (symlinks)"
  find "$NESTED_ROOT/dev-train" -maxdepth 1 -type f -name '*.wav' -print0 | while IFS= read -r -d '' f; do
    base="$(basename "$f")"
    rel="$(realpath --relative-to="$OUTDIR" "$f")"
    ln -sfn "$rel" "$OUTDIR/$base"
  done
fi

# Ensure legacy path alias data/TAU2021 -> data/tau2021
ln -sfn tau2021 "$ROOT_DIR/data/TAU2021"

# Optionally ingest metadata if provided alongside
for ZIP in metadata_dev.zip metadata_eval.zip; do
  if [ -f "$SRC_DIR/$ZIP" ]; then
    SUBDIR="${ZIP%.zip}"
    DEST="$ROOT_DIR/data/tau2021/$SUBDIR"
    mkdir -p "$DEST"
    echo "[ingest_offline_tau] extracting $ZIP -> $DEST"
    if [ -x "$SEVENZ_BIN" ]; then
      "$SEVENZ_BIN" x -y "$SRC_DIR/$ZIP" -o"$DEST"
    else
      unzip -o "$SRC_DIR/$ZIP" -d "$DEST"
    fi
  fi
done

echo "[ingest_offline_tau] done -> $OUTDIR"
