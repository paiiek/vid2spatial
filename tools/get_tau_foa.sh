#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
DATA_DIR="$ROOT_DIR/data/tau2021"
ARCH_DIR="$DATA_DIR/_archives"
mkdir -p "$ARCH_DIR" "$DATA_DIR"
cd "$ARCH_DIR"
# Multi-part zip
wget -c https://zenodo.org/api/records/5476980/files/foa_dev.z01/content -O foa_dev.z01
wget -c https://zenodo.org/api/records/5476980/files/foa_dev.zip/content -O foa_dev.zip
# Extract
mkdir -p "$DATA_DIR/foa_dev"
unzip -o -q foa_dev.zip -d "$DATA_DIR/foa_dev"
echo "[TAU2021] FOA dev extracted to $DATA_DIR/foa_dev"
