#!/usr/bin/env bash
set -euo pipefail

WHEEL_DIR=${1:?"Usage: $0 /path/to/wheels_dir (contains *.whl)"}
shift || true
EXTRA_ARGS=($@)

if ls "$WHEEL_DIR"/*.whl >/dev/null 2>&1; then
  echo "[install_wheels] installing from $WHEEL_DIR"
  python3 -m pip install --no-index --find-links "$WHEEL_DIR" "${EXTRA_ARGS[@]}" || true
else
  echo "[install_wheels] no wheels found in $WHEEL_DIR"
fi

