#!/usr/bin/env bash
set -euo pipefail

# Downloads TAU-NIGENS Spatial Sound Events 2021 metadata and (optionally) FOA audio.
# FOA dev is multi-part zip (foa_dev.z01 + foa_dev.zip). Ensure enough disk space.

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
DATA_DIR="$ROOT_DIR/data/tau2021"
ARCH_DIR="$DATA_DIR/_archives"
mkdir -p "$ARCH_DIR" "$DATA_DIR"

REC_API="https://zenodo.org/api/records/5476980"

echo "[TAU2021] fetching file list..."
python3 - "$REC_API" << 'PY'
import sys, json, urllib.request, os
api=sys.argv[1]
d=json.loads(urllib.request.urlopen(api, timeout=20).read().decode('utf-8'))
files=d['files']
entries=files if isinstance(files,list) else files.get('entries',[])
urls={ f['key']: f['links']['self'] for f in entries }
print('\n'.join(f"{k} {v}" for k,v in urls.items()))
PY

echo "[TAU2021] downloading metadata_dev.zip"
cd "$ARCH_DIR"
wget -c "https://zenodo.org/api/records/5476980/files/metadata_dev.zip/content" -O metadata_dev.zip
unzip -o -q metadata_dev.zip -d "$DATA_DIR/metadata_dev"

echo "[TAU2021] To download FOA dev, run the following (uncomment):"
cat << 'EOF'
# wget -c https://zenodo.org/api/records/5476980/files/foa_dev.z01/content -O foa_dev.z01
# wget -c https://zenodo.org/api/records/5476980/files/foa_dev.zip/content -O foa_dev.zip
# unzip -o foa_dev.zip -d ../foa_dev
EOF

echo "[TAU2021] metadata ready at $DATA_DIR/metadata_dev"

