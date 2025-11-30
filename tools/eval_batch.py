#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import librosa

from ..evaluate import seg_snr, log_spectral_distance, estimate_rt60


def load_audio_mono(path: Path, sr: int | None = None) -> Tuple[np.ndarray, int]:
    y, sra = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32), int(sra)


def eval_one(ref_mono: Path, bin_stereo: Path) -> Dict[str, float]:
    x, sr = load_audio_mono(ref_mono, sr=None)
    y, srb = sf.read(str(bin_stereo), always_2d=True)
    if srb != sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=srb, target_sr=sr, axis=0)
    y_mono = y.mean(axis=1).astype(np.float32)
    L = min(len(x), len(y_mono))
    x = x[:L]
    y_mono = y_mono[:L]
    try:
        lsd = float(log_spectral_distance(x, y_mono))
    except Exception:
        lsd = float('nan')
    try:
        snr = float(seg_snr(x, y_mono, sr=sr))
    except Exception:
        snr = float('nan')
    try:
        rt = float(estimate_rt60(y_mono, sr))
    except Exception:
        rt = float('nan')
    # spectral centroid (rough brightness proxy)
    try:
        cent = float(np.mean(librosa.feature.spectral_centroid(y=y_mono, sr=sr)))
    except Exception:
        cent = float('nan')
    return {"lsd": lsd, "seg_snr": snr, "rt60_est": rt, "centroid": cent}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate batch of binaural renders vs. input mono")
    ap.add_argument("--ref-audio", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True, help="report.json from sweep_spatial")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    items = json.loads(args.report.read_text(encoding="utf-8"))
    rows: List[Dict[str, object]] = []
    for it in items:
        binp = Path(it.get("out_bin"))
        if not binp.exists():
            continue
        m = eval_one(args.ref_audio, binp)
        rows.append({**it, **m})

    out = args.out or args.report.parent / "metrics.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("Saved:", out)


if __name__ == "__main__":
    main()
