#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import List, Optional


def run_cmd(cmd: List[str]) -> int:
    import subprocess
    print("$", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep vid2spatial rendering hyperparameters and render outputs")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--traj-json", type=Path, default=None, help="precomputed trajectory JSON")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--sofa", type=Path, default=None, help="SOFA HRIR path (.sofa)")
    ap.add_argument("--ang-smooth-ms", type=str, default="50,80")
    ap.add_argument("--max-deg-per-s", type=str, default="120,180")
    ap.add_argument("--dist-gain-k", type=str, default="1.0,1.3")
    ap.add_argument("--dist-lpf-min", type=str, default="600,800")
    ap.add_argument("--dist-lpf-max", type=str, default="8000,10000")
    ap.add_argument("--binaural-mode", type=str, default="crossfeed,sofa")
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    smooths = [float(x) for x in args.ang_smooth_ms.split(",") if x]
    maxdegs = [None if x=="none" else float(x) for x in args.max_deg_per_s.split(",") if x]
    gks = [float(x) for x in args.dist_gain_k.split(",") if x]
    lmins = [float(x) for x in args.dist_lpf_min.split(",") if x]
    lmaxs = [float(x) for x in args.dist_lpf_max.split(",") if x]
    modes = [m.strip() for m in args.binaural_mode.split(",") if m.strip()]

    report = []
    for sm, md, gk, lmin, lmax, mode in itertools.product(smooths, maxdegs, gks, lmins, lmaxs, modes):
        tag = f"s{sm}_m{md if md is not None else 'none'}_g{gk}_lp{int(lmin)}-{int(lmax)}_{mode}".replace(".", "p")
        outdir = args.out_root / tag
        outdir.mkdir(parents=True, exist_ok=True)
        out_foa = outdir / "out.foa.wav"
        out_bin = outdir / "out.bin.wav"
        cmd = [
            "python", "-m", "mmhoa.vid2spatial.run_demo",
            "--video", str(args.video),
            "--audio", str(args.audio),
            "--out_foa", str(out_foa),
            "--out_bin", str(out_bin),
            "--ang_smooth_ms", str(sm),
            "--dist_gain_k", str(gk),
            "--dist_lpf_min_hz", str(lmin),
            "--dist_lpf_max_hz", str(lmax),
        ]
        if md is not None:
            cmd += ["--max_deg_per_s", str(md)]
        if args.traj_json:
            cmd += ["--traj_json", str(args.traj_json)]
        if mode == "sofa" and args.sofa and args.sofa.exists():
            cmd += ["--binaural_mode", "sofa", "--sofa", str(args.sofa)]
        else:
            cmd += ["--binaural_mode", "crossfeed"]
        rc = run_cmd(cmd)
        report.append({
            "tag": tag,
            "rc": rc,
            "out_foa": str(out_foa),
            "out_bin": str(out_bin),
            "params": {
                "ang_smooth_ms": sm,
                "max_deg_per_s": md,
                "dist_gain_k": gk,
                "dist_lpf_min_hz": lmin,
                "dist_lpf_max_hz": lmax,
                "binaural_mode": mode,
                "sofa": str(args.sofa) if args.sofa else None,
            }
        })

    (args.out_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved:", args.out_root / "report.json")


if __name__ == "__main__":
    main()

