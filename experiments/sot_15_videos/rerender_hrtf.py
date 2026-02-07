#!/usr/bin/env python3
"""
Re-render all 30 SOT binaural files using HRTF (KEMAR SOFA).
Also re-renders the 15 original pipeline outputs.

Replaces the simple crossfeed with proper HRTF convolution via KEMAR.

Date: 2026-02-07
"""

import sys, os, json, time
sys.path.insert(0, "/home/seung/mmhoa/vid2spatial")

import numpy as np
import soundfile as sf
from pathlib import Path

from vid2spatial_pkg.foa_render import (
    render_foa_from_trajectory,
    foa_to_binaural_sofa,
    foa_to_binaural,
)

SOFA_PATH = "/home/seung/mmhoa/text2hoa/renderer/hrtf/kemar.sofa"
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"

# HRTF output directories
HRTF_A_DIR = BASE_DIR / "render_A_instrument_hrtf"
HRTF_B_DIR = BASE_DIR / "render_B_foley_hrtf"
HRTF_ORIG_DIR = BASE_DIR / "render_orig_hrtf"

for d in [HRTF_A_DIR, HRTF_B_DIR, HRTF_ORIG_DIR]:
    d.mkdir(exist_ok=True, parents=True)


def render_hrtf_binaural(audio_path: str, trajectory_path: str, output_path: str):
    """Render FOA then decode to binaural with KEMAR HRTF."""
    with open(trajectory_path) as f:
        traj_data = json.load(f)

    if "frames" in traj_data:
        frames = traj_data["frames"]
        fps = traj_data.get("fps", 30.0)
    else:
        frames = traj_data
        fps = 30.0

    foa_path = output_path.replace("_binaural.wav", "_foa.wav")

    # Render FOA with depth smoothing fix + HRTF binaural
    result = render_foa_from_trajectory(
        audio_path=audio_path,
        trajectory={"frames": frames, "fps": fps},
        output_path=foa_path,
        smooth_ms=30.0,
        apply_reverb=True,
        rt60=0.5,
        output_stereo=False,  # we'll do binaural separately with HRTF
        sofa_path=SOFA_PATH,
    )

    # Load FOA and decode with HRTF
    foa_data, foa_sr = sf.read(foa_path, dtype="float32")
    foa_data = foa_data.T  # (4, T)

    binaural = foa_to_binaural_sofa(foa_data, foa_sr, SOFA_PATH)
    sf.write(output_path, binaural.T, foa_sr, subtype="FLOAT")

    # Cleanup intermediate FOA
    if os.path.exists(foa_path):
        os.remove(foa_path)

    return output_path


def main():
    print("=" * 70)
    print("  Re-render with HRTF (KEMAR SOFA)")
    print(f"  SOFA: {SOFA_PATH}")
    print("=" * 70)

    # Verify SOFA loads
    import h5py
    with h5py.File(SOFA_PATH, 'r') as sofa:
        n_meas = sofa['Data.IR'].shape[0]
        fs = float(sofa['Data.SamplingRate'][0])
    print(f"  KEMAR: {n_meas} measurements, {fs:.0f} Hz\n")

    scenarios = sorted([d.name for d in OUTPUT_DIR.iterdir() if d.is_dir()])
    print(f"  Found {len(scenarios)} scenarios\n")

    # ============================
    # Part 1: Original pipeline audio → HRTF binaural
    # ============================
    print("--- Part 1: Original Pipeline Audio → HRTF ---\n")
    ok_orig = 0
    for vid_id in scenarios:
        traj_path = str(OUTPUT_DIR / vid_id / "trajectory_3d.json")
        audio_path = str(OUTPUT_DIR / vid_id / "audio_mono.wav")

        if not os.path.exists(traj_path) or not os.path.exists(audio_path):
            print(f"  [{vid_id}] SKIP")
            continue

        out_path = str(HRTF_ORIG_DIR / f"{vid_id}_hrtf_binaural.wav")
        print(f"  [{vid_id}]...", end=" ", flush=True)
        t0 = time.time()
        try:
            render_hrtf_binaural(audio_path, traj_path, out_path)
            dt = time.time() - t0
            print(f"OK ({dt:.1f}s)")
            ok_orig += 1
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\n  Part 1: {ok_orig}/{len(scenarios)}\n")

    # ============================
    # Part 2: Instrument audio → HRTF binaural
    # ============================
    print("--- Part 2: Instrument Audio → HRTF ---\n")
    inst_dir = BASE_DIR / "audio_instrument"
    ok_a = 0
    for vid_id in scenarios:
        traj_path = str(OUTPUT_DIR / vid_id / "trajectory_3d.json")
        if not os.path.exists(traj_path):
            continue

        # Find instrument audio
        inst_files = list(inst_dir.glob(f"{vid_id}_*.wav"))
        if not inst_files:
            print(f"  [{vid_id}] SKIP - no instrument audio")
            continue

        audio_path = str(inst_files[0])
        inst_name = inst_files[0].stem.replace(f"{vid_id}_", "")
        out_path = str(HRTF_A_DIR / f"{vid_id}_{inst_name}_hrtf_binaural.wav")

        print(f"  [{vid_id}] {inst_name}...", end=" ", flush=True)
        t0 = time.time()
        try:
            render_hrtf_binaural(audio_path, traj_path, out_path)
            dt = time.time() - t0
            print(f"OK ({dt:.1f}s)")
            ok_a += 1
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\n  Part 2: {ok_a}/{len(scenarios)}\n")

    # ============================
    # Part 3: Foley audio → HRTF binaural
    # ============================
    print("--- Part 3: Foley Audio → HRTF ---\n")
    foley_dir = BASE_DIR / "audio_foley"
    ok_b = 0
    for vid_id in scenarios:
        traj_path = str(OUTPUT_DIR / vid_id / "trajectory_3d.json")
        if not os.path.exists(traj_path):
            continue

        # Find foley audio
        foley_files = list(foley_dir.glob(f"{vid_id}_*.wav"))
        if not foley_files:
            print(f"  [{vid_id}] SKIP - no foley audio")
            continue

        audio_path = str(foley_files[0])
        foley_name = foley_files[0].stem.replace(f"{vid_id}_", "")
        out_path = str(HRTF_B_DIR / f"{vid_id}_{foley_name}_hrtf_binaural.wav")

        print(f"  [{vid_id}] {foley_name}...", end=" ", flush=True)
        t0 = time.time()
        try:
            render_hrtf_binaural(audio_path, traj_path, out_path)
            dt = time.time() - t0
            print(f"OK ({dt:.1f}s)")
            ok_b += 1
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\n  Part 3: {ok_b}/{len(scenarios)}\n")

    # Summary
    total = ok_orig + ok_a + ok_b
    print("=" * 70)
    print("  HRTF Re-render Summary")
    print("=" * 70)
    print(f"  Original pipeline audio:  {ok_orig}/15  → {HRTF_ORIG_DIR}")
    print(f"  Instrument audio:         {ok_a}/15  → {HRTF_A_DIR}")
    print(f"  Foley audio:              {ok_b}/15  → {HRTF_B_DIR}")
    print(f"  Total:                    {total}/45")
    print("=" * 70)


if __name__ == "__main__":
    main()
