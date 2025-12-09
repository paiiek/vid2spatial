"""
Multi-source spatial audio demo.

Demonstrates processing video with multiple sound sources.

Usage:
    python3 scripts/demo_multisource.py --video INPUT.mp4 --audio1 SRC1.wav --audio2 SRC2.wav --output OUTPUT.foa.wav
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import soundfile as sf

from vid2spatial_pkg.multi_source import process_multi_source_video


def main():
    parser = argparse.ArgumentParser(description="Multi-source spatial audio demo")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--audio1", required=True, help="Audio source 1 (mono WAV)")
    parser.add_argument("--audio2", required=True, help="Audio source 2 (mono WAV)")
    parser.add_argument("--audio3", default=None, help="Audio source 3 (optional, mono WAV)")
    parser.add_argument("--output", default="multi_source.foa.wav", help="Output FOA file")
    parser.add_argument("--fov", type=float, default=60.0, help="Camera FOV (degrees)")
    args = parser.parse_args()

    print("Multi-Source Spatial Audio Demo")
    print("="*60)

    # Load audio sources
    audio_sources = []

    print(f"\nLoading audio source 1: {args.audio1}")
    audio1, sr1 = sf.read(args.audio1, always_2d=False)
    if audio1.ndim > 1:
        audio1 = audio1[:, 0]  # Take first channel if stereo
    audio_sources.append(audio1)
    sr = sr1

    print(f"Loading audio source 2: {args.audio2}")
    audio2, sr2 = sf.read(args.audio2, always_2d=False)
    if audio2.ndim > 1:
        audio2 = audio2[:, 0]
    if sr2 != sr:
        print(f"[warn] Sample rate mismatch: {sr2} vs {sr}, using {sr}")
    audio_sources.append(audio2)

    if args.audio3:
        print(f"Loading audio source 3: {args.audio3}")
        audio3, sr3 = sf.read(args.audio3, always_2d=False)
        if audio3.ndim > 1:
            audio3 = audio3[:, 0]
        if sr3 != sr:
            print(f"[warn] Sample rate mismatch: {sr3} vs {sr}, using {sr}")
        audio_sources.append(audio3)

    print(f"\nLoaded {len(audio_sources)} audio sources")
    for i, audio in enumerate(audio_sources):
        print(f"  Source {i+1}: {len(audio)} samples ({len(audio)/sr:.2f}s)")

    # Process multi-source
    result = process_multi_source_video(
        video_path=args.video,
        audio_sources=audio_sources,
        sr=sr,
        num_sources=len(audio_sources),
        fov_deg=args.fov,
        output_path=args.output
    )

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Tracked {result['num_sources']} sources")
    print(f"Duration: {result['duration_sec']:.2f}s")
    print(f"Output: {result['output_path']}")

    for i, traj in enumerate(result['trajectories']):
        print(f"\nSource {i+1}:")
        print(f"  Track ID: {traj['track_id']}")
        print(f"  Frames: {len(traj['frames'])}")

    print("\n✅ Multi-source processing complete!")


if __name__ == "__main__":
    main()
