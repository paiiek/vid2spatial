"""
Baseline Systems for Comparison

Implements simple baseline methods for spatial audio generation:
1. Mono: No spatialization (duplicate mono to both channels)
2. SimplePan: Basic L-R panning based on horizontal position
3. RandomPan: Random panning (sanity check, should perform worst)

For academic comparison to show our full system's advantage.
"""
import sys
sys.path.insert(0, '/home/seung')

import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Dict, Tuple
import cv2
import json


def create_mono_baseline(mono_audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Baseline 1: Mono (no spatialization)

    Simply duplicates mono to both channels.
    This is the simplest possible baseline.

    Args:
        mono_audio: (N,) mono audio
        sr: sample rate

    Returns:
        binaural: (2, N) stereo with identical channels
    """
    return np.stack([mono_audio, mono_audio])


def create_simple_pan_baseline(
    mono_audio: np.ndarray,
    video_path: str,
    sr: int,
    init_bbox: Tuple[int, int, int, int] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Baseline 2: Simple Pan (basic L-R panning)

    Tracks object using KCF, applies simple stereo panning based on
    horizontal position. No depth, no IR, no distance effects.

    Pan law: Equal power panning
    - Left: -45° → full left (1.0, 0.0)
    - Center: 0° → center (0.707, 0.707)
    - Right: +45° → full right (0.0, 1.0)

    Args:
        mono_audio: (N,) mono audio
        video_path: path to video file
        sr: sample rate
        init_bbox: (x, y, w, h) initial bounding box

    Returns:
        binaural: (2, N) stereo audio
        trajectory: dict with tracking info
    """
    # Load video and initialize tracker
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Default init_bbox: center of frame
    if init_bbox is None:
        init_bbox = (width//2 - 40, height//2 - 60, 80, 120)

    # Initialize KCF tracker (legacy API compatibility)
    try:
        tracker = cv2.TrackerKCF_create()
    except AttributeError:
        # Newer OpenCV versions
        tracker = cv2.legacy.TrackerKCF_create()

    # Read first frame and initialize
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Cannot read first frame")

    tracker.init(frame, init_bbox)

    # Track through video
    trajectory = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Track
        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = bbox
            cx = x + w / 2.0
            cy = y + h / 2.0

            # Normalize horizontal position to [-1, 1]
            # -1 = left edge, 0 = center, +1 = right edge
            norm_x = (cx / width) * 2.0 - 1.0

            trajectory.append({
                'frame': frame_idx,
                'time_s': frame_idx / fps,
                'bbox': [float(x), float(y), float(w), float(h)],
                'center': [float(cx), float(cy)],
                'norm_x': float(norm_x)
            })
        else:
            # Lost tracking, use last known position
            if trajectory:
                last = trajectory[-1].copy()
                last['frame'] = frame_idx
                last['time_s'] = frame_idx / fps
                trajectory.append(last)

        frame_idx += 1

    cap.release()

    # Apply panning to audio
    audio_len = len(mono_audio)
    audio_dur = audio_len / sr

    # Interpolate trajectory to audio timeline
    traj_times = np.array([t['time_s'] for t in trajectory])
    traj_norm_x = np.array([t['norm_x'] for t in trajectory])

    audio_times = np.linspace(0, audio_dur, audio_len)
    norm_x_interp = np.interp(audio_times, traj_times, traj_norm_x)

    # Convert normalized position to pan angle
    # norm_x = -1 → -45°, 0 → 0°, +1 → +45°
    pan_angles = norm_x_interp * 45.0  # degrees

    # Apply equal-power panning
    pan_rad = np.deg2rad(pan_angles)

    # Equal power pan law (constant energy)
    # θ = 0 (center): L=R=0.707 (√2/2)
    # θ = -45° (left): L=1.0, R=0.0
    # θ = +45° (right): L=0.0, R=1.0

    # Map angle to pan position: -45° → 0, 0° → 0.5, +45° → 1.0
    pan_pos = (pan_angles + 45.0) / 90.0
    pan_pos = np.clip(pan_pos, 0.0, 1.0)

    # Equal power gains
    gain_R = np.sqrt(pan_pos)
    gain_L = np.sqrt(1.0 - pan_pos)

    # Apply gains
    left = mono_audio * gain_L
    right = mono_audio * gain_R

    binaural = np.stack([left, right])

    trajectory_dict = {
        'fps': fps,
        'total_frames': len(trajectory),
        'init_bbox': list(init_bbox),
        'frames': trajectory
    }

    return binaural, trajectory_dict


def create_random_pan_baseline(mono_audio: np.ndarray, sr: int, seed: int = 42) -> np.ndarray:
    """
    Baseline 3: Random Pan (sanity check)

    Random panning that changes over time. Should perform worst.
    Useful to verify evaluation metrics are working correctly.

    Args:
        mono_audio: (N,) mono audio
        sr: sample rate
        seed: random seed

    Returns:
        binaural: (2, N) stereo audio
    """
    np.random.seed(seed)

    audio_len = len(mono_audio)

    # Generate random pan positions at 10 Hz
    num_control_points = int(audio_len / sr * 10)
    random_pan = np.random.uniform(0.0, 1.0, num_control_points)

    # Interpolate to audio length
    control_times = np.linspace(0, audio_len - 1, num_control_points)
    audio_times = np.arange(audio_len)
    pan_pos = np.interp(audio_times, control_times, random_pan)

    # Apply equal power panning
    gain_R = np.sqrt(pan_pos)
    gain_L = np.sqrt(1.0 - pan_pos)

    left = mono_audio * gain_L
    right = mono_audio * gain_R

    return np.stack([left, right])


def evaluate_baseline_systems(
    fairplay_loader,
    num_samples: int = 20,
    output_dir: str = "baseline_eval"
):
    """
    Evaluate all baseline systems on FAIR-Play dataset.

    Args:
        fairplay_loader: FairPlayDataset instance
        num_samples: number of samples to evaluate
        output_dir: output directory
    """
    from evaluation_v2 import evaluate_spatial_audio_v2

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    baselines = ['mono', 'simple_pan', 'random_pan']
    results = {b: [] for b in baselines}

    for i in range(num_samples):
        sample = fairplay_loader.get_sample(i)
        sample_id = sample['sample_id']
        mono = sample['mono_audio']
        gt_binaural = sample['gt_binaural']
        sr = sample['sample_rate']
        video_path = sample['video_path']

        print(f"\n[{i+1}/{num_samples}] {sample_id}")

        # Get video dimensions for init_bbox
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        init_bbox = (width//2 - 40, height//2 - 60, 80, 120)

        # Baseline 1: Mono
        print(f"  [mono] ", end='', flush=True)
        mono_binaural = create_mono_baseline(mono, sr)

        # Convert to FOA for fair comparison
        W = (mono_binaural[0] + mono_binaural[1]) / 2.0
        Y = np.zeros_like(W)
        Z = np.zeros_like(W)
        X = np.zeros_like(W)
        mono_foa = np.stack([W, Y, Z, X])

        metrics_mono = evaluate_spatial_audio_v2(mono_foa, gt_binaural, sr)
        results['mono'].append(metrics_mono)
        print(f"Corr={metrics_mono['correlation_L']:.3f}")

        # Save
        sf.write(
            str(output_path / f"{sample_id}_mono.wav"),
            mono_binaural.T,
            sr
        )

        # Baseline 2: Simple Pan
        print(f"  [simple_pan] ", end='', flush=True)
        pan_binaural, trajectory = create_simple_pan_baseline(mono, video_path, sr, init_bbox)

        # Convert to FOA
        W = (pan_binaural[0] + pan_binaural[1]) / 2.0
        Y = (pan_binaural[0] - pan_binaural[1]) / 2.0  # Approximate Y from L-R
        Z = np.zeros_like(W)
        X = np.zeros_like(W)
        pan_foa = np.stack([W, Y, Z, X])

        metrics_pan = evaluate_spatial_audio_v2(pan_foa, gt_binaural, sr)
        results['simple_pan'].append(metrics_pan)
        print(f"Corr={metrics_pan['correlation_L']:.3f}")

        # Save
        sf.write(
            str(output_path / f"{sample_id}_simple_pan.wav"),
            pan_binaural.T,
            sr
        )
        with open(output_path / f"{sample_id}_simple_pan_traj.json", 'w') as f:
            json.dump(trajectory, f, indent=2)

        # Baseline 3: Random Pan
        print(f"  [random_pan] ", end='', flush=True)
        random_binaural = create_random_pan_baseline(mono, sr, seed=i)

        # Convert to FOA
        W = (random_binaural[0] + random_binaural[1]) / 2.0
        Y = (random_binaural[0] - random_binaural[1]) / 2.0
        Z = np.zeros_like(W)
        X = np.zeros_like(W)
        random_foa = np.stack([W, Y, Z, X])

        metrics_random = evaluate_spatial_audio_v2(random_foa, gt_binaural, sr)
        results['random_pan'].append(metrics_random)
        print(f"Corr={metrics_random['correlation_L']:.3f}")

        # Save
        sf.write(
            str(output_path / f"{sample_id}_random_pan.wav"),
            random_binaural.T,
            sr
        )

    # Aggregate results
    print("\n" + "="*60)
    print("BASELINE EVALUATION SUMMARY")
    print("="*60)

    summary = {}
    for baseline in baselines:
        # Compute means and stds
        metrics_list = results[baseline]

        # Get all metric names from first sample
        metric_names = list(metrics_list[0].keys())

        baseline_summary = {}
        for metric_name in metric_names:
            values = [m[metric_name] for m in metrics_list]
            baseline_summary[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }

        summary[baseline] = {
            'num_samples': len(metrics_list),
            'metrics': baseline_summary
        }

        # Print key metrics
        print(f"\n{baseline.upper()}")
        print(f"  Correlation L: {baseline_summary['correlation_L']['mean']:.3f} ± {baseline_summary['correlation_L']['std']:.3f}")
        print(f"  Correlation R: {baseline_summary['correlation_R']['mean']:.3f} ± {baseline_summary['correlation_R']['std']:.3f}")
        print(f"  ILD Error:     {baseline_summary['ild_error_db']['mean']:.2f} ± {baseline_summary['ild_error_db']['std']:.2f} dB")
        print(f"  SI-SDR L:      {baseline_summary['si_sdr_L']['mean']:.2f} ± {baseline_summary['si_sdr_L']['std']:.2f} dB")

    # Save summary
    with open(output_path / 'baseline_summary.json', 'w') as f:
        json.dump({
            'summary': summary,
            'all_results': results
        }, f, indent=2)

    print(f"\nResults saved to {output_path}/")

    return summary, results


if __name__ == '__main__':
    import argparse
    from fairplay_loader import FairPlayDataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='baseline_eval')
    args = parser.parse_args()

    loader = FairPlayDataset()
    print(f"Loaded {len(loader)} samples from FAIR-Play dataset")

    summary, results = evaluate_baseline_systems(
        loader,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
