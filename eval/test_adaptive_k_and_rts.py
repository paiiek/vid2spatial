#!/usr/bin/env python3
"""
Test Adaptive K and RTS Smoother.

1. Adaptive K: motion-based triggered re-detection
2. RTS Smoother: optimal backward-pass smoothing
"""

import sys
sys.path.insert(0, "/home/seung/mmhoa/vid2spatial")

import os
import time
import json
import numpy as np
from pathlib import Path
import torch

from vid2spatial_pkg.hybrid_tracker import HybridTracker
from vid2spatial_pkg.trajectory_stabilizer import rts_smooth_trajectory
from vid2spatial_pkg.vision import CameraIntrinsics, pixel_to_ray, ray_to_angles


def compute_metrics(frames, gt_cx=None, width=640, height=480, fov_deg=60.0):
    """Compute trajectory quality metrics."""
    if len(frames) < 3:
        return None

    K = CameraIntrinsics(width, height, fov_deg)

    cx = np.array([f.center[0] for f in frames])
    cy = np.array([f.center[1] for f in frames])
    conf = np.array([f.confidence for f in frames])

    az_list, el_list = [], []
    for f in frames:
        ray = pixel_to_ray(f.center[0], f.center[1], K)
        az, el = ray_to_angles(ray)
        az_list.append(np.degrees(az))
        el_list.append(np.degrees(el))

    az = np.array(az_list)
    el = np.array(el_list)

    # Jerk
    if len(az) > 3:
        az_jerk = np.abs(np.diff(np.diff(np.diff(az))))
        el_jerk = np.abs(np.diff(np.diff(np.diff(el))))
        jerk_avg = (np.mean(az_jerk) + np.mean(el_jerk)) / 2
    else:
        jerk_avg = 0

    # Direction changes
    if len(az) > 2:
        dir_changes_az = int(np.sum(np.diff(np.sign(np.diff(az))) != 0))
    else:
        dir_changes_az = 0

    # Amplitude
    amp_x = (np.max(cx) - np.min(cx)) / 2

    # GT metrics if provided
    result = {
        "jerk_avg": float(jerk_avg),
        "dir_changes_az": dir_changes_az,
        "amplitude_x_px": float(amp_x),
        "num_frames": len(frames),
    }

    if gt_cx is not None:
        min_len = min(len(cx), len(gt_cx))
        mae = np.mean(np.abs(cx[:min_len] - gt_cx[:min_len]))
        gt_vel = np.diff(gt_cx[:min_len])
        pred_vel = np.diff(cx[:min_len])
        vel_corr = np.corrcoef(gt_vel, pred_vel)[0, 1] if len(gt_vel) > 1 else 0

        gt_amp = (np.max(gt_cx) - np.min(gt_cx)) / 2
        result["mae_px"] = float(mae)
        result["velocity_correlation"] = float(vel_corr)
        result["amplitude_ratio"] = float(amp_x / gt_amp) if gt_amp > 0 else 0

    return result


def test_adaptive_k_synthetic():
    """Test Adaptive K on synthetic oscillating video."""
    print("=" * 70)
    print("TEST 1: Adaptive K on Synthetic Oscillating (0.6 Hz)")
    print("=" * 70)

    video_path = "/home/seung/mmhoa/vid2spatial/eval/test_outputs/synthetic/oscillating.mp4"
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return None

    # Ground truth
    num_frames = 150
    fps = 30.0
    freq = 0.6
    width = 640
    cx_gt = np.array([width * (0.5 + 0.35 * np.sin(2 * np.pi * freq * i / fps)) for i in range(num_frames)])

    results = {}

    methods = [
        ("redetect_k5", "redetect", {"redetect_interval": 5}),
        ("adaptive_k", "adaptive_k", {}),
    ]

    for name, method, params in methods:
        print(f"\n[{name}]")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tracker = HybridTracker(device="cuda", scene_type="auto", box_threshold=0.15)

        start = time.time()
        result = tracker.track(
            video_path=video_path,
            text_prompt="orange circle",
            tracking_method=method,
            estimate_depth=False,
            **params
        )
        elapsed = time.time() - start

        cx_pred = np.array([f.center[0] for f in result.frames])
        m = compute_metrics(result.frames, cx_gt)
        m["fps"] = len(result.frames) / elapsed
        m["time_sec"] = elapsed

        print(f"  Amplitude ratio: {m.get('amplitude_ratio', 0)*100:.1f}%")
        print(f"  MAE: {m.get('mae_px', 0):.1f}px")
        print(f"  Velocity correlation: {m.get('velocity_correlation', 0):.3f}")
        print(f"  Jerk avg: {m['jerk_avg']:.4f}")
        print(f"  Dir changes (Az): {m['dir_changes_az']}")
        print(f"  FPS: {m['fps']:.1f}")

        results[name] = m

    return results


def test_rts_smoother():
    """Test RTS smoother on a tracked trajectory."""
    print("\n" + "=" * 70)
    print("TEST 2: RTS Smoother on Real Video (marker_hd)")
    print("=" * 70)

    video_path = "/home/seung/mmhoa/vid2spatial/test_videos/marker_hd.mp4"
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return None

    tracker = HybridTracker(device="cuda", scene_type="auto", box_threshold=0.15)

    # Track with redetect mode
    result = tracker.track(
        video_path=video_path,
        text_prompt="colored marker",
        tracking_method="redetect",
        redetect_interval=5,
        estimate_depth=True,
        depth_stride=5,
    )

    # Get trajectory
    traj_3d = result.get_trajectory_3d(smooth=False)
    frames_raw = traj_3d["frames"]

    print(f"\n[Raw trajectory] {len(frames_raw)} frames")

    # Apply RTS smoother
    frames_rts = rts_smooth_trajectory(
        frames_raw,
        process_noise=0.01,
        measurement_noise=0.1,
    )

    print(f"[RTS smoothed] {len(frames_rts)} frames")

    # Compare jerk
    def compute_jerk(frames, key='az'):
        vals = np.array([f[key] for f in frames])
        if len(vals) > 3:
            jerk = np.abs(np.diff(np.diff(np.diff(vals))))
            return np.mean(jerk), np.max(jerk)
        return 0, 0

    raw_jerk_avg, raw_jerk_max = compute_jerk(frames_raw, 'az')
    rts_jerk_avg, rts_jerk_max = compute_jerk(frames_rts, 'az')

    print(f"\n  Jerk (az) - Raw:  avg={raw_jerk_avg:.6f}, max={raw_jerk_max:.6f}")
    print(f"  Jerk (az) - RTS:  avg={rts_jerk_avg:.6f}, max={rts_jerk_max:.6f}")
    print(f"  Improvement: {(1 - rts_jerk_avg/raw_jerk_avg)*100:.1f}% reduction")

    # Direction changes
    def count_dir_changes(frames, key='az'):
        vals = np.array([f[key] for f in frames])
        if len(vals) > 2:
            return int(np.sum(np.diff(np.sign(np.diff(vals))) != 0))
        return 0

    raw_dir = count_dir_changes(frames_raw, 'az')
    rts_dir = count_dir_changes(frames_rts, 'az')

    print(f"\n  Dir changes (az) - Raw: {raw_dir}")
    print(f"  Dir changes (az) - RTS: {rts_dir}")

    return {
        "raw_jerk_avg": raw_jerk_avg,
        "rts_jerk_avg": rts_jerk_avg,
        "jerk_reduction_pct": (1 - rts_jerk_avg/raw_jerk_avg)*100 if raw_jerk_avg > 0 else 0,
        "raw_dir_changes": raw_dir,
        "rts_dir_changes": rts_dir,
    }


def test_adaptive_k_real_videos():
    """Test Adaptive K on real videos."""
    print("\n" + "=" * 70)
    print("TEST 3: Adaptive K on Real Videos")
    print("=" * 70)

    test_videos = [
        ("marker_hd", "/home/seung/mmhoa/vid2spatial/test_videos/marker_hd.mp4", "colored marker"),
        ("daw_hd", "/home/seung/mmhoa/vid2spatial/test_videos/daw_hd.mp4", "colored marker"),
    ]

    # Add benchmark videos
    benchmark_dir = "/home/seung/mmhoa/vid2spatial/test_videos/benchmark"
    if os.path.exists(benchmark_dir):
        for f in sorted(os.listdir(benchmark_dir))[:3]:
            if f.endswith(".mp4"):
                test_videos.append((f.replace(".mp4", ""), os.path.join(benchmark_dir, f), "person"))

    results = {}

    for name, path, prompt in test_videos:
        if not os.path.exists(path):
            continue

        print(f"\n--- {name} ---")
        results[name] = {}

        for method_name, method, params in [
            ("redetect_k5", "redetect", {"redetect_interval": 5}),
            ("adaptive_k", "adaptive_k", {}),
        ]:
            print(f"  [{method_name}]", end=" ", flush=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            tracker = HybridTracker(device="cuda", scene_type="auto", box_threshold=0.15)

            try:
                start = time.time()
                result = tracker.track(
                    video_path=path,
                    text_prompt=prompt,
                    tracking_method=method,
                    estimate_depth=False,
                    **params
                )
                elapsed = time.time() - start

                m = compute_metrics(result.frames)
                m["fps"] = len(result.frames) / elapsed

                print(f"Jerk={m['jerk_avg']:.4f}, DirChg={m['dir_changes_az']}, FPS={m['fps']:.1f}")
                results[name][method_name] = m

            except Exception as e:
                print(f"ERROR: {e}")

    return results


def main():
    output_dir = Path("/home/seung/mmhoa/vid2spatial/eval/comprehensive_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: Adaptive K on synthetic
    adaptive_synthetic = test_adaptive_k_synthetic()

    # Test 2: RTS smoother
    rts_results = test_rts_smoother()

    # Test 3: Adaptive K on real videos
    adaptive_real = test_adaptive_k_real_videos()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if adaptive_synthetic:
        print("\n### Adaptive K vs Fixed K=5 (Synthetic)")
        print(f"\n{'Method':<15} {'Amp%':>8} {'MAE(px)':>10} {'VelCorr':>10} {'Jerk':>10} {'FPS':>8}")
        print("-" * 65)
        for method, m in adaptive_synthetic.items():
            print(f"{method:<15} {m.get('amplitude_ratio', 0)*100:>7.1f}% {m.get('mae_px', 0):>10.1f} "
                  f"{m.get('velocity_correlation', 0):>10.3f} {m['jerk_avg']:>10.4f} {m['fps']:>8.1f}")

    if rts_results:
        print("\n### RTS Smoother Effect")
        print(f"  Jerk reduction: {rts_results['jerk_reduction_pct']:.1f}%")
        print(f"  Dir changes: {rts_results['raw_dir_changes']} → {rts_results['rts_dir_changes']}")

    if adaptive_real:
        print("\n### Adaptive K vs Fixed K=5 (Real Videos)")
        print(f"\n{'Video':<20} {'K=5 Jerk':>12} {'Adaptive Jerk':>15} {'Winner':>10}")
        print("-" * 60)

        for video, methods in adaptive_real.items():
            k5_jerk = methods.get("redetect_k5", {}).get("jerk_avg", float('inf'))
            adap_jerk = methods.get("adaptive_k", {}).get("jerk_avg", float('inf'))
            winner = "adaptive" if adap_jerk < k5_jerk else "k5"
            print(f"{video:<20} {k5_jerk:>12.4f} {adap_jerk:>15.4f} {winner:>10}")

    # Save results
    all_results = {
        "adaptive_k_synthetic": adaptive_synthetic,
        "rts_smoother": rts_results,
        "adaptive_k_real": adaptive_real,
    }

    with open(output_dir / "adaptive_k_and_rts_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_dir}/adaptive_k_and_rts_results.json")


if __name__ == "__main__":
    main()
