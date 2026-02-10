#!/usr/bin/env python3
"""
Test Robustness Layer: Confidence gating + Jump reject

Compare:
1. redetect (no robustness) vs redetect (with robustness)
2. adaptive_k (with robustness)
3. Full pipeline: adaptive_k + RTS smoother
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

    # Jerk
    if len(az) > 3:
        az_jerk = np.abs(np.diff(np.diff(np.diff(az))))
        jerk_avg = np.mean(az_jerk)
        jerk_max = np.max(az_jerk)
    else:
        jerk_avg = jerk_max = 0

    # Direction changes
    if len(az) > 2:
        dir_changes = int(np.sum(np.diff(np.sign(np.diff(az))) != 0))
    else:
        dir_changes = 0

    # Amplitude
    amp_x = (np.max(cx) - np.min(cx)) / 2

    # Low confidence ratio
    low_conf_ratio = np.sum(conf < 0.4) / len(conf) * 100

    result = {
        "jerk_avg": float(jerk_avg),
        "jerk_max": float(jerk_max),
        "dir_changes": dir_changes,
        "amplitude_x_px": float(amp_x),
        "num_frames": len(frames),
        "low_conf_pct": float(low_conf_ratio),
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


def test_on_synthetic():
    """Test on synthetic oscillating video."""
    print("=" * 70)
    print("SYNTHETIC VIDEO TEST (0.6 Hz oscillation)")
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

    configs = [
        ("adaptive_k", "adaptive_k", {}),
        ("adaptive_k + RTS", "adaptive_k", {"apply_rts": True}),
    ]

    for name, method, extra in configs:
        print(f"\n[{name}]")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tracker = HybridTracker(device="cuda", scene_type="auto", box_threshold=0.15)

        start = time.time()
        result = tracker.track(
            video_path=video_path,
            text_prompt="orange circle",
            tracking_method=method,
            estimate_depth=True,
            depth_stride=5,
        )
        elapsed = time.time() - start

        # Get metrics before RTS
        m = compute_metrics(result.frames, cx_gt)
        m["fps"] = len(result.frames) / elapsed

        # Apply RTS if requested
        if extra.get("apply_rts"):
            traj_3d = result.get_trajectory_3d(smooth=False)
            frames_smoothed = rts_smooth_trajectory(traj_3d["frames"])

            # Compute jerk on smoothed trajectory
            az_smoothed = np.array([f["az"] for f in frames_smoothed])
            if len(az_smoothed) > 3:
                jerk_rts = np.mean(np.abs(np.diff(np.diff(np.diff(np.degrees(az_smoothed))))))
                m["jerk_after_rts"] = float(jerk_rts)
                m["rts_jerk_reduction_pct"] = (1 - jerk_rts / m["jerk_avg"]) * 100 if m["jerk_avg"] > 0 else 0

        print(f"  Amplitude ratio: {m.get('amplitude_ratio', 0)*100:.1f}%")
        print(f"  MAE: {m.get('mae_px', 0):.1f}px")
        print(f"  Velocity correlation: {m.get('velocity_correlation', 0):.3f}")
        print(f"  Jerk avg: {m['jerk_avg']:.4f}")
        if "jerk_after_rts" in m:
            print(f"  Jerk after RTS: {m['jerk_after_rts']:.6f} ({m['rts_jerk_reduction_pct']:.1f}% reduction)")
        print(f"  Dir changes: {m['dir_changes']}")
        print(f"  Low conf %: {m['low_conf_pct']:.1f}%")
        print(f"  FPS: {m['fps']:.1f}")

        results[name] = m

    return results


def test_on_real_videos():
    """Test on real videos."""
    print("\n" + "=" * 70)
    print("REAL VIDEO TEST")
    print("=" * 70)

    test_videos = [
        ("marker_hd", "/home/seung/mmhoa/vid2spatial/test_videos/marker_hd.mp4", "colored marker"),
        ("daw_hd", "/home/seung/mmhoa/vid2spatial/test_videos/daw_hd.mp4", "colored marker"),
    ]

    # Add benchmark videos
    benchmark_dir = "/home/seung/mmhoa/vid2spatial/test_videos/benchmark"
    if os.path.exists(benchmark_dir):
        for f in sorted(os.listdir(benchmark_dir))[:4]:
            if f.endswith(".mp4"):
                test_videos.append((f.replace(".mp4", ""), os.path.join(benchmark_dir, f), "person"))

    results = {}

    for video_name, path, prompt in test_videos:
        if not os.path.exists(path):
            continue

        print(f"\n--- {video_name} ---")
        results[video_name] = {}

        for config_name, method in [
            ("adaptive_k", "adaptive_k"),
        ]:
            print(f"  [{config_name}]", end=" ", flush=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            try:
                tracker = HybridTracker(device="cuda", scene_type="auto", box_threshold=0.15)

                start = time.time()
                result = tracker.track(
                    video_path=path,
                    text_prompt=prompt,
                    tracking_method=method,
                    estimate_depth=True,
                    depth_stride=5,
                )
                elapsed = time.time() - start

                m = compute_metrics(result.frames)
                m["fps"] = len(result.frames) / elapsed

                # Apply RTS
                traj_3d = result.get_trajectory_3d(smooth=False)
                frames_smoothed = rts_smooth_trajectory(traj_3d["frames"])

                az_smoothed = np.array([f["az"] for f in frames_smoothed])
                if len(az_smoothed) > 3:
                    jerk_rts = np.mean(np.abs(np.diff(np.diff(np.diff(np.degrees(az_smoothed))))))
                    m["jerk_after_rts"] = float(jerk_rts)

                print(f"Jerk={m['jerk_avg']:.4f} → RTS={m.get('jerk_after_rts', 0):.6f}, "
                      f"LowConf={m['low_conf_pct']:.0f}%, FPS={m['fps']:.1f}")

                results[video_name][config_name] = m

            except Exception as e:
                print(f"ERROR: {e}")

    return results


def print_summary(synth_results, real_results):
    """Print summary table."""
    print("\n" + "=" * 70)
    print("SUMMARY: Full Pipeline (Adaptive K + Robustness + RTS)")
    print("=" * 70)

    if synth_results:
        print("\n### Synthetic (0.6 Hz)")
        m = synth_results.get("adaptive_k + RTS", {})
        print(f"  Amplitude: {m.get('amplitude_ratio', 0)*100:.1f}%")
        print(f"  Velocity correlation: {m.get('velocity_correlation', 0):.3f}")
        print(f"  Jerk: {m.get('jerk_avg', 0):.4f} → {m.get('jerk_after_rts', 0):.6f} (RTS)")
        print(f"  FPS: {m.get('fps', 0):.1f}")

    if real_results:
        print("\n### Real Videos (Jerk before → after RTS)")
        print(f"\n{'Video':<25} {'Jerk':>10} {'→ RTS':>12} {'LowConf%':>10} {'FPS':>8}")
        print("-" * 70)

        for video, methods in real_results.items():
            m = methods.get("adaptive_k", {})
            jerk_before = m.get("jerk_avg", 0)
            jerk_after = m.get("jerk_after_rts", 0)
            low_conf = m.get("low_conf_pct", 0)
            fps = m.get("fps", 0)

            print(f"{video:<25} {jerk_before:>10.4f} {jerk_after:>12.6f} {low_conf:>10.0f}% {fps:>8.1f}")


def main():
    output_dir = Path("/home/seung/mmhoa/vid2spatial/eval/comprehensive_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test synthetic
    synth_results = test_on_synthetic()

    # Test real videos
    real_results = test_on_real_videos()

    # Summary
    print_summary(synth_results, real_results)

    # Save
    all_results = {
        "synthetic": synth_results,
        "real_videos": real_results,
    }

    with open(output_dir / "robustness_layer_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_dir}/robustness_layer_results.json")


if __name__ == "__main__":
    main()
