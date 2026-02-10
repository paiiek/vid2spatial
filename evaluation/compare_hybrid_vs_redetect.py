#!/usr/bin/env python3
"""
Compare tracking modes:
1. redetect: DINO K-frame + linear interpolation (current best)
2. hybrid: DINO K-frame + SAM2 propagation + EMA/Kalman (new)

Metrics:
- Amplitude ratio (synthetic)
- MAE, Velocity correlation (synthetic)
- Jerk, Direction changes (real videos)
- FPS
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
from vid2spatial_pkg.vision import CameraIntrinsics, pixel_to_ray, ray_to_angles


def compute_trajectory_metrics(frames, fps=30.0, width=640, height=480, fov_deg=60.0):
    """Compute all trajectory quality metrics."""
    if len(frames) < 3:
        return None

    K = CameraIntrinsics(width, height, fov_deg)

    # Extract data
    cx = np.array([f.center[0] for f in frames])
    cy = np.array([f.center[1] for f in frames])
    conf = np.array([f.confidence for f in frames])

    # Compute az/el
    az_list = []
    el_list = []
    for f in frames:
        ray = pixel_to_ray(f.center[0], f.center[1], K)
        az, el = ray_to_angles(ray)
        az_list.append(np.degrees(az))
        el_list.append(np.degrees(el))

    az = np.array(az_list)
    el = np.array(el_list)

    # 1. Jerk (3rd derivative)
    if len(az) > 3:
        az_jerk = np.abs(np.diff(np.diff(np.diff(az))))
        el_jerk = np.abs(np.diff(np.diff(np.diff(el))))
        jerk_avg = (np.mean(az_jerk) + np.mean(el_jerk)) / 2
        jerk_max = max(np.max(az_jerk), np.max(el_jerk))
    else:
        jerk_avg = jerk_max = 0

    # 2. Direction changes
    if len(az) > 2:
        az_diff = np.diff(az)
        el_diff = np.diff(el)
        dir_changes_az = int(np.sum(np.diff(np.sign(az_diff)) != 0))
        dir_changes_el = int(np.sum(np.diff(np.sign(el_diff)) != 0))
    else:
        dir_changes_az = dir_changes_el = 0

    # 3. Coverage
    total_frames = len(frames)
    high_conf_frames = int(np.sum(conf > 0.7))
    high_conf_pct = high_conf_frames / total_frames * 100

    # 4. Amplitude (px)
    amp_x = (np.max(cx) - np.min(cx)) / 2
    amp_y = (np.max(cy) - np.min(cy)) / 2

    return {
        "jerk_avg": float(jerk_avg),
        "jerk_max": float(jerk_max),
        "dir_changes_az": dir_changes_az,
        "dir_changes_el": dir_changes_el,
        "total_frames": total_frames,
        "high_conf_pct": float(high_conf_pct),
        "amplitude_x_px": float(amp_x),
        "amplitude_y_px": float(amp_y),
        "cx": cx.tolist(),
    }


def track_and_measure(video_path, text_prompt, method, k_value=5, ema_alpha=0.3, use_kalman=False):
    """Track video and measure metrics + FPS."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tracker = HybridTracker(
        device="cuda",
        scene_type="auto",
        box_threshold=0.15,
    )

    start = time.time()

    if method == "redetect":
        result = tracker.track(
            video_path=video_path,
            text_prompt=text_prompt,
            tracking_method="redetect",
            redetect_interval=k_value,
            estimate_depth=False,
        )
    elif method == "hybrid":
        result = tracker.track(
            video_path=video_path,
            text_prompt=text_prompt,
            tracking_method="hybrid",
            redetect_interval=k_value,
            ema_alpha=ema_alpha,
            use_kalman=use_kalman,
            estimate_depth=False,
        )
    elif method == "sam2":
        result = tracker.track(
            video_path=video_path,
            text_prompt=text_prompt,
            tracking_method="sam2",
            estimate_depth=False,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    elapsed = time.time() - start
    fps = len(result.frames) / elapsed if elapsed > 0 else 0

    metrics = compute_trajectory_metrics(result.frames)
    if metrics:
        metrics["fps"] = fps
        metrics["num_frames"] = len(result.frames)
        metrics["time_sec"] = elapsed

    return metrics, result


def run_synthetic_comparison():
    """Compare on synthetic oscillating video."""
    print("=" * 70)
    print("SYNTHETIC OSCILLATING (0.6 Hz) - Ground Truth Comparison")
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

    cx_gt = []
    for i in range(num_frames):
        t = i / fps
        cx = width * (0.5 + 0.35 * np.sin(2 * np.pi * freq * t))
        cx_gt.append(cx)
    cx_gt = np.array(cx_gt)
    gt_amp = (np.max(cx_gt) - np.min(cx_gt)) / 2

    results = {}

    methods = [
        ("sam2", {}),
        ("redetect", {"k_value": 5}),
        ("hybrid_ema", {"k_value": 5, "ema_alpha": 0.3, "use_kalman": False}),
        ("hybrid_kalman", {"k_value": 5, "use_kalman": True}),
    ]

    for method_name, params in methods:
        print(f"\n[{method_name}]")

        if method_name == "sam2":
            m, result = track_and_measure(video_path, "orange circle", "sam2")
        elif method_name == "redetect":
            m, result = track_and_measure(video_path, "orange circle", "redetect", **params)
        elif method_name.startswith("hybrid"):
            m, result = track_and_measure(video_path, "orange circle", "hybrid", **params)

        if m is None:
            print("  ERROR")
            continue

        # Compute GT metrics
        cx_pred = np.array(m["cx"])
        min_len = min(len(cx_pred), len(cx_gt))
        pred_amp = m["amplitude_x_px"]

        mae = np.mean(np.abs(cx_pred[:min_len] - cx_gt[:min_len]))
        gt_vel = np.diff(cx_gt[:min_len])
        pred_vel = np.diff(cx_pred[:min_len])
        vel_corr = np.corrcoef(gt_vel, pred_vel)[0, 1] if len(gt_vel) > 1 else 0

        m["amplitude_ratio"] = pred_amp / gt_amp if gt_amp > 0 else 0
        m["mae_px"] = mae
        m["velocity_correlation"] = vel_corr

        print(f"  Amplitude ratio: {m['amplitude_ratio']*100:.1f}%")
        print(f"  MAE: {m['mae_px']:.1f}px")
        print(f"  Velocity correlation: {m['velocity_correlation']:.3f}")
        print(f"  Jerk avg: {m['jerk_avg']:.4f}")
        print(f"  Dir changes (Az): {m['dir_changes_az']}")
        print(f"  FPS: {m['fps']:.1f}")

        # Remove cx array for JSON
        m_clean = {k: v for k, v in m.items() if k != "cx"}
        results[method_name] = m_clean

    return results


def run_real_video_comparison():
    """Compare on real videos."""
    print("\n" + "=" * 70)
    print("REAL VIDEOS - Jerk & Smoothness Comparison")
    print("=" * 70)

    test_videos = [
        ("marker_hd", "/home/seung/mmhoa/vid2spatial/test_videos/marker_hd.mp4", "colored marker"),
        ("daw_hd", "/home/seung/mmhoa/vid2spatial/test_videos/daw_hd.mp4", "colored marker"),
    ]

    # Add benchmark videos
    benchmark_dir = "/home/seung/mmhoa/vid2spatial/test_videos/benchmark"
    if os.path.exists(benchmark_dir):
        for f in sorted(os.listdir(benchmark_dir))[:5]:  # Limit to 5 for speed
            if f.endswith(".mp4"):
                test_videos.append((
                    f.replace(".mp4", ""),
                    os.path.join(benchmark_dir, f),
                    "person"
                ))

    results = {}

    for name, path, prompt in test_videos:
        if not os.path.exists(path):
            continue

        print(f"\n--- {name} ---")
        results[name] = {}

        methods = [
            ("sam2", {}),
            ("redetect", {"k_value": 5}),
            ("hybrid_ema", {"k_value": 5, "ema_alpha": 0.3}),
        ]

        for method_name, params in methods:
            print(f"  [{method_name}]", end=" ", flush=True)

            try:
                if method_name == "sam2":
                    m, _ = track_and_measure(path, prompt, "sam2")
                elif method_name == "redetect":
                    m, _ = track_and_measure(path, prompt, "redetect", **params)
                else:
                    m, _ = track_and_measure(path, prompt, "hybrid", **params)

                if m:
                    print(f"Jerk={m['jerk_avg']:.4f}, DirChg={m['dir_changes_az']}, FPS={m['fps']:.1f}")
                    m_clean = {k: v for k, v in m.items() if k != "cx"}
                    results[name][method_name] = m_clean
                else:
                    print("ERROR")

            except Exception as e:
                print(f"ERROR: {e}")
                results[name][method_name] = {"error": str(e)}

    return results


def print_comparison_table(synth_results, real_results):
    """Print comparison table."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    if synth_results:
        print("\n### Synthetic Oscillating (0.6 Hz)")
        print(f"\n{'Method':<20} {'Amp%':>8} {'MAE(px)':>10} {'VelCorr':>10} {'Jerk':>10} {'FPS':>8}")
        print("-" * 70)

        for method, m in synth_results.items():
            if "error" not in m:
                print(f"{method:<20} {m.get('amplitude_ratio', 0)*100:>7.1f}% {m.get('mae_px', 0):>10.1f} {m.get('velocity_correlation', 0):>10.3f} {m.get('jerk_avg', 0):>10.4f} {m.get('fps', 0):>8.1f}")

    if real_results:
        print("\n### Real Videos (Jerk - lower is better)")
        print(f"\n{'Video':<20} {'SAM2':>10} {'Redetect':>10} {'Hybrid':>10} {'Winner':>12}")
        print("-" * 70)

        for video, methods in real_results.items():
            sam2_jerk = methods.get("sam2", {}).get("jerk_avg", float('inf'))
            redetect_jerk = methods.get("redetect", {}).get("jerk_avg", float('inf'))
            hybrid_jerk = methods.get("hybrid_ema", {}).get("jerk_avg", float('inf'))

            jerks = {"sam2": sam2_jerk, "redetect": redetect_jerk, "hybrid": hybrid_jerk}
            winner = min(jerks, key=jerks.get)

            print(f"{video:<20} {sam2_jerk:>10.4f} {redetect_jerk:>10.4f} {hybrid_jerk:>10.4f} {winner:>12}")


def main():
    output_dir = Path("/home/seung/mmhoa/vid2spatial/eval/comprehensive_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic comparison
    synth_results = run_synthetic_comparison()

    # Real video comparison
    real_results = run_real_video_comparison()

    # Print summary
    print_comparison_table(synth_results, real_results)

    # Save results
    all_results = {
        "synthetic_oscillating": synth_results,
        "real_videos": real_results,
    }

    with open(output_dir / "hybrid_vs_redetect_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_dir}/hybrid_vs_redetect_comparison.json")


if __name__ == "__main__":
    main()
