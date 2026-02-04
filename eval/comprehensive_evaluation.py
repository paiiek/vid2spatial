#!/usr/bin/env python3
"""
Comprehensive Evaluation for Vid2Spatial with New Hybrid Tracker (DINO K=5).

This script evaluates ALL metrics required for ISMAR submission:

1A) Trajectory Quality Metrics:
    - Amplitude ratio
    - MAE (px) / Angular MAE (deg)
    - Velocity correlation
    - Jerk avg / max
    - Direction changes (Az/El)
    - Coverage / high-conf coverage
    - Latency / lag

1B) Performance Benchmarks:
    - FPS (end-to-end)
    - GPU VRAM peak
    - K value scaling (K=1/5/10)

2C) Depth/Projection Validation:
    - Depth range sanity check
    - Depth jerk / discontinuity check

2D) Spatial Rendering Demo:
    - Before/after comparison

3) Comparison Table:
    - SAM2-only vs YOLO vs DINO-per-frame vs DINO-K=5
"""

import sys
sys.path.insert(0, "/home/seung/mmhoa/vid2spatial")

import os
import cv2
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import torch

from vid2spatial_pkg.hybrid_tracker import HybridTracker, HybridTrackingResult
from vid2spatial_pkg.vision import CameraIntrinsics, pixel_to_ray, ray_to_angles


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TrajectoryMetrics:
    """All trajectory quality metrics."""
    # Basic
    amplitude_ratio: float
    mae_px: float
    angular_mae_deg: float
    velocity_correlation: float

    # Jerk
    jerk_avg: float
    jerk_max: float

    # Direction changes
    direction_changes_az: int
    direction_changes_el: int

    # Coverage
    total_frames: int
    tracked_frames: int
    coverage_pct: float
    high_conf_frames: int
    high_conf_coverage_pct: float

    # Latency
    peak_delay_frames: float
    peak_delay_ms: float

    # Additional
    pred_amplitude_px: float
    gt_amplitude_px: float


@dataclass
class PerformanceMetrics:
    """Performance benchmark metrics."""
    method: str
    k_value: int
    total_time_sec: float
    fps: float
    vram_peak_mb: float
    num_frames: int


@dataclass
class DepthMetrics:
    """Depth validation metrics."""
    depth_min: float
    depth_max: float
    depth_mean: float
    depth_std: float
    depth_jerk_avg: float
    depth_discontinuities: int  # jumps > 2m


# =============================================================================
# SYNTHETIC VIDEO GENERATION
# =============================================================================

def generate_synthetic_videos(output_dir: Path) -> Dict[str, Dict]:
    """Generate synthetic videos with known GT."""
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = {}

    # Parameters
    width, height = 640, 480
    fps = 30.0
    duration = 5.0
    num_frames = int(fps * duration)
    fov_deg = 60.0
    K = CameraIntrinsics(width, height, fov_deg)

    # 1. Horizontal sweep
    video_path = output_dir / "horizontal_sweep.mp4"
    if not video_path.exists():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        gt_frames = []

        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            t = i / (num_frames - 1)
            cx = int(width * (0.1 + 0.8 * t))
            cy = height // 2
            cv2.circle(frame, (cx, cy), 30, (0, 0, 255), -1)
            out.write(frame)

            ray = pixel_to_ray(cx, cy, K)
            az, el = ray_to_angles(ray)
            gt_frames.append({"frame": i, "cx": cx, "cy": cy, "az_deg": float(np.degrees(az)), "el_deg": float(np.degrees(el))})

        out.release()
        print(f"  Generated: horizontal_sweep.mp4")
    else:
        # Load GT
        gt_frames = []
        for i in range(num_frames):
            t = i / (num_frames - 1)
            cx = int(width * (0.1 + 0.8 * t))
            cy = height // 2
            ray = pixel_to_ray(cx, cy, K)
            az, el = ray_to_angles(ray)
            gt_frames.append({"frame": i, "cx": cx, "cy": cy, "az_deg": float(np.degrees(az)), "el_deg": float(np.degrees(el))})

    videos["horizontal_sweep"] = {
        "path": str(video_path),
        "gt": gt_frames,
        "prompt": "red circle",
        "motion_type": "linear"
    }

    # 2. Diagonal sweep
    video_path = output_dir / "diagonal_sweep.mp4"
    if not video_path.exists():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        gt_frames = []

        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            t = i / (num_frames - 1)
            cx = int(width * (0.1 + 0.8 * t))
            cy = int(height * (0.1 + 0.8 * t))
            cv2.circle(frame, (cx, cy), 30, (0, 255, 0), -1)
            out.write(frame)

            ray = pixel_to_ray(cx, cy, K)
            az, el = ray_to_angles(ray)
            gt_frames.append({"frame": i, "cx": cx, "cy": cy, "az_deg": float(np.degrees(az)), "el_deg": float(np.degrees(el))})

        out.release()
        print(f"  Generated: diagonal_sweep.mp4")
    else:
        gt_frames = []
        for i in range(num_frames):
            t = i / (num_frames - 1)
            cx = int(width * (0.1 + 0.8 * t))
            cy = int(height * (0.1 + 0.8 * t))
            ray = pixel_to_ray(cx, cy, K)
            az, el = ray_to_angles(ray)
            gt_frames.append({"frame": i, "cx": cx, "cy": cy, "az_deg": float(np.degrees(az)), "el_deg": float(np.degrees(el))})

    videos["diagonal_sweep"] = {
        "path": str(video_path),
        "gt": gt_frames,
        "prompt": "green circle",
        "motion_type": "linear"
    }

    # 3. Oscillating (0.6 Hz)
    video_path = output_dir / "oscillating.mp4"
    frequency_hz = 0.6
    if not video_path.exists():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        gt_frames = []

        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            t = i / fps
            cx = int(width * (0.5 + 0.35 * np.sin(2 * np.pi * frequency_hz * t)))
            cy = height // 2
            cv2.circle(frame, (cx, cy), 30, (0, 165, 255), -1)  # Orange
            out.write(frame)

            ray = pixel_to_ray(cx, cy, K)
            az, el = ray_to_angles(ray)
            gt_frames.append({"frame": i, "cx": cx, "cy": cy, "az_deg": float(np.degrees(az)), "el_deg": float(np.degrees(el))})

        out.release()
        print(f"  Generated: oscillating.mp4")
    else:
        gt_frames = []
        for i in range(num_frames):
            t = i / fps
            cx = int(width * (0.5 + 0.35 * np.sin(2 * np.pi * frequency_hz * t)))
            cy = height // 2
            ray = pixel_to_ray(cx, cy, K)
            az, el = ray_to_angles(ray)
            gt_frames.append({"frame": i, "cx": cx, "cy": cy, "az_deg": float(np.degrees(az)), "el_deg": float(np.degrees(el))})

    videos["oscillating"] = {
        "path": str(video_path),
        "gt": gt_frames,
        "prompt": "orange circle",
        "motion_type": "oscillating",
        "frequency_hz": frequency_hz
    }

    return videos


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def compute_trajectory_metrics(
    pred_frames: List[Dict],
    gt_frames: List[Dict],
    fps: float = 30.0,
    fov_deg: float = 60.0,
    width: int = 640,
    height: int = 480,
) -> TrajectoryMetrics:
    """Compute all trajectory quality metrics."""

    K = CameraIntrinsics(width, height, fov_deg)

    # Align frames
    gt_by_frame = {g["frame"]: g for g in gt_frames}

    matched_pred = []
    matched_gt = []

    for p in pred_frames:
        fidx = p.get("frame", p.get("frame_idx"))
        if fidx in gt_by_frame:
            matched_pred.append(p)
            matched_gt.append(gt_by_frame[fidx])

    if len(matched_pred) < 2:
        return None

    # Extract arrays
    pred_cx = np.array([p.get("cx", p.get("center", [0, 0])[0] if isinstance(p.get("center"), (list, tuple)) else 0) for p in matched_pred])
    pred_cy = np.array([p.get("cy", p.get("center", [0, 0])[1] if isinstance(p.get("center"), (list, tuple)) else 0) for p in matched_pred])
    gt_cx = np.array([g["cx"] for g in matched_gt])
    gt_cy = np.array([g["cy"] for g in matched_gt])

    # Compute predicted az/el
    pred_az = []
    pred_el = []
    for p in matched_pred:
        if "az_deg" in p:
            pred_az.append(p["az_deg"])
            pred_el.append(p.get("el_deg", 0))
        else:
            cx = p.get("cx", p.get("center", [width/2, height/2])[0] if isinstance(p.get("center"), (list, tuple)) else width/2)
            cy = p.get("cy", p.get("center", [width/2, height/2])[1] if isinstance(p.get("center"), (list, tuple)) else height/2)
            ray = pixel_to_ray(cx, cy, K)
            az, el = ray_to_angles(ray)
            pred_az.append(np.degrees(az))
            pred_el.append(np.degrees(el))

    pred_az = np.array(pred_az)
    pred_el = np.array(pred_el)
    gt_az = np.array([g["az_deg"] for g in matched_gt])
    gt_el = np.array([g["el_deg"] for g in matched_gt])

    # 1. Amplitude ratio
    gt_amp_px = (np.max(gt_cx) - np.min(gt_cx)) / 2
    pred_amp_px = (np.max(pred_cx) - np.min(pred_cx)) / 2
    amplitude_ratio = pred_amp_px / gt_amp_px if gt_amp_px > 0 else 0

    # 2. MAE (px)
    mae_px = np.mean(np.sqrt((pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2))

    # 3. Angular MAE (deg)
    angular_mae_deg = np.mean(np.abs(pred_az - gt_az))

    # 4. Velocity correlation
    if len(pred_cx) > 1:
        gt_vel = np.diff(gt_cx)
        pred_vel = np.diff(pred_cx)
        if np.std(gt_vel) > 0 and np.std(pred_vel) > 0:
            velocity_correlation = np.corrcoef(gt_vel, pred_vel)[0, 1]
        else:
            velocity_correlation = 0
    else:
        velocity_correlation = 0

    # 5. Jerk (3rd derivative of azimuth)
    if len(pred_az) > 3:
        az_rad = np.radians(pred_az)
        jerk = np.abs(np.diff(np.diff(np.diff(az_rad))))
        jerk_avg = np.mean(jerk)
        jerk_max = np.max(jerk)
    else:
        jerk_avg = 0
        jerk_max = 0

    # 6. Direction changes
    if len(pred_az) > 2:
        az_diff = np.diff(pred_az)
        az_sign_changes = np.sum(np.diff(np.sign(az_diff)) != 0)

        el_diff = np.diff(pred_el)
        el_sign_changes = np.sum(np.diff(np.sign(el_diff)) != 0)
    else:
        az_sign_changes = 0
        el_sign_changes = 0

    # 7. Coverage
    total_frames = len(gt_frames)
    tracked_frames = len(matched_pred)
    coverage_pct = tracked_frames / total_frames * 100 if total_frames > 0 else 0

    # High confidence coverage (conf > 0.7)
    high_conf_frames = sum(1 for p in matched_pred if p.get("confidence", 1.0) > 0.7)
    high_conf_coverage_pct = high_conf_frames / total_frames * 100 if total_frames > 0 else 0

    # 8. Latency / Peak delay
    peak_delay_frames = compute_peak_delay(gt_az, pred_az)
    peak_delay_ms = peak_delay_frames / fps * 1000

    return TrajectoryMetrics(
        amplitude_ratio=amplitude_ratio,
        mae_px=mae_px,
        angular_mae_deg=angular_mae_deg,
        velocity_correlation=velocity_correlation,
        jerk_avg=jerk_avg,
        jerk_max=jerk_max,
        direction_changes_az=az_sign_changes,
        direction_changes_el=el_sign_changes,
        total_frames=total_frames,
        tracked_frames=tracked_frames,
        coverage_pct=coverage_pct,
        high_conf_frames=high_conf_frames,
        high_conf_coverage_pct=high_conf_coverage_pct,
        peak_delay_frames=peak_delay_frames,
        peak_delay_ms=peak_delay_ms,
        pred_amplitude_px=pred_amp_px,
        gt_amplitude_px=gt_amp_px,
    )


def compute_peak_delay(gt: np.ndarray, pred: np.ndarray, search_range: int = 15) -> float:
    """Compute peak alignment delay in frames."""
    from scipy.signal import find_peaks

    # Find peaks in GT
    gt_peaks, _ = find_peaks(gt, prominence=3)
    gt_troughs, _ = find_peaks(-gt, prominence=3)
    all_extrema = sorted(list(gt_peaks) + list(gt_troughs))

    if len(all_extrema) == 0:
        return 0.0

    delays = []
    for gt_idx in all_extrema:
        search_start = max(0, gt_idx - search_range)
        search_end = min(len(pred) - 1, gt_idx + search_range)

        if gt_idx in gt_peaks:
            pred_idx = search_start + np.argmax(pred[search_start:search_end+1])
        else:
            pred_idx = search_start + np.argmin(pred[search_start:search_end+1])

        delays.append(pred_idx - gt_idx)

    return np.mean(delays)


def compute_depth_metrics(frames: List[Dict]) -> DepthMetrics:
    """Compute depth validation metrics."""
    depths = [f.get("depth_m", f.get("dist_m", 2.0)) for f in frames]
    depths = np.array(depths)

    if len(depths) < 2:
        return DepthMetrics(0, 0, 0, 0, 0, 0)

    # Jerk
    if len(depths) > 3:
        depth_jerk = np.abs(np.diff(np.diff(np.diff(depths))))
        depth_jerk_avg = np.mean(depth_jerk)
    else:
        depth_jerk_avg = 0

    # Discontinuities (jumps > 2m)
    depth_diff = np.abs(np.diff(depths))
    discontinuities = np.sum(depth_diff > 2.0)

    return DepthMetrics(
        depth_min=float(np.min(depths)),
        depth_max=float(np.max(depths)),
        depth_mean=float(np.mean(depths)),
        depth_std=float(np.std(depths)),
        depth_jerk_avg=float(depth_jerk_avg),
        depth_discontinuities=int(discontinuities),
    )


# =============================================================================
# TRACKING METHODS
# =============================================================================

def track_with_method(
    video_path: str,
    text_prompt: str,
    method: str,
    k_value: int = 5,
    estimate_depth: bool = False,
) -> Tuple[List[Dict], PerformanceMetrics]:
    """Track video with specified method and measure performance."""

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Initialize tracker based on method
    if method == "sam2":
        tracker = HybridTracker(device="cuda", scene_type="indoor", box_threshold=0.1)
        tracking_method = "sam2"
        redetect_interval = None
    elif method == "yolo":
        tracker = HybridTracker(device="cuda", scene_type="indoor", box_threshold=0.1)
        tracking_method = "yolo"
        redetect_interval = None
    elif method.startswith("dino_k"):
        tracker = HybridTracker(
            device="cuda",
            scene_type="indoor",
            box_threshold=0.1,
            redetect_interval=k_value,
            trajectory_source="detection",
        )
        tracking_method = "redetect"
        redetect_interval = k_value
    else:
        raise ValueError(f"Unknown method: {method}")

    # Track
    start_time = time.time()

    try:
        result = tracker.track(
            video_path=video_path,
            text_prompt=text_prompt,
            sample_stride=1,
            tracking_method=tracking_method,
            redetect_interval=redetect_interval,
            estimate_depth=estimate_depth,
        )
    except Exception as e:
        print(f"    Tracking failed: {e}")
        return [], None

    elapsed = time.time() - start_time

    # Get VRAM
    if torch.cuda.is_available():
        vram_peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        vram_peak_mb = 0

    # Convert to dict format
    frames = []
    for f in result.frames:
        frames.append({
            "frame": f.frame_idx,
            "cx": f.center[0],
            "cy": f.center[1],
            "confidence": f.confidence,
            "depth_m": f.depth_m,
            "bbox": f.bbox,
        })

    perf = PerformanceMetrics(
        method=method,
        k_value=k_value,
        total_time_sec=elapsed,
        fps=len(frames) / elapsed if elapsed > 0 else 0,
        vram_peak_mb=vram_peak_mb,
        num_frames=len(frames),
    )

    return frames, perf


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_1a_trajectory_metrics(videos: Dict, output_dir: Path) -> Dict:
    """1A) Trajectory quality metrics for all methods."""
    print("\n" + "=" * 70)
    print("1A) TRAJECTORY QUALITY METRICS")
    print("=" * 70)

    methods = [
        ("sam2", 0),
        ("yolo", 0),
        ("dino_k1", 1),
        ("dino_k5", 5),
        ("dino_k10", 10),
    ]

    all_results = {}

    for video_name, video_info in videos.items():
        print(f"\n--- {video_name} ({video_info['motion_type']}) ---")
        all_results[video_name] = {}

        for method_name, k_value in methods:
            print(f"\n  [{method_name}]")

            frames, perf = track_with_method(
                video_info["path"],
                video_info["prompt"],
                method_name,
                k_value,
                estimate_depth=False,
            )

            if not frames:
                print(f"    FAILED")
                continue

            metrics = compute_trajectory_metrics(
                frames,
                video_info["gt"],
                fps=30.0,
            )

            if metrics:
                print(f"    Amplitude ratio: {metrics.amplitude_ratio*100:.1f}%")
                print(f"    MAE (px): {metrics.mae_px:.1f}")
                print(f"    Angular MAE: {metrics.angular_mae_deg:.2f}°")
                print(f"    Velocity corr: {metrics.velocity_correlation:.3f}")
                print(f"    Jerk avg: {metrics.jerk_avg:.6f}")
                print(f"    Dir changes (Az): {metrics.direction_changes_az}")
                print(f"    Coverage: {metrics.coverage_pct:.1f}%")
                print(f"    Peak delay: {metrics.peak_delay_frames:.1f} frames ({metrics.peak_delay_ms:.0f}ms)")

                all_results[video_name][method_name] = asdict(metrics)
            else:
                print(f"    Not enough matched frames")

    # Save results (convert numpy types)
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    with open(output_dir / "1a_trajectory_metrics.json", "w") as f:
        json.dump(convert_numpy(all_results), f, indent=2)

    print(f"\nResults saved to: {output_dir}/1a_trajectory_metrics.json")
    return all_results


def run_1b_performance_benchmark(videos: Dict, output_dir: Path) -> Dict:
    """1B) Performance benchmarks (FPS, VRAM, K scaling)."""
    print("\n" + "=" * 70)
    print("1B) PERFORMANCE BENCHMARKS")
    print("=" * 70)

    # Use oscillating video for benchmark (most demanding)
    video_info = videos["oscillating"]

    methods = [
        ("sam2", 0),
        ("yolo", 0),
        ("dino_k1", 1),
        ("dino_k5", 5),
        ("dino_k10", 10),
    ]

    results = []

    for method_name, k_value in methods:
        print(f"\n  [{method_name}]")

        # Run 3 times and average
        fps_list = []
        vram_list = []

        for run in range(3):
            frames, perf = track_with_method(
                video_info["path"],
                video_info["prompt"],
                method_name,
                k_value,
                estimate_depth=False,
            )

            if perf:
                fps_list.append(perf.fps)
                vram_list.append(perf.vram_peak_mb)

        if fps_list:
            avg_fps = np.mean(fps_list)
            avg_vram = np.mean(vram_list)
            print(f"    FPS: {avg_fps:.1f} (±{np.std(fps_list):.1f})")
            print(f"    VRAM: {avg_vram:.0f} MB")

            results.append({
                "method": method_name,
                "k_value": k_value,
                "fps_mean": avg_fps,
                "fps_std": np.std(fps_list),
                "vram_mb": avg_vram,
                "runs": len(fps_list),
            })

    # Save results
    with open(output_dir / "1b_performance_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}/1b_performance_benchmark.json")
    return results


def run_2c_depth_validation(videos: Dict, output_dir: Path) -> Dict:
    """2C) Depth/Projection validation."""
    print("\n" + "=" * 70)
    print("2C) DEPTH/PROJECTION VALIDATION")
    print("=" * 70)

    # Test on oscillating with depth estimation enabled
    video_info = videos["oscillating"]

    results = {}

    for method_name, k_value in [("sam2", 0), ("dino_k5", 5)]:
        print(f"\n  [{method_name}] with depth estimation")

        frames, perf = track_with_method(
            video_info["path"],
            video_info["prompt"],
            method_name,
            k_value,
            estimate_depth=True,
        )

        if frames:
            depth_metrics = compute_depth_metrics(frames)
            print(f"    Depth range: {depth_metrics.depth_min:.2f}m - {depth_metrics.depth_max:.2f}m")
            print(f"    Depth mean: {depth_metrics.depth_mean:.2f}m (±{depth_metrics.depth_std:.2f})")
            print(f"    Depth jerk: {depth_metrics.depth_jerk_avg:.6f}")
            print(f"    Discontinuities: {depth_metrics.depth_discontinuities}")

            results[method_name] = asdict(depth_metrics)

    # Save results
    with open(output_dir / "2c_depth_validation.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}/2c_depth_validation.json")
    return results


def run_3_comparison_table(results_1a: Dict, results_1b: Dict, output_dir: Path) -> str:
    """3) Generate comparison table for paper."""
    print("\n" + "=" * 70)
    print("3) COMPARISON TABLE (PAPER READY)")
    print("=" * 70)

    # Build comparison table
    table_lines = []
    table_lines.append("| Method | Video | Amp Ratio | MAE(px) | Ang MAE(°) | Vel Corr | Jerk | Dir Chg | Delay(ms) |")
    table_lines.append("|--------|-------|-----------|---------|------------|----------|------|---------|-----------|")

    for video_name in ["horizontal_sweep", "diagonal_sweep", "oscillating"]:
        if video_name not in results_1a:
            continue

        for method in ["sam2", "yolo", "dino_k1", "dino_k5", "dino_k10"]:
            if method not in results_1a[video_name]:
                continue

            m = results_1a[video_name][method]
            table_lines.append(
                f"| {method} | {video_name[:10]} | "
                f"{m['amplitude_ratio']*100:.1f}% | "
                f"{m['mae_px']:.1f} | "
                f"{m['angular_mae_deg']:.2f} | "
                f"{m['velocity_correlation']:.3f} | "
                f"{m['jerk_avg']:.5f} | "
                f"{m['direction_changes_az']} | "
                f"{m['peak_delay_ms']:.0f} |"
            )

    table = "\n".join(table_lines)
    print(table)

    # Performance table
    print("\n--- Performance ---")
    perf_lines = []
    perf_lines.append("| Method | FPS | VRAM (MB) |")
    perf_lines.append("|--------|-----|-----------|")

    for r in results_1b:
        perf_lines.append(f"| {r['method']} | {r['fps_mean']:.1f} | {r['vram_mb']:.0f} |")

    perf_table = "\n".join(perf_lines)
    print(perf_table)

    # Save
    with open(output_dir / "3_comparison_table.md", "w") as f:
        f.write("# Comparison Table\n\n")
        f.write("## Trajectory Metrics\n\n")
        f.write(table)
        f.write("\n\n## Performance\n\n")
        f.write(perf_table)

    print(f"\nTable saved to: {output_dir}/3_comparison_table.md")
    return table


def main():
    """Run comprehensive evaluation."""
    print("=" * 70)
    print("COMPREHENSIVE VID2SPATIAL EVALUATION")
    print("DINO K=5 Hybrid Tracker")
    print("=" * 70)

    output_dir = Path("/home/seung/mmhoa/vid2spatial/eval/comprehensive_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_dir = Path("/home/seung/mmhoa/vid2spatial/eval/test_outputs/synthetic")

    # Generate synthetic videos
    print("\n[0] Generating/loading synthetic videos...")
    videos = generate_synthetic_videos(synthetic_dir)

    # 1A) Trajectory metrics
    results_1a = run_1a_trajectory_metrics(videos, output_dir)

    # 1B) Performance benchmarks
    results_1b = run_1b_performance_benchmark(videos, output_dir)

    # 2C) Depth validation
    results_2c = run_2c_depth_validation(videos, output_dir)

    # 3) Comparison table
    run_3_comparison_table(results_1a, results_1b, output_dir)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
