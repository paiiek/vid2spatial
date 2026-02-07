#!/usr/bin/env python3
"""
Ground-Truth Evaluation: Synthetic 3D → Video → Pipeline → Compare vs GT

15 synthetic scenes (5 seconds each) with known GT trajectories.
Pipeline output parameters (az, el, dist) are compared against GT.
Both GT and predicted trajectories render to binaural audio.

Date: 2026-02-06
"""

import sys, os
sys.path.insert(0, "/home/seung/mmhoa/vid2spatial")

import json
import time
import math
import traceback
import numpy as np
import cv2
import soundfile as sf
import torch
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).parent
VIDEO_DIR = BASE_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "outputs"
VIDEO_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

DURATION = 5.0       # seconds
FPS = 30
SR = 48000
W, H = 640, 480      # video resolution
FOV_DEG = 60.0        # must match pipeline


# =============================================================================
# Synthetic Scene Definition
# =============================================================================

@dataclass
class SyntheticScene:
    """A 3D scene with a known object trajectory."""
    id: str
    name: str
    object_label: str          # text prompt for pipeline
    object_color: Tuple[int,int,int]  # BGR
    object_radius: int         # pixels at z=1m
    bg_color: Tuple[int,int,int]
    motion_type: str
    # GT parameters per frame will be generated
    description: str = ""


SCENES: List[SyntheticScene] = [
    # === Category 1: Horizontal sweep ===
    SyntheticScene("01_ball_lr_slow", "Ball - Slow L/R", "ball", (0,0,255), 30, (40,40,40),
                   "horizontal_slow", "Red ball, slow left-right at 2m"),
    SyntheticScene("02_ball_lr_fast", "Ball - Fast L/R", "ball", (0,255,0), 30, (40,40,40),
                   "horizontal_fast", "Green ball, fast left-right at 2m"),
    SyntheticScene("03_box_lr_mid", "Box - Medium L/R", "box", (255,128,0), 40, (30,30,50),
                   "horizontal_mid", "Orange box, medium lateral at 3m"),

    # === Category 2: Depth motion ===
    SyntheticScene("04_sphere_approach", "Sphere - Approaching", "ball", (255,0,0), 25, (50,50,50),
                   "approaching", "Blue sphere, 8m→1m straight approach"),
    SyntheticScene("05_sphere_recede", "Sphere - Receding", "ball", (0,255,255), 25, (50,50,50),
                   "receding", "Cyan sphere, 1m→8m receding"),
    SyntheticScene("06_cube_depth_osc", "Cube - Depth Pulse", "box", (128,0,255), 35, (40,40,40),
                   "depth_oscillation", "Purple cube, 2m↔6m depth oscillation"),

    # === Category 3: Diagonal / combined ===
    SyntheticScene("07_ball_diagonal", "Ball - Diagonal", "ball", (255,255,0), 28, (30,30,30),
                   "diagonal", "Yellow ball, bottom-left to top-right"),
    SyntheticScene("08_circle_arc", "Circle - Arc", "ball", (255,0,255), 26, (40,40,40),
                   "arc", "Magenta circle, arc motion"),
    SyntheticScene("09_box_zigzag", "Box - Zigzag", "box", (0,200,100), 35, (35,35,35),
                   "zigzag", "Teal box, zigzag pattern"),

    # === Category 4: Circular ===
    SyntheticScene("10_ball_circle_slow", "Ball - Slow Circle", "ball", (100,200,255), 30, (30,30,30),
                   "circle_slow", "Light blue ball, slow full rotation at 3m"),
    SyntheticScene("11_ball_circle_fast", "Ball - Fast Circle", "ball", (255,100,50), 28, (30,30,30),
                   "circle_fast", "Orange ball, fast rotation at 2m"),

    # === Category 5: Static + micro ===
    SyntheticScene("12_ball_static_center", "Ball - Static Center", "ball", (200,200,200), 35, (20,20,20),
                   "static_center", "White ball, stationary at center 3m"),
    SyntheticScene("13_ball_static_left", "Ball - Static Left", "ball", (255,150,150), 30, (20,20,20),
                   "static_left", "Pink ball, stationary at left 4m"),

    # === Category 6: Complex ===
    SyntheticScene("14_ball_figure8", "Ball - Figure-8", "ball", (50,255,50), 28, (35,35,35),
                   "figure8", "Green ball, figure-8 pattern at 3m"),
    SyntheticScene("15_ball_spiral_in", "Ball - Spiral In", "ball", (255,200,0), 25, (30,30,30),
                   "spiral_in", "Gold ball, spiral inward 6m→2m"),
]


# =============================================================================
# GT Trajectory Generation
# =============================================================================

def pixel_to_az_el(cx: float, cy: float, w: int, h: int, fov_deg: float):
    """Convert pixel coordinate to azimuth/elevation (radians).

    Matches pipeline convention (vision.py pixel_to_ray + ray_to_angles):
      x = (px - cx) / f,  y = (py - cy) / f,  z = 1
      az = atan2(x, z),   el = arcsin(y / ||ray||)

    Note: pixel y-down → positive el means below horizon.
    This matches the pipeline's convention.
    """
    fov_rad = math.radians(fov_deg)
    f = (w / 2) / math.tan(fov_rad / 2)
    x = (cx - w/2) / f
    y = (cy - h/2) / f
    z = 1.0
    norm = math.sqrt(x*x + y*y + z*z)
    az = math.atan2(x, z)
    el = math.asin(y / norm)
    return az, el


def generate_gt_trajectory(scene: SyntheticScene) -> List[Dict]:
    """Generate ground-truth trajectory for a scene.

    Returns list of {frame, cx, cy, az, el, dist_m, radius_px} per frame.
    """
    n_frames = int(DURATION * FPS)
    t = np.linspace(0, DURATION, n_frames)
    motion = scene.motion_type
    base_r = scene.object_radius

    # Default: center, 3m
    cx_arr = np.full(n_frames, W / 2.0)
    cy_arr = np.full(n_frames, H / 2.0)
    dist_arr = np.full(n_frames, 3.0)

    # --- Motion patterns (pixel-space + distance) ---
    if motion == "horizontal_slow":
        cx_arr = W/2 + 200 * np.sin(2 * np.pi * 0.2 * t)  # ±200px, 0.2Hz
        dist_arr[:] = 2.0

    elif motion == "horizontal_fast":
        cx_arr = W/2 + 250 * np.sin(2 * np.pi * 1.0 * t)  # ±250px, 1Hz
        dist_arr[:] = 2.0

    elif motion == "horizontal_mid":
        cx_arr = W/2 + 180 * np.sin(2 * np.pi * 0.5 * t)
        dist_arr[:] = 3.0

    elif motion == "approaching":
        progress = t / DURATION
        dist_arr = 8.0 - 7.0 * progress  # 8→1m
        cx_arr[:] = W / 2  # Center

    elif motion == "receding":
        progress = t / DURATION
        dist_arr = 1.0 + 7.0 * progress  # 1→8m
        cx_arr[:] = W / 2

    elif motion == "depth_oscillation":
        dist_arr = 4.0 + 2.0 * np.sin(2 * np.pi * 0.3 * t)  # 2↔6m
        cx_arr[:] = W / 2

    elif motion == "diagonal":
        progress = t / DURATION
        cx_arr = 100 + 440 * progress  # left → right
        cy_arr = 380 - 280 * progress  # bottom → top
        dist_arr = 4.0 - 1.5 * progress

    elif motion == "arc":
        angle = np.pi * 0.8 * t / DURATION  # 0 → ~144°
        cx_arr = W/2 + 200 * np.cos(angle - np.pi/2)
        cy_arr = H/2 - 120 * np.sin(angle)
        dist_arr[:] = 3.0

    elif motion == "zigzag":
        # 3 segments
        seg_len = n_frames // 3
        for i in range(n_frames):
            seg = i // seg_len
            frac = (i % seg_len) / seg_len
            if seg == 0:
                cx_arr[i] = 120 + 400 * frac
                cy_arr[i] = H/2 - 100 * frac
            elif seg == 1:
                cx_arr[i] = 520 - 400 * frac
                cy_arr[i] = H/2 - 100 + 200 * frac
            else:
                cx_arr[i] = 120 + 400 * frac
                cy_arr[i] = H/2 + 100 - 100 * frac
        dist_arr[:] = 3.0

    elif motion == "circle_slow":
        angle = 2 * np.pi * 0.15 * t  # 0.15 Hz
        cx_arr = W/2 + 180 * np.cos(angle)
        cy_arr = H/2 + 100 * np.sin(angle)
        dist_arr[:] = 3.0

    elif motion == "circle_fast":
        angle = 2 * np.pi * 0.6 * t
        cx_arr = W/2 + 200 * np.cos(angle)
        cy_arr = H/2 + 120 * np.sin(angle)
        dist_arr[:] = 2.0

    elif motion == "static_center":
        cx_arr[:] = W / 2
        cy_arr[:] = H / 2
        dist_arr[:] = 3.0

    elif motion == "static_left":
        cx_arr[:] = W / 4
        cy_arr[:] = H / 2
        dist_arr[:] = 4.0

    elif motion == "figure8":
        cx_arr = W/2 + 180 * np.sin(2 * np.pi * 0.3 * t)
        cy_arr = H/2 + 80 * np.sin(4 * np.pi * 0.3 * t)
        dist_arr[:] = 3.0

    elif motion == "spiral_in":
        angle = 2 * np.pi * 0.4 * t
        progress = t / DURATION
        radius = 200 * (1 - 0.7 * progress)
        cx_arr = W/2 + radius * np.cos(angle)
        cy_arr = H/2 + radius * 0.6 * np.sin(angle)
        dist_arr = 6.0 - 4.0 * progress  # 6→2m

    # Compute az, el from pixel positions
    frames = []
    for i in range(n_frames):
        cx, cy = float(cx_arr[i]), float(cy_arr[i])
        d = float(dist_arr[i])
        az, el = pixel_to_az_el(cx, cy, W, H, FOV_DEG)

        # Object apparent size: inversely proportional to distance
        r_px = max(5, int(base_r * (2.0 / max(d, 0.5))))

        frames.append({
            "frame": i,
            "cx": round(cx, 2),
            "cy": round(cy, 2),
            "az": round(az, 5),
            "el": round(el, 5),
            "dist_m": round(d, 3),
            "radius_px": r_px,
        })

    return frames


# =============================================================================
# Synthetic Video Rendering
# =============================================================================

def render_synthetic_video(scene: SyntheticScene, gt_frames: List[Dict], output_path: str):
    """Render a synthetic video with a moving object on solid background."""
    n_frames = len(gt_frames)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, FPS, (W, H))

    for f in gt_frames:
        # Background
        frame = np.full((H, W, 3), scene.bg_color, dtype=np.uint8)

        # Add subtle grid for depth cue
        for gx in range(0, W, 80):
            cv2.line(frame, (gx, 0), (gx, H), tuple(c + 15 for c in scene.bg_color), 1)
        for gy in range(0, H, 80):
            cv2.line(frame, (0, gy), (W, gy), tuple(c + 15 for c in scene.bg_color), 1)

        # Draw object
        cx, cy = int(f["cx"]), int(f["cy"])
        r = f["radius_px"]

        if "box" in scene.object_label:
            cv2.rectangle(frame, (cx - r, cy - r), (cx + r, cy + r), scene.object_color, -1)
            cv2.rectangle(frame, (cx - r, cy - r), (cx + r, cy + r), (255, 255, 255), 1)
        else:
            cv2.circle(frame, (cx, cy), r, scene.object_color, -1)
            cv2.circle(frame, (cx, cy), r, (255, 255, 255), 1)

        out.write(frame)

    out.release()


# =============================================================================
# Pipeline Runner
# =============================================================================

def run_pipeline(video_path: str, prompt: str) -> Dict:
    """Run tracking pipeline on a video. Returns trajectory dict or raises."""
    from vid2spatial_pkg.hybrid_tracker import HybridTracker

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tracker = HybridTracker(
        device="cuda" if torch.cuda.is_available() else "cpu",
        box_threshold=0.15,
        fov_deg=FOV_DEG,
    )

    result = tracker.track(
        video_path=video_path,
        text_prompt=prompt,
        tracking_method="adaptive_k",
        depth_stride=5,
    )

    trajectory = result.get_trajectory_3d(smooth=False, enhance_depth=True)

    del tracker
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return trajectory, result


# =============================================================================
# Comparison Metrics
# =============================================================================

def compare_gt_vs_predicted(gt_frames: List[Dict], pred_frames: List[Dict]) -> Dict:
    """
    Compare GT trajectory against pipeline prediction.

    Metrics:
    - az_mae: Mean absolute error in azimuth (degrees)
    - el_mae: Mean absolute error in elevation (degrees)
    - dist_mae: Mean absolute error in distance (meters)
    - az_correlation: Pearson correlation of azimuth series
    - el_correlation: Pearson correlation of elevation series
    - dist_correlation: Pearson correlation of distance series
    - position_rmse: RMS error of (az, el) in degrees
    """
    # Align frames by index: resample predicted to GT length
    gt_az = np.array([f["az"] for f in gt_frames])
    gt_el = np.array([f["el"] for f in gt_frames])
    gt_dist = np.array([f["dist_m"] for f in gt_frames])

    pred_az = np.array([f["az"] for f in pred_frames])
    pred_el = np.array([f["el"] for f in pred_frames])
    pred_dist = np.array([f.get("dist_m", f.get("depth_blended", 1.0)) for f in pred_frames])

    # Resample predicted to GT length
    n_gt = len(gt_az)
    n_pred = len(pred_az)

    if n_pred != n_gt:
        pred_idx = np.linspace(0, n_pred - 1, n_gt)
        pred_az = np.interp(pred_idx, np.arange(n_pred), pred_az)
        pred_el = np.interp(pred_idx, np.arange(n_pred), pred_el)
        pred_dist = np.interp(pred_idx, np.arange(n_pred), pred_dist)

    # Convert to degrees for interpretability
    gt_az_deg = np.degrees(gt_az)
    gt_el_deg = np.degrees(gt_el)
    pred_az_deg = np.degrees(pred_az)
    pred_el_deg = np.degrees(pred_el)

    # MAE
    az_mae = float(np.mean(np.abs(gt_az_deg - pred_az_deg)))
    el_mae = float(np.mean(np.abs(gt_el_deg - pred_el_deg)))
    dist_mae = float(np.mean(np.abs(gt_dist - pred_dist)))

    # Correlation (handle constant arrays)
    def safe_corr(a, b):
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            # If both constant and close, perfect match
            if np.std(a) < 1e-8 and np.std(b) < 1e-8:
                return 1.0 if np.abs(np.mean(a) - np.mean(b)) < 1.0 else 0.0
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    az_corr = safe_corr(gt_az_deg, pred_az_deg)
    el_corr = safe_corr(gt_el_deg, pred_el_deg)
    dist_corr = safe_corr(gt_dist, pred_dist)

    # RMSE of angular position
    angle_errors = np.sqrt((gt_az_deg - pred_az_deg)**2 + (gt_el_deg - pred_el_deg)**2)
    position_rmse = float(np.sqrt(np.mean(angle_errors**2)))

    # Amplitude ratio for periodic motions
    gt_az_amp = (np.max(gt_az_deg) - np.min(gt_az_deg)) / 2
    pred_az_amp = (np.max(pred_az_deg) - np.min(pred_az_deg)) / 2
    az_amp_ratio = float(pred_az_amp / (gt_az_amp + 1e-6)) * 100

    gt_dist_amp = (np.max(gt_dist) - np.min(gt_dist)) / 2
    pred_dist_amp = (np.max(pred_dist) - np.min(pred_dist)) / 2
    dist_amp_ratio = float(pred_dist_amp / (gt_dist_amp + 1e-6)) * 100

    return {
        "az_mae_deg": round(az_mae, 3),
        "el_mae_deg": round(el_mae, 3),
        "dist_mae_m": round(dist_mae, 3),
        "az_correlation": round(az_corr, 4),
        "el_correlation": round(el_corr, 4),
        "dist_correlation": round(dist_corr, 4),
        "position_rmse_deg": round(position_rmse, 3),
        "az_amplitude_ratio_pct": round(az_amp_ratio, 1),
        "dist_amplitude_ratio_pct": round(dist_amp_ratio, 1),
        "n_gt_frames": n_gt,
        "n_pred_frames": n_pred,

        # Raw arrays for plotting
        "_gt_az_deg": gt_az_deg.tolist(),
        "_pred_az_deg": pred_az_deg.tolist(),
        "_gt_el_deg": gt_el_deg.tolist(),
        "_pred_el_deg": pred_el_deg.tolist(),
        "_gt_dist": gt_dist.tolist(),
        "_pred_dist": pred_dist.tolist(),
    }


# =============================================================================
# Audio Rendering
# =============================================================================

def render_audio_from_trajectory(
    traj_frames: List[Dict],
    output_path: str,
    sr: int = SR,
    duration: float = DURATION,
) -> str:
    """Render FOA then binaural from trajectory frames."""
    from vid2spatial_pkg.foa_render import (
        encode_mono_to_foa, foa_to_binaural, write_foa_wav,
        apply_distance_gain_lpf, smooth_limit_angles,
        build_wet_curve_from_dist_occ, apply_timevarying_reverb_foa,
    )

    n_samples = int(sr * duration)
    n_frames = len(traj_frames)

    # Generate a rich test tone
    t = np.linspace(0, duration, n_samples)
    audio = np.zeros(n_samples, dtype=np.float32)
    for f0 in [261.63, 329.63, 392.0, 523.25]:  # C major chord
        for h in range(1, 4):
            audio += (0.3 ** h) * np.sin(2 * np.pi * f0 * h * t).astype(np.float32)
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.7

    # Interpolate trajectory to sample rate
    frame_idx = np.array([f["frame"] for f in traj_frames], dtype=np.float32)
    az_vals = np.array([f["az"] for f in traj_frames], dtype=np.float32)
    el_vals = np.array([f["el"] for f in traj_frames], dtype=np.float32)
    dist_vals = np.array([f.get("dist_m", f.get("depth_blended", 2.0)) for f in traj_frames], dtype=np.float32)

    s = np.linspace(frame_idx[0], frame_idx[-1], n_samples)
    az_s = np.interp(s, frame_idx, az_vals).astype(np.float32)
    el_s = np.interp(s, frame_idx, el_vals).astype(np.float32)
    dist_s = np.interp(s, frame_idx, dist_vals).astype(np.float32)

    d_rel_s = np.clip((dist_s - 0.5) / 9.5, 0.0, 1.0).astype(np.float32)

    # Distance processing
    audio_proc = apply_distance_gain_lpf(audio, sr, dist_s, d_rel_s)

    # Smooth angles
    az_smooth, el_smooth = smooth_limit_angles(az_s, el_s, sr, smooth_ms=20.0)

    # FOA encode
    foa = encode_mono_to_foa(audio_proc, az_smooth, el_smooth)

    # Reverb
    wet = build_wet_curve_from_dist_occ(d_rel_s, wet_min=0.03, wet_max=0.25)
    foa = apply_timevarying_reverb_foa(foa, sr, wet, rt60=0.4)

    # Save FOA
    foa_path = output_path.replace("_binaural.wav", "_foa.wav")
    write_foa_wav(foa_path, foa, sr)

    # Binaural
    binaural = foa_to_binaural(foa, sr)
    sf.write(output_path, binaural.T, sr, subtype="FLOAT")

    return output_path


# =============================================================================
# Main
# =============================================================================

def process_scene(scene: SyntheticScene) -> Dict:
    """Process a single synthetic scene end-to-end."""
    sid = scene.id
    scene_out = OUTPUT_DIR / sid
    scene_out.mkdir(exist_ok=True)

    result = {"scene_id": sid, "name": scene.name, "motion": scene.motion_type}

    # Step 1: Generate GT trajectory
    gt_frames = generate_gt_trajectory(scene)
    gt_path = str(scene_out / "gt_trajectory.json")
    with open(gt_path, "w") as f:
        json.dump(gt_frames, f, indent=2)
    result["n_gt_frames"] = len(gt_frames)

    # Step 2: Render synthetic video
    video_path = str(VIDEO_DIR / f"{sid}.mp4")
    render_synthetic_video(scene, gt_frames, video_path)
    result["video_path"] = video_path

    # Step 3: Run pipeline
    t0 = time.time()
    try:
        trajectory, tracking_result = run_pipeline(video_path, scene.object_label)
        pred_frames = trajectory.get("frames", [])
        result["pipeline_time_sec"] = round(time.time() - t0, 2)
        result["n_pred_frames"] = len(pred_frames)
        result["pipeline_status"] = "ok"

        # Save predicted trajectory
        pred_path = str(scene_out / "pred_trajectory.json")
        with open(pred_path, "w") as f:
            json.dump(pred_frames, f, indent=2, default=str)

    except Exception as e:
        result["pipeline_status"] = "error"
        result["pipeline_error"] = str(e)
        traceback.print_exc()
        return result

    # Step 4: Compare GT vs Predicted
    try:
        metrics = compare_gt_vs_predicted(gt_frames, pred_frames)

        # Remove raw arrays from summary (keep for detailed file)
        detailed_metrics = metrics.copy()
        summary_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
        result["metrics"] = summary_metrics

        # Save detailed comparison
        detail_path = str(scene_out / "comparison_detail.json")
        with open(detail_path, "w") as f:
            json.dump(detailed_metrics, f, indent=2)

    except Exception as e:
        result["metrics"] = {"error": str(e)}
        traceback.print_exc()

    # Step 5: Render audio - GT trajectory
    try:
        gt_audio_path = str(scene_out / f"{sid}_gt_binaural.wav")
        render_audio_from_trajectory(gt_frames, gt_audio_path)
        result["gt_audio"] = gt_audio_path
    except Exception as e:
        result["gt_audio_error"] = str(e)
        traceback.print_exc()

    # Step 6: Render audio - Predicted trajectory
    try:
        pred_audio_path = str(scene_out / f"{sid}_pred_binaural.wav")
        render_audio_from_trajectory(pred_frames, pred_audio_path)
        result["pred_audio"] = pred_audio_path
    except Exception as e:
        result["pred_audio_error"] = str(e)
        traceback.print_exc()

    return result


def main():
    print("=" * 70)
    print("  GT Evaluation: Synthetic 3D Scenes → Pipeline → Compare")
    print("  15 scenes × 5s each | Date: 2026-02-06")
    print("=" * 70)

    all_results = []

    for i, scene in enumerate(SCENES):
        print(f"\n{'='*70}")
        print(f"[{i+1}/15] {scene.id}: {scene.name}")
        print(f"  Prompt: \"{scene.object_label}\" | Motion: {scene.motion_type}")
        print(f"  {scene.description}")
        print(f"{'='*70}")

        result = process_scene(scene)
        all_results.append(result)

        # Print metrics
        m = result.get("metrics", {})
        if "error" not in m:
            print(f"  Pipeline: {result.get('pipeline_time_sec', 0):.1f}s "
                  f"({result.get('n_pred_frames', 0)} frames)")
            print(f"  Az  MAE: {m.get('az_mae_deg', -1):.2f}°  | corr: {m.get('az_correlation', -1):.3f} "
                  f"| amp: {m.get('az_amplitude_ratio_pct', -1):.0f}%")
            print(f"  El  MAE: {m.get('el_mae_deg', -1):.2f}°  | corr: {m.get('el_correlation', -1):.3f}")
            print(f"  Dist MAE: {m.get('dist_mae_m', -1):.2f}m | corr: {m.get('dist_correlation', -1):.3f} "
                  f"| amp: {m.get('dist_amplitude_ratio_pct', -1):.0f}%")
            print(f"  Pos RMSE: {m.get('position_rmse_deg', -1):.2f}°")
        else:
            print(f"  Metrics ERROR: {m['error']}")

    # === Save all results ===
    results_path = BASE_DIR / "gt_eval_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "date": "2026-02-06",
            "n_scenes": len(SCENES),
            "results": all_results,
        }, f, indent=2, default=str)

    # === Print Summary Table ===
    print("\n\n" + "=" * 90)
    print("  SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Scene':<28} {'Az MAE':>7} {'Az Corr':>8} {'El MAE':>7} {'Dist MAE':>9} {'Pos RMSE':>9} {'Time':>6}")
    print("-" * 90)

    ok_results = [r for r in all_results if r.get("pipeline_status") == "ok"]

    for r in all_results:
        sid = r["scene_id"]
        m = r.get("metrics", {})
        if "error" in m or r.get("pipeline_status") != "ok":
            print(f"{sid:<28} {'FAILED':>7}")
            continue
        print(f"{sid:<28} {m['az_mae_deg']:>6.2f}° {m['az_correlation']:>7.3f} "
              f"{m['el_mae_deg']:>6.2f}° {m['dist_mae_m']:>7.2f}m "
              f"{m['position_rmse_deg']:>7.2f}° {r.get('pipeline_time_sec',0):>5.1f}s")

    # Averages
    if ok_results:
        avg_az = np.mean([r["metrics"]["az_mae_deg"] for r in ok_results])
        avg_el = np.mean([r["metrics"]["el_mae_deg"] for r in ok_results])
        avg_dist = np.mean([r["metrics"]["dist_mae_m"] for r in ok_results])
        avg_rmse = np.mean([r["metrics"]["position_rmse_deg"] for r in ok_results])
        avg_az_corr = np.mean([r["metrics"]["az_correlation"] for r in ok_results])
        avg_time = np.mean([r.get("pipeline_time_sec", 0) for r in ok_results])

        print("-" * 90)
        print(f"{'AVERAGE':<28} {avg_az:>6.2f}° {avg_az_corr:>7.3f} "
              f"{avg_el:>6.2f}° {avg_dist:>7.2f}m "
              f"{avg_rmse:>7.2f}° {avg_time:>5.1f}s")

    print(f"\nSuccess: {len(ok_results)}/{len(all_results)}")
    print(f"Results: {results_path}")
    print(f"Audio pairs: {OUTPUT_DIR}/*/")
    print("\nListen: *_gt_binaural.wav = GT audio, *_pred_binaural.wav = pipeline audio")
    print("=" * 90)


if __name__ == "__main__":
    main()
