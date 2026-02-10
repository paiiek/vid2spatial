#!/usr/bin/env python3
"""
Trajectory overlay visualization tool for Vid2Spatial.

Generates annotated videos showing:
- GT trajectory (green) vs Predicted trajectory (red) for synthetic scenes
- Predicted trajectory (red) with confidence/depth HUD for real videos
- Trail history, per-frame metrics, error arrows

Usage:
    # Synthetic GT overlay (all 15 scenes)
    python evaluation/visualize_trajectory.py --mode gt

    # Real video overlay (E2E + SOT)
    python evaluation/visualize_trajectory.py --mode real

    # Single video
    python evaluation/visualize_trajectory.py --video path/to/video.mp4 --trajectory path/to/traj.json

    # All
    python evaluation/visualize_trajectory.py --mode all
"""
import os
import sys
import json
import argparse
import math
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = Path(__file__).resolve().parent.parent
GT_DIR = ROOT / "experiments" / "gt_eval_synthetic"
E2E_DIR = ROOT / "experiments" / "e2e_20_videos"
SOT_DIR = ROOT / "experiments" / "sot_15_videos"
OUTPUT_DIR = ROOT / "evaluation" / "trajectory_videos"


# ── Coordinate conversion ──────────────────────────────────────────

def angles_to_pixel(az: float, el: float, width: int, height: int,
                    fov_deg: float = 60.0) -> Tuple[float, float]:
    """Convert (az, el) in radians to pixel (cx, cy).

    Inverse of the pipeline's pixel_to_ray + ray_to_angles:
      az = atan2(x, z), el = arcsin(y / norm)
    where x = (px - W/2)/f, y = (py - H/2)/f, z = 1.
    """
    f = width / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
    x = math.tan(az)
    # el = arcsin(y / sqrt(x^2 + y^2 + 1))  →  y = tan(el) * sqrt(x^2 + 1)
    y = math.tan(el) * math.sqrt(x * x + 1.0)
    cx = width / 2.0 + f * x
    cy = height / 2.0 + f * y  # pixel y-down convention
    return cx, cy


def bbox_to_rect(cx: float, cy: float, w: int, h: int) -> Tuple[int, int, int, int]:
    """Convert center + size to (x1, y1, x2, y2)."""
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    return x1, y1, x1 + w, y1 + h


# ── Drawing utilities ──────────────────────────────────────────────

def draw_circle(frame, cx, cy, radius, color, thickness=2, label=None):
    """Draw circle with optional label."""
    pt = (int(cx), int(cy))
    cv2.circle(frame, pt, radius, color, thickness)
    cv2.circle(frame, pt, 3, color, -1)  # center dot
    if label:
        cv2.putText(frame, label, (pt[0] + radius + 4, pt[1] - 4),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def draw_bbox(frame, cx, cy, w, h, color, thickness=1):
    """Draw bounding box from center + size."""
    x1, y1, x2, y2 = bbox_to_rect(cx, cy, w, h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_trail(frame, trail, color, max_len=30):
    """Draw fading trail of past positions."""
    n = len(trail)
    for i in range(1, min(n, max_len)):
        alpha = i / min(n, max_len)
        c = tuple(int(v * alpha) for v in color)
        pt1 = (int(trail[-(i+1)][0]), int(trail[-(i+1)][1]))
        pt2 = (int(trail[-i][0]), int(trail[-i][1]))
        cv2.line(frame, pt1, pt2, c, max(1, int(2 * alpha)), cv2.LINE_AA)


def draw_error_arrow(frame, pred_cx, pred_cy, gt_cx, gt_cy, color=(0, 255, 255)):
    """Draw arrow from predicted to GT position."""
    pt1 = (int(pred_cx), int(pred_cy))
    pt2 = (int(gt_cx), int(gt_cy))
    dist = math.hypot(pred_cx - gt_cx, pred_cy - gt_cy)
    if dist > 3:
        cv2.arrowedLine(frame, pt1, pt2, color, 1, cv2.LINE_AA, tipLength=0.3)


def draw_hud(frame, frame_idx, total_frames, metrics: dict, y_start=20):
    """Draw heads-up display with per-frame metrics."""
    y = y_start
    h, w = frame.shape[:2]

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, y_start - 15), (260, y_start + len(metrics) * 18 + 5),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Frame counter
    cv2.putText(frame, f"Frame {frame_idx}/{total_frames}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    y += 18

    for key, val in metrics.items():
        if isinstance(val, float):
            text = f"{key}: {val:.3f}"
        else:
            text = f"{key}: {val}"
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        y += 18


def draw_legend(frame, items: List[Tuple[str, Tuple[int, int, int]]]):
    """Draw color legend in bottom-left."""
    h, w = frame.shape[:2]
    y = h - 15 * len(items) - 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, y - 5), (150, h - 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for label, color in items:
        cv2.circle(frame, (15, y + 5), 5, color, -1)
        cv2.putText(frame, label, (25, y + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        y += 15


def draw_spatial_gauge(frame, az_deg, el_deg, dist_m, conf):
    """Draw spatial parameter gauge in top-right."""
    h, w = frame.shape[:2]
    x0 = w - 180
    y0 = 15

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 5, y0 - 12), (w - 5, y0 + 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"Az: {az_deg:+6.1f} deg", (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, f"El: {el_deg:+6.1f} deg", (x0, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Dist: {dist_m:5.2f} m", (x0, y0 + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1, cv2.LINE_AA)

    # Confidence bar
    bar_w = 100
    bar_h = 8
    cv2.rectangle(frame, (x0, y0 + 42), (x0 + bar_w, y0 + 42 + bar_h), (80, 80, 80), -1)
    fill = int(bar_w * conf)
    bar_color = (0, 255, 0) if conf > 0.6 else (0, 200, 255) if conf > 0.35 else (0, 0, 255)
    cv2.rectangle(frame, (x0, y0 + 42), (x0 + fill, y0 + 42 + bar_h), bar_color, -1)
    cv2.putText(frame, f"Conf: {conf:.2f}", (x0 + bar_w + 5, y0 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)


# ── Trajectory plotting (bottom panel) ─────────────────────────────

def draw_trajectory_plot(frame, gt_trail, pred_trail, current_idx, total_frames):
    """Draw az/el timeseries plot as bottom strip."""
    h, w = frame.shape[:2]
    plot_h = 80
    plot_y0 = h - plot_h

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, plot_y0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Axis
    cv2.line(frame, (30, plot_y0 + plot_h // 2), (w - 10, plot_y0 + plot_h // 2),
             (60, 60, 60), 1)
    cv2.putText(frame, "Az", (5, plot_y0 + plot_h // 2 + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

    if not pred_trail:
        return

    # Scale x to frame width
    def x_pos(idx):
        return int(30 + (w - 40) * idx / max(total_frames, 1))

    def y_pos(az, range_min=-0.6, range_max=0.6):
        norm = (az - range_min) / (range_max - range_min + 1e-8)
        return int(plot_y0 + plot_h - plot_h * np.clip(norm, 0, 1))

    # Determine az range from all data
    all_az = [t[2] for t in pred_trail]
    if gt_trail:
        all_az += [t[2] for t in gt_trail]
    az_min = min(all_az) - 0.05
    az_max = max(all_az) + 0.05

    # Draw GT line
    if gt_trail and len(gt_trail) > 1:
        for i in range(1, len(gt_trail)):
            pt1 = (x_pos(gt_trail[i-1][3]), y_pos(gt_trail[i-1][2], az_min, az_max))
            pt2 = (x_pos(gt_trail[i][3]), y_pos(gt_trail[i][2], az_min, az_max))
            cv2.line(frame, pt1, pt2, (0, 180, 0), 1, cv2.LINE_AA)

    # Draw pred line
    if len(pred_trail) > 1:
        for i in range(1, len(pred_trail)):
            pt1 = (x_pos(pred_trail[i-1][3]), y_pos(pred_trail[i-1][2], az_min, az_max))
            pt2 = (x_pos(pred_trail[i][3]), y_pos(pred_trail[i][2], az_min, az_max))
            cv2.line(frame, pt1, pt2, (0, 0, 200), 1, cv2.LINE_AA)

    # Current position marker
    cv2.line(frame, (x_pos(current_idx), plot_y0), (x_pos(current_idx), h),
             (255, 255, 255), 1)


# ── Main overlay functions ─────────────────────────────────────────

def overlay_gt_scene(scene_id: str, output_dir: Path) -> Optional[dict]:
    """Generate overlay video for a synthetic GT scene."""
    video_path = GT_DIR / "videos" / f"{scene_id}.mp4"
    gt_path = GT_DIR / "outputs" / scene_id / "gt_trajectory.json"
    pred_path = GT_DIR / "outputs" / scene_id / "pred_trajectory.json"

    if not video_path.exists() or not gt_path.exists() or not pred_path.exists():
        print(f"  [skip] Missing files for {scene_id}")
        return None

    with open(gt_path) as f:
        gt_frames = json.load(f)
    with open(pred_path) as f:
        pred_frames = json.load(f)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [skip] Cannot open {video_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = output_dir / f"{scene_id}_overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height + 80))

    # Build lookup dicts
    gt_by_frame = {f['frame']: f for f in gt_frames}
    pred_by_frame = {f['frame']: f for f in pred_frames}

    gt_trail = []   # (cx, cy, az, frame_idx)
    pred_trail = []
    errors = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gt = gt_by_frame.get(frame_idx)
        pred = pred_by_frame.get(frame_idx)

        # GT position (direct pixel)
        if gt:
            gt_cx, gt_cy = gt['cx'], gt['cy']
            gt_r = gt.get('radius_px', 15)
            draw_circle(frame, gt_cx, gt_cy, gt_r, (0, 200, 0), 2, "GT")
            gt_trail.append((gt_cx, gt_cy, gt['az'], frame_idx))
            draw_trail(frame, [(t[0], t[1]) for t in gt_trail], (0, 180, 0))

        # Pred position (convert az/el to pixel)
        if pred:
            pred_cx, pred_cy = angles_to_pixel(pred['az'], pred['el'], width, height)
            pred_w = pred.get('w', 30)
            pred_h = pred.get('h', 30)
            draw_circle(frame, pred_cx, pred_cy, max(pred_w, pred_h) // 2,
                        (0, 0, 220), 2, "Pred")
            draw_bbox(frame, pred_cx, pred_cy, pred_w, pred_h, (0, 0, 180), 1)
            pred_trail.append((pred_cx, pred_cy, pred['az'], frame_idx))
            draw_trail(frame, [(t[0], t[1]) for t in pred_trail], (0, 0, 200))

            # Error arrow
            if gt:
                draw_error_arrow(frame, pred_cx, pred_cy, gt_cx, gt_cy)
                px_err = math.hypot(pred_cx - gt_cx, pred_cy - gt_cy)
                az_err = abs(math.degrees(pred['az'] - gt['az']))
                el_err = abs(math.degrees(pred['el'] - gt['el']))
                dist_err = abs(pred['dist_m'] - gt['dist_m'])
                errors.append(px_err)

                hud = {
                    "px_err": f"{px_err:.1f} px",
                    "az_err": f"{az_err:.2f} deg",
                    "el_err": f"{el_err:.2f} deg",
                    "dist_err": f"{dist_err:.2f} m",
                    "conf": pred.get('confidence', 0),
                }
                draw_hud(frame, frame_idx, total, hud)

            # Spatial gauge
            draw_spatial_gauge(frame, math.degrees(pred['az']), math.degrees(pred['el']),
                               pred['dist_m'], pred.get('confidence', 0))

        # Legend
        draw_legend(frame, [("GT", (0, 200, 0)), ("Pred", (0, 0, 220)), ("Error", (0, 255, 255))])

        # Extend frame for bottom plot
        plot_strip = np.zeros((80, width, 3), dtype=np.uint8)
        combined = np.vstack([frame, plot_strip])
        draw_trajectory_plot(combined, gt_trail, pred_trail, frame_idx, total)

        out.write(combined)
        frame_idx += 1

    cap.release()
    out.release()

    metrics = {}
    if errors:
        metrics = {
            "scene_id": scene_id,
            "frames": frame_idx,
            "px_err_mean": float(np.mean(errors)),
            "px_err_max": float(np.max(errors)),
            "px_err_std": float(np.std(errors)),
        }
    print(f"  [done] {scene_id}: {frame_idx} frames, px_err={metrics.get('px_err_mean', 0):.1f} px → {out_path.name}")
    return metrics


def overlay_real_video(video_path: str, traj_path: str, raw_path: Optional[str],
                       output_dir: Path, video_id: str) -> Optional[dict]:
    """Generate overlay video for a real video."""
    video_p = Path(video_path)
    traj_p = Path(traj_path)

    if not video_p.exists() or not traj_p.exists():
        print(f"  [skip] Missing files for {video_id}")
        return None

    with open(traj_p) as f:
        traj_data = json.load(f)

    frames_data = traj_data.get("frames", traj_data)
    intrinsics = traj_data.get("intrinsics", {})

    # Load raw tracking if available (has pixel coords directly)
    raw_by_frame = {}
    if raw_path and Path(raw_path).exists():
        with open(raw_path) as f:
            raw_data = json.load(f)
        raw_by_frame = {r['frame_idx']: r for r in raw_data}

    traj_by_frame = {f['frame']: f for f in frames_data}

    cap = cv2.VideoCapture(str(video_p))
    if not cap.isOpened():
        print(f"  [skip] Cannot open {video_p}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fov = intrinsics.get('fov_deg', 60.0)

    out_path = output_dir / f"{video_id}_overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height + 80))

    pred_trail = []
    confs = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = traj_by_frame.get(frame_idx)
        raw = raw_by_frame.get(frame_idx)

        if t:
            # Use raw tracking center if available (exact pixel), otherwise convert from angles
            if raw and 'center' in raw:
                pred_cx, pred_cy = raw['center']
            else:
                pred_cx, pred_cy = angles_to_pixel(t['az'], t['el'], width, height, fov)

            tw = t.get('w', 40)
            th = t.get('h', 40)
            conf = t.get('confidence', raw.get('confidence', 0) if raw else 0)

            # Color based on confidence
            if conf > 0.6:
                color = (0, 200, 0)   # green = high confidence
            elif conf > 0.35:
                color = (0, 200, 255)  # orange = medium
            else:
                color = (0, 0, 255)    # red = low

            # Draw bbox and center
            draw_bbox(frame, pred_cx, pred_cy, tw, th, color, 2)
            cv2.circle(frame, (int(pred_cx), int(pred_cy)), 4, color, -1)

            pred_trail.append((pred_cx, pred_cy, t['az'], frame_idx))
            draw_trail(frame, [(pt[0], pt[1]) for pt in pred_trail], color)
            confs.append(conf)

            # HUD
            hud = {
                "conf": f"{conf:.3f}",
                "dist_m": f"{t['dist_m']:.2f} m",
                "d_rel": f"{t.get('d_rel', 0):.3f}",
                "az": f"{math.degrees(t['az']):+.1f} deg",
                "el": f"{math.degrees(t['el']):+.1f} deg",
            }
            draw_hud(frame, frame_idx, total, hud)

            # Spatial gauge
            draw_spatial_gauge(frame, math.degrees(t['az']), math.degrees(t['el']),
                               t['dist_m'], conf)

        # Legend
        draw_legend(frame, [("High conf", (0, 200, 0)), ("Med conf", (0, 200, 255)),
                             ("Low conf", (0, 0, 255))])

        # Bottom plot
        plot_strip = np.zeros((80, width, 3), dtype=np.uint8)
        combined = np.vstack([frame, plot_strip])
        draw_trajectory_plot(combined, None, pred_trail, frame_idx, total)

        out.write(combined)
        frame_idx += 1

    cap.release()
    out.release()

    metrics = {
        "video_id": video_id,
        "frames": frame_idx,
        "mean_conf": float(np.mean(confs)) if confs else 0,
        "min_conf": float(np.min(confs)) if confs else 0,
        "tracked_pct": len(confs) / max(frame_idx, 1) * 100,
    }
    print(f"  [done] {video_id}: {frame_idx} frames, conf={metrics['mean_conf']:.3f} → {out_path.name}")
    return metrics


# ── Batch runners ──────────────────────────────────────────────────

def run_gt_overlays(output_dir: Path) -> List[dict]:
    """Generate overlay videos for all synthetic GT scenes."""
    print("\n=== Synthetic GT Overlay Videos ===")
    gt_out = output_dir / "gt_synthetic"
    gt_out.mkdir(parents=True, exist_ok=True)

    scenes = sorted([d.name for d in (GT_DIR / "outputs").iterdir() if d.is_dir()])
    results = []
    for scene_id in scenes:
        m = overlay_gt_scene(scene_id, gt_out)
        if m:
            results.append(m)
    return results


def run_e2e_overlays(output_dir: Path) -> List[dict]:
    """Generate overlay videos for E2E real videos."""
    print("\n=== E2E Real Video Overlay Videos ===")
    e2e_out = output_dir / "e2e_real"
    e2e_out.mkdir(parents=True, exist_ok=True)

    videos_dir = E2E_DIR / "videos"
    outputs_dir = E2E_DIR / "outputs"
    if not videos_dir.exists():
        print("  [skip] No E2E videos directory")
        return []

    results = []
    for vf in sorted(videos_dir.glob("*.mp4")):
        vid_id = vf.stem
        traj_p = outputs_dir / vid_id / "trajectory_3d.json"
        raw_p = outputs_dir / vid_id / "raw_tracking.json"
        m = overlay_real_video(str(vf), str(traj_p),
                               str(raw_p) if raw_p.exists() else None,
                               e2e_out, vid_id)
        if m:
            results.append(m)
    return results


def run_sot_overlays(output_dir: Path) -> List[dict]:
    """Generate overlay videos for SOT benchmark videos."""
    print("\n=== SOT Benchmark Overlay Videos ===")
    sot_out = output_dir / "sot_benchmark"
    sot_out.mkdir(parents=True, exist_ok=True)

    videos_dir = SOT_DIR / "videos"
    outputs_dir = SOT_DIR / "outputs"
    if not videos_dir.exists():
        print("  [skip] No SOT videos directory")
        return []

    results = []
    for vf in sorted(videos_dir.glob("*.mp4")):
        vid_id = vf.stem
        traj_p = outputs_dir / vid_id / "trajectory_3d.json"
        raw_p = outputs_dir / vid_id / "raw_tracking.json"
        m = overlay_real_video(str(vf), str(traj_p),
                               str(raw_p) if raw_p.exists() else None,
                               sot_out, vid_id)
        if m:
            results.append(m)
    return results


def generate_summary_report(gt_results, e2e_results, sot_results, output_dir: Path):
    """Generate a markdown summary report of all evaluations."""
    report_path = output_dir / "TRAJECTORY_VISUAL_REPORT.md"
    all_results = {
        "gt_synthetic": gt_results,
        "e2e_real": e2e_results,
        "sot_benchmark": sot_results,
    }

    # Save JSON
    json_path = output_dir / "trajectory_visual_results.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Markdown report
    lines = ["# Trajectory Visual Evaluation Report\n"]
    lines.append(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # GT results
    if gt_results:
        lines.append("\n## 1. Synthetic GT Evaluation (Pixel-Accurate)\n")
        lines.append("| Scene | Frames | Px Err Mean | Px Err Max | Px Err Std |")
        lines.append("|-------|--------|-------------|------------|------------|")
        for r in gt_results:
            lines.append(f"| {r['scene_id']} | {r['frames']} | "
                        f"{r['px_err_mean']:.1f} | {r['px_err_max']:.1f} | "
                        f"{r['px_err_std']:.1f} |")

        avg_err = np.mean([r['px_err_mean'] for r in gt_results])
        max_err = max(r['px_err_max'] for r in gt_results)
        lines.append(f"\n**Average pixel error: {avg_err:.1f} px** | "
                    f"**Max pixel error: {max_err:.1f} px**\n")

    # E2E results
    if e2e_results:
        lines.append("\n## 2. E2E Real Video Tracking\n")
        lines.append("| Video | Frames | Mean Conf | Min Conf | Tracked % |")
        lines.append("|-------|--------|-----------|----------|-----------|")
        for r in e2e_results:
            lines.append(f"| {r['video_id']} | {r['frames']} | "
                        f"{r['mean_conf']:.3f} | {r['min_conf']:.3f} | "
                        f"{r['tracked_pct']:.1f}% |")

    # SOT results
    if sot_results:
        lines.append("\n## 3. SOT Benchmark Tracking\n")
        lines.append("| Video | Frames | Mean Conf | Min Conf | Tracked % |")
        lines.append("|-------|--------|-----------|----------|-----------|")
        for r in sot_results:
            lines.append(f"| {r['video_id']} | {r['frames']} | "
                        f"{r['mean_conf']:.3f} | {r['min_conf']:.3f} | "
                        f"{r['tracked_pct']:.1f}% |")

    lines.append("\n---\n")
    lines.append("Videos are in `evaluation/trajectory_videos/`\n")

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n[report] {report_path}")
    print(f"[json]   {json_path}")


# ── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vid2Spatial trajectory overlay visualization")
    parser.add_argument("--mode", choices=["gt", "real", "all"], default="all",
                        help="Which videos to process")
    parser.add_argument("--video", help="Single video file path")
    parser.add_argument("--trajectory", help="Single trajectory JSON path")
    parser.add_argument("--gt-trajectory", help="Optional GT trajectory JSON (for comparison)")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Single video mode
    if args.video and args.trajectory:
        vid_id = Path(args.video).stem
        m = overlay_real_video(args.video, args.trajectory, None, output_dir, vid_id)
        if m:
            print(json.dumps(m, indent=2))
        return

    gt_results = []
    e2e_results = []
    sot_results = []

    if args.mode in ("gt", "all"):
        gt_results = run_gt_overlays(output_dir)

    if args.mode in ("real", "all"):
        e2e_results = run_e2e_overlays(output_dir)
        sot_results = run_sot_overlays(output_dir)

    generate_summary_report(gt_results, e2e_results, sot_results, output_dir)


if __name__ == "__main__":
    main()
