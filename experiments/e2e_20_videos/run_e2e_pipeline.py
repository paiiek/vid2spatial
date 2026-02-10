#!/usr/bin/env python3
"""
End-to-End Pipeline: 20 diverse real videos → spatial audio.

Full pipeline per video:
1. Extract audio from video (ffmpeg)
2. HybridTracker.track() → 2D tracking + depth
3. get_trajectory_3d(enhance_depth=True) → 3D trajectory
4. rts_smooth_trajectory() → smoothed trajectory
5. render_foa_from_trajectory() → FOA audio
6. foa_to_binaural() → binaural for headphone listening

Date: 2026-02-06
"""

import sys
import os

# Project paths
PROJECT_ROOT = "/home/seung/mmhoa/vid2spatial"
sys.path.insert(0, PROJECT_ROOT)

import json
import time
import traceback
import subprocess
import numpy as np
import soundfile as sf
import torch
from pathlib import Path

FFMPEG = "/home/seung/miniforge3/bin/ffmpeg"
FFPROBE = "/home/seung/miniforge3/bin/ffprobe"

# Scenario base directory
BASE_DIR = Path(__file__).parent
VIDEO_DIR = BASE_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def extract_audio(video_path: str, output_wav: str, sr: int = 48000) -> bool:
    """Extract audio track from video using ffmpeg. Generate synthetic if no audio."""
    env = {**os.environ, "PATH": f"/home/seung/miniforge3/bin:/usr/bin:{os.environ.get('PATH', '')}"}

    # Try extracting real audio
    cmd = [
        FFMPEG, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_f32le",
        "-ar", str(sr), "-ac", "1",
        output_wav,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)

    if result.returncode == 0 and os.path.exists(output_wav):
        info = sf.info(output_wav)
        if info.duration > 0.5:
            return True

    # No audio track: generate synthetic tone matching video duration
    dur_cmd = [
        FFPROBE, "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    dur_result = subprocess.run(dur_cmd, capture_output=True, text=True, timeout=10, env=env)
    duration = float(dur_result.stdout.strip()) if dur_result.stdout.strip() else 10.0

    # Generate a rich synthetic source (harmonics + noise for spatial perception)
    t = np.linspace(0, duration, int(sr * duration))
    # C major chord: C4 + E4 + G4 with harmonics
    audio = np.zeros_like(t)
    for f0 in [261.63, 329.63, 392.0]:
        for h in range(1, 5):
            audio += (0.4 ** h) * np.sin(2 * np.pi * f0 * h * t)
    # Add envelope for natural feel
    env_curve = (1 - np.exp(-t * 5)) * np.exp(-t * 0.1)
    audio = audio * env_curve
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.7
    sf.write(output_wav, audio.astype(np.float32), sr)
    return True


def run_pipeline_for_video(
    video_path: str,
    text_prompt: str,
    video_id: str,
    output_dir: Path,
) -> dict:
    """
    Run full end-to-end pipeline for a single video.

    Returns dict with status, paths, timing, and metrics.
    """
    result = {
        "video_id": video_id,
        "prompt": text_prompt,
        "video_path": video_path,
        "status": "pending",
        "stages": {},
    }

    sr = 48000
    video_output_dir = output_dir / video_id
    video_output_dir.mkdir(exist_ok=True)

    # === Stage 0: Extract audio ===
    t0 = time.time()
    audio_path = str(video_output_dir / "audio_mono.wav")
    try:
        extract_audio(video_path, audio_path, sr)
        result["stages"]["audio_extract"] = {
            "status": "ok",
            "time_sec": round(time.time() - t0, 2),
            "path": audio_path,
        }
    except Exception as e:
        result["stages"]["audio_extract"] = {"status": "error", "error": str(e)}
        result["status"] = "failed_audio"
        return result

    # === Stage 1: Tracking ===
    t1 = time.time()
    try:
        from vid2spatial_pkg.hybrid_tracker import HybridTracker

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tracker = HybridTracker(
            device="cuda" if torch.cuda.is_available() else "cpu",
            box_threshold=0.15,
            fov_deg=60.0,
        )

        tracking_result = tracker.track(
            video_path=video_path,
            text_prompt=text_prompt,
            tracking_method="adaptive_k",
            depth_stride=5,
        )

        n_frames = len(tracking_result.frames)
        tracking_time = round(time.time() - t1, 2)

        result["stages"]["tracking"] = {
            "status": "ok",
            "time_sec": tracking_time,
            "frames_tracked": n_frames,
            "fps": tracking_result.fps,
            "video_size": f"{tracking_result.video_width}x{tracking_result.video_height}",
        }

        # Save raw tracking data
        raw_traj_path = str(video_output_dir / "raw_tracking.json")
        raw_data = []
        for f in tracking_result.frames:
            raw_data.append({
                "frame_idx": f.frame_idx,
                "bbox": list(f.bbox),
                "center": list(f.center),
                "confidence": round(f.confidence, 4),
                "depth_m": round(f.depth_m, 4),
            })
        with open(raw_traj_path, "w") as f:
            json.dump(raw_data, f, indent=2)

        # Free tracker GPU memory
        del tracker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        result["stages"]["tracking"] = {"status": "error", "error": str(e)}
        result["status"] = "failed_tracking"
        traceback.print_exc()
        return result

    # === Stage 2: 3D Trajectory with depth enhancement ===
    t2 = time.time()
    try:
        trajectory = tracking_result.get_trajectory_3d(
            smooth=False,
            enhance_depth=True,
        )

        frames = trajectory.get("frames", [])
        n_frames_3d = len(frames)

        # Extract stats
        if frames:
            az_vals = [f["az"] for f in frames]
            el_vals = [f["el"] for f in frames]
            dist_vals = [f.get("dist_m", 1.0) for f in frames]
            d_rel_vals = [f.get("d_rel", 0.5) for f in frames]
            conf_vals = [f.get("confidence", 0.5) for f in frames]

            traj_stats = {
                "az_range": [round(min(az_vals), 2), round(max(az_vals), 2)],
                "el_range": [round(min(el_vals), 2), round(max(el_vals), 2)],
                "dist_range": [round(min(dist_vals), 2), round(max(dist_vals), 2)],
                "d_rel_range": [round(min(d_rel_vals), 3), round(max(d_rel_vals), 3)],
                "mean_confidence": round(np.mean(conf_vals), 3),
            }
        else:
            traj_stats = {}

        result["stages"]["trajectory_3d"] = {
            "status": "ok",
            "time_sec": round(time.time() - t2, 2),
            "frames": n_frames_3d,
            "stats": traj_stats,
        }

        # Save trajectory
        traj_path = str(video_output_dir / "trajectory_3d.json")
        with open(traj_path, "w") as f:
            json.dump(trajectory, f, indent=2, default=str)

    except Exception as e:
        result["stages"]["trajectory_3d"] = {"status": "error", "error": str(e)}
        result["status"] = "failed_trajectory"
        traceback.print_exc()
        return result

    # === Stage 3: RTS Smoothing ===
    t3 = time.time()
    try:
        from vid2spatial_pkg.trajectory_stabilizer import rts_smooth_trajectory

        smoothed_frames = rts_smooth_trajectory(frames)

        # Compute jerk reduction
        def compute_jerk(frame_list, key):
            values = [f.get(key, 0) for f in frame_list]
            if len(values) < 4:
                return 0
            vel = np.diff(values)
            acc = np.diff(vel)
            jerk = np.diff(acc)
            return float(np.mean(np.abs(jerk)))

        raw_jerk = compute_jerk(frames, "az")
        smooth_jerk = compute_jerk(smoothed_frames, "az")
        jerk_reduction = round((1 - smooth_jerk / (raw_jerk + 1e-9)) * 100, 1)

        result["stages"]["smoothing"] = {
            "status": "ok",
            "time_sec": round(time.time() - t3, 2),
            "jerk_reduction_pct": jerk_reduction,
            "raw_jerk": round(raw_jerk, 6),
            "smooth_jerk": round(smooth_jerk, 6),
        }

        # Save smoothed trajectory
        smooth_traj_path = str(video_output_dir / "trajectory_smoothed.json")
        smooth_traj = {
            "frames": smoothed_frames,
            "fps": trajectory.get("fps", tracking_result.fps),
        }
        with open(smooth_traj_path, "w") as f:
            json.dump(smooth_traj, f, indent=2, default=str)

    except Exception as e:
        result["stages"]["smoothing"] = {"status": "error", "error": str(e)}
        # Fall back to unsmoothed
        smoothed_frames = frames
        smooth_traj = trajectory

    # === Stage 4: FOA Rendering ===
    t4 = time.time()
    try:
        from vid2spatial_pkg.foa_render import render_foa_from_trajectory

        foa_path = str(video_output_dir / f"{video_id}_foa.wav")

        render_result = render_foa_from_trajectory(
            audio_path=audio_path,
            trajectory={
                "frames": smoothed_frames,
                "fps": trajectory.get("fps", tracking_result.fps),
            },
            output_path=foa_path,
            smooth_ms=30.0,
            dist_gain_k=1.0,
            apply_reverb=True,
            rt60=0.5,
            output_stereo=False,  # We'll do binaural separately
        )

        foa_info = sf.info(foa_path)
        result["stages"]["foa_render"] = {
            "status": "ok",
            "time_sec": round(time.time() - t4, 2),
            "channels": foa_info.channels,
            "duration_sec": round(foa_info.duration, 2),
            "sample_rate": foa_info.samplerate,
            "path": foa_path,
        }

    except Exception as e:
        result["stages"]["foa_render"] = {"status": "error", "error": str(e)}
        result["status"] = "failed_render"
        traceback.print_exc()
        return result

    # === Stage 5: FOA → Binaural ===
    t5 = time.time()
    try:
        from vid2spatial_pkg.foa_render import foa_to_binaural

        foa_data, foa_sr = sf.read(foa_path, dtype="float32")
        foa_data = foa_data.T  # [4, T]

        binaural = foa_to_binaural(foa_data, foa_sr)
        binaural_path = str(video_output_dir / f"{video_id}_binaural.wav")
        sf.write(binaural_path, binaural.T, foa_sr, subtype="FLOAT")

        result["stages"]["binaural"] = {
            "status": "ok",
            "time_sec": round(time.time() - t5, 2),
            "path": binaural_path,
        }

    except Exception as e:
        result["stages"]["binaural"] = {"status": "error", "error": str(e)}

    # === Compute total timing ===
    total_time = sum(
        s.get("time_sec", 0) for s in result["stages"].values()
        if isinstance(s, dict)
    )
    result["total_time_sec"] = round(total_time, 2)
    result["status"] = "success"

    return result


def main():
    """Run E2E pipeline for all 20 videos."""
    print("=" * 70)
    print("  Vid2Spatial End-to-End Pipeline: 20 Real Videos")
    print("  Date: 2026-02-06")
    print("=" * 70)

    # Load manifest
    manifest_path = BASE_DIR / "video_manifest.json"
    if not manifest_path.exists():
        print("ERROR: video_manifest.json not found. Run download_videos.py first.")
        return

    with open(manifest_path) as f:
        videos = json.load(f)

    # Filter to downloaded videos
    available = [v for v in videos if v.get("downloaded", False)]
    print(f"\nAvailable videos: {len(available)}/20")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Category distribution
    cats = {}
    for v in available:
        cat = v["category"]
        cats[cat] = cats.get(cat, 0) + 1
    print("Category distribution:")
    for cat, cnt in sorted(cats.items()):
        print(f"  {cat}: {cnt}")
    print()

    all_results = []

    for i, video in enumerate(available):
        vid_id = video["id"]
        video_path = video.get("local_path", "")
        prompt = video["prompt"]

        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(available)}] {vid_id}")
        print(f"  Prompt: \"{prompt}\"")
        print(f"  Video: {os.path.basename(video_path)}")
        print(f"  Category: {video['category']}")
        print(f"  Expected motion: {video['expected_motion']}")
        print(f"{'='*70}")

        if not os.path.exists(video_path):
            print(f"  SKIP: Video file not found")
            all_results.append({"video_id": vid_id, "status": "missing_video"})
            continue

        try:
            result = run_pipeline_for_video(
                video_path=video_path,
                text_prompt=prompt,
                video_id=vid_id,
                output_dir=OUTPUT_DIR,
            )
            all_results.append(result)

            # Print stage summary
            for stage_name, stage_data in result.get("stages", {}).items():
                status = stage_data.get("status", "?")
                t = stage_data.get("time_sec", 0)
                icon = "OK" if status == "ok" else "FAIL"
                extra = ""
                if stage_name == "tracking":
                    extra = f" ({stage_data.get('frames_tracked', 0)} frames)"
                elif stage_name == "smoothing":
                    extra = f" (jerk -{stage_data.get('jerk_reduction_pct', 0)}%)"
                elif stage_name == "foa_render":
                    extra = f" ({stage_data.get('duration_sec', 0)}s, {stage_data.get('channels', 0)}ch)"
                print(f"  [{icon}] {stage_name}: {t:.1f}s{extra}")

            print(f"\n  Total: {result.get('total_time_sec', 0):.1f}s | Status: {result['status']}")

        except Exception as e:
            print(f"  FATAL ERROR: {e}")
            traceback.print_exc()
            all_results.append({"video_id": vid_id, "status": "fatal_error", "error": str(e)})

    # === Save comprehensive results ===
    results_path = BASE_DIR / "e2e_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "date": "2026-02-06",
            "total_videos": len(available),
            "results": all_results,
        }, f, indent=2, default=str)

    # === Print Summary ===
    print("\n\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    success = sum(1 for r in all_results if r.get("status") == "success")
    failed = len(all_results) - success

    print(f"\n  Success: {success}/{len(all_results)}")
    if failed:
        print(f"  Failed:  {failed}")
        for r in all_results:
            if r.get("status") != "success":
                print(f"    - {r['video_id']}: {r['status']}")

    # Timing summary
    times = [r.get("total_time_sec", 0) for r in all_results if r.get("status") == "success"]
    if times:
        print(f"\n  Timing (successful):")
        print(f"    Mean: {np.mean(times):.1f}s per video")
        print(f"    Min:  {np.min(times):.1f}s")
        print(f"    Max:  {np.max(times):.1f}s")
        print(f"    Total: {np.sum(times):.1f}s")

    # Per-category success
    print(f"\n  Per-category:")
    cat_results = {}
    for r, v in zip(all_results, available[:len(all_results)]):
        cat = v["category"]
        if cat not in cat_results:
            cat_results[cat] = {"success": 0, "total": 0}
        cat_results[cat]["total"] += 1
        if r.get("status") == "success":
            cat_results[cat]["success"] += 1
    for cat, cr in sorted(cat_results.items()):
        print(f"    {cat}: {cr['success']}/{cr['total']}")

    print(f"\n  Results saved: {results_path}")
    print(f"  Outputs: {OUTPUT_DIR}/")

    # List binaural files for listening
    print(f"\n  Binaural files for headphone listening:")
    for r in all_results:
        if r.get("status") == "success":
            binaural = r.get("stages", {}).get("binaural", {}).get("path", "")
            if binaural:
                print(f"    {os.path.basename(binaural)}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
