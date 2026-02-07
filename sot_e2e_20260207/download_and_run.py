#!/usr/bin/env python3
"""
Single-Object Tracking E2E Evaluation.

Downloads 15 diverse single-object videos from YouTube
(sourced from DAVIS-style / segmentation benchmark categories)
and runs the full Vid2Spatial pipeline on each.

Object diversity (no skew):
  3 Animals, 3 Vehicles, 3 Sports, 3 People/Dance, 3 Objects

Date: 2026-02-07
"""

import sys, os
sys.path.insert(0, "/home/seung/mmhoa/vid2spatial")

import json
import time
import subprocess
import traceback
import numpy as np
import soundfile as sf
import torch
from pathlib import Path

FFMPEG = "/home/seung/miniforge3/bin/ffmpeg"
FFPROBE = "/home/seung/miniforge3/bin/ffprobe"
PYTHON = "/home/seung/miniforge3/bin/python3"

BASE_DIR = Path(__file__).parent
VIDEO_DIR = BASE_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "outputs"
VIDEO_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# 15 curated single-object tracking scenarios
# Queries designed to find videos with a single prominent object + clear motion
SCENARIOS = [
    # === Animals (3) - classic SOT targets ===
    {"id": "01_dog_frisbee",     "query": "dog catching frisbee slow motion single dog",
     "prompt": "dog",            "category": "animal"},
    {"id": "02_cat_jumping",     "query": "cat jumping single cat close up slow motion",
     "prompt": "cat",            "category": "animal"},
    {"id": "03_horse_running",   "query": "single horse running field side view tracking shot",
     "prompt": "horse",          "category": "animal"},

    # === Vehicles (3) - typical segmentation targets ===
    {"id": "04_car_highway",     "query": "single car driving highway tracking shot drone",
     "prompt": "car",            "category": "vehicle"},
    {"id": "05_motorbike_road",  "query": "single motorcycle riding road tracking follow",
     "prompt": "motorcycle",     "category": "vehicle"},
    {"id": "06_boat_water",      "query": "single boat moving on water tracking shot",
     "prompt": "boat",           "category": "vehicle"},

    # === Sports (3) - fast motion tracking ===
    {"id": "07_skier_downhill",  "query": "single skier downhill skiing tracking shot",
     "prompt": "skier",          "category": "sports"},
    {"id": "08_surfer_wave",     "query": "single surfer riding wave tracking shot",
     "prompt": "surfer",         "category": "sports"},
    {"id": "09_bmx_rider",      "query": "single bmx rider trick close up tracking",
     "prompt": "person on bicycle", "category": "sports"},

    # === People/Dance (3) - articulated motion ===
    {"id": "10_dancer_solo",     "query": "solo dancer performing contemporary dance studio",
     "prompt": "dancer",         "category": "people"},
    {"id": "11_gymnast_floor",   "query": "single gymnast floor exercise close up",
     "prompt": "gymnast",        "category": "people"},
    {"id": "12_parkour_runner",  "query": "single parkour runner freerunning city",
     "prompt": "person",         "category": "people"},

    # === Objects (3) - rigid body tracking ===
    {"id": "13_drone_flying",    "query": "single drone flying sky tracking shot close up",
     "prompt": "drone",          "category": "object"},
    {"id": "14_kite_flying",     "query": "single kite flying in sky close up tracking",
     "prompt": "kite",           "category": "object"},
    {"id": "15_ball_bouncing",   "query": "single ball bouncing slow motion close up",
     "prompt": "ball",           "category": "object"},
]


def download_video(scenario: dict, duration: int = 10) -> str:
    """Search YouTube and download a 10-second clip."""
    vid_id = scenario["id"]
    query = scenario["query"]
    output_path = str(VIDEO_DIR / f"{vid_id}.mp4")
    env = {**os.environ, "PATH": f"/home/seung/miniforge3/bin:/usr/bin:{os.environ.get('PATH', '')}"}

    if os.path.exists(output_path):
        print(f"    [skip] Already exists")
        return output_path

    print(f"    [search] '{query}'...")

    try:
        # Search YouTube
        search_cmd = [
            PYTHON, "-m", "yt_dlp",
            f"ytsearch1:{query}",
            "--get-id", "--no-playlist",
            "-f", "best[height<=480][ext=mp4]/best[height<=480]/best",
        ]
        result = subprocess.run(search_cmd, capture_output=True, text=True, timeout=30, env=env)
        if result.returncode != 0:
            print(f"    [warn] Search failed")
            return ""

        video_yt_id = result.stdout.strip().split("\n")[0]
        if not video_yt_id or len(video_yt_id) > 15:
            print(f"    [warn] No valid ID found")
            return ""

        yt_url = f"https://www.youtube.com/watch?v={video_yt_id}"
        print(f"    [found] {yt_url}")

        # Download
        temp_path = str(VIDEO_DIR / f"_temp_{vid_id}.mp4")
        dl_cmd = [
            PYTHON, "-m", "yt_dlp", yt_url,
            "-f", "best[height<=480][ext=mp4]/best[height<=480]/best",
            "--no-playlist",
            "--ffmpeg-location", "/home/seung/miniforge3/bin",
            "-o", temp_path, "--quiet",
        ]
        subprocess.run(dl_cmd, capture_output=True, text=True, timeout=60, env=env)

        if not os.path.exists(temp_path):
            temp_webm = temp_path.replace(".mp4", ".webm")
            if os.path.exists(temp_webm):
                temp_path = temp_webm
            else:
                print(f"    [warn] Download failed")
                return ""

        # Trim to 10 seconds, skip first 5s to avoid intros
        trim_cmd = [
            FFMPEG, "-y", "-i", temp_path,
            "-ss", "5", "-t", str(duration),
            "-c:v", "libx264", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ]
        subprocess.run(trim_cmd, capture_output=True, text=True, timeout=60, env=env)

        # Cleanup
        for tmp in [temp_path, temp_path.replace(".mp4", ".webm")]:
            if os.path.exists(tmp):
                os.remove(tmp)

        if os.path.exists(output_path):
            # Get duration
            probe = subprocess.run(
                [FFPROBE, "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", output_path],
                capture_output=True, text=True, timeout=10, env=env
            )
            dur = probe.stdout.strip()
            print(f"    [ok] {vid_id}.mp4 ({dur}s)")
            return output_path
        else:
            print(f"    [warn] Trim failed")
            return ""

    except Exception as e:
        print(f"    [error] {e}")
        return ""


def extract_audio(video_path: str, output_wav: str, sr: int = 48000) -> bool:
    """Extract mono audio from video. Generate synthetic if no audio track."""
    env = {**os.environ, "PATH": f"/home/seung/miniforge3/bin:/usr/bin:{os.environ.get('PATH', '')}"}

    cmd = [FFMPEG, "-y", "-i", video_path, "-vn", "-acodec", "pcm_f32le",
           "-ar", str(sr), "-ac", "1", output_wav]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)

    if result.returncode == 0 and os.path.exists(output_wav):
        info = sf.info(output_wav)
        if info.duration > 0.5:
            return True

    # Fallback: synthetic
    probe = subprocess.run(
        [FFPROBE, "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, timeout=10, env=env
    )
    duration = float(probe.stdout.strip()) if probe.stdout.strip() else 10.0

    t = np.linspace(0, duration, int(sr * duration))
    audio = np.zeros_like(t)
    for f0 in [261.63, 329.63, 392.0, 523.25]:
        for h in range(1, 4):
            audio += (0.3 ** h) * np.sin(2 * np.pi * f0 * h * t)
    env_c = (1 - np.exp(-t * 5)) * np.exp(-t * 0.05)
    audio = audio * env_c
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.7
    sf.write(output_wav, audio.astype(np.float32), sr)
    return True


def run_pipeline(video_path: str, text_prompt: str, scene_out: Path, vid_id: str) -> dict:
    """Full pipeline: track → trajectory → smooth → FOA → binaural."""
    from vid2spatial_pkg.hybrid_tracker import HybridTracker
    from vid2spatial_pkg.trajectory_stabilizer import rts_smooth_trajectory
    from vid2spatial_pkg.foa_render import render_foa_from_trajectory, foa_to_binaural

    sr = 48000
    result = {"video_id": vid_id, "status": "pending", "stages": {}}

    # Audio extract
    audio_path = str(scene_out / "audio_mono.wav")
    extract_audio(video_path, audio_path, sr)

    # Stage 1: Tracking
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tracker = HybridTracker(
        device="cuda" if torch.cuda.is_available() else "cpu",
        box_threshold=0.15, fov_deg=60.0,
    )

    tracking_result = tracker.track(
        video_path=video_path,
        text_prompt=text_prompt,
        tracking_method="adaptive_k",
        depth_stride=5,
    )

    n_tracked = len(tracking_result.frames)
    result["stages"]["tracking"] = {
        "time_sec": round(time.time() - t0, 2),
        "frames": n_tracked,
        "fps": tracking_result.fps,
        "size": f"{tracking_result.video_width}x{tracking_result.video_height}",
    }

    # Save raw tracking
    raw_data = [{"frame_idx": f.frame_idx, "bbox": list(f.bbox),
                 "center": list(f.center), "confidence": round(f.confidence, 4),
                 "depth_m": round(f.depth_m, 4)} for f in tracking_result.frames]
    with open(str(scene_out / "raw_tracking.json"), "w") as f:
        json.dump(raw_data, f, indent=2)

    del tracker
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Stage 2: 3D Trajectory
    t1 = time.time()
    trajectory = tracking_result.get_trajectory_3d(smooth=False, enhance_depth=True)
    frames = trajectory.get("frames", [])

    if frames:
        az_vals = [f["az"] for f in frames]
        el_vals = [f["el"] for f in frames]
        dist_vals = [f.get("dist_m", 1.0) for f in frames]
        conf_vals = [f.get("confidence", 0.5) for f in frames]
        result["stages"]["trajectory"] = {
            "time_sec": round(time.time() - t1, 2),
            "frames": len(frames),
            "az_range": [round(min(az_vals), 3), round(max(az_vals), 3)],
            "el_range": [round(min(el_vals), 3), round(max(el_vals), 3)],
            "dist_range": [round(min(dist_vals), 2), round(max(dist_vals), 2)],
            "mean_conf": round(np.mean(conf_vals), 3),
            "az_span_deg": round(np.degrees(max(az_vals) - min(az_vals)), 1),
        }

    with open(str(scene_out / "trajectory_3d.json"), "w") as f:
        json.dump(trajectory, f, indent=2, default=str)

    # Stage 3: RTS Smoothing
    t2 = time.time()
    smoothed_frames = rts_smooth_trajectory(frames)

    def jerk(frame_list, key):
        v = [f.get(key, 0) for f in frame_list]
        if len(v) < 4: return 0
        return float(np.mean(np.abs(np.diff(np.diff(np.diff(v))))))

    raw_j = jerk(frames, "az")
    smooth_j = jerk(smoothed_frames, "az")
    result["stages"]["smoothing"] = {
        "time_sec": round(time.time() - t2, 2),
        "jerk_reduction_pct": round((1 - smooth_j / (raw_j + 1e-9)) * 100, 1),
    }

    # Stage 4: FOA Render
    t3 = time.time()
    foa_path = str(scene_out / f"{vid_id}_foa.wav")
    render_foa_from_trajectory(
        audio_path=audio_path,
        trajectory={"frames": smoothed_frames, "fps": trajectory.get("fps", tracking_result.fps)},
        output_path=foa_path,
        smooth_ms=30.0, apply_reverb=True, rt60=0.5,
        output_stereo=False,
    )
    foa_info = sf.info(foa_path)
    result["stages"]["foa_render"] = {
        "time_sec": round(time.time() - t3, 2),
        "channels": foa_info.channels,
        "duration_sec": round(foa_info.duration, 2),
    }

    # Stage 5: Binaural
    t4 = time.time()
    foa_data, foa_sr = sf.read(foa_path, dtype="float32")
    foa_data = foa_data.T
    binaural = foa_to_binaural(foa_data, foa_sr)
    binaural_path = str(scene_out / f"{vid_id}_binaural.wav")
    sf.write(binaural_path, binaural.T, foa_sr, subtype="FLOAT")
    result["stages"]["binaural"] = {"time_sec": round(time.time() - t4, 2), "path": binaural_path}

    total = sum(s.get("time_sec", 0) for s in result["stages"].values() if isinstance(s, dict))
    result["total_time_sec"] = round(total, 2)
    result["status"] = "success"
    return result


def main():
    print("=" * 70)
    print("  Single-Object Tracking E2E: 15 Real Videos → Spatial Audio")
    print("  Date: 2026-02-07")
    print("=" * 70)

    # Step 1: Download all videos
    print("\n--- PHASE 1: Download Videos ---\n")
    for i, sc in enumerate(SCENARIOS):
        print(f"[{i+1}/15] {sc['id']} ({sc['prompt']}, {sc['category']})")
        path = download_video(sc)
        sc["local_path"] = path
        sc["downloaded"] = bool(path)

    downloaded = sum(1 for s in SCENARIOS if s.get("downloaded"))
    print(f"\nDownloaded: {downloaded}/15")

    # Save manifest
    with open(BASE_DIR / "manifest.json", "w") as f:
        json.dump(SCENARIOS, f, indent=2)

    # Step 2: Run pipeline on each
    print("\n--- PHASE 2: E2E Pipeline ---\n")
    all_results = []

    for i, sc in enumerate(SCENARIOS):
        vid_id = sc["id"]
        video_path = sc.get("local_path", "")
        prompt = sc["prompt"]

        print(f"\n{'='*70}")
        print(f"[{i+1}/15] {vid_id} | prompt=\"{prompt}\" | {sc['category']}")
        print(f"{'='*70}")

        if not video_path or not os.path.exists(video_path):
            print("  SKIP: video not found")
            all_results.append({"video_id": vid_id, "status": "missing"})
            continue

        scene_out = OUTPUT_DIR / vid_id
        scene_out.mkdir(exist_ok=True)

        # Symlink video into output dir for reference
        video_link = scene_out / f"{vid_id}.mp4"
        if not video_link.exists():
            os.symlink(os.path.abspath(video_path), str(video_link))

        try:
            result = run_pipeline(video_path, prompt, scene_out, vid_id)
            all_results.append(result)

            for stage, data in result["stages"].items():
                if isinstance(data, dict):
                    t = data.get("time_sec", 0)
                    extra = ""
                    if stage == "tracking":
                        extra = f" ({data.get('frames',0)}f)"
                    elif stage == "trajectory":
                        extra = f" (az:{data.get('az_span_deg',0):.0f}° conf:{data.get('mean_conf',0):.2f})"
                    elif stage == "smoothing":
                        extra = f" (jerk-{data.get('jerk_reduction_pct',0):.0f}%)"
                    elif stage == "foa_render":
                        extra = f" ({data.get('duration_sec',0)}s {data.get('channels',0)}ch)"
                    print(f"  {stage:<15}: {t:5.1f}s{extra}")

            print(f"  TOTAL: {result.get('total_time_sec',0):.1f}s")

        except Exception as e:
            print(f"  FATAL: {e}")
            traceback.print_exc()
            all_results.append({"video_id": vid_id, "status": "error", "error": str(e)})

    # Save results
    results_path = BASE_DIR / "sot_e2e_results.json"
    with open(results_path, "w") as f:
        json.dump({"date": "2026-02-07", "results": all_results}, f, indent=2, default=str)

    # Summary
    print("\n\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    success = [r for r in all_results if r.get("status") == "success"]
    print(f"\n  Success: {len(success)}/{len(all_results)}")

    if success:
        times = [r["total_time_sec"] for r in success]
        print(f"  Mean time: {np.mean(times):.1f}s | Total: {sum(times):.0f}s")

    print(f"\n  {'ID':<25} {'Cat':<10} {'Prompt':<18} {'AzSpan':>7} {'Conf':>6} {'Time':>6} {'Status'}")
    print("  " + "-" * 85)
    for r in all_results:
        vid_id = r.get("video_id", "?")
        sc = next((s for s in SCENARIOS if s["id"] == vid_id), {})
        cat = sc.get("category", "?")
        prompt = sc.get("prompt", "?")
        status = r.get("status", "?")

        if status == "success":
            traj = r["stages"].get("trajectory", {})
            az_span = f"{traj.get('az_span_deg', 0):.1f}°"
            conf = f"{traj.get('mean_conf', 0):.2f}"
            t = f"{r['total_time_sec']:.1f}s"
        else:
            az_span = conf = t = "-"

        print(f"  {vid_id:<25} {cat:<10} {prompt:<18} {az_span:>7} {conf:>6} {t:>6} {status}")

    # Binaural file list
    print(f"\n  Binaural outputs:")
    for r in success:
        bp = r["stages"].get("binaural", {}).get("path", "")
        if bp:
            print(f"    {os.path.basename(bp)}")

    print(f"\n  Results: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
