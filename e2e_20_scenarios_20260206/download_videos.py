#!/usr/bin/env python3
"""
Download 20 diverse 10-second video clips for end-to-end evaluation.

Balanced categories (no object skew):
- 4 Musical instruments (guitar, piano, drums, violin)
- 4 Sports/action (soccer, basketball, tennis, running)
- 4 Vehicles/transport (car, motorcycle, bicycle, train)
- 4 Animals/nature (dog, bird, cat, horse)
- 4 Daily life/people (walking, dancing, cooking, street)

Each video: ~10 seconds, with clear single-object motion.
"""

import os
import subprocess
import sys
import json

# Output directory
VIDEO_DIR = os.path.join(os.path.dirname(__file__), "videos")
os.makedirs(VIDEO_DIR, exist_ok=True)

FFMPEG = "/home/seung/miniforge3/bin/ffmpeg"

# 20 diverse scenarios - Pexels/Pixabay free stock videos via direct URLs
# Using YouTube Creative Commons or stock footage
VIDEOS = [
    # === Musical Instruments (4) ===
    {
        "id": "01_guitar_acoustic",
        "query": "acoustic guitar playing close up",
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # placeholder
        "prompt": "acoustic guitar",
        "category": "instrument",
        "expected_motion": "small lateral + finger movement",
    },
    {
        "id": "02_piano_playing",
        "query": "piano playing hands close up",
        "url": "",
        "prompt": "piano",
        "category": "instrument",
        "expected_motion": "lateral hand movement",
    },
    {
        "id": "03_drums_playing",
        "query": "drums playing performance",
        "url": "",
        "prompt": "drums",
        "category": "instrument",
        "expected_motion": "fast percussive movement",
    },
    {
        "id": "04_violin_playing",
        "query": "violin playing solo performance",
        "url": "",
        "prompt": "violin",
        "category": "instrument",
        "expected_motion": "slow lateral bowing",
    },

    # === Sports/Action (4) ===
    {
        "id": "05_soccer_kick",
        "query": "soccer player kicking ball",
        "url": "",
        "prompt": "soccer ball",
        "category": "sports",
        "expected_motion": "fast multi-directional",
    },
    {
        "id": "06_basketball_dribble",
        "query": "basketball dribbling player",
        "url": "",
        "prompt": "basketball",
        "category": "sports",
        "expected_motion": "vertical bouncing + lateral",
    },
    {
        "id": "07_tennis_serve",
        "query": "tennis serve slow motion",
        "url": "",
        "prompt": "tennis ball",
        "category": "sports",
        "expected_motion": "fast parabolic arc",
    },
    {
        "id": "08_runner_track",
        "query": "runner sprinting track",
        "url": "",
        "prompt": "runner",
        "category": "sports",
        "expected_motion": "fast lateral pass-by",
    },

    # === Vehicles (4) ===
    {
        "id": "09_car_driving",
        "query": "car driving street passing by",
        "url": "",
        "prompt": "car",
        "category": "vehicle",
        "expected_motion": "fast lateral pass-by",
    },
    {
        "id": "10_motorcycle_road",
        "query": "motorcycle riding road",
        "url": "",
        "prompt": "motorcycle",
        "category": "vehicle",
        "expected_motion": "fast approaching/receding",
    },
    {
        "id": "11_bicycle_park",
        "query": "bicycle riding in park",
        "url": "",
        "prompt": "bicycle",
        "category": "vehicle",
        "expected_motion": "medium lateral",
    },
    {
        "id": "12_train_passing",
        "query": "train passing by station",
        "url": "",
        "prompt": "train",
        "category": "vehicle",
        "expected_motion": "fast lateral pass-by",
    },

    # === Animals (4) ===
    {
        "id": "13_dog_running",
        "query": "dog running in park",
        "url": "",
        "prompt": "dog",
        "category": "animal",
        "expected_motion": "fast random movement",
    },
    {
        "id": "14_bird_flying",
        "query": "bird flying sky",
        "url": "",
        "prompt": "bird",
        "category": "animal",
        "expected_motion": "3D flight path",
    },
    {
        "id": "15_cat_walking",
        "query": "cat walking room",
        "url": "",
        "prompt": "cat",
        "category": "animal",
        "expected_motion": "slow meandering",
    },
    {
        "id": "16_horse_galloping",
        "query": "horse galloping field",
        "url": "",
        "prompt": "horse",
        "category": "animal",
        "expected_motion": "fast lateral",
    },

    # === Daily Life/People (4) ===
    {
        "id": "17_person_walking",
        "query": "person walking sidewalk city",
        "url": "",
        "prompt": "person",
        "category": "people",
        "expected_motion": "steady lateral walk",
    },
    {
        "id": "18_dancer_stage",
        "query": "dancer performing on stage",
        "url": "",
        "prompt": "dancer",
        "category": "people",
        "expected_motion": "complex 2D movement",
    },
    {
        "id": "19_chef_cooking",
        "query": "chef cooking kitchen",
        "url": "",
        "prompt": "chef",
        "category": "people",
        "expected_motion": "small movement, depth changes",
    },
    {
        "id": "20_skateboarder_park",
        "query": "skateboarder trick park",
        "url": "",
        "prompt": "skateboarder",
        "category": "people",
        "expected_motion": "fast multi-directional",
    },
]


def search_and_download(video_info: dict, duration: int = 10) -> str:
    """Search YouTube for a video matching the query and download 10 seconds."""
    vid_id = video_info["id"]
    query = video_info["query"]
    output_path = os.path.join(VIDEO_DIR, f"{vid_id}.mp4")

    if os.path.exists(output_path):
        print(f"  [skip] Already exists: {vid_id}.mp4")
        return output_path

    print(f"  [search] '{query}'...")

    # Use yt-dlp to search and download
    # Search YouTube for short CC videos
    try:
        # First: search and get URL
        search_cmd = [
            sys.executable, "-m", "yt_dlp",
            f"ytsearch1:{query} short clip",
            "--get-url",
            "--get-id",
            "--no-playlist",
            "-f", "best[height<=480][ext=mp4]/best[height<=480]/best",
        ]
        result = subprocess.run(
            search_cmd,
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PATH": f"/home/seung/miniforge3/bin:/usr/bin:{os.environ.get('PATH', '')}"}
        )

        if result.returncode != 0:
            print(f"    [warn] Search failed: {result.stderr[:200]}")
            return ""

        lines = result.stdout.strip().split("\n")
        video_yt_id = None
        for line in lines:
            if len(line) == 11 and not line.startswith("http"):
                video_yt_id = line
                break

        if not video_yt_id:
            print(f"    [warn] No video ID found")
            return ""

        yt_url = f"https://www.youtube.com/watch?v={video_yt_id}"
        print(f"    Found: {yt_url}")

        # Download with yt-dlp + ffmpeg trim to 10 seconds
        temp_path = os.path.join(VIDEO_DIR, f"_temp_{vid_id}.mp4")
        dl_cmd = [
            sys.executable, "-m", "yt_dlp",
            yt_url,
            "-f", "best[height<=480][ext=mp4]/best[height<=480]/best",
            "--no-playlist",
            "--ffmpeg-location", "/home/seung/miniforge3/bin",
            "-o", temp_path,
            "--quiet",
        ]

        subprocess.run(
            dl_cmd,
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "PATH": f"/home/seung/miniforge3/bin:/usr/bin:{os.environ.get('PATH', '')}"}
        )

        if not os.path.exists(temp_path):
            # try .webm extension
            temp_webm = temp_path.replace(".mp4", ".webm")
            if os.path.exists(temp_webm):
                temp_path = temp_webm
            else:
                print(f"    [warn] Download failed")
                return ""

        # Trim to 10 seconds using ffmpeg
        trim_cmd = [
            FFMPEG,
            "-y", "-i", temp_path,
            "-t", str(duration),
            "-c:v", "libx264", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ]

        subprocess.run(
            trim_cmd,
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "PATH": f"/home/seung/miniforge3/bin:/usr/bin:{os.environ.get('PATH', '')}"}
        )

        # Cleanup temp
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if os.path.exists(output_path):
            # Get actual duration
            probe_cmd = [
                "/home/seung/miniforge3/bin/ffprobe",
                "-v", "quiet", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                output_path,
            ]
            dur_result = subprocess.run(
                probe_cmd, capture_output=True, text=True, timeout=10,
                env={**os.environ, "PATH": f"/home/seung/miniforge3/bin:/usr/bin:{os.environ.get('PATH', '')}"}
            )
            dur = dur_result.stdout.strip()
            print(f"    [ok] Saved: {vid_id}.mp4 ({dur}s)")
            return output_path
        else:
            print(f"    [warn] Trim failed")
            return ""

    except subprocess.TimeoutExpired:
        print(f"    [warn] Timeout")
        return ""
    except Exception as e:
        print(f"    [error] {e}")
        return ""


def main():
    print("=" * 60)
    print("Downloading 20 Diverse Video Clips (10s each)")
    print("=" * 60)
    print(f"Output: {VIDEO_DIR}\n")

    # Category summary
    categories = {}
    for v in VIDEOS:
        cat = v["category"]
        categories[cat] = categories.get(cat, 0) + 1
    print("Category distribution:")
    for cat, cnt in categories.items():
        print(f"  {cat}: {cnt} videos")
    print()

    results = []
    for i, video in enumerate(VIDEOS):
        print(f"[{i+1}/20] {video['id']}: {video['prompt']}")
        path = search_and_download(video)
        video["local_path"] = path
        video["downloaded"] = bool(path)
        results.append(video)

    # Save manifest
    manifest_path = os.path.join(os.path.dirname(__file__), "video_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    success = sum(1 for r in results if r["downloaded"])
    print(f"\n{'='*60}")
    print(f"Downloaded: {success}/20 videos")
    print(f"Manifest: {manifest_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
