#!/usr/bin/env python3
"""
Prepare stimuli for vid2spatial perceptual listening test.

Expert-advised design:
  - 8 strategic clips (3 fast, 3 moderate, 2 slow motion)
  - Conditions: A (Stereo Pan+Reverb baseline), C (Proposed HRTF binaural)
  - 7-point MOS, 4 evaluation dimensions
  - Video + audio simultaneous playback

Generates:
  stimuli/
    config.json              - Test configuration
    {scene_id}/
      video.mp4              - Source video (symlink)
      overlay.mp4            - Trajectory overlay (symlink)
      proposed.wav           - Proposed method (HRTF binaural)
      baseline.wav           - Stereo pan + reverb baseline
      mono.wav               - Mono anchor (dual-mono, no spatialization)

Usage:
    cd evaluation/listening_test
    python prepare_stimuli.py
"""
import sys
import json
import shutil
import numpy as np
import soundfile as sf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from vid2spatial_pkg.foa_render import (
    render_stereo_pan_reverb_baseline,
    interpolate_angles_distance,
)

SOT_DIR = ROOT / "experiments" / "sot_15_videos"
OVERLAY_DIR = ROOT / "evaluation" / "trajectory_videos" / "sot_benchmark"
STIMULI_DIR = Path(__file__).resolve().parent / "stimuli"

# 8 strategic clips: 3 fast, 3 moderate, 2 slow
SELECTED_CLIPS = [
    # Fast motion — challenges tracking + spatial rendering
    {"id": "08_surfer_wave",    "motion": "fast",     "category": "sports"},
    {"id": "12_parkour_runner", "motion": "fast",     "category": "person"},
    {"id": "07_skier_downhill", "motion": "fast",     "category": "sports"},
    # Moderate motion — typical use cases
    {"id": "10_dancer_solo",    "motion": "moderate", "category": "person"},
    {"id": "01_dog_frisbee",    "motion": "moderate", "category": "animal"},
    {"id": "09_bmx_rider",      "motion": "moderate", "category": "sports"},
    # Slow / predictable — easy for tracking, tests spatial quality
    {"id": "06_boat_water",     "motion": "slow",     "category": "vehicle"},
    {"id": "15_ball_bouncing",  "motion": "slow",     "category": "object"},
]


def make_mono_anchor(stereo_path: str, output_path: str):
    """Create mono (no spatialization) version as dual-mono stereo."""
    data, sr = sf.read(stereo_path)
    if data.ndim == 2:
        mono = data.mean(axis=1)
    else:
        mono = data
    stereo_mono = np.stack([mono, mono], axis=1)
    sf.write(output_path, stereo_mono, sr, subtype='FLOAT')


def render_baseline(audio_mono_path: str, trajectory_path: str, output_path: str):
    """Render stereo pan + reverb baseline from trajectory."""
    audio, sr = sf.read(audio_mono_path, dtype='float32')
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    T = audio.shape[0]

    with open(trajectory_path) as f:
        traj = json.load(f)
    frames = traj["frames"]

    az_s, el_s, dist_s, d_rel_s = interpolate_angles_distance(frames, T, sr)
    stereo = render_stereo_pan_reverb_baseline(audio, sr, az_s, dist_s, d_rel_s)
    sf.write(output_path, stereo.T, sr, subtype='FLOAT')
    return sr, T


def prepare():
    STIMULI_DIR.mkdir(parents=True, exist_ok=True)

    videos_dir = SOT_DIR / "videos"
    outputs_dir = SOT_DIR / "outputs"
    hrtf_dir = SOT_DIR / "render_orig_hrtf"

    scenes = []

    for clip in SELECTED_CLIPS:
        scene_id = clip["id"]
        scene_dir = STIMULI_DIR / scene_id
        scene_dir.mkdir(exist_ok=True)

        # Source files
        video_mp4 = videos_dir / f"{scene_id}.mp4"
        overlay_mp4 = OVERLAY_DIR / f"{scene_id}_overlay.mp4"
        hrtf_wav = hrtf_dir / f"{scene_id}_hrtf_binaural.wav"
        audio_mono = outputs_dir / scene_id / "audio_mono.wav"
        traj_json = outputs_dir / scene_id / "trajectory_3d.json"

        required = [video_mp4, overlay_mp4, hrtf_wav, audio_mono, traj_json]
        if not all(p.exists() for p in required):
            missing = [p.name for p in required if not p.exists()]
            print(f"  [skip] {scene_id}: missing {missing}")
            continue

        # Symlink video and overlay
        for name, src in [("video.mp4", video_mp4), ("overlay.mp4", overlay_mp4)]:
            dst = scene_dir / name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())

        # Condition C: Proposed (HRTF binaural) — copy existing render
        shutil.copy2(hrtf_wav, scene_dir / "proposed.wav")

        # Condition A: Stereo Pan + Reverb baseline — generate from trajectory
        sr, T = render_baseline(
            str(audio_mono), str(traj_json), str(scene_dir / "baseline.wav")
        )

        # Mono anchor (low anchor for sanity check)
        make_mono_anchor(str(hrtf_wav), str(scene_dir / "mono.wav"))

        info = sf.info(str(hrtf_wav))
        scenes.append({
            "id": scene_id,
            "motion": clip["motion"],
            "category": clip["category"],
            "duration_s": round(info.duration, 1),
            "sample_rate": info.samplerate,
        })
        print(f"  [ok] {scene_id} ({clip['motion']}): {info.duration:.1f}s")

    # Build test config — expert-advised 7-point MOS, 4 dimensions
    config = {
        "title": "Vid2Spatial Perceptual Evaluation",
        "description": "Evaluate spatial audio quality for video-guided spatial audio rendering",
        "instructions": {
            "general": "Please use headphones for the entire test. For each clip, watch the video and listen to the audio simultaneously. You will rate two versions (presented in random order as 'Version 1' and 'Version 2') on four dimensions using a 7-point scale.",
            "environment": "Use over-ear headphones in a quiet room. Adjust volume to a comfortable level before starting."
        },
        "scenes": scenes,
        "conditions": [
            {
                "id": "proposed",
                "file": "proposed.wav",
                "blind_label": None,  # assigned randomly per trial
                "description": "Proposed: FOA + KEMAR HRTF binaural"
            },
            {
                "id": "baseline",
                "file": "baseline.wav",
                "blind_label": None,
                "description": "Baseline: Stereo pan + distance reverb"
            },
            {
                "id": "mono",
                "file": "mono.wav",
                "blind_label": "Mono",
                "description": "Mono anchor (no spatialization)"
            }
        ],
        "questions": [
            {
                "id": "alignment",
                "text": "Audiovisual Alignment",
                "instruction": "How well does the sound position match the visual object location?",
                "anchors": {
                    "1": "Audio seems unrelated to video position",
                    "4": "Approximate match, some noticeable offset",
                    "7": "Audio clearly follows the object in the scene"
                }
            },
            {
                "id": "smoothness",
                "text": "Motion Smoothness",
                "instruction": "How smooth and continuous is the spatial audio movement?",
                "anchors": {
                    "1": "Frequent jumps, pops, or instability",
                    "4": "Mostly smooth with occasional artifacts",
                    "7": "Continuous, stable, artifact-free motion"
                }
            },
            {
                "id": "depth",
                "text": "Depth Plausibility",
                "instruction": "How natural are the near/far distance changes in the audio?",
                "anchors": {
                    "1": "No sense of distance or depth",
                    "4": "Some distance perception but not always convincing",
                    "7": "Natural and convincing near/far changes"
                }
            },
            {
                "id": "overall",
                "text": "Overall Usefulness",
                "instruction": "How useful would this spatial audio be for immersive content creation?",
                "anchors": {
                    "1": "Unusable — distracting or incorrect",
                    "4": "Usable with noticeable limitations",
                    "7": "Ready for production — enhances the experience"
                }
            }
        ],
        "scale": {
            "min": 1,
            "max": 7,
            "type": "mos"
        },
        "design": {
            "conditions_per_clip": ["proposed", "baseline"],
            "mono_anchor": True,
            "blind": True,
            "randomize_condition_order": True,
            "randomize_clip_order": True,
            "uses_video": True,
            "note": "Mono anchor shown once per clip as reference point, not rated in pairwise comparison"
        }
    }

    config_path = STIMULI_DIR / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*50}")
    print(f"[done] {len(scenes)} clips prepared → {STIMULI_DIR}")
    print(f"[config] {config_path}")
    print(f"[questions] {len(config['questions'])} evaluation dimensions (7-point MOS)")
    print(f"[conditions] proposed (HRTF) vs baseline (stereo pan+reverb) + mono anchor")
    print(f"[clips] fast:{sum(1 for s in scenes if s['motion']=='fast')} "
          f"moderate:{sum(1 for s in scenes if s['motion']=='moderate')} "
          f"slow:{sum(1 for s in scenes if s['motion']=='slow')}")


if __name__ == "__main__":
    prepare()
