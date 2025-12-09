"""
Multi-source spatial audio extension.

Extends Vid2Spatial to handle 2-3 simultaneous sound sources:
- Track multiple objects in video
- Separate audio sources (externally provided or estimated)
- Encode each source to FOA independently
- Mix FOA streams to create multi-source spatial audio
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
import soundfile as sf

from .vision import yolo_bytetrack_traj, CameraIntrinsics, pixel_to_ray, ray_to_angles
from .foa_render import encode_mono_to_foa, interpolate_angles_distance


def track_multiple_sources(
    video_path: str,
    num_sources: int = 2,
    class_names: Optional[List[str]] = None,
    track_ids: Optional[List[int]] = None,
    fov_deg: float = 60.0,
    sample_stride: int = 1,
) -> List[Dict]:
    """
    Track multiple objects in video and compute 3D trajectories.

    Args:
        video_path: Path to input video
        num_sources: Number of sources to track (2-3)
        class_names: List of class names for each source (default: all 'person')
        track_ids: Specific track IDs to select for each source
        fov_deg: Camera horizontal FOV in degrees
        sample_stride: Frame sampling stride

    Returns:
        List of trajectory dicts, one per source:
        [{
            "source_id": 0,
            "intrinsics": {"width": int, "height": int, "fov_deg": float},
            "frames": [{"frame": int, "az": float, "el": float, "dist_m": float, ...}, ...]
        }, ...]
    """
    if class_names is None:
        class_names = ["person"] * num_sources

    if track_ids is None:
        track_ids = [None] * num_sources

    import cv2

    # Get video dimensions
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    K = CameraIntrinsics(width=W, height=H, fov_deg=fov_deg)

    # Track all objects with YOLO
    print(f"[multi-source] Tracking {num_sources} sources...")
    all_tracks = yolo_bytetrack_traj(
        video_path,
        cls_name="person",  # Track all people first
        select_track_id=None,
        sample_stride=sample_stride
    )

    # Group tracks by track_id
    tracks_by_id = {}
    for rec in all_tracks:
        tid = rec.get("track_id", 0)
        if tid not in tracks_by_id:
            tracks_by_id[tid] = []
        tracks_by_id[tid].append(rec)

    print(f"[multi-source] Found {len(tracks_by_id)} unique tracks")

    # Select top N tracks by frame count
    sorted_tracks = sorted(
        tracks_by_id.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    if len(sorted_tracks) < num_sources:
        print(f"[warn] Only found {len(sorted_tracks)} tracks, requested {num_sources}")
        num_sources = len(sorted_tracks)

    selected_track_ids = [tid for tid, _ in sorted_tracks[:num_sources]]
    print(f"[multi-source] Selected track IDs: {selected_track_ids}")

    # Compute 3D trajectory for each source
    trajectories = []

    for i, tid in enumerate(selected_track_ids):
        traj_2d = tracks_by_id[tid]
        print(f"[multi-source] Source {i}: {len(traj_2d)} frames, track_id={tid}")

        # Convert 2D trajectory to 3D
        frames_3d = []
        for rec in traj_2d:
            cx = rec["x"] + rec["w"] / 2
            cy = rec["y"] + rec["h"] / 2

            # Simple depth estimation (constant for now)
            depth_rel = 0.5  # Mid-range

            # Pixel to ray
            ray = pixel_to_ray(cx, cy, K)

            # Ray to angles
            az, el = ray_to_angles(ray)

            # Estimate distance (simple heuristic based on bbox size)
            # Larger bbox = closer object
            bbox_area = rec["w"] * rec["h"]
            frame_area = W * H
            relative_size = bbox_area / frame_area
            # Map relative size [0.001, 0.1] to distance [3.0, 1.0] meters
            dist_m = np.clip(3.0 / (relative_size * 30 + 0.1), 1.0, 3.0)

            # 3D position
            x = ray[0] * dist_m
            y = ray[1] * dist_m
            z = ray[2] * dist_m

            frames_3d.append({
                "frame": rec["frame"],
                "az": float(az),
                "el": float(el),
                "dist_m": float(dist_m),
                "x": float(x),
                "y": float(y),
                "z": float(z),
            })

        trajectories.append({
            "source_id": i,
            "track_id": tid,
            "intrinsics": {"width": W, "height": H, "fov_deg": fov_deg},
            "frames": frames_3d
        })

    return trajectories


def encode_multi_source_foa(
    audio_sources: List[np.ndarray],
    trajectories: List[Dict],
    sr: int = 48000,
) -> np.ndarray:
    """
    Encode multiple mono audio sources to FOA and mix.

    Args:
        audio_sources: List of mono audio signals [source1, source2, ...]
        trajectories: List of trajectory dicts (one per source)
        sr: Sample rate

    Returns:
        Mixed FOA audio [4, T]
    """
    if len(audio_sources) != len(trajectories):
        raise ValueError(f"Number of audio sources ({len(audio_sources)}) must match trajectories ({len(trajectories)})")

    # Find max length
    max_len = max(len(audio) for audio in audio_sources)

    # Initialize mixed FOA
    foa_mixed = np.zeros((4, max_len), dtype=np.float32)

    for i, (audio, traj) in enumerate(zip(audio_sources, trajectories)):
        print(f"[multi-source] Encoding source {i}...")

        T = len(audio)

        # Interpolate trajectory to audio timeline
        az_s, el_s, dist_s = interpolate_angles_distance(traj["frames"], T=T, sr=sr)

        # Encode to FOA
        foa = encode_mono_to_foa(audio, az_s, el_s)

        # Mix into combined FOA
        foa_mixed[:, :T] += foa

    return foa_mixed


def process_multi_source_video(
    video_path: str,
    audio_sources: List[np.ndarray],
    sr: int = 48000,
    num_sources: int = 2,
    fov_deg: float = 60.0,
    output_path: str = "multi_source.foa.wav",
) -> Dict:
    """
    End-to-end multi-source spatial audio from video.

    Args:
        video_path: Path to input video
        audio_sources: List of mono audio signals
        sr: Sample rate
        num_sources: Number of sources to track
        fov_deg: Camera FOV
        output_path: Output FOA file path

    Returns:
        Result dict with trajectories and output info
    """
    print("="*60)
    print("Multi-Source Spatial Audio Processing")
    print("="*60)

    # Track multiple sources
    trajectories = track_multiple_sources(
        video_path=video_path,
        num_sources=num_sources,
        fov_deg=fov_deg
    )

    print(f"\n[multi-source] Tracked {len(trajectories)} sources")

    # Encode to FOA
    foa_mixed = encode_multi_source_foa(
        audio_sources=audio_sources,
        trajectories=trajectories,
        sr=sr
    )

    print(f"[multi-source] Generated mixed FOA: {foa_mixed.shape}")

    # Write output
    sf.write(output_path, foa_mixed.T, sr)
    print(f"[multi-source] Saved to {output_path}")

    print("="*60)
    print("Multi-source processing complete!")
    print("="*60)

    return {
        "trajectories": trajectories,
        "output_path": output_path,
        "num_sources": len(trajectories),
        "duration_sec": foa_mixed.shape[1] / sr
    }


__all__ = [
    "track_multiple_sources",
    "encode_multi_source_foa",
    "process_multi_source_video",
]
