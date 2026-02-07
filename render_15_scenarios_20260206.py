#!/usr/bin/env python3
"""
Render 15 diverse spatial audio scenarios for perceptual evaluation.
Date: 2026-02-06

Each scenario tests different:
- Objects (instruments, vehicles, voices)
- Motion patterns (oscillation, circular, approaching, receding)
- Speed (slow, medium, fast)
- Depth variations (near, far, depth change)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Import FOA renderer
from vid2spatial_pkg.foa_render import (
    encode_mono_to_foa,
    write_foa_wav,
    foa_to_binaural,
    apply_distance_gain_lpf,
    build_wet_curve_from_dist_occ,
    apply_timevarying_reverb_foa,
    smooth_limit_angles,
)


# =============================================================================
# Scenario Definitions
# =============================================================================

SCENARIOS = [
    # Category 1: Horizontal Motion (Left-Right)
    {
        "id": "01_guitar_slow_lr",
        "name": "Guitar - Slow Left-Right Pan",
        "object": "guitar",
        "motion": "horizontal_oscillation",
        "speed": "slow",  # 0.2 Hz
        "depth": "fixed_near",  # 2m
        "description": "Acoustic guitar slowly panning left to right",
    },
    {
        "id": "02_drum_fast_lr",
        "name": "Drum - Fast Left-Right Pan",
        "object": "drum",
        "motion": "horizontal_oscillation",
        "speed": "fast",  # 1.0 Hz
        "depth": "fixed_mid",  # 4m
        "description": "Percussion rapidly moving left to right",
    },
    {
        "id": "03_voice_medium_lr",
        "name": "Voice - Medium Left-Right Pan",
        "object": "voice",
        "motion": "horizontal_oscillation",
        "speed": "medium",  # 0.5 Hz
        "depth": "fixed_near",  # 1.5m
        "description": "Human voice at conversation distance, moving laterally",
    },

    # Category 2: Depth Motion (Near-Far)
    {
        "id": "04_piano_approaching",
        "name": "Piano - Approaching",
        "object": "piano",
        "motion": "approaching",
        "speed": "slow",
        "depth": "far_to_near",  # 8m → 1m
        "description": "Piano sound approaching from far to near",
    },
    {
        "id": "05_violin_receding",
        "name": "Violin - Receding",
        "object": "violin",
        "motion": "receding",
        "speed": "medium",
        "depth": "near_to_far",  # 1m → 10m
        "description": "Violin sound moving away into the distance",
    },
    {
        "id": "06_synth_depth_oscillation",
        "name": "Synth - Depth Oscillation",
        "object": "synth",
        "motion": "depth_oscillation",
        "speed": "slow",
        "depth": "oscillating",  # 2m ↔ 6m
        "description": "Synthesizer pulsing in depth (near-far-near)",
    },

    # Category 3: Circular Motion
    {
        "id": "07_cello_circular_slow",
        "name": "Cello - Slow Circular",
        "object": "cello",
        "motion": "circular",
        "speed": "slow",  # 0.1 Hz (10s per rotation)
        "depth": "fixed_mid",  # 3m
        "description": "Cello circling around the listener slowly",
    },
    {
        "id": "08_flute_circular_fast",
        "name": "Flute - Fast Circular",
        "object": "flute",
        "motion": "circular",
        "speed": "fast",  # 0.5 Hz (2s per rotation)
        "depth": "fixed_near",  # 2m
        "description": "Flute rapidly circling the listener",
    },

    # Category 4: Combined Motion (3D)
    {
        "id": "09_brass_spiral",
        "name": "Brass - Spiral Motion",
        "object": "brass",
        "motion": "spiral",
        "speed": "medium",
        "depth": "spiral_in",  # Circling while approaching
        "description": "Brass section spiraling inward toward listener",
    },
    {
        "id": "10_strings_figure8",
        "name": "Strings - Figure-8 Pattern",
        "object": "strings",
        "motion": "figure8",
        "speed": "slow",
        "depth": "varying",
        "description": "String ensemble moving in figure-8 pattern",
    },

    # Category 5: Elevation Changes
    {
        "id": "11_bird_ascending",
        "name": "Bird - Ascending",
        "object": "nature",
        "motion": "ascending",
        "speed": "medium",
        "depth": "fixed_mid",
        "description": "Bird sound rising from ground level to overhead",
    },
    {
        "id": "12_helicopter_descending",
        "name": "Helicopter - Descending",
        "object": "vehicle",
        "motion": "descending",
        "speed": "slow",
        "depth": "far_to_near",
        "description": "Helicopter sound descending from above",
    },

    # Category 6: Complex Real-World Scenarios
    {
        "id": "13_car_passby",
        "name": "Car - Drive-by",
        "object": "vehicle",
        "motion": "passby",
        "speed": "fast",
        "depth": "passby_curve",  # Far → Near → Far
        "description": "Car driving past the listener (Doppler-like)",
    },
    {
        "id": "14_crowd_random",
        "name": "Crowd - Random Movement",
        "object": "crowd",
        "motion": "random_walk",
        "speed": "slow",
        "depth": "random",
        "description": "Ambient crowd noise with random spatial movement",
    },
    {
        "id": "15_orchestra_static_far",
        "name": "Orchestra - Static Far",
        "object": "orchestra",
        "motion": "static",
        "speed": "none",
        "depth": "fixed_far",  # 8m, slight variation
        "description": "Full orchestra at concert hall distance",
    },
]


# =============================================================================
# Motion Pattern Generators
# =============================================================================

def generate_trajectory(
    scenario: Dict,
    duration_sec: float = 10.0,
    fps: float = 30.0,
    sr: int = 48000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate trajectory based on scenario specification.

    Returns:
        az_s: Azimuth per sample (radians)
        el_s: Elevation per sample (radians)
        dist_s: Distance per sample (meters)
        d_rel_s: Normalized distance [0,1]
    """
    n_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n_samples)

    motion = scenario["motion"]
    speed = scenario["speed"]
    depth = scenario["depth"]

    # Speed mapping (Hz for oscillation)
    speed_map = {"slow": 0.2, "medium": 0.5, "fast": 1.0, "none": 0.0}
    freq = speed_map.get(speed, 0.5)

    # Initialize outputs
    az = np.zeros(n_samples)
    el = np.zeros(n_samples)
    dist = np.ones(n_samples) * 3.0  # Default 3m

    # --- Azimuth patterns ---
    if motion == "horizontal_oscillation":
        # Left-right pan: ±60 degrees
        az = np.deg2rad(60) * np.sin(2 * np.pi * freq * t)

    elif motion == "circular":
        # Full 360 rotation
        az = 2 * np.pi * freq * t
        az = np.mod(az + np.pi, 2 * np.pi) - np.pi  # Wrap to [-pi, pi]

    elif motion == "figure8":
        # Figure-8 pattern (horizontal)
        az = np.deg2rad(45) * np.sin(2 * np.pi * freq * t)
        el = np.deg2rad(20) * np.sin(4 * np.pi * freq * t)  # Twice the frequency

    elif motion == "spiral":
        # Spiral inward
        az = 2 * np.pi * freq * t
        az = np.mod(az + np.pi, 2 * np.pi) - np.pi

    elif motion == "passby":
        # Car passby: start from left, end at right
        # Position: -90° → 0° → +90° (relative to listener)
        # But in audio convention: +90° = left, -90° = right
        progress = t / duration_sec  # 0 to 1
        az = np.deg2rad(90) * (1 - 2 * progress)  # +90° to -90°

    elif motion == "random_walk":
        # Random walk with smoothing
        np.random.seed(42)  # Reproducibility
        az_raw = np.cumsum(np.random.randn(n_samples) * 0.01)
        az_raw = np.clip(az_raw, -np.pi/2, np.pi/2)
        # Smooth
        from scipy.ndimage import gaussian_filter1d
        az = gaussian_filter1d(az_raw, sigma=sr*0.1)

    elif motion in ["approaching", "receding", "depth_oscillation", "ascending", "descending"]:
        # Primarily depth/elevation motion, keep azimuth stable with slight variation
        az = np.deg2rad(10) * np.sin(2 * np.pi * 0.1 * t)  # Slight drift

    elif motion == "static":
        # Static position with minimal variation
        az = np.deg2rad(5) * np.sin(2 * np.pi * 0.05 * t)

    # --- Elevation patterns ---
    if motion == "ascending":
        # Ground (0°) to overhead (+60°)
        progress = t / duration_sec
        el = np.deg2rad(60) * progress

    elif motion == "descending":
        # Overhead (+60°) to ground (0°)
        progress = t / duration_sec
        el = np.deg2rad(60) * (1 - progress)

    elif motion == "spiral":
        # Slight elevation change during spiral
        el = np.deg2rad(15) * np.sin(2 * np.pi * freq * 0.5 * t)

    elif motion == "circular":
        # Keep on horizon
        el = np.zeros(n_samples)

    # --- Distance patterns ---
    if depth == "fixed_near":
        dist = np.ones(n_samples) * 1.5
    elif depth == "fixed_mid":
        dist = np.ones(n_samples) * 4.0
    elif depth == "fixed_far":
        dist = np.ones(n_samples) * 8.0
    elif depth == "far_to_near":
        progress = t / duration_sec
        dist = 8.0 - 7.0 * progress  # 8m → 1m
    elif depth == "near_to_far":
        progress = t / duration_sec
        dist = 1.0 + 9.0 * progress  # 1m → 10m
    elif depth == "oscillating":
        dist = 4.0 + 2.0 * np.sin(2 * np.pi * freq * t)  # 2m ↔ 6m
    elif depth == "spiral_in":
        progress = t / duration_sec
        dist = 6.0 - 4.0 * progress  # 6m → 2m
    elif depth == "passby_curve":
        # Closest at midpoint
        progress = t / duration_sec
        dist = 8.0 - 6.0 * np.sin(np.pi * progress)  # 8m → 2m → 8m
    elif depth == "varying":
        dist = 3.0 + 1.5 * np.sin(2 * np.pi * freq * 0.5 * t)
    elif depth == "random":
        np.random.seed(43)
        dist_raw = 3.0 + np.cumsum(np.random.randn(n_samples) * 0.005)
        dist_raw = np.clip(dist_raw, 1.0, 8.0)
        from scipy.ndimage import gaussian_filter1d
        dist = gaussian_filter1d(dist_raw, sigma=sr*0.2)

    # Compute d_rel (normalized distance)
    d_min, d_max = 0.5, 10.0
    d_rel = np.clip((dist - d_min) / (d_max - d_min), 0.0, 1.0)

    return az.astype(np.float32), el.astype(np.float32), dist.astype(np.float32), d_rel.astype(np.float32)


# =============================================================================
# Audio Source Selection
# =============================================================================

def get_sample_audio(scenario: Dict, sr: int = 48000, duration_sec: float = 10.0) -> np.ndarray:
    """
    Get or generate sample audio for the scenario.
    Uses real audio from FAIR-Play dataset or generates synthetic.
    """
    # Map object types to potential audio sources
    audio_mapping = {
        "guitar": [1, 2, 3, 4, 5],  # FAIR-Play indices with guitar-like sounds
        "drum": [10, 11, 12],
        "voice": [20, 21, 22],
        "piano": [30, 31, 32],
        "violin": [40, 41, 42],
        "synth": None,  # Generate synthetic
        "cello": [50, 51],
        "flute": [60, 61],
        "brass": [70, 71],
        "strings": [80, 81],
        "nature": None,  # Generate
        "vehicle": None,  # Generate
        "crowd": None,  # Generate
        "orchestra": [100, 101],
    }

    obj_type = scenario["object"]
    n_samples = int(duration_sec * sr)

    # Try to load from FAIR-Play
    fairplay_dir = Path("/home/seung/data/fairplay/binaural_audios")

    # Simple approach: use first available audio and process it
    audio_files = list(fairplay_dir.glob("*.wav"))
    if audio_files:
        # Select based on scenario index
        idx = int(scenario["id"].split("_")[0]) % len(audio_files)
        audio_path = audio_files[idx]

        try:
            audio, file_sr = sf.read(audio_path, dtype='float32')
            # Mix to mono if stereo
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            # Resample if needed
            if file_sr != sr:
                from scipy.signal import resample
                audio = resample(audio, int(len(audio) * sr / file_sr))
            # Trim or pad to duration
            if len(audio) > n_samples:
                audio = audio[:n_samples]
            elif len(audio) < n_samples:
                # Loop
                repeats = int(np.ceil(n_samples / len(audio)))
                audio = np.tile(audio, repeats)[:n_samples]
            return audio.astype(np.float32)
        except Exception as e:
            print(f"  Warning: Could not load {audio_path}: {e}")

    # Fallback: generate synthetic audio
    return generate_synthetic_audio(obj_type, sr, duration_sec)


def generate_synthetic_audio(obj_type: str, sr: int, duration_sec: float) -> np.ndarray:
    """Generate synthetic audio for testing."""
    n_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n_samples)

    if obj_type in ["guitar", "piano", "violin", "cello", "strings"]:
        # Harmonic rich tone
        f0 = 220.0  # A3
        audio = np.zeros(n_samples)
        for h in range(1, 8):
            audio += (0.5 ** h) * np.sin(2 * np.pi * f0 * h * t)
        # Add envelope
        env = np.exp(-t * 0.3) * (1 - np.exp(-t * 20))
        audio = audio * env
        # Add some noise
        audio += 0.02 * np.random.randn(n_samples)

    elif obj_type == "drum":
        # Percussive with noise
        audio = np.zeros(n_samples)
        beat_interval = sr // 2  # 2 beats per second
        for i in range(0, n_samples, beat_interval):
            if i + sr//4 < n_samples:
                t_beat = np.linspace(0, 0.25, sr//4)
                beat = np.sin(2 * np.pi * 100 * t_beat) * np.exp(-t_beat * 30)
                beat += 0.5 * np.random.randn(sr//4) * np.exp(-t_beat * 40)
                audio[i:i+sr//4] += beat

    elif obj_type in ["synth", "brass"]:
        # Saw-like wave
        f0 = 330.0
        audio = 2 * (t * f0 - np.floor(t * f0 + 0.5))
        audio *= 0.3
        # LFO
        audio *= 1 + 0.3 * np.sin(2 * np.pi * 0.5 * t)

    elif obj_type == "flute":
        # Pure tone with breathiness
        f0 = 880.0
        audio = 0.6 * np.sin(2 * np.pi * f0 * t)
        audio += 0.1 * np.random.randn(n_samples)
        # Vibrato
        vibrato = 1 + 0.02 * np.sin(2 * np.pi * 5 * t)
        audio = np.sin(2 * np.pi * f0 * vibrato * t) * 0.6

    elif obj_type == "voice":
        # Vowel-like formants
        f0 = 150.0
        audio = np.zeros(n_samples)
        formants = [800, 1200, 2500]  # 'ah' vowel
        for f in formants:
            audio += np.sin(2 * np.pi * f * t) * np.exp(-abs(f-1000)/500)
        # Modulate with f0
        audio *= np.sin(2 * np.pi * f0 * t)
        audio += 0.05 * np.random.randn(n_samples)

    elif obj_type in ["nature", "crowd"]:
        # Pink noise
        audio = np.random.randn(n_samples)
        # Simple pink filter (1/f)
        from scipy.signal import butter, filtfilt
        b, a = butter(2, 2000 / (sr/2), btype='low')
        audio = filtfilt(b, a, audio)
        audio *= 0.3

    elif obj_type == "vehicle":
        # Low rumble with harmonics
        f0 = 60.0
        audio = np.zeros(n_samples)
        for h in range(1, 10):
            audio += (0.7 ** h) * np.sin(2 * np.pi * f0 * h * t)
        # Modulate for engine variation
        audio *= 1 + 0.2 * np.sin(2 * np.pi * 2 * t)
        audio += 0.1 * np.random.randn(n_samples)

    elif obj_type == "orchestra":
        # Complex mix
        audio = np.zeros(n_samples)
        for f in [220, 330, 440, 554, 660]:
            audio += np.sin(2 * np.pi * f * t) * (0.5 + 0.3 * np.random.rand())
        audio += 0.05 * np.random.randn(n_samples)

    else:
        # Default: simple tone
        audio = np.sin(2 * np.pi * 440 * t)

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.8
    return audio.astype(np.float32)


# =============================================================================
# Main Rendering Function
# =============================================================================

def render_scenario(
    scenario: Dict,
    output_dir: Path,
    sr: int = 48000,
    duration_sec: float = 10.0,
    apply_reverb: bool = True,
) -> Dict:
    """
    Render a single scenario to FOA and binaural audio.

    Returns:
        Dict with output paths and metadata
    """
    scenario_id = scenario["id"]
    print(f"\n[{scenario_id}] {scenario['name']}")
    print(f"  Motion: {scenario['motion']}, Speed: {scenario['speed']}, Depth: {scenario['depth']}")

    # Generate trajectory
    az_s, el_s, dist_s, d_rel_s = generate_trajectory(scenario, duration_sec, sr=sr)
    print(f"  Trajectory: az=[{np.rad2deg(az_s.min()):.1f}, {np.rad2deg(az_s.max()):.1f}] deg, "
          f"el=[{np.rad2deg(el_s.min()):.1f}, {np.rad2deg(el_s.max()):.1f}] deg, "
          f"dist=[{dist_s.min():.1f}, {dist_s.max():.1f}] m")

    # Get audio
    audio = get_sample_audio(scenario, sr, duration_sec)
    print(f"  Audio: {len(audio)/sr:.1f}s @ {sr}Hz")

    # Apply distance-based processing (gain + LPF)
    audio_proc = apply_distance_gain_lpf(
        audio, sr, dist_s, d_rel_s,
        gain_k=1.0,
        lpf_min_hz=600.0,
        lpf_max_hz=10000.0,
    )

    # Smooth angles
    az_smooth, el_smooth = smooth_limit_angles(az_s, el_s, sr, smooth_ms=30.0)

    # Encode to FOA
    foa = encode_mono_to_foa(audio_proc, az_smooth, el_smooth)

    # Apply reverb
    if apply_reverb:
        wet_curve = build_wet_curve_from_dist_occ(d_rel_s, wet_min=0.05, wet_max=0.30)
        foa = apply_timevarying_reverb_foa(foa, sr, wet_curve, rt60=0.5)

    # Save FOA
    foa_path = output_dir / f"{scenario_id}_foa.wav"
    write_foa_wav(str(foa_path), foa, sr)
    print(f"  Saved FOA: {foa_path.name}")

    # Convert to binaural
    binaural = foa_to_binaural(foa, sr)
    binaural_path = output_dir / f"{scenario_id}_binaural.wav"
    sf.write(str(binaural_path), binaural.T, sr, subtype="FLOAT")
    print(f"  Saved Binaural: {binaural_path.name}")

    # Save trajectory for reference
    traj_path = output_dir / f"{scenario_id}_trajectory.json"
    n_frames = int(duration_sec * 30)  # 30 fps
    frame_indices = np.linspace(0, len(az_s)-1, n_frames).astype(int)
    traj_data = {
        "scenario": scenario,
        "frames": [
            {
                "frame": int(i),
                "az": float(az_s[frame_indices[i]]),
                "el": float(el_s[frame_indices[i]]),
                "dist_m": float(dist_s[frame_indices[i]]),
                "d_rel": float(d_rel_s[frame_indices[i]]),
            }
            for i in range(n_frames)
        ]
    }
    with open(traj_path, 'w') as f:
        json.dump(traj_data, f, indent=2)

    return {
        "scenario_id": scenario_id,
        "foa_path": str(foa_path),
        "binaural_path": str(binaural_path),
        "trajectory_path": str(traj_path),
        "duration_sec": duration_sec,
        "sample_rate": sr,
    }


def main():
    """Render all 15 scenarios."""
    # Output directory
    output_dir = Path(__file__).parent / "render_outputs_20260206"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Vid2Spatial: 15 Scenario Audio Rendering")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Parameters
    sr = 48000
    duration_sec = 10.0

    results = []
    for scenario in SCENARIOS:
        try:
            result = render_scenario(
                scenario,
                output_dir,
                sr=sr,
                duration_sec=duration_sec,
                apply_reverb=True,
            )
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save summary
    summary_path = output_dir / "render_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "date": "2026-02-06",
            "scenarios": SCENARIOS,
            "results": results,
        }, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Completed: {len(results)}/{len(SCENARIOS)} scenarios")
    print(f"Summary: {summary_path}")
    print("=" * 60)

    # Print listening instructions
    print("\n** Listening Instructions **")
    print("1. Use headphones for proper binaural spatialization")
    print("2. Binaural files (*_binaural.wav) are ready for headphone listening")
    print("3. FOA files (*_foa.wav) can be decoded with any Ambisonics decoder")
    print("\nScenario Summary:")
    for s in SCENARIOS:
        print(f"  {s['id']}: {s['name']} - {s['description']}")


if __name__ == "__main__":
    main()
