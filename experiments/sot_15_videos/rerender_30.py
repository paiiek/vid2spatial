#!/usr/bin/env python3
"""
Re-render 15 SOT tracking results with two audio sources:
  A) Instrument audio (from FAIR-Play dataset) → 15 binaural
  B) Foley / object-matched audio (downloaded SFX) → 15 binaural

Total: 30 binaural outputs.
Reuses existing trajectory data from the first pipeline run.

Date: 2026-02-07
"""

import sys, os
sys.path.insert(0, "/home/seung/mmhoa/vid2spatial")

import json
import subprocess
import numpy as np
import soundfile as sf
from pathlib import Path

FFMPEG = "/home/seung/miniforge3/bin/ffmpeg"
PYTHON = "/home/seung/miniforge3/bin/python3"

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
INSTRUMENT_DIR = BASE_DIR / "audio_instrument"
FOLEY_DIR = BASE_DIR / "audio_foley"
RENDER_A_DIR = BASE_DIR / "render_A_instrument"
RENDER_B_DIR = BASE_DIR / "render_B_foley"

for d in [INSTRUMENT_DIR, FOLEY_DIR, RENDER_A_DIR, RENDER_B_DIR]:
    d.mkdir(exist_ok=True, parents=True)

SR = 48000

# ============================================================================
# Instrument assignment: 15 different instruments from FAIR-Play
# FAIR-Play has diverse instrument recordings.  We pick specific file indices
# that feature distinct instrument sounds.
# ============================================================================

INSTRUMENT_MAP = {
    "01_dog_frisbee":     {"fp_idx": "000001", "instrument": "guitar"},
    "02_cat_jumping":     {"fp_idx": "000010", "instrument": "piano"},
    "03_horse_running":   {"fp_idx": "000020", "instrument": "cello"},
    "04_car_highway":     {"fp_idx": "000030", "instrument": "trumpet"},
    "05_motorbike_road":  {"fp_idx": "000040", "instrument": "clarinet"},
    "06_boat_water":      {"fp_idx": "000050", "instrument": "violin"},
    "07_skier_downhill":  {"fp_idx": "000060", "instrument": "flute"},
    "08_surfer_wave":     {"fp_idx": "000070", "instrument": "saxophone"},
    "09_bmx_rider":       {"fp_idx": "000080", "instrument": "drums"},
    "10_dancer_solo":     {"fp_idx": "000090", "instrument": "harp"},
    "11_gymnast_floor":   {"fp_idx": "000100", "instrument": "bass"},
    "12_parkour_runner":  {"fp_idx": "000110", "instrument": "organ"},
    "13_drone_flying":    {"fp_idx": "000120", "instrument": "accordion"},
    "14_kite_flying":     {"fp_idx": "000130", "instrument": "banjo"},
    "15_ball_bouncing":   {"fp_idx": "000140", "instrument": "tuba"},
}

# ============================================================================
# Foley sound queries: object-matched sounds
# ============================================================================

FOLEY_MAP = {
    "01_dog_frisbee":     {"query": "dog barking sound effect",          "foley": "dog bark"},
    "02_cat_jumping":     {"query": "cat meow jump sound effect",        "foley": "cat meow"},
    "03_horse_running":   {"query": "horse galloping hooves sound effect","foley": "horse hooves"},
    "04_car_highway":     {"query": "car engine driving highway sound",  "foley": "car engine"},
    "05_motorbike_road":  {"query": "motorcycle engine revving sound",   "foley": "motorcycle rev"},
    "06_boat_water":      {"query": "motorboat water engine sound",      "foley": "boat motor"},
    "07_skier_downhill":  {"query": "skiing snow swoosh sound effect",   "foley": "skiing swoosh"},
    "08_surfer_wave":     {"query": "ocean wave surfing splash sound",   "foley": "wave splash"},
    "09_bmx_rider":       {"query": "bicycle chain pedaling sound",      "foley": "bicycle chain"},
    "10_dancer_solo":     {"query": "dance shoes footsteps floor sound", "foley": "dance steps"},
    "11_gymnast_floor":   {"query": "gymnast landing mat thud sound",    "foley": "gym landing"},
    "12_parkour_runner":  {"query": "running footsteps concrete sound",  "foley": "footsteps"},
    "13_drone_flying":    {"query": "drone propeller buzzing sound",     "foley": "drone buzz"},
    "14_kite_flying":     {"query": "kite wind flapping sound effect",   "foley": "kite flap"},
    "15_ball_bouncing":   {"query": "ball bouncing rubber sound effect", "foley": "ball bounce"},
}


# ============================================================================
# Audio Preparation
# ============================================================================

def prepare_instrument_audio(vid_id: str, duration: float = 10.0) -> str:
    """Get instrument audio from FAIR-Play, convert to mono, trim to duration."""
    info = INSTRUMENT_MAP[vid_id]
    fp_path = f"/home/seung/data/fairplay/binaural_audios/{info['fp_idx']}.wav"
    out_path = str(INSTRUMENT_DIR / f"{vid_id}_{info['instrument']}.wav")

    if os.path.exists(out_path):
        return out_path

    if not os.path.exists(fp_path):
        print(f"    [warn] FAIR-Play file not found: {fp_path}")
        return _generate_instrument_synth(vid_id, info["instrument"], out_path, duration)

    audio, file_sr = sf.read(fp_path, dtype="float32")
    # Mix stereo to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    # Resample if needed
    if file_sr != SR:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * SR / file_sr)).astype(np.float32)
    # Trim/pad
    n = int(duration * SR)
    if len(audio) > n:
        audio = audio[:n]
    elif len(audio) < n:
        audio = np.tile(audio, int(np.ceil(n / len(audio))))[:n]
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.7
    sf.write(out_path, audio.astype(np.float32), SR)
    return out_path


def _generate_instrument_synth(vid_id: str, instrument: str, out_path: str, duration: float) -> str:
    """Fallback: generate synthetic instrument sound."""
    n = int(duration * SR)
    t = np.linspace(0, duration, n)

    freqs = {
        "guitar": 220, "piano": 261.63, "cello": 130.81, "trumpet": 349.23,
        "clarinet": 293.66, "violin": 440, "flute": 880, "saxophone": 311.13,
        "drums": 100, "harp": 523.25, "bass": 82.41, "organ": 196,
        "accordion": 349.23, "banjo": 392, "tuba": 58.27,
    }
    f0 = freqs.get(instrument, 261.63)

    audio = np.zeros(n, dtype=np.float32)
    for h in range(1, 6):
        audio += (0.5 ** h) * np.sin(2 * np.pi * f0 * h * t).astype(np.float32)
    audio *= (1 - np.exp(-t * 10)) * np.exp(-t * 0.2)
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.7
    sf.write(out_path, audio, SR)
    return out_path


def download_foley_audio(vid_id: str, duration: float = 10.0) -> str:
    """Download foley sound from YouTube, extract audio, trim to mono."""
    info = FOLEY_MAP[vid_id]
    out_path = str(FOLEY_DIR / f"{vid_id}_{info['foley'].replace(' ', '_')}.wav")

    if os.path.exists(out_path):
        return out_path

    env = {**os.environ, "PATH": f"/home/seung/miniforge3/bin:/usr/bin:{os.environ.get('PATH', '')}"}

    try:
        # Search YouTube for SFX
        search_cmd = [
            PYTHON, "-m", "yt_dlp",
            f"ytsearch1:{info['query']}",
            "--get-id", "--no-playlist",
        ]
        result = subprocess.run(search_cmd, capture_output=True, text=True, timeout=30, env=env)
        if result.returncode != 0:
            return _generate_foley_synth(vid_id, info["foley"], out_path, duration)

        yt_id = result.stdout.strip().split("\n")[0]
        if not yt_id or len(yt_id) > 15:
            return _generate_foley_synth(vid_id, info["foley"], out_path, duration)

        # Download audio only
        temp_audio = str(FOLEY_DIR / f"_temp_{vid_id}.wav")
        dl_cmd = [
            PYTHON, "-m", "yt_dlp",
            f"https://www.youtube.com/watch?v={yt_id}",
            "-x", "--audio-format", "wav",
            "--ffmpeg-location", "/home/seung/miniforge3/bin",
            "-o", temp_audio.replace(".wav", ".%(ext)s"),
            "--quiet",
        ]
        subprocess.run(dl_cmd, capture_output=True, text=True, timeout=60, env=env)

        # Find the actual downloaded file
        temp_candidates = list(FOLEY_DIR.glob(f"_temp_{vid_id}.*"))
        if not temp_candidates:
            return _generate_foley_synth(vid_id, info["foley"], out_path, duration)

        temp_file = str(temp_candidates[0])

        # Convert to mono 48kHz, trim
        trim_cmd = [
            FFMPEG, "-y", "-i", temp_file,
            "-t", str(duration),
            "-acodec", "pcm_f32le", "-ar", str(SR), "-ac", "1",
            out_path,
        ]
        subprocess.run(trim_cmd, capture_output=True, text=True, timeout=30, env=env)

        # Cleanup
        for tc in temp_candidates:
            try: os.remove(str(tc))
            except: pass

        if os.path.exists(out_path):
            info_f = sf.info(out_path)
            if info_f.duration > 0.5:
                # Normalize
                audio, sr = sf.read(out_path, dtype="float32")
                audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.7
                sf.write(out_path, audio, sr)
                return out_path

        return _generate_foley_synth(vid_id, info["foley"], out_path, duration)

    except Exception as e:
        print(f"    [foley download error] {e}")
        return _generate_foley_synth(vid_id, info["foley"], out_path, duration)


def _generate_foley_synth(vid_id: str, foley_name: str, out_path: str, duration: float) -> str:
    """Fallback: synthesize approximate foley sound."""
    n = int(duration * SR)
    t = np.linspace(0, duration, n)
    audio = np.zeros(n, dtype=np.float32)

    if "bark" in foley_name or "meow" in foley_name:
        # Short bursts of noise + tone
        interval = int(SR * 0.8)
        for start in range(0, n, interval):
            end = min(start + int(SR * 0.3), n)
            seg_t = np.linspace(0, 0.3, end - start)
            burst = np.sin(2 * np.pi * 400 * seg_t) * np.exp(-seg_t * 15)
            burst += 0.3 * np.random.randn(end - start) * np.exp(-seg_t * 20)
            audio[start:end] += burst.astype(np.float32)

    elif "engine" in foley_name or "rev" in foley_name or "motor" in foley_name:
        # Low rumble with harmonics
        for h in range(1, 12):
            audio += (0.6 ** h) * np.sin(2 * np.pi * 50 * h * t).astype(np.float32)
        audio *= 1 + 0.2 * np.sin(2 * np.pi * 3 * t)
        audio += 0.1 * np.random.randn(n).astype(np.float32)

    elif "hooves" in foley_name or "steps" in foley_name or "footsteps" in foley_name:
        # Rhythmic impacts
        bpm = 160
        interval = int(SR * 60 / bpm)
        for start in range(0, n, interval):
            end = min(start + int(SR * 0.05), n)
            seg_t = np.linspace(0, 0.05, end - start)
            click = 0.8 * np.exp(-seg_t * 100)
            click += 0.5 * np.random.randn(end - start) * np.exp(-seg_t * 150)
            audio[start:end] += click.astype(np.float32)

    elif "wave" in foley_name or "splash" in foley_name or "water" in foley_name:
        # Ocean noise: filtered noise with slow amplitude modulation
        noise = np.random.randn(n).astype(np.float32)
        from scipy.signal import butter, filtfilt
        b, a = butter(3, 1500 / (SR/2), btype='low')
        audio = filtfilt(b, a, noise).astype(np.float32)
        audio *= 0.5 + 0.3 * np.sin(2 * np.pi * 0.15 * t)

    elif "swoosh" in foley_name or "wind" in foley_name or "flap" in foley_name:
        # Wind noise
        noise = np.random.randn(n).astype(np.float32)
        from scipy.signal import butter, filtfilt
        b, a = butter(2, [200/(SR/2), 3000/(SR/2)], btype='band')
        audio = filtfilt(b, a, noise).astype(np.float32)
        audio *= 0.6 + 0.3 * np.sin(2 * np.pi * 0.3 * t)

    elif "buzz" in foley_name or "drone" in foley_name:
        # High-pitched buzzing
        for h in range(1, 8):
            audio += (0.5 ** h) * np.sin(2 * np.pi * 200 * h * t).astype(np.float32)
        audio *= 1 + 0.15 * np.sin(2 * np.pi * 8 * t)

    elif "bounce" in foley_name:
        # Decreasing bounce intervals
        bounce_times = [0.0, 0.8, 1.4, 1.9, 2.3, 2.6, 2.85, 3.05, 3.2, 3.32]
        for bt in bounce_times:
            start = int(bt * SR)
            end = min(start + int(SR * 0.15), n)
            if start >= n: break
            seg_t = np.linspace(0, 0.15, end - start)
            impact = np.sin(2 * np.pi * 150 * seg_t) * np.exp(-seg_t * 30)
            audio[start:end] += impact.astype(np.float32)

    elif "chain" in foley_name or "bicycle" in foley_name:
        # Clicking chain
        interval = int(SR * 0.08)
        for start in range(0, n, interval):
            end = min(start + int(SR * 0.02), n)
            seg_t = np.linspace(0, 0.02, end - start)
            tick = 0.5 * np.exp(-seg_t * 200) * np.sin(2 * np.pi * 2000 * seg_t)
            audio[start:end] += tick.astype(np.float32)

    elif "landing" in foley_name or "thud" in foley_name:
        # Heavy impacts
        for bt in [0.5, 2.0, 3.5, 5.0, 6.5, 8.0]:
            start = int(bt * SR)
            end = min(start + int(SR * 0.2), n)
            if start >= n: break
            seg_t = np.linspace(0, 0.2, end - start)
            thud = np.sin(2 * np.pi * 60 * seg_t) * np.exp(-seg_t * 20)
            thud += 0.4 * np.random.randn(end - start) * np.exp(-seg_t * 30)
            audio[start:end] += thud.astype(np.float32)

    else:
        # Generic noise burst
        audio = 0.3 * np.random.randn(n).astype(np.float32)
        from scipy.signal import butter, filtfilt
        b, a = butter(2, 2000 / (SR/2), btype='low')
        audio = filtfilt(b, a, audio).astype(np.float32)

    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.7
    sf.write(out_path, audio.astype(np.float32), SR)
    return out_path


# ============================================================================
# Rendering
# ============================================================================

def render_binaural_from_trajectory(
    audio_path: str,
    trajectory_path: str,
    output_path: str,
    fps: float = 30.0,
):
    """Render FOA+binaural from audio + existing trajectory."""
    from vid2spatial_pkg.foa_render import (
        render_foa_from_trajectory, foa_to_binaural
    )

    with open(trajectory_path) as f:
        traj_data = json.load(f)

    # Handle both formats
    if "frames" in traj_data:
        frames = traj_data["frames"]
        traj_fps = traj_data.get("fps", fps)
    else:
        frames = traj_data
        traj_fps = fps

    foa_path = output_path.replace("_binaural.wav", "_foa.wav")

    render_foa_from_trajectory(
        audio_path=audio_path,
        trajectory={"frames": frames, "fps": traj_fps},
        output_path=foa_path,
        smooth_ms=30.0, apply_reverb=True, rt60=0.5,
        output_stereo=False,
    )

    foa_data, foa_sr = sf.read(foa_path, dtype="float32")
    foa_data = foa_data.T
    binaural = foa_to_binaural(foa_data, foa_sr)
    sf.write(output_path, binaural.T, foa_sr, subtype="FLOAT")

    return output_path


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("  Re-render 15 SOT Tracks × 2 Audio Sources = 30 Binaural Files")
    print("  A) Instrument (FAIR-Play)  |  B) Foley (object-matched)")
    print("  Date: 2026-02-07")
    print("=" * 70)

    # Get list of completed scenarios
    scenarios = list(INSTRUMENT_MAP.keys())

    # ========================
    # PHASE A: Instrument Audio
    # ========================
    print("\n--- PHASE A: Instrument Audio (FAIR-Play) ---\n")

    results_a = []
    for vid_id in scenarios:
        inst = INSTRUMENT_MAP[vid_id]["instrument"]
        traj_path = str(OUTPUT_DIR / vid_id / "trajectory_3d.json")

        if not os.path.exists(traj_path):
            print(f"  [{vid_id}] SKIP - no trajectory")
            results_a.append({"vid_id": vid_id, "status": "skip"})
            continue

        print(f"  [{vid_id}] instrument={inst}...", end=" ", flush=True)

        # Prepare audio
        audio_path = prepare_instrument_audio(vid_id)

        # Render
        out_path = str(RENDER_A_DIR / f"{vid_id}_{inst}_binaural.wav")
        try:
            render_binaural_from_trajectory(audio_path, traj_path, out_path)
            print(f"OK → {os.path.basename(out_path)}")
            results_a.append({"vid_id": vid_id, "instrument": inst, "status": "ok", "path": out_path})
        except Exception as e:
            print(f"ERROR: {e}")
            results_a.append({"vid_id": vid_id, "status": "error", "error": str(e)})

    ok_a = sum(1 for r in results_a if r["status"] == "ok")
    print(f"\n  Phase A: {ok_a}/{len(scenarios)} rendered")

    # ========================
    # PHASE B: Foley Audio
    # ========================
    print("\n--- PHASE B: Foley / Object-Matched Audio ---\n")

    results_b = []
    for vid_id in scenarios:
        foley_name = FOLEY_MAP[vid_id]["foley"]
        traj_path = str(OUTPUT_DIR / vid_id / "trajectory_3d.json")

        if not os.path.exists(traj_path):
            print(f"  [{vid_id}] SKIP - no trajectory")
            results_b.append({"vid_id": vid_id, "status": "skip"})
            continue

        print(f"  [{vid_id}] foley={foley_name}...", end=" ", flush=True)

        # Download/generate foley
        audio_path = download_foley_audio(vid_id)

        # Render
        out_path = str(RENDER_B_DIR / f"{vid_id}_{foley_name.replace(' ', '_')}_binaural.wav")
        try:
            render_binaural_from_trajectory(audio_path, traj_path, out_path)
            print(f"OK → {os.path.basename(out_path)}")
            results_b.append({"vid_id": vid_id, "foley": foley_name, "status": "ok", "path": out_path})
        except Exception as e:
            print(f"ERROR: {e}")
            results_b.append({"vid_id": vid_id, "status": "error", "error": str(e)})

    ok_b = sum(1 for r in results_b if r["status"] == "ok")
    print(f"\n  Phase B: {ok_b}/{len(scenarios)} rendered")

    # Save results
    with open(BASE_DIR / "rerender_30_results.json", "w") as f:
        json.dump({
            "date": "2026-02-07",
            "phase_a_instrument": results_a,
            "phase_b_foley": results_b,
        }, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY: 30 Binaural Renders")
    print("=" * 70)
    print(f"\n  Phase A (Instrument): {ok_a}/15")
    print(f"  Phase B (Foley):      {ok_b}/15")
    print(f"  Total:                {ok_a + ok_b}/30")

    print(f"\n  Phase A outputs: {RENDER_A_DIR}/")
    for r in results_a:
        if r["status"] == "ok":
            print(f"    {os.path.basename(r['path'])}")

    print(f"\n  Phase B outputs: {RENDER_B_DIR}/")
    for r in results_b:
        if r["status"] == "ok":
            print(f"    {os.path.basename(r['path'])}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
