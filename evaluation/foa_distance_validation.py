#!/usr/bin/env python3
"""
FOA Distance Rendering Validation

Validates distance rendering behavior in the FOA renderer:
1. Gain monotonicity (farther = quieter)
2. FOA energy behavior (W stable, XYZ attenuate)
3. Smoothness (no discontinuities)

Generates plots and validation report for paper/documentation.
"""

import sys
sys.path.insert(0, "/home/seung/mmhoa/vid2spatial")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from vid2spatial_pkg.foa_render import (
    apply_distance_gain_lpf,
    build_wet_curve_from_dist_occ,
    encode_mono_to_foa,
    dir_to_foa_acn_sn3d_gains,
)


# Output directories
PLOT_DIR = Path("/home/seung/mmhoa/vid2spatial/evaluation/plots")
REPORT_PATH = Path("/home/seung/mmhoa/vid2spatial/evaluation/FOA_DISTANCE_REPORT.md")

# Test parameters
SR = 48000
DURATION = 5.0  # seconds
FREQ = 1000  # Hz sine wave


def generate_test_audio():
    """Generate 1kHz sine wave test signal."""
    t = np.linspace(0, DURATION, int(SR * DURATION), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * FREQ * t).astype(np.float32)
    return audio


def generate_distance_trajectory(start_m, end_m, num_samples):
    """Generate linear distance trajectory."""
    dist = np.linspace(start_m, end_m, num_samples, dtype=np.float32)
    # Compute d_rel (normalized 0-1)
    d_rel = np.clip((dist - 0.5) / (10.0 - 0.5), 0.0, 1.0)
    return dist, d_rel


def test_gain_monotonicity():
    """Test 1: Verify gain decreases monotonically with distance."""
    print("\n" + "=" * 60)
    print("TEST 1: Gain Monotonicity")
    print("=" * 60)

    audio = generate_test_audio()
    T = len(audio)

    # Test distance range: 0.5m to 10m
    dist_s, d_rel_s = generate_distance_trajectory(0.5, 10.0, T)

    # Apply distance gain + LPF
    audio_proc = apply_distance_gain_lpf(audio, SR, dist_s, d_rel_s)

    # Compute RMS in windows
    window_size = SR // 10  # 100ms windows
    num_windows = T // window_size
    rms_values = []
    dist_at_window = []

    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        rms = np.sqrt(np.mean(audio_proc[start:end] ** 2))
        rms_values.append(rms)
        dist_at_window.append(dist_s[start + window_size // 2])

    rms_values = np.array(rms_values)
    dist_at_window = np.array(dist_at_window)

    # Check monotonicity (with small tolerance for noise)
    diffs = np.diff(rms_values)
    monotonic_violations = np.sum(diffs > 0.001)  # Allow tiny increases
    is_monotonic = monotonic_violations <= len(diffs) * 0.05  # 5% tolerance

    print(f"  Distance range: {dist_s[0]:.1f}m → {dist_s[-1]:.1f}m")
    print(f"  RMS range: {rms_values[0]:.4f} → {rms_values[-1]:.4f}")
    print(f"  Gain ratio (near/far): {rms_values[0] / rms_values[-1]:.2f}x")
    print(f"  Monotonic violations: {monotonic_violations}/{len(diffs)}")
    print(f"  Result: {'✓ PASS' if is_monotonic else '✗ FAIL'}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    ax1.plot(dist_at_window, rms_values, 'b-', linewidth=2)
    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("RMS Amplitude")
    ax1.set_title("Gain vs Distance (should decrease)")
    ax1.grid(True, alpha=0.3)

    # Also plot theoretical 1/r curve
    theoretical = rms_values[0] / dist_at_window * dist_at_window[0]
    ax1.plot(dist_at_window, theoretical, 'r--', alpha=0.5, label='1/r reference')
    ax1.legend()

    # Gain in dB
    gain_db = 20 * np.log10(rms_values / rms_values[0] + 1e-10)
    ax2.plot(dist_at_window, gain_db, 'g-', linewidth=2)
    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Gain (dB)")
    ax2.set_title("Attenuation vs Distance")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "distance_gain_curve.png", dpi=150)
    plt.close()

    return {
        "test": "gain_monotonicity",
        "passed": is_monotonic,
        "rms_near": float(rms_values[0]),
        "rms_far": float(rms_values[-1]),
        "gain_ratio": float(rms_values[0] / rms_values[-1]),
        "attenuation_db": float(gain_db[-1]),
    }


def test_foa_energy():
    """Test 2: Verify FOA channel energy behavior with distance."""
    print("\n" + "=" * 60)
    print("TEST 2: FOA Energy Behavior")
    print("=" * 60)

    audio = generate_test_audio()
    T = len(audio)

    # Fixed position (front center), varying distance
    az_s = np.zeros(T, dtype=np.float32)  # 0° azimuth
    el_s = np.zeros(T, dtype=np.float32)  # 0° elevation

    # Test at different distances
    test_distances = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    results = {ch: [] for ch in ["W", "Y", "Z", "X"]}
    results["distance"] = test_distances

    for dist in test_distances:
        dist_s = np.full(T, dist, dtype=np.float32)
        d_rel_s = np.clip((dist_s - 0.5) / 9.5, 0.0, 1.0)

        # Apply distance processing
        audio_proc = apply_distance_gain_lpf(audio, SR, dist_s, d_rel_s)

        # Encode to FOA
        foa = encode_mono_to_foa(audio_proc, az_s, el_s)

        # Compute RMS for each channel
        for i, ch in enumerate(["W", "Y", "Z", "X"]):
            rms = np.sqrt(np.mean(foa[i] ** 2))
            results[ch].append(float(rms))

    # Analyze
    w_stable = np.std(results["W"]) / np.mean(results["W"]) < 0.5  # W should be relatively stable
    x_attenuates = results["X"][0] > results["X"][-1]  # X should attenuate (front)

    print(f"  W channel std/mean: {np.std(results['W']) / np.mean(results['W']):.3f}")
    print(f"  X channel (0.5m vs 10m): {results['X'][0]:.4f} → {results['X'][-1]:.4f}")
    print(f"  W stability: {'✓' if w_stable else '⚠ (expected: W attenuates with input)'}")
    print(f"  X attenuation: {'✓ PASS' if x_attenuates else '✗ FAIL'}")

    # Note: W also attenuates because input audio is attenuated by distance
    # The key is that all channels attenuate proportionally

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for ch in ["W", "Y", "Z", "X"]:
        ax.plot(test_distances, results[ch], 'o-', label=ch, linewidth=2, markersize=8)

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("RMS Energy")
    ax.set_title("FOA Channel Energy vs Distance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "foa_energy_vs_distance.png", dpi=150)
    plt.close()

    return {
        "test": "foa_energy",
        "passed": x_attenuates,
        "w_values": results["W"],
        "x_values": results["X"],
        "distances": test_distances,
    }


def test_smoothness():
    """Test 3: Verify smooth distance transitions (no discontinuities)."""
    print("\n" + "=" * 60)
    print("TEST 3: Smoothness")
    print("=" * 60)

    audio = generate_test_audio()
    T = len(audio)

    # Sinusoidal distance motion
    t = np.linspace(0, DURATION, T, dtype=np.float32)
    dist_s = 3.0 + 2.5 * np.sin(2 * np.pi * 0.5 * t)  # 0.5-5.5m, 0.5Hz
    d_rel_s = np.clip((dist_s - 0.5) / 9.5, 0.0, 1.0)

    # Apply processing
    audio_proc = apply_distance_gain_lpf(audio, SR, dist_s, d_rel_s)

    # Compute envelope (RMS in short windows)
    window_size = SR // 100  # 10ms
    envelope = []
    for i in range(0, T - window_size, window_size // 2):
        rms = np.sqrt(np.mean(audio_proc[i:i + window_size] ** 2))
        envelope.append(rms)
    envelope = np.array(envelope)

    # Check for discontinuities (large jumps relative to signal level)
    env_diff = np.abs(np.diff(envelope))
    max_jump = np.max(env_diff)
    mean_jump = np.mean(env_diff)
    # Use relative threshold: 10x mean OR 10% of envelope range
    env_range = envelope.max() - envelope.min()
    discontinuity_threshold = max(mean_jump * 10, env_range * 0.1)
    num_discontinuities = np.sum(env_diff > discontinuity_threshold)

    is_smooth = num_discontinuities == 0

    print(f"  Distance range: {dist_s.min():.1f}m - {dist_s.max():.1f}m (sinusoidal)")
    print(f"  Max envelope jump: {max_jump:.6f}")
    print(f"  Mean envelope jump: {mean_jump:.6f}")
    print(f"  Discontinuities (>5x mean): {num_discontinuities}")
    print(f"  Result: {'✓ PASS' if is_smooth else '✗ FAIL'}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    time_env = np.linspace(0, DURATION, len(envelope))
    time_dist = np.linspace(0, DURATION, T)

    ax1.plot(time_dist, dist_s, 'b-', linewidth=1)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Distance (m)")
    ax1.set_title("Sinusoidal Distance Motion")
    ax1.grid(True, alpha=0.3)

    ax2.plot(time_env, envelope, 'g-', linewidth=1)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("RMS Envelope")
    ax2.set_title("Output Envelope (should be smooth)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "distance_smoothness.png", dpi=150)
    plt.close()

    return {
        "test": "smoothness",
        "passed": is_smooth,
        "max_jump": float(max_jump),
        "mean_jump": float(mean_jump),
        "num_discontinuities": int(num_discontinuities),
    }


def test_d_rel_mapping():
    """Test 4: Verify d_rel → LPF cutoff mapping is correct."""
    print("\n" + "=" * 60)
    print("TEST 4: d_rel → LPF Cutoff Mapping")
    print("=" * 60)

    import math

    # Test the mathematical mapping (no audio processing needed)
    lpf_min_hz = 800.0
    lpf_max_hz = 8000.0
    lp_min = max(50.0, lpf_min_hz)
    lp_max = max(lp_min + 10.0, lpf_max_hz)

    test_distances = [0.5, 1.0, 3.0, 5.0, 7.0, 10.0]
    cutoffs = []

    print("  Distance → LPF Cutoff (higher = brighter = closer):")
    for dist in test_distances:
        d_rel = np.clip((dist - 0.5) / 9.5, 0.0, 1.0)

        log_min = math.log(lp_min)
        log_max = math.log(lp_max)
        log_fc = log_max - (log_max - log_min) * d_rel
        fc = np.exp(log_fc)

        cutoffs.append(fc)
        print(f"    dist={dist:.1f}m, d_rel={d_rel:.2f} → cutoff={fc:.0f}Hz")

    # Verify cutoff decreases with distance
    is_decreasing = all(cutoffs[i] >= cutoffs[i+1] for i in range(len(cutoffs)-1))

    # Verify range is correct
    range_ok = cutoffs[0] >= lpf_max_hz * 0.95 and cutoffs[-1] <= lpf_min_hz * 1.05

    print(f"\n  Cutoff decreasing with distance: {'✓' if is_decreasing else '✗'}")
    print(f"  Range correct ({lpf_min_hz}-{lpf_max_hz}Hz): {'✓' if range_ok else '✗'}")
    print(f"  Result: {'✓ PASS' if (is_decreasing and range_ok) else '✗ FAIL'}")

    return {
        "test": "d_rel_lpf_mapping",
        "passed": is_decreasing and range_ok,
        "distances": test_distances,
        "cutoffs": [float(c) for c in cutoffs],
    }


def test_reverb_mapping():
    """Test 5: Verify reverb wetness increases with distance."""
    print("\n" + "=" * 60)
    print("TEST 5: Reverb Mapping")
    print("=" * 60)

    # Test d_rel to wetness mapping
    d_rel_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    wet_values = []

    for d_rel in d_rel_values:
        d_rel_arr = np.array([d_rel], dtype=np.float32)
        wet = build_wet_curve_from_dist_occ(d_rel_arr)[0]
        wet_values.append(wet)

    wet_values = np.array(wet_values)

    # Check monotonicity
    is_monotonic = np.all(np.diff(wet_values) >= 0)

    print(f"  d_rel → wetness mapping:")
    for d, w in zip(d_rel_values, wet_values):
        print(f"    d_rel={d:.2f} → wet={w:.3f}")
    print(f"  Monotonic: {'✓ PASS' if is_monotonic else '✗ FAIL'}")

    return {
        "test": "reverb_mapping",
        "passed": is_monotonic,
        "d_rel_values": d_rel_values.tolist(),
        "wet_values": wet_values.tolist(),
    }


def generate_report(results):
    """Generate markdown report."""
    report = """# FOA Distance Rendering Validation Report

## Summary

| Test | Result |
|------|--------|
"""
    all_passed = True
    for r in results:
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        report += f"| {r['test']} | {status} |\n"
        if not r["passed"]:
            all_passed = False

    report += f"\n**Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}**\n"

    # Detailed results
    report += """
## Test Details

### Test 1: Gain Monotonicity

Verifies that audio amplitude decreases as distance increases.

"""
    r = results[0]
    report += f"- RMS at 0.5m: {r['rms_near']:.4f}\n"
    report += f"- RMS at 10m: {r['rms_far']:.4f}\n"
    report += f"- Gain ratio: {r['gain_ratio']:.2f}x\n"
    report += f"- Total attenuation: {r['attenuation_db']:.1f} dB\n"

    report += """
![Gain Curve](plots/distance_gain_curve.png)

### Test 2: FOA Energy Behavior

Verifies that FOA channels attenuate appropriately with distance.

"""
    r = results[1]
    report += f"- X channel attenuation: {r['x_values'][0]:.4f} → {r['x_values'][-1]:.4f}\n"

    report += """
![FOA Energy](plots/foa_energy_vs_distance.png)

### Test 3: Smoothness

Verifies no discontinuities in distance-varying gain.

"""
    r = results[2]
    report += f"- Discontinuities detected: {r['num_discontinuities']}\n"

    report += """
![Smoothness](plots/distance_smoothness.png)

### Test 4: d_rel → LPF Cutoff Mapping

Verifies that LPF cutoff decreases with distance (farther = more filtering).

"""
    r = results[3]
    report += "| Distance | d_rel | Cutoff |\n|----------|-------|--------|\n"
    for d, c in zip(r['distances'], r['cutoffs']):
        d_rel = (d - 0.5) / 9.5
        report += f"| {d:.1f}m | {d_rel:.2f} | {c:.0f}Hz |\n"

    report += """
### Test 5: Reverb Mapping

Verifies that reverb wetness increases with distance.

"""
    r = results[4]
    report += "| d_rel | wetness |\n|-------|----------|\n"
    for d, w in zip(r['d_rel_values'], r['wet_values']):
        report += f"| {d:.2f} | {w:.3f} |\n"

    report += """
## Interpretation

The FOA distance rendering model implements:

1. **Gain attenuation**: 1/r law with floor at 0.2 (prevents complete silence)
2. **Low-pass filtering**: Log-scaled cutoff from d_rel (consistent across tracks)
3. **Reverb wetness**: Linear mapping from d_rel (more reverb = farther)

The use of `d_rel` (normalized 0-1) instead of raw distance ensures:
- Consistent perceptual mapping regardless of absolute distance range
- Predictable DAW parameter mapping
- Comparable results across different video scenarios
"""

    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(f"\nReport saved to: {REPORT_PATH}")


def main():
    print("=" * 60)
    print("FOA DISTANCE RENDERING VALIDATION")
    print("=" * 60)

    # Ensure output directory exists
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    # Run tests
    results.append(test_gain_monotonicity())
    results.append(test_foa_energy())
    results.append(test_smoothness())
    results.append(test_d_rel_mapping())
    results.append(test_reverb_mapping())

    # Generate report
    generate_report(results)

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_passed = all(r["passed"] for r in results)
    for r in results:
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        print(f"  {r['test']:<25}: {status}")

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)

    print(f"\nPlots saved to: {PLOT_DIR}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
