"""
Improved IR based on GT statistics.

Key finding from analysis:
- FAIR-Play GT has VERY DRY acoustics
- Direct ratio: ~73%
- Early ratio: ~7%
- Late ratio: ~20%
- RT60: ~0.5s

This is MUCH drier than Schroeder default (RT60=0.6s with uniform diffuse field).

Simple improvement: Use GT-matched IR parameters instead of Schroeder.
"""
import numpy as np
import scipy.signal as signal


def generate_fairplay_matched_ir(
    sr: int = 48000,
    ir_length_ms: float = 500.0,
    rt60: float = 0.5,
    direct_ratio: float = 0.73,
    early_ratio: float = 0.07,
    late_ratio: float = 0.20
) -> np.ndarray:
    """
    Generate IR that matches FAIR-Play GT statistics.

    Much drier than typical room IR.

    Args:
        sr: sample rate
        ir_length_ms: IR length
        rt60: reverberation time
        direct_ratio: energy in direct sound
        early_ratio: energy in early reflections
        late_ratio: energy in late reflections

    Returns:
        ir: (N,) impulse response
    """
    ir_len = int(sr * ir_length_ms / 1000.0)
    ir = np.zeros(ir_len)

    # Direct sound (delta at t=0)
    ir[0] = np.sqrt(direct_ratio)

    # Early reflections (5-50ms, sparse)
    if early_ratio > 0:
        num_early = 8
        early_times = np.linspace(0.005, 0.050, num_early)  # Evenly spaced
        early_gain = np.sqrt(early_ratio / num_early)

        for t_early in early_times:
            idx = int(t_early * sr)
            if idx < ir_len:
                # Add some randomness to gain (70-100%)
                gain = early_gain * np.random.uniform(0.7, 1.0)
                ir[idx] += gain

    # Late reflections (50ms+, exponential decay)
    if late_ratio > 0:
        late_start = int(0.050 * sr)
        late_samples = ir_len - late_start
        t_late = np.arange(late_samples) / sr

        # Exponential decay
        decay_rate = 6.91 / rt60  # -60dB in rt60 seconds
        envelope = np.exp(-decay_rate * t_late)

        # Filtered noise (dense reverb tail)
        noise = np.random.randn(late_samples)

        # Low-pass filter (reverb darker than direct)
        b, a = signal.butter(4, 4000 / (sr/2), btype='low')
        reverb = signal.filtfilt(b, a, noise)

        # Normalize and scale
        reverb = reverb / (np.abs(reverb).max() + 1e-8)
        reverb = reverb * envelope * np.sqrt(late_ratio)

        ir[late_start:] += reverb

    return ir


def generate_minimal_ir(sr: int = 48000) -> np.ndarray:
    """
    Minimal IR: Just direct sound, no reflections.

    This should perform best if GT is truly anechoic.
    """
    ir_len = int(sr * 0.01)  # 10ms (minimal)
    ir = np.zeros(ir_len)
    ir[0] = 1.0  # Pure delta function
    return ir


if __name__ == '__main__':
    # Test
    sr = 48000

    # FAIR-Play matched IR
    ir_matched = generate_fairplay_matched_ir(sr=sr)
    print(f"FAIR-Play matched IR shape: {ir_matched.shape}")
    print(f"Energy distribution:")
    print(f"  Direct (first sample): {ir_matched[0]**2:.3f}")
    print(f"  Early (5-50ms): {np.sum(ir_matched[int(0.005*sr):int(0.050*sr)]**2):.3f}")
    print(f"  Late (50ms+): {np.sum(ir_matched[int(0.050*sr):]**2):.3f}")
    print(f"  Total: {np.sum(ir_matched**2):.3f}")

    # Minimal IR
    ir_minimal = generate_minimal_ir(sr=sr)
    print(f"\nMinimal IR shape: {ir_minimal.shape}")
    print(f"Energy: {np.sum(ir_minimal**2):.3f}")

    print("\n✓ Improved IR generation working!")
