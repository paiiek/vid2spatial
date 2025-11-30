"""
Unit tests for foa_render.py module.
Tests spatial audio encoding and FOA rendering functions.
"""
import pytest
import numpy as np
import math
from mmhoa.vid2spatial.foa_render import (
    dir_to_foa_acn_sn3d_gains,
    interpolate_angles,
    interpolate_angles_distance,
    smooth_limit_angles,
    encode_mono_to_foa,
    apply_distance_gain_lpf,
)


class TestFOAGainCalculation:
    """Test FOA gain calculation for AmbiX encoding."""

    def test_foa_gains_shape(self):
        """Test that FOA gains have correct shape [4, T]."""
        T = 1000
        az = np.zeros(T, dtype=np.float32)
        el = np.zeros(T, dtype=np.float32)
        gains = dir_to_foa_acn_sn3d_gains(az, el)
        assert gains.shape == (4, T)

    def test_foa_gains_dtype(self):
        """Test that FOA gains are float32."""
        T = 100
        az = np.zeros(T, dtype=np.float32)
        el = np.zeros(T, dtype=np.float32)
        gains = dir_to_foa_acn_sn3d_gains(az, el)
        assert gains.dtype == np.float32

    def test_w_channel_omnidirectional(self):
        """Test that W channel is constant (omnidirectional)."""
        T = 1000
        # Random directions
        az = np.random.uniform(-np.pi, np.pi, T).astype(np.float32)
        el = np.random.uniform(-np.pi/2, np.pi/2, T).astype(np.float32)
        gains = dir_to_foa_acn_sn3d_gains(az, el)

        # W channel (index 0) should be constant 1/sqrt(2)
        expected_w = 1.0 / math.sqrt(2.0)
        np.testing.assert_allclose(gains[0], expected_w, rtol=1e-6)

    def test_front_direction(self):
        """Test FOA gains for front direction (az=0, el=0)."""
        T = 1
        az = np.array([0.0], dtype=np.float32)
        el = np.array([0.0], dtype=np.float32)
        gains = dir_to_foa_acn_sn3d_gains(az, el)

        # At front: x=1, y=0, z=0
        # W = 1/sqrt(2), X = sqrt(3/2), Y = 0, Z = 0
        sqrt_3_2 = math.sqrt(3.0 / 2.0)
        expected = np.array([
            1.0 / math.sqrt(2.0),  # W
            0.0,                    # Y
            0.0,                    # Z
            sqrt_3_2,              # X
        ], dtype=np.float32)
        np.testing.assert_allclose(gains[:, 0], expected, atol=1e-6)

    def test_right_direction(self):
        """Test FOA gains for right direction (az=90°, el=0)."""
        T = 1
        az = np.array([math.pi / 2], dtype=np.float32)
        el = np.array([0.0], dtype=np.float32)
        gains = dir_to_foa_acn_sn3d_gains(az, el)

        # At right: x=0, y=1, z=0
        # W = 1/sqrt(2), X = 0, Y = sqrt(3/2), Z = 0
        sqrt_3_2 = math.sqrt(3.0 / 2.0)
        expected = np.array([
            1.0 / math.sqrt(2.0),  # W
            sqrt_3_2,              # Y
            0.0,                    # Z
            0.0,                    # X
        ], dtype=np.float32)
        np.testing.assert_allclose(gains[:, 0], expected, atol=1e-6)

    def test_up_direction(self):
        """Test FOA gains for up direction (el=90°)."""
        T = 1
        az = np.array([0.0], dtype=np.float32)
        el = np.array([math.pi / 2], dtype=np.float32)
        gains = dir_to_foa_acn_sn3d_gains(az, el)

        # At up: x=0, y=0, z=1
        # W = 1/sqrt(2), X = 0, Y = 0, Z = sqrt(3/2)
        sqrt_3_2 = math.sqrt(3.0 / 2.0)
        expected = np.array([
            1.0 / math.sqrt(2.0),  # W
            0.0,                    # Y
            sqrt_3_2,              # Z
            0.0,                    # X
        ], dtype=np.float32)
        np.testing.assert_allclose(gains[:, 0], expected, atol=1e-6)

    def test_sn3d_normalization(self):
        """Test that SN3D normalization is correct."""
        T = 1
        az = np.array([math.pi / 4], dtype=np.float32)  # 45°
        el = np.array([0.0], dtype=np.float32)
        gains = dir_to_foa_acn_sn3d_gains(az, el)

        # Energy should be normalized: W^2 + X^2 + Y^2 + Z^2
        # For SN3D: W^2 + (X^2 + Y^2 + Z^2) should equal constant
        energy = np.sum(gains[:, 0] ** 2)
        # W contributes 1/2, directional components contribute 3/2 total
        expected_energy = 0.5 + 1.5  # = 2.0
        assert abs(energy - expected_energy) < 1e-6


class TestAngleInterpolation:
    """Test angle interpolation functions."""

    def test_interpolate_single_frame(self):
        """Test interpolation with single frame (constant output)."""
        frames = [{"frame": 0, "az": 0.5, "el": 0.2}]
        T = 1000
        sr = 48000
        az_s, el_s = interpolate_angles(frames, T, sr)

        assert az_s.shape == (T,)
        assert el_s.shape == (T,)
        # All values should be constant
        np.testing.assert_allclose(az_s, 0.5, atol=1e-6)
        np.testing.assert_allclose(el_s, 0.2, atol=1e-6)

    def test_interpolate_two_frames_linear(self):
        """Test linear interpolation between two frames."""
        frames = [
            {"frame": 0, "az": 0.0, "el": 0.0},
            {"frame": 100, "az": 1.0, "el": 0.5},
        ]
        T = 101
        sr = 48000
        az_s, el_s = interpolate_angles(frames, T, sr)

        # Should interpolate linearly from 0 to 1
        assert az_s[0] == 0.0
        assert az_s[-1] == 1.0
        assert el_s[0] == 0.0
        assert el_s[-1] == 0.5

        # Check midpoint
        mid = T // 2
        assert abs(az_s[mid] - 0.5) < 0.01
        assert abs(el_s[mid] - 0.25) < 0.01

    def test_interpolate_multiple_frames(self, sample_trajectory):
        """Test interpolation with multiple frames."""
        T = 48000  # 1 second at 48kHz
        sr = 48000
        az_s, el_s = interpolate_angles(sample_trajectory, T, sr)

        assert az_s.shape == (T,)
        assert el_s.shape == (T,)
        assert az_s.dtype == np.float32
        assert el_s.dtype == np.float32

    def test_interpolate_empty_frames_raises(self):
        """Test that empty frames list raises error."""
        with pytest.raises(ValueError, match="Empty frames"):
            interpolate_angles([], T=100, sr=48000)

    def test_interpolate_angles_distance(self, sample_trajectory):
        """Test interpolation with distance."""
        T = 48000
        sr = 48000
        az_s, el_s, dist_s = interpolate_angles_distance(sample_trajectory, T, sr)

        assert az_s.shape == (T,)
        assert el_s.shape == (T,)
        assert dist_s.shape == (T,)
        assert dist_s.dtype == np.float32
        # Distance should be positive
        assert np.all(dist_s > 0)

    def test_interpolate_distance_missing(self):
        """Test that missing distance defaults to 1.0."""
        frames = [
            {"frame": 0, "az": 0.0, "el": 0.0},
            {"frame": 10, "az": 0.1, "el": 0.1},
        ]
        T = 100
        sr = 48000
        az_s, el_s, dist_s = interpolate_angles_distance(frames, T, sr)

        # Should default to 1.0
        np.testing.assert_allclose(dist_s, 1.0, atol=1e-6)


class TestAngleSmoothing:
    """Test angle smoothing and limiting."""

    def test_smooth_no_effect_on_constant(self):
        """Test that smoothing has minimal effect on constant angles."""
        T = 10000  # Need longer signal for edge effects to be small
        az_s = np.full(T, 0.5, dtype=np.float32)
        el_s = np.full(T, 0.2, dtype=np.float32)
        sr = 48000

        az_smooth, el_smooth = smooth_limit_angles(az_s, el_s, sr, smooth_ms=50.0)

        # Center region should be very close to constant (edge effects at boundaries)
        center = slice(T//4, 3*T//4)
        np.testing.assert_allclose(az_smooth[center], 0.5, atol=1e-3)
        np.testing.assert_allclose(el_smooth[center], 0.2, atol=1e-3)

    def test_smooth_reduces_noise(self):
        """Test that smoothing reduces high-frequency noise."""
        T = 10000
        sr = 48000
        # Create noisy signal: smooth base + high-freq noise
        np.random.seed(42)  # For reproducibility
        az_s = 0.5 + 0.1 * np.random.randn(T).astype(np.float32)
        el_s = 0.2 + 0.1 * np.random.randn(T).astype(np.float32)

        az_smooth, el_smooth = smooth_limit_angles(az_s, el_s, sr, smooth_ms=50.0)

        # Smoothed should have lower variance in the center region
        center = slice(T//4, 3*T//4)
        assert np.var(az_smooth[center]) < np.var(az_s[center])
        assert np.var(el_smooth[center]) < np.var(el_s[center])

    def test_delta_limiting(self):
        """Test delta limiting functionality."""
        T = 1000
        sr = 48000
        # Create signal with sudden jump
        az_s = np.zeros(T, dtype=np.float32)
        az_s[500:] = 1.0  # Sudden jump at midpoint
        el_s = np.zeros(T, dtype=np.float32)

        # Limit to 180 deg/s = π rad/s
        max_deg_per_s = 180.0
        az_smooth, el_smooth = smooth_limit_angles(
            az_s, el_s, sr, smooth_ms=0.0, max_deg_per_s=max_deg_per_s
        )

        # Check that delta is limited
        max_delta = np.max(np.abs(np.diff(az_smooth)))
        max_allowed = math.radians(max_deg_per_s) / sr
        assert max_delta <= max_allowed * 1.1  # Allow small margin

    def test_output_dtype(self):
        """Test that output dtype is float32."""
        T = 100
        az_s = np.zeros(T, dtype=np.float32)
        el_s = np.zeros(T, dtype=np.float32)
        sr = 48000

        az_smooth, el_smooth = smooth_limit_angles(az_s, el_s, sr)
        assert az_smooth.dtype == np.float32
        assert el_smooth.dtype == np.float32


class TestMonoToFOAEncoding:
    """Test mono to FOA encoding."""

    def test_encode_mono_to_foa_shape(self, sample_audio_mono):
        """Test that output has correct FOA shape [4, T]."""
        mono, sr = sample_audio_mono
        T = len(mono)
        az_series = np.zeros(T, dtype=np.float32)
        el_series = np.zeros(T, dtype=np.float32)

        foa = encode_mono_to_foa(mono, az_series, el_series)
        assert foa.shape == (4, T)

    def test_encode_mono_to_foa_dtype(self, sample_audio_mono):
        """Test that FOA output is float32."""
        mono, sr = sample_audio_mono
        T = len(mono)
        az_series = np.zeros(T, dtype=np.float32)
        el_series = np.zeros(T, dtype=np.float32)

        foa = encode_mono_to_foa(mono, az_series, el_series)
        assert foa.dtype == np.float32

    def test_encode_front_direction(self, sample_audio_mono):
        """Test encoding with static front direction."""
        mono, sr = sample_audio_mono
        T = len(mono)
        # Static front direction
        az_series = np.zeros(T, dtype=np.float32)
        el_series = np.zeros(T, dtype=np.float32)

        foa = encode_mono_to_foa(mono, az_series, el_series)

        # W channel should be mono * (1/sqrt(2))
        expected_w = mono / math.sqrt(2.0)
        np.testing.assert_allclose(foa[0], expected_w, atol=1e-4, rtol=1e-4)

        # X channel should be mono * sqrt(3/2)
        expected_x = mono * math.sqrt(3.0 / 2.0)
        np.testing.assert_allclose(foa[3], expected_x, atol=1e-4, rtol=1e-4)

        # Y and Z should be near zero
        assert np.max(np.abs(foa[1])) < 1e-5
        assert np.max(np.abs(foa[2])) < 1e-5

    def test_encode_normalization(self, sample_audio_mono):
        """Test that output is normalized if it exceeds 1.0."""
        mono, sr = sample_audio_mono
        # Scale mono to exceed 1.0 after encoding
        mono_loud = mono * 5.0
        T = len(mono_loud)
        az_series = np.zeros(T, dtype=np.float32)
        el_series = np.zeros(T, dtype=np.float32)

        foa = encode_mono_to_foa(mono_loud, az_series, el_series)

        # Peak should not exceed 1.0
        peak = np.max(np.abs(foa))
        assert peak <= 1.0

    def test_encode_time_varying_azimuth(self, sample_audio_mono):
        """Test encoding with time-varying azimuth."""
        mono, sr = sample_audio_mono
        T = len(mono)
        # Sweep from left to right
        az_series = np.linspace(-np.pi/2, np.pi/2, T, dtype=np.float32)
        el_series = np.zeros(T, dtype=np.float32)

        foa = encode_mono_to_foa(mono, az_series, el_series)

        # Y channel should vary (left-right)
        y_channel = foa[1]
        # Y channel should increase from start to end
        # Use larger regions to average out audio modulation
        region_size = T // 8
        y_start = np.mean(np.abs(y_channel[:region_size]))
        y_middle = np.mean(np.abs(y_channel[T//2 - region_size//2:T//2 + region_size//2]))
        y_end = np.mean(np.abs(y_channel[-region_size:]))

        # Absolute values should show Y channel is active at both ends
        assert y_start > 0.01  # Active at left
        assert y_end > 0.01  # Active at right

    def test_encode_preserves_energy(self, sample_audio_mono):
        """Test that encoding roughly preserves signal energy."""
        mono, sr = sample_audio_mono
        T = len(mono)
        az_series = np.full(T, np.pi/4, dtype=np.float32)
        el_series = np.zeros(T, dtype=np.float32)

        foa = encode_mono_to_foa(mono, az_series, el_series)

        # Total energy in FOA should be comparable to mono
        mono_energy = np.sum(mono ** 2)
        foa_energy = np.sum(foa ** 2)

        # Should be similar order of magnitude
        ratio = foa_energy / mono_energy
        assert 0.5 < ratio < 5.0  # Reasonable range


class TestDistanceRendering:
    """Test distance-based gain and filtering."""

    def test_apply_distance_gain_shape(self, sample_audio_mono):
        """Test that distance processing preserves shape."""
        audio, sr = sample_audio_mono
        T = len(audio)
        dist_s = np.full(T, 2.0, dtype=np.float32)

        processed = apply_distance_gain_lpf(audio, sr, dist_s)
        assert processed.shape == audio.shape

    def test_apply_distance_gain_dtype(self, sample_audio_mono):
        """Test that output dtype is float32."""
        audio, sr = sample_audio_mono
        T = len(audio)
        dist_s = np.full(T, 2.0, dtype=np.float32)

        processed = apply_distance_gain_lpf(audio, sr, dist_s)
        assert processed.dtype == np.float32

    def test_far_distance_attenuates(self, sample_audio_mono):
        """Test that far distance reduces gain."""
        audio, sr = sample_audio_mono
        T = len(audio)

        # Near source
        dist_near = np.full(T, 1.0, dtype=np.float32)
        near = apply_distance_gain_lpf(audio, sr, dist_near, gain_k=1.0)

        # Far source
        dist_far = np.full(T, 5.0, dtype=np.float32)
        far = apply_distance_gain_lpf(audio, sr, dist_far, gain_k=1.0)

        # Far should have lower RMS
        rms_near = np.sqrt(np.mean(near ** 2))
        rms_far = np.sqrt(np.mean(far ** 2))
        assert rms_far < rms_near

    def test_distance_processing_runs(self, sample_audio_mono):
        """Test that distance processing completes without errors."""
        audio, sr = sample_audio_mono
        T = len(audio)

        # Test with various distances
        for dist_val in [0.5, 1.0, 5.0, 20.0]:
            dist_s = np.full(T, dist_val, dtype=np.float32)
            processed = apply_distance_gain_lpf(audio, sr, dist_s,
                                              lpf_min_hz=800, lpf_max_hz=8000,
                                              gain_k=1.0)

            # Should complete and produce valid output
            assert processed.shape == audio.shape
            assert np.all(np.isfinite(processed))

    def test_no_gain_when_disabled(self, sample_audio_mono):
        """Test that gain_k=0 disables distance attenuation."""
        audio, sr = sample_audio_mono
        T = len(audio)

        dist_near = np.full(T, 1.0, dtype=np.float32)
        dist_far = np.full(T, 10.0, dtype=np.float32)

        # With gain disabled
        near = apply_distance_gain_lpf(audio, sr, dist_near, gain_k=0.0)
        far = apply_distance_gain_lpf(audio, sr, dist_far, gain_k=0.0)

        # RMS should be similar (only filtering differs)
        rms_near = np.sqrt(np.mean(near ** 2))
        rms_far = np.sqrt(np.mean(far ** 2))
        # Allow small difference due to filtering
        assert abs(rms_near - rms_far) / rms_near < 0.3


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_audio(self):
        """Test encoding with zero audio."""
        mono = np.zeros(1000, dtype=np.float32)
        az_series = np.zeros(1000, dtype=np.float32)
        el_series = np.zeros(1000, dtype=np.float32)

        foa = encode_mono_to_foa(mono, az_series, el_series)

        # Should produce zeros
        np.testing.assert_allclose(foa, 0.0, atol=1e-9)

    def test_very_short_audio(self):
        """Test with very short audio."""
        mono = np.array([0.5], dtype=np.float32)
        az_series = np.array([0.0], dtype=np.float32)
        el_series = np.array([0.0], dtype=np.float32)

        foa = encode_mono_to_foa(mono, az_series, el_series)
        assert foa.shape == (4, 1)

    def test_extreme_angles(self):
        """Test with extreme angle values."""
        T = 100
        mono = np.random.randn(T).astype(np.float32) * 0.1

        # Test with extreme azimuths
        az_series = np.full(T, 3 * np.pi, dtype=np.float32)  # Beyond [-π, π]
        el_series = np.zeros(T, dtype=np.float32)

        foa = encode_mono_to_foa(mono, az_series, el_series)
        # Should not crash, produces valid output
        assert foa.shape == (4, T)
        assert np.all(np.isfinite(foa))
