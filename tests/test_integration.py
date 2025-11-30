"""
Integration tests for the full vid2spatial pipeline.
Tests end-to-end workflows combining multiple modules.
"""
import pytest
import numpy as np
import cv2
import tempfile
from pathlib import Path
from mmhoa.vid2spatial.vision import (
    CameraIntrinsics,
    pixel_to_ray,
    ray_to_angles,
)
from mmhoa.vid2spatial.foa_render import (
    interpolate_angles_distance,
    smooth_limit_angles,
    encode_mono_to_foa,
    apply_distance_gain_lpf,
)


@pytest.fixture
def synthetic_video(test_data_dir):
    """Create a synthetic video with a moving object."""
    video_path = test_data_dir / "test_video.mp4"
    width, height = 640, 480
    fps = 30
    duration = 2  # seconds
    num_frames = fps * duration

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    # Generate frames with moving rectangle
    for i in range(num_frames):
        # Create blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Gray background

        # Moving rectangle from left to right
        x = int(50 + (width - 150) * (i / num_frames))
        y = height // 2 - 25
        cv2.rectangle(frame, (x, y), (x + 50, y + 50), (0, 255, 0), -1)

        writer.write(frame)

    writer.release()
    return video_path


@pytest.fixture
def synthetic_trajectory():
    """Create a synthetic 3D trajectory matching the video."""
    frames = []
    num_frames = 60  # 2 seconds at 30fps
    K = CameraIntrinsics(width=640, height=480, fov_deg=60.0)

    for i in range(num_frames):
        # Object moves from left to right
        u = 50 + (640 - 150) * (i / num_frames) + 25  # Center of rectangle
        v = 480 // 2  # Vertical center

        # Convert to 3D
        ray = pixel_to_ray(u, v, K)
        az, el = ray_to_angles(ray)

        # Assume constant distance
        dist_m = 2.0

        # Compute 3D position
        x = dist_m * np.cos(az) * np.cos(el)
        y = dist_m * np.sin(az) * np.cos(el)
        z = dist_m * np.sin(el)

        frames.append({
            "frame": i,
            "az": float(az),
            "el": float(el),
            "dist_m": float(dist_m),
            "x": float(x),
            "y": float(y),
            "z": float(z),
        })

    return frames


class TestPixelToSpatialPipeline:
    """Test the complete pixel → 3D → spatial audio pipeline."""

    def test_pixel_to_spatial_conversion(self, sample_camera_intrinsics):
        """Test converting pixel coordinates to spatial audio parameters."""
        K = sample_camera_intrinsics

        # Test sequence: object moves from left to right
        test_positions = [
            (K.cx - 300, K.cy),  # Left
            (K.cx, K.cy),         # Center
            (K.cx + 300, K.cy),  # Right
        ]

        azimuths = []
        for u, v in test_positions:
            ray = pixel_to_ray(u, v, K)
            az, el = ray_to_angles(ray)
            azimuths.append(az)

        # Should go from negative → zero → positive
        assert azimuths[0] < 0  # Left
        assert abs(azimuths[1]) < 0.01  # Center
        assert azimuths[2] > 0  # Right

        # Should be monotonically increasing
        assert azimuths[0] < azimuths[1] < azimuths[2]

    def test_trajectory_to_foa_pipeline(self, synthetic_trajectory, sample_audio_mono):
        """Test converting trajectory to FOA audio."""
        mono, sr = sample_audio_mono
        T = len(mono)

        # Interpolate trajectory to audio samples
        az_s, el_s, dist_s = interpolate_angles_distance(synthetic_trajectory, T, sr)

        # Apply smoothing
        az_s, el_s = smooth_limit_angles(az_s, el_s, sr, smooth_ms=50.0)

        # Apply distance processing
        mono_dist = apply_distance_gain_lpf(mono, sr, dist_s)

        # Encode to FOA
        foa = encode_mono_to_foa(mono_dist, az_s, el_s)

        # Verify output
        assert foa.shape == (4, T)
        assert foa.dtype == np.float32
        assert np.all(np.isfinite(foa))
        assert np.max(np.abs(foa)) <= 1.0  # Normalized

    def test_moving_source_panning(self, synthetic_trajectory, sample_audio_mono):
        """Test that moving source produces correct panning in FOA."""
        mono, sr = sample_audio_mono
        T = len(mono)

        # Interpolate trajectory
        az_s, el_s, dist_s = interpolate_angles_distance(synthetic_trajectory, T, sr)

        # Encode to FOA
        foa = encode_mono_to_foa(mono, az_s, el_s)

        # Y channel (left-right) should vary as source moves
        y_channel = foa[1]

        # Find first and last quarter
        quarter = T // 4
        y_start = np.mean(y_channel[:quarter])
        y_end = np.mean(y_channel[-quarter:])

        # Object moves left to right, so Y should increase
        # (trajectory goes from negative to positive azimuth)
        assert y_end > y_start


class TestSpatialConsistency:
    """Test spatial audio encoding consistency."""

    def test_static_source_produces_static_foa(self):
        """Test that static source produces time-invariant FOA gains."""
        sr = 48000
        duration = 0.1
        T = int(sr * duration)

        # Constant audio
        mono = np.ones(T, dtype=np.float32) * 0.5

        # Static position (front)
        az_s = np.zeros(T, dtype=np.float32)
        el_s = np.zeros(T, dtype=np.float32)

        foa = encode_mono_to_foa(mono, az_s, el_s)

        # Each channel should be constant (proportional to mono)
        for ch in range(4):
            # Check variance is low
            variance = np.var(foa[ch])
            assert variance < 1e-10

    def test_energy_conservation_during_movement(self):
        """Test that total energy is conserved as source moves."""
        sr = 48000
        duration = 0.5
        T = int(sr * duration)

        # Constant amplitude mono
        mono = np.ones(T, dtype=np.float32) * 0.5

        # Moving azimuth
        az_s = np.linspace(-np.pi/2, np.pi/2, T, dtype=np.float32)
        el_s = np.zeros(T, dtype=np.float32)

        foa = encode_mono_to_foa(mono, az_s, el_s)

        # Total energy per sample should be relatively constant
        # Energy = W^2 + X^2 + Y^2 + Z^2
        energy_per_sample = np.sum(foa ** 2, axis=0)

        # Energy should not vary much
        energy_std = np.std(energy_per_sample)
        energy_mean = np.mean(energy_per_sample)
        cv = energy_std / energy_mean  # Coefficient of variation

        assert cv < 0.1  # Less than 10% variation


class TestDistanceEffects:
    """Test distance-based audio effects."""

    def test_distance_gain_attenuation(self):
        """Test that distance correctly attenuates audio."""
        sr = 48000
        duration = 0.1
        T = int(sr * duration)

        # Constant audio
        mono = np.ones(T, dtype=np.float32) * 0.5

        # Test different distances
        distances = [1.0, 2.0, 5.0, 10.0]
        rms_values = []

        for dist in distances:
            dist_s = np.full(T, dist, dtype=np.float32)
            processed = apply_distance_gain_lpf(mono, sr, dist_s, gain_k=1.0)
            rms = np.sqrt(np.mean(processed ** 2))
            rms_values.append(rms)

        # RMS should decrease with distance
        for i in range(len(rms_values) - 1):
            assert rms_values[i] > rms_values[i + 1]

    def test_distance_lowpass_effect(self):
        """Test that distance applies appropriate low-pass filtering."""
        sr = 48000
        duration = 0.2
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        T = len(t)

        # High frequency signal
        mono = np.sin(2 * np.pi * 5000 * t).astype(np.float32)

        # Near vs far
        dist_near = np.full(T, 1.0, dtype=np.float32)
        dist_far = np.full(T, 10.0, dtype=np.float32)

        near = apply_distance_gain_lpf(mono, sr, dist_near,
                                       lpf_min_hz=1000, lpf_max_hz=8000)
        far = apply_distance_gain_lpf(mono, sr, dist_far,
                                      lpf_min_hz=1000, lpf_max_hz=8000)

        # Far should have more attenuation of high frequencies
        # Normalize for amplitude differences
        near_norm = near / (np.std(near) + 1e-9)
        far_norm = far / (np.std(far) + 1e-9)

        # Compare high-frequency content
        fft_near = np.abs(np.fft.rfft(near_norm))
        fft_far = np.abs(np.fft.rfft(far_norm))

        # Sum energy above 3kHz
        freqs = np.fft.rfftfreq(T, 1/sr)
        high_freq_mask = freqs > 3000

        energy_near_hf = np.sum(fft_near[high_freq_mask] ** 2)
        energy_far_hf = np.sum(fft_far[high_freq_mask] ** 2)

        # Far should have less high-frequency energy
        assert energy_far_hf < energy_near_hf


class TestAngleInterpolation:
    """Test trajectory interpolation quality."""

    def test_interpolation_smoothness(self, synthetic_trajectory):
        """Test that interpolated angles are smooth."""
        sr = 48000
        duration = 2.0
        T = int(sr * duration)

        az_s, el_s, dist_s = interpolate_angles_distance(synthetic_trajectory, T, sr)

        # Compute differences
        az_diff = np.abs(np.diff(az_s))
        el_diff = np.abs(np.diff(el_s))

        # Should not have sudden jumps
        # Max change per sample at 48kHz should be small
        max_az_change = np.max(az_diff)
        max_el_change = np.max(el_diff)

        # At 48kHz, even fast movement should be < 0.01 rad/sample
        assert max_az_change < 0.01
        assert max_el_change < 0.01

    def test_smoothing_reduces_jitter(self, synthetic_trajectory):
        """Test that smoothing reduces high-frequency jitter."""
        sr = 48000
        duration = 2.0
        T = int(sr * duration)

        az_s, el_s, dist_s = interpolate_angles_distance(synthetic_trajectory, T, sr)

        # Add artificial jitter
        az_noisy = az_s + 0.01 * np.random.randn(T).astype(np.float32)
        el_noisy = el_s + 0.01 * np.random.randn(T).astype(np.float32)

        # Apply smoothing
        az_smooth, el_smooth = smooth_limit_angles(az_noisy, el_noisy, sr,
                                                    smooth_ms=50.0)

        # Smoothed should have lower variance of differences
        var_noisy = np.var(np.diff(az_noisy))
        var_smooth = np.var(np.diff(az_smooth))

        assert var_smooth < var_noisy


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios."""

    def test_static_front_source(self):
        """Test encoding a static source in front."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        T = len(t)

        # Create audio signal
        mono = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Static front position
        frames = [{"frame": 0, "az": 0.0, "el": 0.0, "dist_m": 2.0}]
        az_s, el_s, dist_s = interpolate_angles_distance(frames, T, sr)

        # Process
        mono_dist = apply_distance_gain_lpf(mono, sr, dist_s)
        foa = encode_mono_to_foa(mono_dist, az_s, el_s)

        # Verify FOA properties
        assert foa.shape == (4, T)
        # W channel should be dominant for omnidirectional
        w_energy = np.sum(foa[0] ** 2)
        total_energy = np.sum(foa ** 2)
        assert w_energy > total_energy * 0.2  # At least 20% in W

    def test_left_to_right_pan(self):
        """Test source panning from left to right."""
        sr = 48000
        duration = 2.0
        T = int(sr * duration)
        t = np.linspace(0, duration, T, dtype=np.float32)

        # Create audio signal
        mono = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Create trajectory: left to right
        num_frames = 60
        frames = []
        for i in range(num_frames):
            az = np.radians(-90 + 180 * (i / num_frames))  # -90° to +90°
            frames.append({
                "frame": i,
                "az": float(az),
                "el": 0.0,
                "dist_m": 2.0,
            })

        # Process
        az_s, el_s, dist_s = interpolate_angles_distance(frames, T, sr)
        az_s, el_s = smooth_limit_angles(az_s, el_s, sr, smooth_ms=50.0)
        foa = encode_mono_to_foa(mono, az_s, el_s)

        # Y channel should transition from negative to positive
        y_channel = foa[1]
        assert np.mean(y_channel[:T//4]) < 0  # Left at start
        assert np.mean(y_channel[-T//4:]) > 0  # Right at end

    def test_approaching_source(self):
        """Test source approaching from far to near."""
        sr = 48000
        duration = 2.0
        T = int(sr * duration)
        t = np.linspace(0, duration, T, dtype=np.float32)

        # Create audio signal with both low and high frequencies
        mono = (0.2 * np.sin(2 * np.pi * 200 * t) +
                0.2 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)

        # Create trajectory: far to near
        num_frames = 60
        frames = []
        for i in range(num_frames):
            dist = 10.0 - 8.0 * (i / num_frames)  # 10m to 2m
            frames.append({
                "frame": i,
                "az": 0.0,
                "el": 0.0,
                "dist_m": float(dist),
            })

        # Process
        az_s, el_s, dist_s = interpolate_angles_distance(frames, T, sr)
        mono_dist = apply_distance_gain_lpf(mono, sr, dist_s, gain_k=1.0)
        foa = encode_mono_to_foa(mono_dist, az_s, el_s)

        # Audio should get louder over time
        rms_start = np.sqrt(np.mean(foa[:, :T//4] ** 2))
        rms_end = np.sqrt(np.mean(foa[:, -T//4:] ** 2))
        assert rms_end > rms_start


@pytest.mark.slow
class TestPerformance:
    """Test performance characteristics."""

    def test_large_audio_processing(self):
        """Test processing large audio files."""
        sr = 48000
        duration = 10.0  # 10 seconds
        T = int(sr * duration)

        mono = np.random.randn(T).astype(np.float32) * 0.1

        # Create simple trajectory
        frames = [
            {"frame": 0, "az": -0.5, "el": 0.0, "dist_m": 2.0},
            {"frame": 100, "az": 0.5, "el": 0.0, "dist_m": 3.0},
        ]

        # Process - should complete without errors
        az_s, el_s, dist_s = interpolate_angles_distance(frames, T, sr)
        mono_dist = apply_distance_gain_lpf(mono, sr, dist_s)
        foa = encode_mono_to_foa(mono_dist, az_s, el_s)

        assert foa.shape == (4, T)
        assert np.all(np.isfinite(foa))
