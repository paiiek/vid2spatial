"""
Pytest configuration and shared fixtures for vid2spatial tests.
"""
import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_audio_mono():
    """Generate a simple mono audio signal for testing."""
    sr = 48000
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    # 440 Hz sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sr


@pytest.fixture
def sample_trajectory():
    """Generate a sample 3D trajectory for testing."""
    frames = []
    num_frames = 30
    for i in range(num_frames):
        # Object moving from left (-30°) to right (+30°) at constant elevation
        az = np.radians(-30 + 60 * (i / num_frames))
        el = np.radians(0)
        dist_m = 2.0 + 0.5 * np.sin(2 * np.pi * i / num_frames)  # oscillating distance

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


@pytest.fixture
def sample_camera_intrinsics():
    """Standard 1920x1080 camera with 60° FOV."""
    from mmhoa.vid2spatial.vision import CameraIntrinsics
    return CameraIntrinsics(width=1920, height=1080, fov_deg=60.0)


@pytest.fixture
def test_data_dir(tmp_path):
    """Temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir
