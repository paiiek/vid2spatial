"""
Utility functions shared across vid2spatial modules.

This module consolidates common functionality to eliminate code duplication:
- File I/O (JSONL, manifests)
- Depth predictor builders
- STFT feature extraction
- Common validation and preprocessing
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Tuple

import numpy as np


# ============================================================================
# File I/O Utilities
# ============================================================================

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Read JSONL file (one JSON object per line).

    Args:
        path: Path to JSONL file

    Returns:
        List of dictionaries, one per line

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If line is not valid JSON
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records = []
    with open(path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON at line {line_num}: {e.msg}",
                    e.doc,
                    e.pos
                )

    return records


def write_jsonl(records: List[Dict[str, Any]], path: str):
    """
    Write list of dicts to JSONL file.

    Args:
        records: List of dictionaries
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')


def read_manifest(path: str) -> List[Dict[str, Any]]:
    """
    Read manifest file (JSONL format).

    Alias for read_jsonl with additional validation.

    Args:
        path: Path to manifest file

    Returns:
        List of manifest entries
    """
    return read_jsonl(path)


# ============================================================================
# Depth Estimation Utilities
# ============================================================================

def build_depth_predictor_unified(
    backend: str = "auto",
    device: Optional[str] = None,
    model_size: str = "small"
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Unified depth predictor builder.

    This is a convenience wrapper around depth_anything_adapter.build_depth_predictor
    that can be imported from utils.

    Args:
        backend: 'auto', 'depth_anything_v2', or 'midas'
        device: 'cpu', 'cuda', or None (auto-detect)
        model_size: For Depth Anything V2: 'small', 'base', 'large'

    Returns:
        Depth prediction function: (H,W,3) BGR -> (H,W) float32 [0,1]
    """
    from .depth_anything_adapter import build_depth_predictor
    return build_depth_predictor(device=device, backend=backend, model_size=model_size)


# ============================================================================
# Audio Feature Extraction
# ============================================================================

def extract_stft_features(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: Optional[int] = None,
    window: str = 'hann'
) -> np.ndarray:
    """
    Extract STFT magnitude features from audio.

    Args:
        audio: Audio signal (T,) or multi-channel (C, T)
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length in samples
        n_mels: If provided, convert to mel scale
        window: Window function name

    Returns:
        STFT features:
        - If n_mels is None: (n_fft//2+1, n_frames) or (C, n_fft//2+1, n_frames)
        - If n_mels is set: (n_mels, n_frames) or (C, n_mels, n_frames)
    """
    import librosa

    # Handle multi-channel
    if audio.ndim == 1:
        # Single channel
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window)
        mag = np.abs(stft)

        if n_mels is not None:
            mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
            mag = mel_basis @ mag

        return mag.astype(np.float32)

    else:
        # Multi-channel
        C = audio.shape[0]
        mags = []
        for ch in range(C):
            stft = librosa.stft(audio[ch], n_fft=n_fft, hop_length=hop_length, window=window)
            mag = np.abs(stft)

            if n_mels is not None:
                mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
                mag = mel_basis @ mag

            mags.append(mag)

        return np.stack(mags, axis=0).astype(np.float32)


def foa_to_stft_features(
    foa: np.ndarray,
    sr: int,
    n_fft: int = 512,
    hop_length: int = 160
) -> np.ndarray:
    """
    Extract STFT features from FOA audio.

    Args:
        foa: FOA audio [4, T]
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length

    Returns:
        STFT magnitude features [4, n_fft//2+1, n_frames]
    """
    if foa.shape[0] != 4:
        raise ValueError(f"Expected 4-channel FOA, got {foa.shape[0]} channels")

    return extract_stft_features(foa, sr, n_fft=n_fft, hop_length=hop_length)


# ============================================================================
# Validation and Preprocessing
# ============================================================================

def validate_audio_shape(audio: np.ndarray, expected_channels: Optional[int] = None):
    """
    Validate audio array shape.

    Args:
        audio: Audio array
        expected_channels: Expected number of channels (None = any)

    Raises:
        ValueError: If shape is invalid
    """
    if audio.ndim not in (1, 2):
        raise ValueError(f"Audio must be 1D or 2D, got {audio.ndim}D")

    if audio.ndim == 2 and expected_channels is not None:
        if audio.shape[0] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} channels, got {audio.shape[0]}"
            )


def normalize_audio(audio: np.ndarray, peak: float = 1.0) -> np.ndarray:
    """
    Normalize audio to peak amplitude.

    Args:
        audio: Input audio
        peak: Target peak amplitude

    Returns:
        Normalized audio
    """
    current_peak = np.max(np.abs(audio))
    if current_peak > 1e-9:
        return audio * (peak / current_peak)
    return audio


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """
    Ensure audio is mono (convert stereo to mono if needed).

    Args:
        audio: Audio array (T,) or (2, T) or (T, 2)

    Returns:
        Mono audio (T,)
    """
    if audio.ndim == 1:
        return audio

    if audio.ndim == 2:
        # Check which dimension is channels
        if audio.shape[0] == 2:
            # (2, T) format
            return audio.mean(axis=0).astype(audio.dtype)
        elif audio.shape[1] == 2:
            # (T, 2) format
            return audio.mean(axis=1).astype(audio.dtype)
        else:
            # Multi-channel, just take mean
            return audio.mean(axis=0).astype(audio.dtype)

    raise ValueError(f"Cannot convert {audio.shape} to mono")


# ============================================================================
# Array Utilities
# ============================================================================

def interpolate_timeline(
    frames: List[Dict[str, Any]],
    key: str,
    T: int,
    default: float = 0.0
) -> np.ndarray:
    """
    Interpolate a timeline value from sparse frame data.

    Args:
        frames: List of frame dicts with 'frame' and key fields
        key: Key to interpolate (e.g., 'az', 'el', 'dist_m')
        T: Number of output samples
        default: Default value if key missing

    Returns:
        Interpolated values of length T
    """
    if not frames:
        return np.full(T, default, dtype=np.float32)

    frame_indices = np.array([f.get("frame", 0) for f in frames], dtype=np.float32)
    values = np.array([float(f.get(key, default)) for f in frames], dtype=np.float32)

    if len(frame_indices) == 1:
        # Single frame: constant value
        return np.full(T, values[0], dtype=np.float32)

    # Linear interpolation
    sample_positions = np.linspace(frame_indices[0], frame_indices[-1], T, dtype=np.float32)
    interpolated = np.interp(sample_positions, frame_indices, values)

    return interpolated.astype(np.float32)


def smooth_signal(
    signal: np.ndarray,
    window_size: int,
    mode: str = 'same'
) -> np.ndarray:
    """
    Smooth signal with moving average.

    Args:
        signal: Input signal
        window_size: Window size for moving average
        mode: Convolution mode ('same', 'valid', 'full')

    Returns:
        Smoothed signal
    """
    if window_size <= 1:
        return signal

    kernel = np.ones(window_size, dtype=np.float32) / window_size
    smoothed = np.convolve(signal, kernel, mode=mode)

    return smoothed.astype(signal.dtype)


# ============================================================================
# Geometry Utilities
# ============================================================================

def cartesian_to_spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert Cartesian to spherical coordinates.

    Args:
        x, y, z: Cartesian coordinates

    Returns:
        (azimuth, elevation, distance) in radians and meters
    """
    import math

    dist = math.sqrt(x*x + y*y + z*z)
    az = math.atan2(y, x)
    el = math.atan2(z, math.sqrt(x*x + y*y))

    return az, el, dist


def spherical_to_cartesian(az: float, el: float, dist: float) -> Tuple[float, float, float]:
    """
    Convert spherical to Cartesian coordinates.

    Args:
        az: Azimuth in radians
        el: Elevation in radians
        dist: Distance

    Returns:
        (x, y, z) Cartesian coordinates
    """
    import math

    x = dist * math.cos(az) * math.cos(el)
    y = dist * math.sin(az) * math.cos(el)
    z = dist * math.sin(el)

    return x, y, z


# ============================================================================
# Debug and Logging
# ============================================================================

def print_array_stats(name: str, array: np.ndarray):
    """
    Print statistics about numpy array (useful for debugging).

    Args:
        name: Array name for display
        array: Numpy array
    """
    print(f"{name}:")
    print(f"  shape: {array.shape}")
    print(f"  dtype: {array.dtype}")
    print(f"  min: {np.min(array):.6f}")
    print(f"  max: {np.max(array):.6f}")
    print(f"  mean: {np.mean(array):.6f}")
    print(f"  std: {np.std(array):.6f}")

    if np.any(np.isnan(array)):
        print(f"  WARNING: Contains NaN values!")
    if np.any(np.isinf(array)):
        print(f"  WARNING: Contains Inf values!")


__all__ = [
    # File I/O
    'read_jsonl',
    'write_jsonl',
    'read_manifest',
    # Depth
    'build_depth_predictor_unified',
    # Audio features
    'extract_stft_features',
    'foa_to_stft_features',
    # Validation
    'validate_audio_shape',
    'normalize_audio',
    'ensure_mono',
    # Array utilities
    'interpolate_timeline',
    'smooth_signal',
    # Geometry
    'cartesian_to_spherical',
    'spherical_to_cartesian',
    # Debug
    'print_array_stats',
]
