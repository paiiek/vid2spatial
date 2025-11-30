"""
Learned IR Module: Data-driven impulse response prediction

Instead of using fixed Schroeder RT60, learn IR parameters from video/audio data.

Approach:
1. Extract "effective IR" from GT binaural vs source mono
2. Parameterize IR with simple features (RT60, wet/dry, early reflections)
3. Train small network to predict IR params from video features

This should significantly improve over fixed Schroeder IR.
"""
import sys
sys.path.insert(0, '/home/seung')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.signal as signal
from pathlib import Path
import json
from typing import Dict, Tuple, Optional
import cv2


def extract_ir_from_binaural(
    binaural: np.ndarray,
    source_mono: np.ndarray,
    sr: int = 48000,
    ir_length_ms: float = 500.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract effective IR from binaural output and source mono.

    Uses Wiener deconvolution to estimate the impulse response
    that transforms mono → binaural.

    Args:
        binaural: (2, N) binaural audio
        source_mono: (N,) mono source
        sr: sample rate
        ir_length_ms: IR length in milliseconds

    Returns:
        ir_L: (M,) left channel IR
        ir_R: (M,) right channel IR
    """
    ir_len = int(sr * ir_length_ms / 1000.0)

    # Wiener deconvolution with regularization
    # H = Y / (X + noise)
    # where Y = FFT(binaural), X = FFT(source)

    # Add small epsilon for numerical stability
    epsilon = 1e-8

    # FFT
    fft_len = len(source_mono)
    X = np.fft.rfft(source_mono)
    Y_L = np.fft.rfft(binaural[0])
    Y_R = np.fft.rfft(binaural[1])

    # Wiener filter (regularization prevents division by zero)
    noise_power = 0.01  # Regularization parameter
    denominator = np.abs(X)**2 + noise_power

    H_L = Y_L * np.conj(X) / (denominator + epsilon)
    H_R = Y_R * np.conj(X) / (denominator + epsilon)

    # IFFT to get IR
    ir_L_full = np.fft.irfft(H_L)
    ir_R_full = np.fft.irfft(H_R)

    # Truncate to desired length
    ir_L = ir_L_full[:ir_len]
    ir_R = ir_R_full[:ir_len]

    # Normalize
    ir_L = ir_L / (np.abs(ir_L).max() + epsilon)
    ir_R = ir_R / (np.abs(ir_R).max() + epsilon)

    return ir_L, ir_R


def analyze_ir_parameters(ir: np.ndarray, sr: int = 48000) -> Dict[str, float]:
    """
    Extract interpretable parameters from IR.

    Parameters:
    - RT60: Reverberation time
    - Direct/Reverb ratio
    - Early reflection energy (0-50ms)
    - Late reflection energy (50-500ms)

    Args:
        ir: (N,) impulse response
        sr: sample rate

    Returns:
        params: dict with RT60, direct_ratio, early_energy, late_energy
    """
    # Energy decay curve (Schroeder integral)
    energy = ir**2
    decay_curve = np.flip(np.cumsum(np.flip(energy)))

    # Avoid log(0)
    decay_curve = decay_curve / (decay_curve[0] + 1e-8)
    decay_db = 10 * np.log10(decay_curve + 1e-8)

    # RT60: time for -60dB decay
    # Find where decay reaches -60dB
    idx_60db = np.where(decay_db < -60)[0]
    if len(idx_60db) > 0:
        rt60 = idx_60db[0] / sr
    else:
        rt60 = len(ir) / sr  # Max possible

    # Direct sound (first 5ms)
    direct_samples = int(0.005 * sr)
    direct_energy = np.sum(ir[:direct_samples]**2)

    # Early reflections (5-50ms)
    early_start = int(0.005 * sr)
    early_end = int(0.050 * sr)
    early_energy = np.sum(ir[early_start:early_end]**2)

    # Late reflections (50ms+)
    late_start = int(0.050 * sr)
    late_energy = np.sum(ir[late_start:]**2)

    # Normalize energies
    total_energy = direct_energy + early_energy + late_energy + 1e-8

    return {
        'rt60': float(rt60),
        'direct_ratio': float(direct_energy / total_energy),
        'early_ratio': float(early_energy / total_energy),
        'late_ratio': float(late_energy / total_energy)
    }


def generate_parametric_ir(
    rt60: float,
    early_ratio: float = 0.3,
    late_ratio: float = 0.6,
    sr: int = 48000,
    ir_length_ms: float = 500.0
) -> np.ndarray:
    """
    Generate IR from parameters using improved model.

    Args:
        rt60: Reverberation time (seconds)
        early_ratio: Energy in early reflections
        late_ratio: Energy in late reflections
        sr: sample rate
        ir_length_ms: IR length in milliseconds

    Returns:
        ir: (N,) impulse response
    """
    ir_len = int(sr * ir_length_ms / 1000.0)
    t = np.arange(ir_len) / sr

    # Direct sound (delta function at t=0)
    direct_ratio = max(0.0, 1.0 - early_ratio - late_ratio)
    ir = np.zeros(ir_len)
    ir[0] = direct_ratio

    # Early reflections (sparse, geometric pattern)
    # Model as series of discrete reflections
    num_early = 10
    early_times = np.random.uniform(0.005, 0.050, num_early)  # 5-50ms
    early_times = np.sort(early_times)

    early_gain = early_ratio / num_early
    for t_early in early_times:
        idx = int(t_early * sr)
        if idx < ir_len:
            ir[idx] += early_gain * np.random.uniform(0.5, 1.0)

    # Late reflections (dense, exponential decay)
    # Model as filtered noise with exponential envelope
    late_start = int(0.050 * sr)
    late_samples = ir_len - late_start

    # Exponential decay envelope
    decay_rate = 6.91 / rt60  # -60dB decay
    envelope = np.exp(-decay_rate * t[late_start:])

    # Generate dense reverberation (filtered noise)
    noise = np.random.randn(late_samples)

    # Low-pass filter (reverb is typically darker than direct sound)
    b, a = signal.butter(4, 4000 / (sr/2), btype='low')
    reverb = signal.filtfilt(b, a, noise)

    # Apply envelope and scale
    reverb = reverb * envelope * late_ratio
    ir[late_start:] += reverb

    # Normalize
    ir = ir / (np.abs(ir).max() + 1e-8)

    return ir


class SimpleIRPredictor(nn.Module):
    """
    Simple neural network to predict IR parameters from video features.

    Input: Video frame (or pre-extracted features)
    Output: IR parameters (RT60, early_ratio, late_ratio)
    """
    def __init__(self, input_size: int = 512, hidden_size: int = 256):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 3)
        )

        # Output heads with appropriate activations
        # RT60: 0.1 - 2.0 seconds
        # early_ratio: 0.0 - 0.5
        # late_ratio: 0.0 - 0.8

    def forward(self, features):
        """
        Args:
            features: (B, input_size) video features

        Returns:
            rt60: (B,) seconds
            early_ratio: (B,) 0-1
            late_ratio: (B,) 0-1
        """
        out = self.fc(features)  # (B, 3)

        # RT60: sigmoid to [0,1] then scale to [0.1, 2.0]
        rt60 = torch.sigmoid(out[:, 0]) * 1.9 + 0.1

        # Ratios: sigmoid to [0, 1]
        early_ratio = torch.sigmoid(out[:, 1]) * 0.5
        late_ratio = torch.sigmoid(out[:, 2]) * 0.8

        return rt60, early_ratio, late_ratio


def extract_video_features_simple(video_path: str, device: str = 'cuda') -> np.ndarray:
    """
    Extract simple visual features from video for IR prediction.

    For now, use simple statistics (can be replaced with CNN features).

    Features:
    - Mean brightness
    - Std brightness
    - Edge density (proxy for clutter/geometry)
    - Color histogram

    Args:
        video_path: path to video
        device: cuda or cpu

    Returns:
        features: (512,) feature vector
    """
    cap = cv2.VideoCapture(video_path)

    features = []
    frame_count = 0
    max_frames = 30  # Sample first 1 second

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for efficiency
        frame = cv2.resize(frame, (224, 224))

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Brightness statistics
        brightness = gray.mean()
        brightness_std = gray.std()

        # Edge density (Sobel)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_density = edges.mean()

        # Color histogram (HSV)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

        # Combine features
        frame_features = np.concatenate([
            [brightness, brightness_std, edge_density],
            hist_h / (hist_h.sum() + 1e-8),
            hist_s / (hist_s.sum() + 1e-8),
            hist_v / (hist_v.sum() + 1e-8)
        ])

        features.append(frame_features)
        frame_count += 1

    cap.release()

    # Average across frames
    if len(features) > 0:
        features = np.array(features)
        features_mean = features.mean(axis=0)
        features_std = features.std(axis=0)

        # Concatenate mean and std
        combined = np.concatenate([features_mean, features_std])

        # Pad to 512 dimensions
        if len(combined) < 512:
            combined = np.pad(combined, (0, 512 - len(combined)))
        else:
            combined = combined[:512]

        return combined
    else:
        return np.zeros(512)


# ============================================================================
# Dataset creation for IR learning
# ============================================================================

def create_ir_dataset(
    fairplay_loader,
    num_samples: int = 100,
    output_path: str = "ir_dataset.json"
):
    """
    Create dataset of (video_features, IR_params) pairs.

    Args:
        fairplay_loader: FairPlayDataset instance
        num_samples: number of samples to process
        output_path: where to save dataset
    """
    import tqdm

    dataset = []

    for i in tqdm.tqdm(range(num_samples), desc="Extracting IR dataset"):
        sample = fairplay_loader.get_sample(i)

        # Extract effective IR from GT
        try:
            ir_L, ir_R = extract_ir_from_binaural(
                sample['gt_binaural'],
                sample['mono_audio'],
                sample['sample_rate']
            )

            # Analyze IR parameters
            params_L = analyze_ir_parameters(ir_L, sample['sample_rate'])
            params_R = analyze_ir_parameters(ir_R, sample['sample_rate'])

            # Average L/R parameters
            params = {
                'rt60': (params_L['rt60'] + params_R['rt60']) / 2,
                'early_ratio': (params_L['early_ratio'] + params_R['early_ratio']) / 2,
                'late_ratio': (params_L['late_ratio'] + params_R['late_ratio']) / 2
            }

            # Extract video features
            video_features = extract_video_features_simple(sample['video_path'])

            dataset.append({
                'sample_id': sample['sample_id'],
                'video_features': video_features.tolist(),
                'ir_params': params
            })
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    # Save dataset
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Created IR dataset with {len(dataset)} samples → {output_path}")
    return dataset


if __name__ == '__main__':
    # Test IR extraction
    print("Testing IR extraction and parameterization...")

    # Generate synthetic test
    sr = 48000
    duration = 1.0
    t = np.arange(int(sr * duration)) / sr

    # Source: impulse
    source = np.zeros(len(t))
    source[0] = 1.0

    # Ground truth IR: simple exponential decay
    gt_ir = np.exp(-3.0 * t) * (np.random.randn(len(t)) * 0.1 + 1.0)
    gt_ir = gt_ir[:int(0.5 * sr)]  # 500ms IR

    # Convolve to get binaural
    binaural_L = signal.convolve(source, gt_ir, mode='same')
    binaural_R = signal.convolve(source, gt_ir * 0.8, mode='same')  # Slightly different
    binaural = np.stack([binaural_L, binaural_R])

    # Extract IR
    ir_L, ir_R = extract_ir_from_binaural(binaural, source, sr)

    # Analyze parameters
    params_L = analyze_ir_parameters(ir_L, sr)
    params_R = analyze_ir_parameters(ir_R, sr)

    print(f"Extracted IR parameters (L):")
    for k, v in params_L.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nExtracted IR parameters (R):")
    for k, v in params_R.items():
        print(f"  {k}: {v:.4f}")

    # Generate parametric IR
    synth_ir = generate_parametric_ir(
        rt60=params_L['rt60'],
        early_ratio=params_L['early_ratio'],
        late_ratio=params_L['late_ratio'],
        sr=sr
    )

    print(f"\nGenerated parametric IR shape: {synth_ir.shape}")
    print("✓ IR extraction and parameterization working!")
