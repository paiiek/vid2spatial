"""
Improved Evaluation Metrics for Spatial Audio

Focus on perceptual and spectral metrics instead of problematic angular error.

Metrics:
1. Spectral Distance (Multi-resolution STFT)
2. Envelope Distance
3. ITD/ILD Similarity (for binaural)
4. Perceptual metrics (PESQ-like)
"""
import numpy as np
import librosa
from scipy import signal
from typing import Dict, Tuple


def multi_resolution_stft_distance(pred: np.ndarray, target: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Compute spectral distance at multiple resolutions.

    Args:
        pred: predicted signal
        target: target signal
        sr: sample rate

    Returns:
        dict with distances at different resolutions
    """
    # Match lengths
    min_len = min(len(pred), len(target))
    pred = pred[:min_len]
    target = target[:min_len]

    distances = {}

    # Multiple FFT sizes for different time-frequency resolutions
    fft_sizes = [512, 1024, 2048, 4096]

    for n_fft in fft_sizes:
        # Compute STFT
        pred_stft = np.abs(librosa.stft(pred, n_fft=n_fft))
        target_stft = np.abs(librosa.stft(target, n_fft=n_fft))

        # Frobenius norm distance
        distance = np.linalg.norm(pred_stft - target_stft, 'fro') / np.linalg.norm(target_stft, 'fro')

        distances[f'stft_{n_fft}'] = float(distance)

    # Average distance
    distances['mean'] = float(np.mean(list(distances.values())))

    return distances


def envelope_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute distance between amplitude envelopes.

    Args:
        pred: predicted signal
        target: target signal

    Returns:
        normalized envelope distance
    """
    # Match lengths
    min_len = min(len(pred), len(target))
    pred = pred[:min_len]
    target = target[:min_len]

    # Compute envelopes using Hilbert transform
    pred_env = np.abs(signal.hilbert(pred))
    target_env = np.abs(signal.hilbert(target))

    # Normalize
    pred_env = pred_env / (np.max(pred_env) + 1e-8)
    target_env = target_env / (np.max(target_env) + 1e-8)

    # L2 distance
    distance = np.sqrt(np.mean((pred_env - target_env)**2))

    return float(distance)


def itd_ild_similarity(pred_binaural: np.ndarray, target_binaural: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Compare Interaural Time Difference (ITD) and Interaural Level Difference (ILD).

    Args:
        pred_binaural: [2, T] predicted binaural
        target_binaural: [2, T] target binaural
        sr: sample rate

    Returns:
        dict with ITD and ILD similarities
    """
    if pred_binaural.ndim == 1 or target_binaural.ndim == 1:
        return {'itd_error': 0.0, 'ild_error': 0.0}

    # Match lengths
    min_len = min(pred_binaural.shape[1], target_binaural.shape[1])
    pred_binaural = pred_binaural[:, :min_len]
    target_binaural = target_binaural[:, :min_len]

    # Compute ILD (Interaural Level Difference)
    pred_L, pred_R = pred_binaural[0], pred_binaural[1]
    target_L, target_R = target_binaural[0], target_binaural[1]

    # RMS levels
    pred_ild = 20 * np.log10((np.sqrt(np.mean(pred_L**2)) + 1e-8) / (np.sqrt(np.mean(pred_R**2)) + 1e-8))
    target_ild = 20 * np.log10((np.sqrt(np.mean(target_L**2)) + 1e-8) / (np.sqrt(np.mean(target_R**2)) + 1e-8))

    ild_error = abs(pred_ild - target_ild)

    # Compute ITD (Interaural Time Difference) using cross-correlation
    correlation = signal.correlate(pred_L, pred_R, mode='same')
    pred_itd_samples = len(pred_L) // 2 - np.argmax(correlation)

    correlation = signal.correlate(target_L, target_R, mode='same')
    target_itd_samples = len(target_L) // 2 - np.argmax(correlation)

    itd_error = abs(pred_itd_samples - target_itd_samples) / sr * 1000  # ms

    return {
        'itd_error_ms': float(itd_error),
        'ild_error_db': float(ild_error),
    }


def log_spectral_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Log-spectral distance (dB).

    Lower is better.
    """
    # Compute power spectra
    pred_spec = np.abs(np.fft.rfft(pred))**2
    target_spec = np.abs(np.fft.rfft(target))**2

    # Log-spectral distance
    lsd = np.sqrt(np.mean((10 * np.log10(pred_spec + 1e-8) - 10 * np.log10(target_spec + 1e-8))**2))

    return float(lsd)


def evaluate_spatial_audio_v2(
    pred_foa: np.ndarray,
    gt_binaural: np.ndarray,
    sr: int = 48000,
) -> Dict[str, float]:
    """
    Improved evaluation focusing on perceptual and spectral metrics.

    Args:
        pred_foa: [4, T] predicted FOA
        gt_binaural: [2, T] ground truth binaural
        sr: sample rate

    Returns:
        dict with evaluation metrics
    """
    metrics = {}

    # Convert FOA to binaural for fair comparison (simple panning)
    # Use geometric decoding at ±30° (standard stereo positions)
    W, Y, Z, X = pred_foa

    # Convert SN3D to N3D
    Y_n3d = Y * np.sqrt(3.0)
    Z_n3d = Z * np.sqrt(3.0)
    X_n3d = X * np.sqrt(3.0)

    # Decode at ±30° azimuth, 0° elevation
    az_L = np.deg2rad(30.0)
    az_R = np.deg2rad(-30.0)

    # Y00 = 1, Y1-1 = sin(az), Y10 = 0 (el=0), Y11 = cos(az)
    L = W + Y_n3d * np.sin(az_L) + X_n3d * np.cos(az_L)
    R = W + Y_n3d * np.sin(az_R) + X_n3d * np.cos(az_R)

    pred_binaural = np.stack([L, R])  # [2, T]

    # Match lengths
    min_len = min(pred_binaural.shape[1], gt_binaural.shape[1])
    pred_binaural = pred_binaural[:, :min_len]
    gt_binaural = gt_binaural[:, :min_len]

    print(f"  Pred binaural shape: {pred_binaural.shape}")
    print(f"  GT binaural shape: {gt_binaural.shape}")

    # 1. Multi-resolution spectral distance (on left channel)
    stft_dist = multi_resolution_stft_distance(pred_binaural[0], gt_binaural[0], sr)
    metrics.update({f'spectral_{k}': v for k, v in stft_dist.items()})

    # 2. Envelope distance
    metrics['envelope_distance_L'] = envelope_distance(pred_binaural[0], gt_binaural[0])
    metrics['envelope_distance_R'] = envelope_distance(pred_binaural[1], gt_binaural[1])

    # 3. ITD/ILD similarity
    itd_ild = itd_ild_similarity(pred_binaural, gt_binaural, sr)
    metrics.update(itd_ild)

    # 4. Log-spectral distance
    metrics['lsd_L'] = log_spectral_distance(pred_binaural[0], gt_binaural[0])
    metrics['lsd_R'] = log_spectral_distance(pred_binaural[1], gt_binaural[1])

    # 5. SI-SDR (Scale-Invariant SDR)
    def si_sdr(pred, target):
        min_len = min(len(pred), len(target))
        pred, target = pred[:min_len], target[:min_len]
        pred = pred - np.mean(pred)
        target = target - np.mean(target)
        alpha = np.dot(pred, target) / (np.dot(target, target) + 1e-8)
        target_scaled = alpha * target
        noise = pred - target_scaled
        return 10 * np.log10((np.dot(target_scaled, target_scaled) + 1e-8) / (np.dot(noise, noise) + 1e-8))

    metrics['si_sdr_L'] = float(si_sdr(pred_binaural[0], gt_binaural[0]))
    metrics['si_sdr_R'] = float(si_sdr(pred_binaural[1], gt_binaural[1]))

    # 6. Correlation
    metrics['correlation_L'] = float(np.corrcoef(pred_binaural[0], gt_binaural[0])[0, 1])
    metrics['correlation_R'] = float(np.corrcoef(pred_binaural[1], gt_binaural[1])[0, 1])

    return metrics


if __name__ == "__main__":
    # Test improved metrics
    print("Testing Improved Evaluation Metrics")
    print("=" * 70)

    # Load FAIR-Play sample
    import soundfile as sf

    # Load our FOA
    foa, sr = sf.read("/home/seung/mmhoa/vid2spatial/fairplay_eval/000001_output.foa.wav")
    foa = foa.T  # [T, 4] -> [4, T]

    # Load GT binaural
    gt_binaural, _ = librosa.load("/home/seung/data/fairplay/binaural_audios/000001.wav", sr=sr, mono=False)

    print(f"\nPred FOA shape: {foa.shape}")
    print(f"GT binaural shape: {gt_binaural.shape}")

    # Compute metrics
    print("\nComputing improved metrics...")
    metrics = evaluate_spatial_audio_v2(foa, gt_binaural, sr)

    print("\nResults:")
    for key, val in sorted(metrics.items()):
        print(f"  {key}: {val:.4f}")

    print("\n" + "=" * 70)
    print("Test complete!")
