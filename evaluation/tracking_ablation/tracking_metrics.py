"""
Tracking Ablation Metrics Module.

Provides standardized metrics for evaluating trajectory extraction quality.
All metrics compare estimated trajectory against ground truth.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import signal
from scipy.stats import pearsonr


@dataclass
class TrajectoryMetrics:
    """Container for all trajectory evaluation metrics."""
    amplitude_ratio: float      # amp_est / amp_gt (1.0 = perfect)
    mae: float                  # Mean Absolute Error in pixels
    velocity_correlation: float # Correlation between velocity signals
    jerk: float                 # Mean jerk (3rd derivative magnitude)
    direction_changes: int      # Sign changes in angular velocity
    coverage: float             # Frames with valid detection / total

    def to_dict(self) -> Dict:
        return {
            "amplitude_ratio": self.amplitude_ratio,
            "mae": self.mae,
            "velocity_correlation": self.velocity_correlation,
            "jerk": self.jerk,
            "direction_changes": self.direction_changes,
            "coverage": self.coverage,
        }


def compute_amplitude(trajectory: np.ndarray) -> float:
    """Compute peak-to-peak amplitude of 1D trajectory."""
    if len(trajectory) < 2:
        return 0.0
    return float(np.max(trajectory) - np.min(trajectory))


def compute_amplitude_ratio(est: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute amplitude preservation ratio.

    Args:
        est: Estimated trajectory (N,) or (N, 2)
        gt: Ground truth trajectory (N,) or (N, 2)

    Returns:
        amp_est / amp_gt (1.0 = perfect preservation)
    """
    if est.ndim == 2:
        # Use primary motion axis (larger amplitude)
        amp_est_x = compute_amplitude(est[:, 0])
        amp_est_y = compute_amplitude(est[:, 1])
        amp_gt_x = compute_amplitude(gt[:, 0])
        amp_gt_y = compute_amplitude(gt[:, 1])

        # Use axis with larger GT amplitude
        if amp_gt_x > amp_gt_y:
            amp_est, amp_gt = amp_est_x, amp_gt_x
        else:
            amp_est, amp_gt = amp_est_y, amp_gt_y
    else:
        amp_est = compute_amplitude(est)
        amp_gt = compute_amplitude(gt)

    if amp_gt < 1e-6:
        return 1.0 if amp_est < 1e-6 else 0.0

    return float(np.clip(amp_est / amp_gt, 0.0, 2.0))


def compute_mae(est: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute Mean Absolute Error between trajectories.

    Args:
        est: Estimated trajectory (N,) or (N, 2)
        gt: Ground truth trajectory (N,) or (N, 2)

    Returns:
        Mean L2 distance in pixels
    """
    if est.shape != gt.shape:
        # Resample to same length
        min_len = min(len(est), len(gt))
        est = est[:min_len]
        gt = gt[:min_len]

    if est.ndim == 1:
        return float(np.mean(np.abs(est - gt)))
    else:
        diff = est - gt
        distances = np.sqrt(np.sum(diff ** 2, axis=1))
        return float(np.mean(distances))


def compute_velocity(trajectory: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """Compute velocity (first derivative) of trajectory."""
    if len(trajectory) < 2:
        return np.array([0.0])

    vel = np.diff(trajectory, axis=0) * fps
    return vel


def compute_velocity_correlation(est: np.ndarray, gt: np.ndarray, fps: float = 30.0) -> float:
    """
    Compute correlation between estimated and GT velocity signals.

    Args:
        est: Estimated trajectory (N,) or (N, 2)
        gt: Ground truth trajectory (N,) or (N, 2)
        fps: Frame rate

    Returns:
        Pearson correlation coefficient (-1 to 1)
    """
    vel_est = compute_velocity(est, fps)
    vel_gt = compute_velocity(gt, fps)

    # Align lengths
    min_len = min(len(vel_est), len(vel_gt))
    vel_est = vel_est[:min_len]
    vel_gt = vel_gt[:min_len]

    if vel_est.ndim == 2:
        # Use magnitude for 2D
        vel_est = np.sqrt(np.sum(vel_est ** 2, axis=1))
        vel_gt = np.sqrt(np.sum(vel_gt ** 2, axis=1))

    # Check for constant signals
    if np.std(vel_est) < 1e-6 or np.std(vel_gt) < 1e-6:
        return 0.0

    corr, _ = pearsonr(vel_est.flatten(), vel_gt.flatten())
    return float(corr) if not np.isnan(corr) else 0.0


def compute_jerk(trajectory: np.ndarray, fps: float = 30.0) -> float:
    """
    Compute mean jerk (3rd derivative magnitude).

    Lower jerk = smoother trajectory.

    Args:
        trajectory: Position trajectory (N,) or (N, 2)
        fps: Frame rate

    Returns:
        Mean jerk magnitude
    """
    if len(trajectory) < 4:
        return 0.0

    # 1st derivative: velocity
    vel = np.diff(trajectory, axis=0) * fps
    # 2nd derivative: acceleration
    acc = np.diff(vel, axis=0) * fps
    # 3rd derivative: jerk
    jerk = np.diff(acc, axis=0) * fps

    if jerk.ndim == 2:
        jerk_mag = np.sqrt(np.sum(jerk ** 2, axis=1))
    else:
        jerk_mag = np.abs(jerk)

    return float(np.mean(jerk_mag))


def compute_direction_changes(trajectory: np.ndarray) -> int:
    """
    Count direction changes in trajectory.

    For 2D: counts sign changes in angular velocity.
    For 1D: counts sign changes in velocity.

    Args:
        trajectory: Position trajectory (N,) or (N, 2)

    Returns:
        Number of direction changes
    """
    if len(trajectory) < 3:
        return 0

    vel = np.diff(trajectory, axis=0)

    if vel.ndim == 2:
        # Compute angle of velocity vector
        angles = np.arctan2(vel[:, 1], vel[:, 0])
        # Angular velocity
        ang_vel = np.diff(angles)
        # Wrap to [-pi, pi]
        ang_vel = np.arctan2(np.sin(ang_vel), np.cos(ang_vel))
        signal_to_check = ang_vel
    else:
        signal_to_check = vel

    # Count sign changes
    signs = np.sign(signal_to_check)
    sign_changes = np.sum(signs[1:] != signs[:-1])

    return int(sign_changes)


def compute_coverage(valid_mask: np.ndarray) -> float:
    """
    Compute tracking coverage ratio.

    Args:
        valid_mask: Boolean array where True = valid detection

    Returns:
        Ratio of valid frames (0 to 1)
    """
    if len(valid_mask) == 0:
        return 0.0
    return float(np.mean(valid_mask))


def evaluate_trajectory(
    est: np.ndarray,
    gt: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    fps: float = 30.0,
) -> TrajectoryMetrics:
    """
    Compute all trajectory metrics.

    Args:
        est: Estimated trajectory (N, 2) for xy or (N,) for 1D
        gt: Ground truth trajectory, same shape as est
        valid_mask: Optional boolean mask for valid detections
        fps: Frame rate

    Returns:
        TrajectoryMetrics dataclass with all metrics
    """
    # Ensure same length
    min_len = min(len(est), len(gt))
    est = est[:min_len]
    gt = gt[:min_len]

    if valid_mask is None:
        valid_mask = np.ones(min_len, dtype=bool)
    else:
        valid_mask = valid_mask[:min_len]

    return TrajectoryMetrics(
        amplitude_ratio=compute_amplitude_ratio(est, gt),
        mae=compute_mae(est, gt),
        velocity_correlation=compute_velocity_correlation(est, gt, fps),
        jerk=compute_jerk(est, fps),
        direction_changes=compute_direction_changes(est),
        coverage=compute_coverage(valid_mask),
    )


def normalize_jerk(jerk: float, amplitude_ratio: float) -> float:
    """
    Normalize jerk by amplitude ratio.

    This is important because a near-stationary trajectory (low amplitude)
    will have artificially low jerk. Normalizing reveals true smoothness.

    Args:
        jerk: Raw jerk value
        amplitude_ratio: Amplitude preservation ratio

    Returns:
        Normalized jerk (jerk / amplitude_ratio)
    """
    if amplitude_ratio < 0.01:
        return float('inf')  # Essentially stationary = infinite normalized jerk
    return jerk / amplitude_ratio


# =============================================================================
# Synthetic Ground Truth Generation
# =============================================================================

def generate_horizontal_sweep(
    n_frames: int = 150,
    img_width: int = 640,
    img_height: int = 480,
    margin: float = 0.1,
) -> np.ndarray:
    """
    Generate horizontal sweep trajectory (left to right).

    Args:
        n_frames: Number of frames
        img_width, img_height: Image dimensions
        margin: Margin from edges (fraction)

    Returns:
        (N, 2) trajectory array
    """
    x_min = img_width * margin
    x_max = img_width * (1 - margin)
    y = img_height / 2

    x = np.linspace(x_min, x_max, n_frames)
    trajectory = np.stack([x, np.full_like(x, y)], axis=1)

    return trajectory


def generate_diagonal_motion(
    n_frames: int = 150,
    img_width: int = 640,
    img_height: int = 480,
    margin: float = 0.1,
) -> np.ndarray:
    """
    Generate diagonal motion trajectory (top-left to bottom-right).
    """
    x_min = img_width * margin
    x_max = img_width * (1 - margin)
    y_min = img_height * margin
    y_max = img_height * (1 - margin)

    x = np.linspace(x_min, x_max, n_frames)
    y = np.linspace(y_min, y_max, n_frames)

    return np.stack([x, y], axis=1)


def generate_oscillating_motion(
    n_frames: int = 150,
    frequency_hz: float = 0.6,
    fps: float = 30.0,
    img_width: int = 640,
    img_height: int = 480,
    amplitude_frac: float = 0.35,
) -> np.ndarray:
    """
    Generate oscillating (sinusoidal) motion trajectory.

    This is the critical test case - SAM2 fails at >0.5Hz motion.

    Args:
        n_frames: Number of frames
        frequency_hz: Oscillation frequency
        fps: Frame rate
        img_width, img_height: Image dimensions
        amplitude_frac: Amplitude as fraction of image width

    Returns:
        (N, 2) trajectory array
    """
    t = np.arange(n_frames) / fps

    center_x = img_width / 2
    center_y = img_height / 2
    amplitude = img_width * amplitude_frac

    x = center_x + amplitude * np.sin(2 * np.pi * frequency_hz * t)
    y = np.full_like(x, center_y)

    return np.stack([x, y], axis=1)


def generate_all_synthetic_trajectories(
    n_frames: int = 150,
    fps: float = 30.0,
) -> Dict[str, np.ndarray]:
    """Generate all synthetic GT trajectories."""
    return {
        "horizontal_sweep": generate_horizontal_sweep(n_frames),
        "diagonal_motion": generate_diagonal_motion(n_frames),
        "oscillating_0.6Hz": generate_oscillating_motion(n_frames, frequency_hz=0.6, fps=fps),
    }


# =============================================================================
# Result Formatting
# =============================================================================

def format_metrics_table(
    results: Dict[str, Dict[str, TrajectoryMetrics]],
    metric_keys: List[str] = None,
) -> str:
    """
    Format metrics as markdown table.

    Args:
        results: {video_name: {method_name: TrajectoryMetrics}}
        metric_keys: Which metrics to include

    Returns:
        Markdown table string
    """
    if metric_keys is None:
        metric_keys = ["amplitude_ratio", "mae", "velocity_correlation", "jerk"]

    # Header
    header = "| Method | Video | " + " | ".join([
        "Amp%" if k == "amplitude_ratio" else
        "MAE(px)" if k == "mae" else
        "VelCorr" if k == "velocity_correlation" else
        "Jerk" if k == "jerk" else
        "DirChg" if k == "direction_changes" else
        "Coverage" if k == "coverage" else k
        for k in metric_keys
    ]) + " |"

    separator = "|" + "|".join(["------"] * (len(metric_keys) + 2)) + "|"

    rows = [header, separator]

    for video_name, methods in results.items():
        for method_name, metrics in methods.items():
            row_values = []
            for k in metric_keys:
                v = getattr(metrics, k)
                if k == "amplitude_ratio":
                    row_values.append(f"{v*100:.1f}%")
                elif k in ["mae", "jerk"]:
                    row_values.append(f"{v:.2f}")
                elif k == "velocity_correlation":
                    row_values.append(f"{v:.3f}")
                elif k == "direction_changes":
                    row_values.append(f"{int(v)}")
                elif k == "coverage":
                    row_values.append(f"{v*100:.1f}%")
                else:
                    row_values.append(f"{v:.3f}")

            row = f"| {method_name} | {video_name} | " + " | ".join(row_values) + " |"
            rows.append(row)

    return "\n".join(rows)


__all__ = [
    "TrajectoryMetrics",
    "evaluate_trajectory",
    "compute_amplitude_ratio",
    "compute_mae",
    "compute_velocity_correlation",
    "compute_jerk",
    "compute_direction_changes",
    "compute_coverage",
    "normalize_jerk",
    "generate_horizontal_sweep",
    "generate_diagonal_motion",
    "generate_oscillating_motion",
    "generate_all_synthetic_trajectories",
    "format_metrics_table",
]
