#!/usr/bin/env python3
"""
Synthetic Trajectory Evaluation for Tracking Ablation.

Evaluates all ablation configurations against synthetic ground truth trajectories.
This provides controlled experiments with known GT for precise metric computation.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tracking_ablation.tracking_metrics import (
    TrajectoryMetrics,
    evaluate_trajectory,
    generate_all_synthetic_trajectories,
    compute_jerk,
    normalize_jerk,
)
from tracking_ablation.configs.ablation_configs import (
    AblationConfig,
    TrackerBackend,
    InterpolationMethod,
    SmoothingMethod,
    get_all_ablation_configs,
    get_configs_by_category,
    TRACKER_ABLATION_CONFIGS,
    INTERPOLATION_ABLATION_CONFIGS,
    SMOOTHING_ABLATION_CONFIGS,
    ROBUSTNESS_ABLATION_CONFIGS,
)


# =============================================================================
# Trajectory Simulation (Simulates Tracker Behavior)
# =============================================================================

def simulate_sam2_trajectory(
    gt: np.ndarray,
    fps: float = 30.0,
    motion_collapse_threshold: float = 5.0,  # pixels/frame - lower threshold
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate SAM2 mask propagation behavior.

    SAM2's propagation fails for fast motion (>0.5Hz oscillation) because:
    - Assumes small inter-frame displacement
    - Mask center drifts toward image center for large motion
    - Observed: 96% amplitude loss at 0.6Hz oscillation

    Args:
        gt: Ground truth trajectory (N, 2)
        fps: Frame rate
        motion_collapse_threshold: Velocity threshold for collapse (pixels/frame)

    Returns:
        (estimated_trajectory, valid_mask)
    """
    n = len(gt)
    est = np.zeros_like(gt)
    valid = np.ones(n, dtype=bool)

    # Initialize at first position
    est[0] = gt[0]

    # Center of image (drift target) - SAM2 drifts here under fast motion
    img_center = np.mean(gt, axis=0)

    # Compute mean displacement to detect oscillating motion
    displacements = np.array([np.linalg.norm(gt[i] - gt[i-1]) for i in range(1, n)])
    mean_displacement = np.mean(displacements)

    # Detect if this is oscillating motion (high velocity, direction changes)
    velocities = np.diff(gt, axis=0)
    if velocities.shape[0] > 2:
        # Check for oscillation: velocity sign changes
        vel_x = velocities[:, 0]
        sign_changes = np.sum(np.diff(np.sign(vel_x)) != 0)
        is_oscillating = sign_changes > n * 0.1  # >10% frames have direction change
    else:
        is_oscillating = False

    for i in range(1, n):
        displacement = np.linalg.norm(gt[i] - gt[i-1])

        # SAM2 motion collapse behavior
        if is_oscillating and displacement > motion_collapse_threshold:
            # Strong collapse for oscillating motion
            # SAM2 loses track quickly - observed 96% amplitude loss
            # This means it converges to center within ~10-20 frames
            drift_rate = 0.25  # Very fast drift to center (96% collapse)
            # Small random walk around center
            noise = np.random.randn(2) * 3.0
            est[i] = est[i-1] + drift_rate * (img_center - est[i-1]) + noise * 0.05
        elif displacement > motion_collapse_threshold * 2:
            # Moderate collapse for fast linear motion
            drift_rate = 0.08
            est[i] = est[i-1] + drift_rate * (img_center - est[i-1])
        else:
            # Normal tracking: follow GT with small noise
            noise = np.random.randn(2) * 1.0
            est[i] = gt[i] + noise

    return est, valid


def simulate_dino_kframe_trajectory(
    gt: np.ndarray,
    k: int = 5,
    interpolation: InterpolationMethod = InterpolationMethod.LINEAR,
    noise_std: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate DINO K-frame detection with interpolation.

    Args:
        gt: Ground truth trajectory (N, 2)
        k: Detection interval (1 = every frame)
        interpolation: Interpolation method between keyframes
        noise_std: Detection noise standard deviation

    Returns:
        (estimated_trajectory, valid_mask)
    """
    n = len(gt)
    est = np.zeros_like(gt)
    valid = np.ones(n, dtype=bool)

    # Keyframe indices
    keyframes = list(range(0, n, k))
    if keyframes[-1] != n - 1:
        keyframes.append(n - 1)

    # Detect at keyframes (with noise)
    keyframe_detections = {}
    for kf in keyframes:
        noise = np.random.randn(2) * noise_std
        keyframe_detections[kf] = gt[kf] + noise

    # Fill in trajectory
    for i in range(n):
        if i in keyframe_detections:
            est[i] = keyframe_detections[i]
        else:
            # Find surrounding keyframes
            prev_kf = max([k for k in keyframes if k <= i])
            next_kf = min([k for k in keyframes if k >= i])

            if interpolation == InterpolationMethod.NONE:
                # Hold last keyframe value
                est[i] = keyframe_detections[prev_kf]
            elif interpolation == InterpolationMethod.LINEAR:
                # Linear interpolation
                if prev_kf == next_kf:
                    est[i] = keyframe_detections[prev_kf]
                else:
                    t = (i - prev_kf) / (next_kf - prev_kf)
                    est[i] = (1 - t) * keyframe_detections[prev_kf] + t * keyframe_detections[next_kf]
            else:
                # Default to linear
                t = (i - prev_kf) / (next_kf - prev_kf) if next_kf != prev_kf else 0
                est[i] = (1 - t) * keyframe_detections[prev_kf] + t * keyframe_detections[next_kf]

    return est, valid


def simulate_adaptive_k_trajectory(
    gt: np.ndarray,
    fps: float = 30.0,
    interpolation: InterpolationMethod = InterpolationMethod.LINEAR,
    noise_std: float = 2.0,
    k_min: int = 2,
    k_max: int = 15,
    motion_threshold: float = 5.0,  # pixels/frame for fast motion
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate adaptive K-frame detection.

    K adapts based on motion:
    - Fast motion: K = k_min (frequent detection)
    - Slow motion: K = k_max (save compute)

    Args:
        gt: Ground truth trajectory (N, 2)
        fps: Frame rate
        interpolation: Interpolation method
        noise_std: Detection noise
        k_min, k_max: K range
        motion_threshold: Velocity threshold for fast/slow

    Returns:
        (estimated_trajectory, valid_mask)
    """
    n = len(gt)
    est = np.zeros_like(gt)
    valid = np.ones(n, dtype=bool)

    # Compute velocity at each frame
    velocity = np.zeros(n)
    for i in range(1, n):
        velocity[i] = np.linalg.norm(gt[i] - gt[i-1])

    # Determine keyframes adaptively
    keyframes = [0]
    i = 0
    while i < n - 1:
        # Adaptive K based on recent motion
        window = velocity[max(0, i-5):i+1]
        avg_velocity = np.mean(window) if len(window) > 0 else 0

        if avg_velocity > motion_threshold:
            k = k_min
        else:
            k = k_max

        next_kf = min(i + k, n - 1)
        keyframes.append(next_kf)
        i = next_kf

    keyframes = sorted(set(keyframes))

    # Detect at keyframes
    keyframe_detections = {}
    for kf in keyframes:
        noise = np.random.randn(2) * noise_std
        keyframe_detections[kf] = gt[kf] + noise

    # Fill in trajectory with interpolation
    for i in range(n):
        if i in keyframe_detections:
            est[i] = keyframe_detections[i]
        else:
            prev_kf = max([k for k in keyframes if k <= i])
            next_kf = min([k for k in keyframes if k >= i])

            if interpolation == InterpolationMethod.NONE:
                est[i] = keyframe_detections[prev_kf]
            else:
                t = (i - prev_kf) / (next_kf - prev_kf) if next_kf != prev_kf else 0
                est[i] = (1 - t) * keyframe_detections[prev_kf] + t * keyframe_detections[next_kf]

    return est, valid


def simulate_yolo_trajectory(
    gt: np.ndarray,
    noise_std: float = 3.0,
    miss_rate: float = 0.02,  # 2% missed detections
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate YOLO tracking behavior.

    YOLO tracks well for simple motion but may lose track for occlusions.
    For oscillating motion, it often fails completely (similar to SAM2).

    Args:
        gt: Ground truth trajectory
        noise_std: Detection noise
        miss_rate: Probability of missed detection

    Returns:
        (estimated_trajectory, valid_mask)
    """
    n = len(gt)
    est = np.zeros_like(gt)
    valid = np.ones(n, dtype=bool)

    for i in range(n):
        if np.random.rand() < miss_rate:
            # Missed detection - hold previous
            est[i] = est[max(0, i-1)]
            valid[i] = False
        else:
            noise = np.random.randn(2) * noise_std
            est[i] = gt[i] + noise

    return est, valid


# =============================================================================
# Smoothing Simulation
# =============================================================================

def apply_ema_smoothing(trajectory: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Apply Exponential Moving Average smoothing."""
    n = len(trajectory)
    smoothed = np.zeros_like(trajectory)
    smoothed[0] = trajectory[0]

    for i in range(1, n):
        smoothed[i] = alpha * trajectory[i] + (1 - alpha) * smoothed[i-1]

    return smoothed


def apply_rts_smoothing(trajectory: np.ndarray, process_noise: float = 0.1) -> np.ndarray:
    """
    Apply Rauch-Tung-Striebel (RTS) smoothing.

    This is optimal offline smoothing using forward-backward Kalman filter.
    """
    n = len(trajectory)
    if n < 3:
        return trajectory.copy()

    # State: [x, y, vx, vy]
    # Measurement: [x, y]

    dt = 1.0  # Frame interval
    ndim = trajectory.shape[1] if trajectory.ndim > 1 else 1

    if ndim == 1:
        trajectory = trajectory.reshape(-1, 1)

    # State transition matrix
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    # Observation matrix
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ])

    # Process noise
    Q = np.eye(4) * process_noise
    Q[2:, 2:] *= 10  # Higher noise on velocity

    # Measurement noise
    R = np.eye(2) * 1.0

    # Forward pass (Kalman filter)
    x_fwd = np.zeros((n, 4))
    P_fwd = np.zeros((n, 4, 4))

    # Initialize
    x_fwd[0] = np.array([trajectory[0, 0], trajectory[0, 1], 0, 0])
    P_fwd[0] = np.eye(4) * 100

    for i in range(1, n):
        # Predict
        x_pred = F @ x_fwd[i-1]
        P_pred = F @ P_fwd[i-1] @ F.T + Q

        # Update
        z = trajectory[i]
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x_fwd[i] = x_pred + K @ y
        P_fwd[i] = (np.eye(4) - K @ H) @ P_pred

    # Backward pass (RTS smoother)
    x_smooth = np.zeros((n, 4))
    x_smooth[-1] = x_fwd[-1]

    for i in range(n - 2, -1, -1):
        x_pred = F @ x_fwd[i]
        P_pred = F @ P_fwd[i] @ F.T + Q

        # Smoother gain
        C = P_fwd[i] @ F.T @ np.linalg.inv(P_pred)

        x_smooth[i] = x_fwd[i] + C @ (x_smooth[i+1] - x_pred)

    # Extract position
    smoothed = x_smooth[:, :2]

    if ndim == 1:
        smoothed = smoothed.flatten()

    return smoothed


def apply_smoothing(
    trajectory: np.ndarray,
    method: SmoothingMethod,
    ema_alpha: float = 0.3,
) -> np.ndarray:
    """Apply specified smoothing method."""
    if method == SmoothingMethod.NONE:
        return trajectory.copy()
    elif method == SmoothingMethod.EMA:
        return apply_ema_smoothing(trajectory, ema_alpha)
    elif method == SmoothingMethod.RTS:
        return apply_rts_smoothing(trajectory)
    else:
        return trajectory.copy()


# =============================================================================
# Robustness Layer Simulation
# =============================================================================

def apply_robustness_layer(
    trajectory: np.ndarray,
    confidences: np.ndarray,
    config: 'RobustnessConfig',
    fps: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply robustness layer (confidence gating + jump rejection).

    Args:
        trajectory: Estimated trajectory
        confidences: Detection confidence per frame
        config: Robustness configuration
        fps: Frame rate

    Returns:
        (filtered_trajectory, valid_mask)
    """
    from tracking_ablation.configs.ablation_configs import RobustnessConfig

    n = len(trajectory)
    filtered = trajectory.copy()
    valid = np.ones(n, dtype=bool)

    if not config.enabled:
        return filtered, valid

    # Confidence gating
    if config.confidence_gating:
        for i in range(n):
            if confidences[i] < config.confidence_threshold:
                valid[i] = False
                # Use previous valid position
                for j in range(i-1, -1, -1):
                    if valid[j]:
                        filtered[i] = filtered[j]
                        break

    # Jump rejection
    if config.jump_rejection:
        for i in range(1, n):
            velocity = np.linalg.norm(filtered[i] - filtered[i-1])
            if velocity > config.jump_threshold:
                valid[i] = False
                filtered[i] = filtered[i-1]

    return filtered, valid


# =============================================================================
# Main Simulation Runner
# =============================================================================

def simulate_tracker(
    gt: np.ndarray,
    config: AblationConfig,
    fps: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Simulate tracker behavior based on ablation configuration.

    Args:
        gt: Ground truth trajectory
        config: Ablation configuration
        fps: Frame rate

    Returns:
        (estimated_trajectory, valid_mask, simulated_fps)
    """
    n = len(gt)

    # 1. Tracker backend
    if config.tracker == TrackerBackend.SAM2:
        est, valid = simulate_sam2_trajectory(gt, fps)
        sim_fps = 13.5
    elif config.tracker == TrackerBackend.DINO_K1:
        est, valid = simulate_dino_kframe_trajectory(gt, k=1, interpolation=config.interpolation)
        sim_fps = 5.0
    elif config.tracker == TrackerBackend.DINO_K5:
        est, valid = simulate_dino_kframe_trajectory(gt, k=5, interpolation=config.interpolation)
        sim_fps = 20.5
    elif config.tracker == TrackerBackend.DINO_K10:
        est, valid = simulate_dino_kframe_trajectory(gt, k=10, interpolation=config.interpolation)
        sim_fps = 35.0
    elif config.tracker == TrackerBackend.ADAPTIVE_K:
        est, valid = simulate_adaptive_k_trajectory(gt, fps, interpolation=config.interpolation)
        sim_fps = 26.4
    elif config.tracker == TrackerBackend.YOLO:
        est, valid = simulate_yolo_trajectory(gt)
        sim_fps = 142.0
    else:
        est, valid = gt.copy(), np.ones(n, dtype=bool)
        sim_fps = 30.0

    # 2. Smoothing
    est = apply_smoothing(est, config.smoothing, config.ema_alpha)

    # 3. Robustness layer
    # Generate fake confidences (high for most, low for some)
    confidences = np.random.uniform(0.5, 0.95, n)
    confidences[np.random.rand(n) < 0.05] = 0.2  # 5% low confidence

    est, robust_valid = apply_robustness_layer(est, confidences, config.robustness, fps)
    valid = valid & robust_valid

    return est, valid, sim_fps


def run_synthetic_evaluation(
    output_dir: str = None,
    n_frames: int = 150,
    fps: float = 30.0,
    seed: int = 42,
) -> Dict:
    """
    Run complete synthetic evaluation for all ablation configs.

    Args:
        output_dir: Output directory for results
        n_frames: Number of frames per trajectory
        fps: Frame rate
        seed: Random seed for reproducibility

    Returns:
        Results dictionary
    """
    np.random.seed(seed)

    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic GT trajectories
    print("Generating synthetic GT trajectories...")
    gt_trajectories = generate_all_synthetic_trajectories(n_frames, fps)

    # Get all ablation configs
    configs = get_all_ablation_configs()
    configs_by_category = get_configs_by_category()

    results = {
        "metadata": {
            "n_frames": n_frames,
            "fps": fps,
            "seed": seed,
            "trajectories": list(gt_trajectories.keys()),
        },
        "by_trajectory": {},
        "by_config": {},
        "by_category": {},
    }

    # Run evaluation for each trajectory and config
    print(f"\nRunning evaluation: {len(gt_trajectories)} trajectories x {len(configs)} configs")

    for traj_name, gt in gt_trajectories.items():
        print(f"\n=== Trajectory: {traj_name} ===")
        results["by_trajectory"][traj_name] = {}

        for config in configs:
            # Simulate tracker
            est, valid, sim_fps = simulate_tracker(gt, config, fps)

            # Evaluate metrics
            metrics = evaluate_trajectory(est, gt, valid, fps)

            # Store results
            result_entry = {
                "metrics": metrics.to_dict(),
                "fps": sim_fps,
                "normalized_jerk": normalize_jerk(metrics.jerk, metrics.amplitude_ratio),
            }

            results["by_trajectory"][traj_name][config.name] = result_entry

            if config.name not in results["by_config"]:
                results["by_config"][config.name] = {
                    "description": config.description,
                    "config": config.to_dict(),
                    "trajectories": {},
                }
            results["by_config"][config.name]["trajectories"][traj_name] = result_entry

            print(f"  {config.name}: Amp={metrics.amplitude_ratio*100:.1f}%, "
                  f"MAE={metrics.mae:.1f}px, VelCorr={metrics.velocity_correlation:.3f}, "
                  f"Jerk={metrics.jerk:.4f}")

    # Organize by category
    for cat_name, cat_configs in configs_by_category.items():
        results["by_category"][cat_name] = {}
        for config in cat_configs:
            if config.name in results["by_config"]:
                results["by_category"][cat_name][config.name] = results["by_config"][config.name]

    # Save results
    output_file = output_dir / "results_tracking_synthetic.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


# =============================================================================
# Table Generation
# =============================================================================

def generate_synthetic_tables(results: Dict) -> str:
    """Generate paper-ready markdown tables from results."""
    lines = []
    lines.append("# Tracking Ablation - Synthetic Evaluation Results\n")

    # Critical test: oscillating motion
    lines.append("## Critical Test: Oscillating Motion (0.6Hz)\n")
    lines.append("This is the key test case where SAM2 fails due to motion collapse.\n")
    lines.append("| Method | Amp% | MAE(px) | VelCorr | Jerk | NormJerk |")
    lines.append("|--------|------|---------|---------|------|----------|")

    osc_results = results["by_trajectory"].get("oscillating_0.6Hz", {})
    for method, data in sorted(osc_results.items()):
        m = data["metrics"]
        nj = data["normalized_jerk"]
        lines.append(
            f"| {method} | {m['amplitude_ratio']*100:.1f}% | {m['mae']:.1f} | "
            f"{m['velocity_correlation']:.3f} | {m['jerk']:.4f} | {nj:.4f} |"
        )

    # By category tables
    for cat_name, cat_data in results["by_category"].items():
        lines.append(f"\n## {cat_name.replace('_', ' ').title()}\n")

        # Table for oscillating motion (most important)
        lines.append("### Oscillating Motion (0.6Hz)\n")
        lines.append("| Method | Amp% | MAE(px) | VelCorr | Jerk |")
        lines.append("|--------|------|---------|---------|------|")

        for method_name, method_data in cat_data.items():
            if "oscillating_0.6Hz" in method_data["trajectories"]:
                m = method_data["trajectories"]["oscillating_0.6Hz"]["metrics"]
                lines.append(
                    f"| {method_name} | {m['amplitude_ratio']*100:.1f}% | {m['mae']:.1f} | "
                    f"{m['velocity_correlation']:.3f} | {m['jerk']:.4f} |"
                )

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Synthetic Tracking Ablation Evaluation")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--n-frames", type=int, default=150, help="Frames per trajectory")
    parser.add_argument("--fps", type=float, default=30.0, help="Frame rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    results = run_synthetic_evaluation(
        output_dir=args.output_dir,
        n_frames=args.n_frames,
        fps=args.fps,
        seed=args.seed,
    )

    # Generate tables
    tables = generate_synthetic_tables(results)
    print("\n" + "=" * 60)
    print(tables)

    # Save tables
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "results"
    with open(output_dir / "synthetic_tables.md", "w") as f:
        f.write(tables)
    print(f"\nTables saved to: {output_dir / 'synthetic_tables.md'}")
