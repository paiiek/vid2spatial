"""
Trajectory3D - First-class trajectory data structure with unified confidence.

This is the core data structure for vid2spatial spatial audio authoring.
All trajectory outputs should eventually be converted to this format.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Tuple
from enum import Enum
import numpy as np


class TrackingState(Enum):
    """Tracking state for each frame."""
    TRACKED = "tracked"      # Normal tracking
    HOLD = "hold"            # Lost tracking, holding last position
    INTERPOLATED = "interpolated"  # Gap filled by interpolation
    LOW_CONFIDENCE = "low_confidence"  # Tracked but uncertain


@dataclass
class TrajectoryFrame:
    """Single frame in a 3D trajectory."""
    frame_idx: int
    time_sec: float

    # Spatial position (spherical)
    azimuth: float      # radians, -pi to pi (left negative, right positive)
    elevation: float    # radians, -pi/2 to pi/2 (down negative, up positive)
    distance: float     # meters

    # Spatial position (cartesian) - derived from spherical
    x: float  # right positive
    y: float  # up positive
    z: float  # forward positive (depth)

    # Unified confidence (0-1)
    confidence: float

    # Tracking state
    state: TrackingState = TrackingState.TRACKED

    # Component confidences (for debugging/analysis)
    detection_conf: float = 1.0
    tracking_conf: float = 1.0
    depth_conf: float = 1.0

    def to_dict(self) -> Dict:
        return {
            "frame": self.frame_idx,
            "time": self.time_sec,
            "az": self.azimuth,
            "el": self.elevation,
            "dist_m": self.distance,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "confidence": self.confidence,
            "state": self.state.value,
        }


@dataclass
class Trajectory3D:
    """
    First-class 3D trajectory with unified confidence.

    This is the core output format for vid2spatial.
    """
    frames: List[TrajectoryFrame]

    # Video metadata
    video_width: int
    video_height: int
    fps: float
    total_video_frames: int

    # Source info
    source_id: str = "source_0"
    text_prompt: str = ""

    # Camera intrinsics
    fov_deg: float = 60.0

    # Quality metrics (computed)
    avg_confidence: float = field(init=False)
    coverage: float = field(init=False)  # % of frames tracked
    continuity: float = field(init=False)  # % without gaps

    def __post_init__(self):
        self._compute_metrics()

    def _compute_metrics(self):
        if not self.frames:
            self.avg_confidence = 0.0
            self.coverage = 0.0
            self.continuity = 0.0
            return

        # Average confidence
        self.avg_confidence = np.mean([f.confidence for f in self.frames])

        # Coverage (tracked frames / total possible frames)
        expected_frames = self.total_video_frames
        self.coverage = len(self.frames) / max(expected_frames, 1)

        # Continuity (non-hold frames / total frames)
        non_hold = sum(1 for f in self.frames if f.state != TrackingState.HOLD)
        self.continuity = non_hold / max(len(self.frames), 1)

    def get_frame(self, frame_idx: int) -> Optional[TrajectoryFrame]:
        """Get frame by index."""
        for f in self.frames:
            if f.frame_idx == frame_idx:
                return f
        return None

    def get_time_range(self) -> tuple:
        """Get (start_time, end_time) in seconds."""
        if not self.frames:
            return (0.0, 0.0)
        return (self.frames[0].time_sec, self.frames[-1].time_sec)

    def to_dict(self) -> Dict:
        """Convert to serializable dict."""
        return {
            "source_id": self.source_id,
            "text_prompt": self.text_prompt,
            "intrinsics": {
                "width": self.video_width,
                "height": self.video_height,
                "fps": self.fps,
                "fov_deg": self.fov_deg,
            },
            "quality": {
                "avg_confidence": self.avg_confidence,
                "coverage": self.coverage,
                "continuity": self.continuity,
                "total_frames": len(self.frames),
            },
            "frames": [f.to_dict() for f in self.frames],
        }

    def get_spatial_arrays(self) -> Dict[str, np.ndarray]:
        """Get numpy arrays for spatial parameters."""
        return {
            "time": np.array([f.time_sec for f in self.frames]),
            "azimuth": np.array([f.azimuth for f in self.frames]),
            "elevation": np.array([f.elevation for f in self.frames]),
            "distance": np.array([f.distance for f in self.frames]),
            "confidence": np.array([f.confidence for f in self.frames]),
        }

    @classmethod
    def from_tracking_result(
        cls,
        tracking_result,  # HybridTrackingResult
        detection_conf: float = 1.0,
        apply_fallback: bool = True,
        auto_stabilize: bool = False,
    ) -> "Trajectory3D":
        """
        Create Trajectory3D from HybridTrackingResult.

        This computes unified confidence from:
        - detection_conf: Initial detection confidence
        - tracking_conf: Per-frame tracking confidence
        - depth_conf: Depth estimation confidence (based on variance + scene change)

        Args:
            tracking_result: HybridTrackingResult from tracker
            detection_conf: Override detection confidence (default: from result)
            apply_fallback: Apply fallback policy for tracking failures
            auto_stabilize: Auto-select and apply stabilizer based on quality
        """
        from .vision import CameraIntrinsics, pixel_to_ray, ray_to_angles

        K = CameraIntrinsics(
            width=tracking_result.video_width,
            height=tracking_result.video_height,
            fov_deg=tracking_result.fov_deg,
        )

        # Use actual detection confidence if available
        det_conf = getattr(tracking_result, 'initial_detection_conf', detection_conf)

        frames = []
        prev_depth = None
        depth_values = []

        for f in tracking_result.frames:
            cx, cy = f.center
            dist_m = f.depth_m if f.depth_m > 0 else 2.0

            # Compute 3D position
            ray = pixel_to_ray(cx, cy, K)
            az, el = ray_to_angles(ray)

            x = ray[0] * dist_m
            y = ray[1] * dist_m
            z = ray[2] * dist_m

            # Compute depth confidence with scene change detection
            depth_conf, is_scene_change = compute_depth_confidence(
                current_depth=dist_m,
                prev_depth=prev_depth,
                prev_depths=depth_values,
                scene_change_threshold=5.0,
                variance_window=5,
            )

            prev_depth = dist_m
            depth_values.append(dist_m)

            # Unified confidence = geometric mean of components
            tracking_conf = f.confidence
            unified_conf = (det_conf * tracking_conf * depth_conf) ** (1/3)

            # Determine state
            if tracking_conf < 0.4:
                state = TrackingState.LOW_CONFIDENCE
            else:
                state = TrackingState.TRACKED

            time_sec = f.frame_idx / tracking_result.fps

            frames.append(TrajectoryFrame(
                frame_idx=f.frame_idx,
                time_sec=time_sec,
                azimuth=float(az),
                elevation=float(el),
                distance=dist_m,
                x=float(x),
                y=float(y),
                z=float(z),
                confidence=unified_conf,
                state=state,
                detection_conf=det_conf,
                tracking_conf=tracking_conf,
                depth_conf=depth_conf,
            ))

        trajectory = cls(
            frames=frames,
            video_width=tracking_result.video_width,
            video_height=tracking_result.video_height,
            fps=tracking_result.fps,
            total_video_frames=tracking_result.total_frames,
            text_prompt=tracking_result.text_prompt,
            fov_deg=tracking_result.fov_deg,
        )

        # Apply fallback policy if enabled
        if apply_fallback:
            trajectory = apply_fallback_policy(trajectory)

        # Apply auto stabilizer if enabled
        if auto_stabilize:
            trajectory, _ = apply_stabilizer_policy(trajectory)

        return trajectory


def compute_unified_confidence(
    detection_conf: float,
    tracking_conf: float,
    depth_conf: float,
    weights: tuple = (0.2, 0.5, 0.3),
) -> float:
    """
    Compute unified confidence from component confidences.

    Args:
        detection_conf: Initial detection confidence (0-1)
        tracking_conf: Frame tracking confidence (0-1)
        depth_conf: Depth estimation confidence (0-1)
        weights: (detection, tracking, depth) weights

    Returns:
        Unified confidence (0-1)
    """
    w_det, w_track, w_depth = weights
    return (
        w_det * detection_conf +
        w_track * tracking_conf +
        w_depth * depth_conf
    )


# ============================================================================
# Fallback Policy
# ============================================================================

@dataclass
class FallbackPolicy:
    """
    Policy for handling tracking failures and depth anomalies.

    Implements:
    1. HOLD: When tracking lost, hold last known position
    2. INTERPOLATE: Fill gaps with linear interpolation
    3. SMOOTH_DEPTH: Clip extreme depth jumps
    """
    # Tracking loss thresholds
    confidence_threshold: float = 0.3  # Below this = tracking lost
    max_hold_frames: int = 15  # Max frames to hold before marking as lost

    # Depth anomaly thresholds
    max_depth_jump_m: float = 3.0  # Max allowed depth change per frame
    depth_smooth_alpha: float = 0.5  # EMA factor for depth smoothing

    # Interpolation settings
    enable_interpolation: bool = True
    max_interpolation_gap: int = 10  # Max frames to interpolate


def apply_fallback_policy(
    trajectory: "Trajectory3D",
    policy: Optional[FallbackPolicy] = None,
) -> "Trajectory3D":
    """
    Apply fallback policy to handle tracking failures and depth anomalies.

    Args:
        trajectory: Input trajectory
        policy: Fallback policy settings

    Returns:
        Trajectory with fallback handling applied
    """
    if policy is None:
        policy = FallbackPolicy()

    if not trajectory.frames:
        return trajectory

    frames = list(trajectory.frames)

    # Pass 1: Detect and mark low-confidence frames
    for i, f in enumerate(frames):
        if f.confidence < policy.confidence_threshold:
            # Mark as low confidence, will be handled in pass 2
            frames[i] = TrajectoryFrame(
                frame_idx=f.frame_idx,
                time_sec=f.time_sec,
                azimuth=f.azimuth,
                elevation=f.elevation,
                distance=f.distance,
                x=f.x, y=f.y, z=f.z,
                confidence=f.confidence,
                state=TrackingState.LOW_CONFIDENCE,
                detection_conf=f.detection_conf,
                tracking_conf=f.tracking_conf,
                depth_conf=f.depth_conf,
            )

    # Pass 2: Apply HOLD for consecutive low-confidence frames
    hold_count = 0
    last_good_frame = None

    for i, f in enumerate(frames):
        if f.state == TrackingState.LOW_CONFIDENCE:
            hold_count += 1
            if last_good_frame is not None and hold_count <= policy.max_hold_frames:
                # HOLD: Use last good position
                frames[i] = TrajectoryFrame(
                    frame_idx=f.frame_idx,
                    time_sec=f.time_sec,
                    azimuth=last_good_frame.azimuth,
                    elevation=last_good_frame.elevation,
                    distance=last_good_frame.distance,
                    x=last_good_frame.x,
                    y=last_good_frame.y,
                    z=last_good_frame.z,
                    confidence=f.confidence * 0.5,  # Reduced confidence for held frames
                    state=TrackingState.HOLD,
                    detection_conf=f.detection_conf,
                    tracking_conf=f.tracking_conf,
                    depth_conf=f.depth_conf * 0.5,
                )
        else:
            hold_count = 0
            last_good_frame = f

    # Pass 3: Smooth extreme depth jumps
    prev_depth = frames[0].distance
    for i in range(1, len(frames)):
        f = frames[i]
        depth_jump = abs(f.distance - prev_depth)

        if depth_jump > policy.max_depth_jump_m and f.state != TrackingState.HOLD:
            # Smooth the depth jump
            smoothed_depth = (
                prev_depth * (1 - policy.depth_smooth_alpha) +
                f.distance * policy.depth_smooth_alpha
            )

            # Recompute xyz with smoothed depth
            ratio = smoothed_depth / f.distance if f.distance > 0 else 1.0
            frames[i] = TrajectoryFrame(
                frame_idx=f.frame_idx,
                time_sec=f.time_sec,
                azimuth=f.azimuth,
                elevation=f.elevation,
                distance=smoothed_depth,
                x=f.x * ratio,
                y=f.y * ratio,
                z=f.z * ratio,
                confidence=f.confidence * 0.8,  # Slightly reduce confidence
                state=f.state,
                detection_conf=f.detection_conf,
                tracking_conf=f.tracking_conf,
                depth_conf=f.depth_conf * 0.7,  # Reduce depth confidence
            )
            prev_depth = smoothed_depth
        else:
            prev_depth = f.distance

    # Create new trajectory with processed frames
    return Trajectory3D(
        frames=frames,
        video_width=trajectory.video_width,
        video_height=trajectory.video_height,
        fps=trajectory.fps,
        total_video_frames=trajectory.total_video_frames,
        source_id=trajectory.source_id,
        text_prompt=trajectory.text_prompt,
        fov_deg=trajectory.fov_deg,
    )


# ============================================================================
# Stabilizer Policy (confidence-based auto selection)
# ============================================================================

@dataclass
class StabilizerPolicy:
    """
    Policy for automatic stabilizer selection based on trajectory quality.

    Decision logic:
    - High quality (avg_conf > 0.7, jerk < threshold): No stabilizer (EMA only)
    - Medium quality (avg_conf > 0.5): One Euro filter
    - Low quality or high jerk: Kalman filter
    """
    # Thresholds for quality assessment
    high_quality_conf: float = 0.7
    medium_quality_conf: float = 0.5

    # Jerk thresholds (normalized by fps)
    low_jerk_threshold: float = 0.5
    high_jerk_threshold: float = 1.5

    # EMA settings (default)
    ema_alpha: float = 0.3

    # Kalman settings (for noisy trajectories)
    kalman_process_noise: float = 0.01
    kalman_measurement_noise: float = 0.1


def compute_trajectory_jerk(trajectory: "Trajectory3D") -> float:
    """Compute average jerk (3rd derivative) of trajectory."""
    if len(trajectory.frames) < 4:
        return 0.0

    # Use azimuth as primary motion indicator
    az = np.array([f.azimuth for f in trajectory.frames])

    if len(az) < 4:
        return 0.0

    vel = np.diff(az)
    acc = np.diff(vel)
    jerk = np.diff(acc)

    return float(np.mean(np.abs(jerk)))


def select_stabilizer(
    trajectory: "Trajectory3D",
    policy: Optional[StabilizerPolicy] = None,
) -> Tuple[str, Dict]:
    """
    Automatically select best stabilizer based on trajectory quality.

    Args:
        trajectory: Input trajectory
        policy: Stabilizer policy settings

    Returns:
        Tuple of (stabilizer_name, stabilizer_params)
        - stabilizer_name: "none", "ema", "one_euro", or "kalman"
        - stabilizer_params: Parameters for the selected stabilizer
    """
    if policy is None:
        policy = StabilizerPolicy()

    avg_conf = trajectory.avg_confidence
    jerk = compute_trajectory_jerk(trajectory)

    # Normalize jerk by fps (higher fps = more samples = lower jerk per sample)
    normalized_jerk = jerk * (trajectory.fps / 30.0)

    # Decision logic
    if avg_conf >= policy.high_quality_conf and normalized_jerk < policy.low_jerk_threshold:
        # High quality: minimal smoothing
        return "ema", {"alpha": policy.ema_alpha}

    elif avg_conf >= policy.medium_quality_conf and normalized_jerk < policy.high_jerk_threshold:
        # Medium quality: adaptive smoothing
        return "one_euro", {
            "min_cutoff": 1.0,
            "beta": 0.007,
        }

    else:
        # Low quality or high jerk: Kalman filter
        return "kalman", {
            "process_noise": policy.kalman_process_noise,
            "measurement_noise": policy.kalman_measurement_noise,
        }


def apply_stabilizer_policy(
    trajectory: "Trajectory3D",
    policy: Optional[StabilizerPolicy] = None,
    force_stabilizer: Optional[str] = None,
) -> Tuple["Trajectory3D", str]:
    """
    Apply stabilizer based on policy or force a specific stabilizer.

    Args:
        trajectory: Input trajectory
        policy: Stabilizer policy settings
        force_stabilizer: Override auto-selection with specific stabilizer

    Returns:
        Tuple of (stabilized_trajectory, stabilizer_used)
    """
    from .trajectory_stabilizer import TrajectoryStabilizer

    if force_stabilizer is not None:
        stabilizer_name = force_stabilizer
        params = {}
    else:
        stabilizer_name, params = select_stabilizer(trajectory, policy)

    if stabilizer_name == "none":
        return trajectory, "none"

    # Convert trajectory to dict format for stabilizer
    frames_dict = [
        {
            "frame": f.frame_idx,
            "az": f.azimuth,
            "el": f.elevation,
            "dist_m": f.distance,
            "x": f.x,
            "y": f.y,
            "z": f.z,
        }
        for f in trajectory.frames
    ]

    if stabilizer_name == "ema":
        # Apply EMA directly
        alpha = params.get("alpha", 0.3)
        smoothed = [frames_dict[0]]
        prev = frames_dict[0]

        for f in frames_dict[1:]:
            smoothed_f = {
                "frame": f["frame"],
                "az": prev["az"] * (1 - alpha) + f["az"] * alpha,
                "el": prev["el"] * (1 - alpha) + f["el"] * alpha,
                "dist_m": prev["dist_m"] * (1 - alpha) + f["dist_m"] * alpha,
                "x": prev["x"] * (1 - alpha) + f["x"] * alpha,
                "y": prev["y"] * (1 - alpha) + f["y"] * alpha,
                "z": prev["z"] * (1 - alpha) + f["z"] * alpha,
            }
            smoothed.append(smoothed_f)
            prev = smoothed_f

        frames_dict = smoothed

    elif stabilizer_name in ("kalman", "one_euro"):
        stabilizer = TrajectoryStabilizer(
            method=stabilizer_name,
            process_noise=params.get("process_noise", 0.01),
            measurement_noise=params.get("measurement_noise", 0.1),
        )
        frames_dict = stabilizer.stabilize_trajectory(frames_dict)

    # Convert back to TrajectoryFrame list
    new_frames = []
    for i, fd in enumerate(frames_dict):
        orig = trajectory.frames[i]
        new_frames.append(TrajectoryFrame(
            frame_idx=fd["frame"],
            time_sec=orig.time_sec,
            azimuth=fd["az"],
            elevation=fd["el"],
            distance=fd["dist_m"],
            x=fd["x"],
            y=fd["y"],
            z=fd["z"],
            confidence=orig.confidence,
            state=orig.state,
            detection_conf=orig.detection_conf,
            tracking_conf=orig.tracking_conf,
            depth_conf=orig.depth_conf,
        ))

    result = Trajectory3D(
        frames=new_frames,
        video_width=trajectory.video_width,
        video_height=trajectory.video_height,
        fps=trajectory.fps,
        total_video_frames=trajectory.total_video_frames,
        source_id=trajectory.source_id,
        text_prompt=trajectory.text_prompt,
        fov_deg=trajectory.fov_deg,
    )

    return result, stabilizer_name


# ============================================================================
# Depth Confidence with Scene Change Detection
# ============================================================================

def compute_depth_confidence(
    current_depth: float,
    prev_depth: Optional[float],
    prev_depths: List[float],
    scene_change_threshold: float = 5.0,
    variance_window: int = 5,
) -> Tuple[float, bool]:
    """
    Compute depth confidence with scene change detection.

    Args:
        current_depth: Current depth in meters
        prev_depth: Previous frame depth
        prev_depths: List of recent depths (for variance computation)
        scene_change_threshold: Depth jump threshold for scene change
        variance_window: Window size for variance computation

    Returns:
        Tuple of (depth_confidence, is_scene_change)
    """
    is_scene_change = False

    if prev_depth is None:
        return 0.8, False

    # Check for scene change (large depth discontinuity)
    depth_jump = abs(current_depth - prev_depth)
    if depth_jump > scene_change_threshold:
        is_scene_change = True
        # Scene change: reset confidence, don't penalize
        return 0.7, True

    # Compute confidence based on stability
    # Large jumps within scene = lower confidence
    jump_penalty = min(1.0, depth_jump / 3.0)  # Normalize by 3m

    # Variance-based confidence
    if len(prev_depths) >= variance_window:
        recent = prev_depths[-variance_window:]
        variance = np.var(recent)
        variance_penalty = min(1.0, variance / 2.0)  # Normalize by 2m^2
    else:
        variance_penalty = 0.2

    # Combined confidence
    confidence = max(0.3, 1.0 - 0.5 * jump_penalty - 0.3 * variance_penalty)

    return confidence, is_scene_change


__all__ = [
    "Trajectory3D",
    "TrajectoryFrame",
    "TrackingState",
    "compute_unified_confidence",
    "FallbackPolicy",
    "apply_fallback_policy",
    "StabilizerPolicy",
    "select_stabilizer",
    "apply_stabilizer_policy",
    "compute_depth_confidence",
    "compute_trajectory_jerk",
]
