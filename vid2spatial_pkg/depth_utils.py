"""
Depth Utilities for Vid2Spatial

Provides:
1. bbox-scale proxy depth with confidence blending
2. d_rel (relative distance) output
3. adaptive depth stride based on motion/variance

Key insight: For spatial audio, relative distance changes matter more than
absolute metric accuracy. Fast-moving objects benefit from bbox-scale proxy.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DepthConfig:
    """Configuration for depth processing."""
    # Blending parameters
    use_bbox_proxy: bool = True
    proxy_blend_by_confidence: bool = True  # blend based on tracking confidence

    # Adaptive stride parameters
    use_adaptive_stride: bool = True
    min_stride: int = 2
    max_stride: int = 30
    default_stride: int = 5

    # Variance thresholds for adaptive stride
    depth_var_low: float = 0.01  # Below this -> increase stride
    depth_var_high: float = 0.5  # Above this -> decrease stride
    bbox_change_low: float = 0.05
    bbox_change_high: float = 0.2

    # Output options
    output_d_rel: bool = True  # Output relative distance (0-1)
    d_rel_min: float = 0.5  # Min distance for d_rel normalization
    d_rel_max: float = 10.0  # Max distance for d_rel normalization


def compute_bbox_scale_proxy(
    bbox_areas: List[float],
    initial_depth_m: float = 2.0,
) -> List[float]:
    """
    Compute depth proxy from bbox scale changes.

    Assumes: depth ∝ 1/sqrt(bbox_area)
    Calibrated using initial frame's metric depth.

    Args:
        bbox_areas: List of bbox areas (w*h) per frame
        initial_depth_m: Metric depth at frame 0 for calibration

    Returns:
        List of proxy depth values in meters
    """
    if not bbox_areas:
        return []

    areas = np.array(bbox_areas, dtype=np.float32)
    areas = np.maximum(areas, 1.0)  # Avoid division by zero

    # Calibrate: z_proxy = z_initial * sqrt(area_initial / area_current)
    initial_area = areas[0]
    proxy_depths = initial_depth_m * np.sqrt(initial_area / areas)

    return proxy_depths.tolist()


def blend_depth_with_proxy(
    metric_depths: List[float],
    proxy_depths: List[float],
    confidences: List[float],
    min_confidence: float = 0.3,
    max_confidence: float = 0.8,
) -> List[float]:
    """
    Blend metric depth with bbox-scale proxy based on confidence.

    Low confidence -> more proxy weight (bbox is still reliable)
    High confidence -> more metric weight (depth estimation reliable)

    Args:
        metric_depths: Metric depth values from Depth Anything V2
        proxy_depths: Proxy depth from bbox scale
        confidences: Tracking confidence per frame
        min_confidence: Below this, use 100% proxy
        max_confidence: Above this, use 100% metric

    Returns:
        Blended depth values
    """
    n = len(metric_depths)
    blended = []

    for i in range(n):
        conf = confidences[i] if i < len(confidences) else 0.5

        # Compute alpha (metric weight) based on confidence
        if conf <= min_confidence:
            alpha = 0.0  # 100% proxy
        elif conf >= max_confidence:
            alpha = 1.0  # 100% metric
        else:
            # Linear interpolation
            alpha = (conf - min_confidence) / (max_confidence - min_confidence)

        metric = metric_depths[i] if i < len(metric_depths) else 2.0
        proxy = proxy_depths[i] if i < len(proxy_depths) else 2.0

        # Blend
        blended_depth = alpha * metric + (1 - alpha) * proxy
        blended.append(blended_depth)

    return blended


def compute_d_rel(
    depths_m: List[float],
    d_min: float = 0.5,
    d_max: float = 10.0,
) -> List[float]:
    """
    Convert metric depth to relative distance (0-1).

    0 = close (d_min), 1 = far (d_max)
    Useful for direct control of gain/reverb in spatial audio.

    Args:
        depths_m: Depth values in meters
        d_min: Minimum distance (maps to 0)
        d_max: Maximum distance (maps to 1)

    Returns:
        Relative distance values (0-1)
    """
    d_rel = []
    for d in depths_m:
        # Clamp and normalize
        clamped = max(d_min, min(d_max, d))
        normalized = (clamped - d_min) / (d_max - d_min)
        d_rel.append(normalized)

    return d_rel


def compute_adaptive_depth_stride(
    recent_depths: List[float],
    recent_bbox_areas: List[float],
    config: DepthConfig = None,
) -> int:
    """
    Compute optimal depth stride based on recent motion/variance.

    Logic:
    - Low variance + low bbox change -> high stride (save computation)
    - High variance or fast motion -> low stride (need accuracy)

    Args:
        recent_depths: Last N depth values
        recent_bbox_areas: Last N bbox areas
        config: DepthConfig with thresholds

    Returns:
        Recommended depth stride
    """
    if config is None:
        config = DepthConfig()

    if len(recent_depths) < 5:
        return config.default_stride

    # Compute depth variance
    depth_var = np.var(recent_depths[-10:]) if len(recent_depths) >= 10 else np.var(recent_depths)

    # Compute bbox scale change rate
    if len(recent_bbox_areas) >= 2:
        areas = np.array(recent_bbox_areas[-10:])
        scale_changes = np.abs(np.diff(areas) / (areas[:-1] + 1e-8))
        bbox_change_rate = np.mean(scale_changes)
    else:
        bbox_change_rate = 0.0

    # Decision logic
    if depth_var < config.depth_var_low and bbox_change_rate < config.bbox_change_low:
        # Very stable -> use max stride
        return config.max_stride
    elif depth_var > config.depth_var_high or bbox_change_rate > config.bbox_change_high:
        # Fast changing -> use min stride
        return config.min_stride
    else:
        # Moderate -> use default
        return config.default_stride


def process_trajectory_depth(
    frames: List[Dict],
    config: DepthConfig = None,
) -> List[Dict]:
    """
    Post-process trajectory with enhanced depth handling.

    Adds:
    - depth_blended: Blended metric+proxy depth
    - d_rel: Relative distance (0-1)
    - depth_proxy: Pure bbox-scale proxy (for debugging)

    Args:
        frames: Trajectory frames with dist_m, confidence, bbox info
        config: DepthConfig

    Returns:
        Enhanced trajectory frames
    """
    if config is None:
        config = DepthConfig()

    if not frames:
        return frames

    # Extract data
    metric_depths = [f.get("dist_m", 2.0) for f in frames]
    confidences = [f.get("confidence", 0.5) for f in frames]

    # Compute bbox areas
    bbox_areas = []
    for f in frames:
        w = f.get("w", f.get("bbox_w", 100))
        h = f.get("h", f.get("bbox_h", 100))
        bbox_areas.append(w * h)

    # 1. Compute bbox-scale proxy
    initial_depth = metric_depths[0] if metric_depths[0] > 0 else 2.0
    proxy_depths = compute_bbox_scale_proxy(bbox_areas, initial_depth)

    # 2. Blend depths based on confidence
    if config.use_bbox_proxy and config.proxy_blend_by_confidence:
        blended_depths = blend_depth_with_proxy(
            metric_depths, proxy_depths, confidences
        )
    else:
        blended_depths = metric_depths

    # 3. Compute d_rel
    if config.output_d_rel:
        d_rel_values = compute_d_rel(blended_depths, config.d_rel_min, config.d_rel_max)
    else:
        d_rel_values = [0.5] * len(frames)

    # 4. Enhance frames
    enhanced_frames = []
    for i, f in enumerate(frames):
        enhanced = f.copy()
        enhanced["depth_proxy"] = proxy_depths[i] if i < len(proxy_depths) else 2.0
        enhanced["depth_blended"] = blended_depths[i] if i < len(blended_depths) else 2.0
        enhanced["d_rel"] = d_rel_values[i] if i < len(d_rel_values) else 0.5
        enhanced_frames.append(enhanced)

    return enhanced_frames


class AdaptiveDepthEstimator:
    """
    Wrapper for adaptive depth estimation during tracking.

    Maintains state to compute adaptive stride on-the-fly.
    """

    def __init__(self, config: DepthConfig = None):
        self.config = config or DepthConfig()
        self.recent_depths: List[float] = []
        self.recent_bbox_areas: List[float] = []
        self.frame_count = 0
        self.last_depth_frame = -1
        self._current_stride = self.config.default_stride

    def should_estimate_depth(self, frame_idx: int) -> bool:
        """Check if we should estimate depth for this frame."""
        if not self.config.use_adaptive_stride:
            # Fixed stride
            return (frame_idx - self.last_depth_frame) >= self.config.default_stride

        # Adaptive stride
        frames_since_last = frame_idx - self.last_depth_frame
        return frames_since_last >= self._current_stride

    def update(self, depth_m: float, bbox_area: float, frame_idx: int):
        """Update state after depth estimation."""
        self.recent_depths.append(depth_m)
        self.recent_bbox_areas.append(bbox_area)
        self.last_depth_frame = frame_idx
        self.frame_count += 1

        # Keep only recent history
        max_history = 30
        if len(self.recent_depths) > max_history:
            self.recent_depths = self.recent_depths[-max_history:]
            self.recent_bbox_areas = self.recent_bbox_areas[-max_history:]

        # Recompute adaptive stride
        if self.config.use_adaptive_stride:
            self._current_stride = compute_adaptive_depth_stride(
                self.recent_depths, self.recent_bbox_areas, self.config
            )

    @property
    def current_stride(self) -> int:
        return self._current_stride

    def get_stats(self) -> Dict:
        """Get statistics for debugging."""
        if not self.recent_depths:
            return {"stride": self._current_stride, "depth_var": 0, "bbox_change": 0}

        depth_var = float(np.var(self.recent_depths))

        if len(self.recent_bbox_areas) >= 2:
            areas = np.array(self.recent_bbox_areas)
            changes = np.abs(np.diff(areas) / (areas[:-1] + 1e-8))
            bbox_change = float(np.mean(changes))
        else:
            bbox_change = 0.0

        return {
            "stride": self._current_stride,
            "depth_var": depth_var,
            "bbox_change": bbox_change,
            "num_samples": len(self.recent_depths),
        }
