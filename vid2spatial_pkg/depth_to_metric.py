"""
Depth to Metric Distance Conversion

문제: Monocular depth estimation은 relative depth만 제공
해결: 다양한 방법으로 metric distance로 변환

Methods:
1. Linear scaling (현재) - 단순하지만 부정확
2. Scale-invariant with reference - reference object 크기로 calibration
3. Temporal consistency - 시간적 일관성 활용
4. Multi-cue fusion - bbox size + depth 결합
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DepthCalibration:
    """Depth calibration parameters."""
    scale: float = 1.0  # depth * scale = meters
    offset: float = 0.0  # depth * scale + offset = meters
    reference_depth: float = 0.5  # reference depth value
    reference_distance: float = 2.0  # known distance at reference depth


class DepthToMetricConverter:
    """
    Convert relative depth to metric distance using multiple cues.

    핵심 아이디어:
    - Monocular depth는 scale ambiguity가 있음
    - 하지만 bbox size, temporal consistency 등 추가 정보로 보정 가능
    """

    def __init__(
        self,
        method: str = "bbox_calibrated",
        near_m: float = 0.5,
        far_m: float = 10.0,
        reference_bbox_height: float = 0.3,  # 프레임 높이의 30% = 약 2m 거리
        reference_distance: float = 2.0,
    ):
        """
        Initialize depth converter.

        Args:
            method: 변환 방법
                - "linear": 단순 linear mapping (현재)
                - "bbox_calibrated": bbox 크기로 calibration
                - "inverse": 1/depth 관계 사용 (더 물리적)
                - "adaptive": 시간에 따라 적응
            near_m: 최소 거리 (meters)
            far_m: 최대 거리 (meters)
            reference_bbox_height: 기준 bbox 높이 (프레임 대비 비율)
            reference_distance: 기준 거리 (meters)
        """
        self.method = method
        self.near_m = near_m
        self.far_m = far_m
        self.reference_bbox_height = reference_bbox_height
        self.reference_distance = reference_distance

        # Adaptive 방법용 history
        self.depth_history: List[float] = []
        self.distance_history: List[float] = []

    def convert(
        self,
        depth_rel: float,
        bbox_height_ratio: Optional[float] = None,
        frame_height: Optional[int] = None,
        bbox_height: Optional[int] = None,
    ) -> float:
        """
        Convert relative depth to metric distance.

        Args:
            depth_rel: Relative depth [0, 1] (0=far, 1=near for most models)
            bbox_height_ratio: bbox 높이 / 프레임 높이
            frame_height: 프레임 높이 (pixels)
            bbox_height: bbox 높이 (pixels)

        Returns:
            distance_m: Metric distance in meters
        """
        # Calculate bbox ratio if components provided
        if bbox_height_ratio is None and frame_height and bbox_height:
            bbox_height_ratio = bbox_height / frame_height

        if self.method == "linear":
            return self._linear_convert(depth_rel)
        elif self.method == "inverse":
            return self._inverse_convert(depth_rel)
        elif self.method == "bbox_calibrated":
            return self._bbox_calibrated_convert(depth_rel, bbox_height_ratio)
        elif self.method == "adaptive":
            return self._adaptive_convert(depth_rel, bbox_height_ratio)
        else:
            return self._linear_convert(depth_rel)

    def _linear_convert(self, depth_rel: float) -> float:
        """
        단순 linear mapping (현재 방식).

        depth가 높을수록 가까움 (대부분의 depth model convention)
        """
        # depth_rel: 0=far, 1=near
        # distance: near_m ~ far_m
        dist = self.near_m + (1.0 - depth_rel) * (self.far_m - self.near_m)
        return float(np.clip(dist, self.near_m, self.far_m))

    def _inverse_convert(self, depth_rel: float) -> float:
        """
        1/depth 관계 사용 (더 물리적).

        실제 depth ∝ 1/distance 관계
        """
        # Avoid division by zero
        depth_rel = max(depth_rel, 0.01)

        # 1/depth relationship
        # depth_rel = k / distance → distance = k / depth_rel
        # k를 near_m, far_m에 맞게 calibrate

        # At depth_rel=1 (closest), distance=near_m
        # At depth_rel=0.1 (far), distance=far_m
        # k = near_m * 1.0 = near_m
        # But we want: k / 0.1 = far_m → k = far_m * 0.1

        # Better: use log scale
        k = self.near_m
        dist = k / depth_rel

        return float(np.clip(dist, self.near_m, self.far_m))

    def _bbox_calibrated_convert(
        self,
        depth_rel: float,
        bbox_height_ratio: Optional[float] = None,
    ) -> float:
        """
        BBox 크기로 calibration.

        핵심 아이디어:
        - 같은 물체의 bbox 크기는 거리에 반비례
        - bbox_height ∝ 1/distance
        - reference 크기와 비교해서 거리 추정
        """
        if bbox_height_ratio is None:
            # Fallback to linear if no bbox info
            return self._linear_convert(depth_rel)

        # BBox 기반 거리 추정
        # bbox_height_ratio = reference_bbox_height * (reference_distance / actual_distance)
        # actual_distance = reference_bbox_height * reference_distance / bbox_height_ratio

        if bbox_height_ratio > 0.01:  # Valid bbox
            dist_bbox = (self.reference_bbox_height * self.reference_distance) / bbox_height_ratio
        else:
            dist_bbox = self.far_m

        # Depth 기반 거리 추정 (inverse)
        dist_depth = self._inverse_convert(depth_rel)

        # 두 추정을 fusion (weighted average)
        # BBox가 클수록 (가까울수록) bbox 추정 신뢰도 높음
        bbox_weight = min(1.0, bbox_height_ratio / self.reference_bbox_height)
        depth_weight = 1.0 - bbox_weight * 0.5  # depth도 어느정도 신뢰

        total_weight = bbox_weight + depth_weight
        dist_fused = (bbox_weight * dist_bbox + depth_weight * dist_depth) / total_weight

        return float(np.clip(dist_fused, self.near_m, self.far_m))

    def _adaptive_convert(
        self,
        depth_rel: float,
        bbox_height_ratio: Optional[float] = None,
    ) -> float:
        """
        시간에 따라 적응하는 변환.

        아이디어:
        - depth와 bbox 변화의 상관관계로 scale 추정
        - 급격한 변화 감지하여 outlier 제거
        """
        # 우선 bbox_calibrated 결과 사용
        dist = self._bbox_calibrated_convert(depth_rel, bbox_height_ratio)

        # History 관리
        self.depth_history.append(depth_rel)
        self.distance_history.append(dist)

        # 최근 N개만 유지
        max_history = 30
        if len(self.depth_history) > max_history:
            self.depth_history = self.depth_history[-max_history:]
            self.distance_history = self.distance_history[-max_history:]

        # Temporal smoothing (simple exponential)
        if len(self.distance_history) > 1:
            alpha = 0.7  # 0.7 = 현재값 70%, 이전값 30%
            prev_dist = self.distance_history[-2]
            dist = alpha * dist + (1 - alpha) * prev_dist

        return float(np.clip(dist, self.near_m, self.far_m))

    def batch_convert(
        self,
        trajectory: List[Dict],
        frame_height: int,
    ) -> List[Dict]:
        """
        전체 trajectory에 대해 depth → metric 변환.

        Args:
            trajectory: [{"frame": int, "depth_rel": float, "h": int, ...}, ...]
            frame_height: 프레임 높이

        Returns:
            trajectory with "dist_m" added
        """
        result = []

        for frame_data in trajectory:
            depth_rel = frame_data.get("depth_rel", 0.5)
            bbox_h = frame_data.get("h", None)

            bbox_ratio = bbox_h / frame_height if bbox_h else None

            dist_m = self.convert(
                depth_rel=depth_rel,
                bbox_height_ratio=bbox_ratio,
            )

            new_frame = frame_data.copy()
            new_frame["dist_m"] = dist_m
            result.append(new_frame)

        return result


def estimate_scale_from_trajectory(
    trajectory: List[Dict],
    known_object_height_m: float = 1.7,  # 사람 키 기준
    frame_height: int = 1080,
) -> float:
    """
    Trajectory에서 scale factor 추정.

    Args:
        trajectory: tracking 결과
        known_object_height_m: 알려진 물체 높이 (사람=1.7m)
        frame_height: 프레임 높이

    Returns:
        scale: depth * scale ≈ meters
    """
    # 가장 가까운 프레임 (bbox가 가장 큰 프레임) 찾기
    max_h = 0
    max_frame = None

    for frame_data in trajectory:
        h = frame_data.get("h", 0)
        if h > max_h:
            max_h = h
            max_frame = frame_data

    if max_frame is None or max_h < 10:
        return 1.0

    # Pinhole camera model:
    # object_height_pixels / focal_length = object_height_m / distance
    # → distance = focal_length * object_height_m / object_height_pixels

    # Assume FOV=60° → focal_length ≈ frame_height / tan(30°) ≈ frame_height * 1.73
    focal_length_approx = frame_height * 1.73

    # 가장 가까운 프레임에서의 거리 추정
    distance_at_max = focal_length_approx * known_object_height_m / max_h

    # 해당 프레임의 depth 값
    depth_at_max = max_frame.get("depth_rel", 0.8)

    # scale factor: depth * scale = distance
    if depth_at_max > 0.1:
        scale = distance_at_max / depth_at_max
    else:
        scale = 1.0

    return float(scale)


def refine_depth_with_bbox(
    depth_trajectory: List[float],
    bbox_heights: List[float],
    frame_height: int,
    reference_distance: float = 2.0,
    reference_bbox_ratio: float = 0.3,
) -> List[float]:
    """
    BBox 정보로 depth trajectory refine.

    Depth model의 temporal inconsistency를 bbox로 보정.

    Args:
        depth_trajectory: relative depth values [0,1]
        bbox_heights: bbox heights in pixels
        frame_height: frame height
        reference_distance: distance at reference_bbox_ratio
        reference_bbox_ratio: bbox_height / frame_height at reference_distance

    Returns:
        refined_distances: metric distances
    """
    refined = []

    for depth, bbox_h in zip(depth_trajectory, bbox_heights):
        # BBox 기반 거리
        bbox_ratio = bbox_h / frame_height if frame_height > 0 else 0
        if bbox_ratio > 0.01:
            dist_bbox = reference_bbox_ratio * reference_distance / bbox_ratio
        else:
            dist_bbox = 10.0  # far

        # Depth 기반 거리 (inverse relationship)
        depth_clamped = max(depth, 0.05)
        dist_depth = 1.0 / depth_clamped  # 단순 inverse

        # Fusion: bbox가 클수록 bbox 신뢰
        bbox_weight = min(1.0, bbox_ratio * 3)  # 0.33 이상이면 weight=1

        dist_fused = bbox_weight * dist_bbox + (1 - bbox_weight) * dist_depth

        # Clamp
        dist_fused = np.clip(dist_fused, 0.5, 10.0)
        refined.append(float(dist_fused))

    return refined


__all__ = [
    "DepthToMetricConverter",
    "DepthCalibration",
    "estimate_scale_from_trajectory",
    "refine_depth_with_bbox",
]
