"""
Visual Room Estimation for Spatial Audio

영상에서 room 특성을 추정하여 적절한 reverb/IR 적용

Methods:
1. Metric depth 기반 room size 추정 (정밀)
2. Semantic segmentation (indoor/outdoor, room type)
3. Visual cues (reflections, materials)
4. Sabine RT60 공식 기반 계산

출력:
- RT60 추정 (Sabine formula based)
- Room dimensions (metric depth based)
- Recommended IR parameters
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import warnings


class RoomType(Enum):
    """Room type classification."""
    OUTDOOR = "outdoor"
    SMALL_ROOM = "small_room"  # bedroom, office
    MEDIUM_ROOM = "medium_room"  # living room, classroom
    LARGE_ROOM = "large_room"  # hall, gym
    CONCERT_HALL = "concert_hall"
    STUDIO = "studio"  # recording studio (dry)
    UNKNOWN = "unknown"


@dataclass
class RoomEstimate:
    """Estimated room parameters."""
    room_type: RoomType = RoomType.UNKNOWN
    dimensions: Tuple[float, float, float] = (6.0, 5.0, 3.0)  # L, W, H in meters
    rt60: float = 0.5  # seconds
    is_outdoor: bool = False
    confidence: float = 0.5

    # Reverb parameters
    direct_ratio: float = 0.7
    early_ratio: float = 0.1
    late_ratio: float = 0.2

    # Additional info
    detected_features: Dict = field(default_factory=dict)


class VisualRoomEstimator:
    """
    Estimate room characteristics from video frames.

    Uses multiple cues:
    - Metric depth → actual room dimensions in meters
    - Scene classification → room type
    - Sabine formula → RT60 calculation
    - Visual features → material/absorption estimation
    """

    # RT60 lookup table by room type (fallback)
    RT60_TABLE = {
        RoomType.OUTDOOR: 0.1,
        RoomType.STUDIO: 0.2,
        RoomType.SMALL_ROOM: 0.4,
        RoomType.MEDIUM_ROOM: 0.6,
        RoomType.LARGE_ROOM: 0.8,
        RoomType.CONCERT_HALL: 1.5,
        RoomType.UNKNOWN: 0.5,
    }

    # Room dimensions by type (L, W, H) - fallback
    DIMENSIONS_TABLE = {
        RoomType.OUTDOOR: (50.0, 50.0, 10.0),
        RoomType.STUDIO: (4.0, 3.0, 2.5),
        RoomType.SMALL_ROOM: (4.0, 4.0, 2.5),
        RoomType.MEDIUM_ROOM: (8.0, 6.0, 3.0),
        RoomType.LARGE_ROOM: (15.0, 10.0, 4.0),
        RoomType.CONCERT_HALL: (30.0, 20.0, 10.0),
        RoomType.UNKNOWN: (6.0, 5.0, 3.0),
    }

    # Average absorption coefficients by material
    ABSORPTION_COEFFICIENTS = {
        "concrete": 0.02,
        "brick": 0.03,
        "plaster": 0.04,
        "glass": 0.03,
        "wood": 0.10,
        "carpet": 0.30,
        "curtain": 0.50,
        "acoustic_panel": 0.80,
        "outdoor": 0.99,  # Almost no reflection
        "default": 0.15,
    }

    def __init__(
        self,
        use_depth: bool = True,
        use_metric_depth: bool = True,
        use_segmentation: bool = False,
        metric_depth_model: Optional[object] = None,
    ):
        """
        Initialize room estimator.

        Args:
            use_depth: Use depth information for estimation
            use_metric_depth: Use metric depth model for precise measurements
            use_segmentation: Use semantic segmentation (requires model)
            metric_depth_model: Pre-loaded metric depth model
        """
        self.use_depth = use_depth
        self.use_metric_depth = use_metric_depth
        self.use_segmentation = use_segmentation
        self.segmentation_model = None
        self.metric_depth_model = metric_depth_model

        # Load metric depth model if needed
        if use_metric_depth and metric_depth_model is None:
            self._load_metric_depth_model()

        if use_segmentation:
            self._load_segmentation_model()

    def _load_metric_depth_model(self):
        """Load metric depth model for precise room measurement."""
        try:
            from .depth_metric import MetricDepthEstimator
            self.metric_depth_model = MetricDepthEstimator(
                scene_type="auto",
                model_size="small",
                device="cuda"
            )
            print("[room_estimator] Loaded metric depth model")
        except Exception as e:
            warnings.warn(f"Failed to load metric depth model: {e}")
            self.metric_depth_model = None

    def _load_segmentation_model(self):
        """Load semantic segmentation model."""
        try:
            # Could use: DeepLabV3, SegFormer, etc.
            import torch
            from torchvision.models.segmentation import deeplabv3_resnet50

            self.segmentation_model = deeplabv3_resnet50(pretrained=True)
            self.segmentation_model.eval()
            print("[room_estimator] Loaded DeepLabV3 for scene segmentation")
        except Exception as e:
            print(f"[room_estimator] Segmentation model failed: {e}")
            self.segmentation_model = None

    def estimate_from_frame(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        fov_deg: float = 60.0,
    ) -> RoomEstimate:
        """
        Estimate room parameters from single frame.

        Args:
            frame: BGR image
            depth_map: Optional depth map (relative [0,1] or metric in meters)
            fov_deg: Camera horizontal FOV

        Returns:
            RoomEstimate with parameters
        """
        features = {}
        is_metric_depth = False
        metric_depth_map = None

        # 1. Basic image analysis
        features.update(self._analyze_image_features(frame))

        # 2. Get metric depth if available
        if self.use_metric_depth and self.metric_depth_model is not None:
            try:
                metric_depth_map = self.metric_depth_model.infer(frame)
                is_metric_depth = True
                features.update(self._analyze_depth(metric_depth_map, is_metric=True))
            except Exception as e:
                warnings.warn(f"Metric depth failed: {e}")

        # 3. Fallback to provided depth map
        if not is_metric_depth and depth_map is not None and self.use_depth:
            features.update(self._analyze_depth(depth_map, is_metric=False))

        # 4. Semantic segmentation
        if self.use_segmentation and self.segmentation_model is not None:
            features.update(self._analyze_segmentation(frame))

        # 5. Determine room type
        room_type, confidence = self._classify_room(features)

        # 6. Calculate precise dimensions if metric depth available
        if is_metric_depth and metric_depth_map is not None and room_type != RoomType.OUTDOOR:
            dimensions = self._estimate_room_dimensions_from_metric_depth(
                metric_depth_map, fov_deg
            )
            # Estimate absorption from visual features
            absorption = self._estimate_absorption(features)
            rt60 = self._calculate_rt60_sabine(dimensions, absorption)
        else:
            # Fallback to lookup table
            dimensions = self.DIMENSIONS_TABLE[room_type]
            rt60 = self.RT60_TABLE[room_type]

        # 7. Build estimate
        estimate = RoomEstimate(
            room_type=room_type,
            dimensions=dimensions,
            rt60=rt60,
            is_outdoor=(room_type == RoomType.OUTDOOR),
            confidence=confidence,
            detected_features=features,
        )

        # Adjust ratios based on room type and dimensions
        estimate = self._adjust_reverb_ratios(estimate)

        return estimate

    def _estimate_absorption(self, features: Dict) -> float:
        """
        Estimate average absorption coefficient from visual features.

        Higher edge density → more furniture → higher absorption
        Higher saturation → less bare walls → higher absorption
        """
        base_absorption = 0.15  # Default

        # Edge density indicates furniture/objects
        edge_density = features.get("edge_density", 0.1)
        if edge_density > 0.2:
            base_absorption += 0.1  # More absorptive materials

        # Outdoor has near-perfect absorption (no reflections)
        if features.get("sky_ratio", 0) > 0.3:
            return 0.95

        # Low brightness might indicate carpeted/soft surfaces
        brightness = features.get("mean_brightness", 128)
        if brightness < 100:
            base_absorption += 0.05

        return np.clip(base_absorption, 0.05, 0.95)

    def estimate_from_video(
        self,
        video_path: str,
        num_samples: int = 5,
        depth_fn: Optional[callable] = None,
    ) -> RoomEstimate:
        """
        Estimate room parameters from video.

        Samples multiple frames and aggregates results.

        Args:
            video_path: Path to video
            num_samples: Number of frames to sample
            depth_fn: Optional depth estimation function

        Returns:
            Aggregated RoomEstimate
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[room_estimator] Failed to open video: {video_path}")
            return RoomEstimate()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return RoomEstimate()

        # Sample frames evenly
        frame_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

        estimates = []

        for fidx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Get depth if available
            depth_map = None
            if depth_fn is not None:
                try:
                    depth_map = depth_fn(frame)
                except Exception:
                    pass

            estimate = self.estimate_from_frame(frame, depth_map)
            estimates.append(estimate)

        cap.release()

        if not estimates:
            return RoomEstimate()

        # Aggregate estimates
        return self._aggregate_estimates(estimates)

    def _analyze_image_features(self, frame: np.ndarray) -> Dict:
        """Analyze basic image features."""
        features = {}

        # Convert to HSV for analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Brightness
        features["mean_brightness"] = float(np.mean(v))
        features["brightness_std"] = float(np.std(v))

        # Color saturation
        features["mean_saturation"] = float(np.mean(s))

        # Edge density (proxy for clutter/texture)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        features["edge_density"] = float(np.mean(edges) / 255.0)

        # Sky detection (outdoor indicator)
        # Top 20% of image, look for high brightness + low saturation
        top_region = frame[:frame.shape[0]//5, :, :]
        top_hsv = cv2.cvtColor(top_region, cv2.COLOR_BGR2HSV)
        top_s = top_hsv[:, :, 1]
        top_v = top_hsv[:, :, 2]

        # Sky typically has high brightness, low saturation
        sky_mask = (top_v > 150) & (top_s < 80)
        features["sky_ratio"] = float(np.mean(sky_mask))

        # Green ratio (outdoor vegetation)
        # Green in HSV: H ≈ 35-85
        green_mask = (hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85) & (hsv[:, :, 1] > 50)
        features["green_ratio"] = float(np.mean(green_mask))

        return features

    def _analyze_depth(self, depth_map: np.ndarray, is_metric: bool = False) -> Dict:
        """
        Analyze depth distribution.

        Args:
            depth_map: Depth values (metric=meters, relative=[0,1])
            is_metric: True if depth is in meters
        """
        features = {}

        # Depth statistics
        features["depth_mean"] = float(np.mean(depth_map))
        features["depth_std"] = float(np.std(depth_map))
        features["depth_range"] = float(np.max(depth_map) - np.min(depth_map))
        features["depth_max"] = float(np.max(depth_map))
        features["depth_min"] = float(np.min(depth_map))
        features["is_metric"] = is_metric

        if is_metric:
            # Metric depth analysis - can estimate actual room dimensions
            features["estimated_room_depth"] = float(np.percentile(depth_map, 95))

            # Estimate room width from depth variance at different y levels
            h, w = depth_map.shape
            center_band = depth_map[h//3:2*h//3, :]
            features["estimated_room_width"] = float(np.std(center_band) * 4)  # Heuristic

            # Floor detection (bottom of image usually has consistent depth)
            floor_region = depth_map[int(h*0.8):, :]
            features["floor_depth_mean"] = float(np.mean(floor_region))
            features["floor_depth_std"] = float(np.std(floor_region))

            # Wall detection (sides of image)
            left_wall = depth_map[:, :w//5]
            right_wall = depth_map[:, -w//5:]
            features["left_wall_depth"] = float(np.median(left_wall))
            features["right_wall_depth"] = float(np.median(right_wall))

        else:
            # Relative depth analysis
            hist, _ = np.histogram(depth_map.flatten(), bins=10, range=(0, 1))
            hist = hist / (hist.sum() + 1e-8)

            # Far depth ratio (outdoor indicator)
            features["far_depth_ratio"] = float(hist[-3:].sum())

            # Near depth ratio (close objects)
            features["near_depth_ratio"] = float(hist[:3].sum())

        # Depth variance in center region
        h, w = depth_map.shape
        center = depth_map[h//4:3*h//4, w//4:3*w//4]
        features["center_depth_std"] = float(np.std(center))

        return features

    def _estimate_room_dimensions_from_metric_depth(
        self,
        depth_map: np.ndarray,
        fov_deg: float = 60.0,
    ) -> Tuple[float, float, float]:
        """
        Estimate room dimensions from metric depth map.

        Uses geometric projection based on FOV.

        Args:
            depth_map: Depth in meters (H, W)
            fov_deg: Camera horizontal FOV in degrees

        Returns:
            (length, width, height) in meters
        """
        h, w = depth_map.shape

        # Estimate room depth (distance to back wall)
        # Use 90th percentile to avoid outliers
        room_depth = float(np.percentile(depth_map, 90))

        # Estimate room width using FOV geometry
        # width = 2 * depth * tan(fov/2)
        fov_rad = np.radians(fov_deg)
        visible_width_at_depth = 2 * room_depth * np.tan(fov_rad / 2)

        # The actual room might be wider than visible
        # Estimate based on depth variation at edges
        left_depth = np.median(depth_map[:, :w//10])
        right_depth = np.median(depth_map[:, -w//10:])

        # If edges show different depth, room extends beyond view
        edge_ratio = max(left_depth, right_depth) / (min(left_depth, right_depth) + 0.1)
        room_width = visible_width_at_depth * min(edge_ratio, 2.0)

        # Estimate height (assume standard proportions if not visible)
        # Typical room height = 2.5-3.5m
        estimated_height = min(3.5, max(2.5, room_depth * 0.4))

        # Clamp to reasonable values
        room_depth = np.clip(room_depth, 2.0, 50.0)
        room_width = np.clip(room_width, 2.0, 30.0)
        estimated_height = np.clip(estimated_height, 2.0, 10.0)

        return float(room_depth), float(room_width), float(estimated_height)

    def _calculate_rt60_sabine(
        self,
        dimensions: Tuple[float, float, float],
        absorption_coeff: float = 0.15,
    ) -> float:
        """
        Calculate RT60 using Sabine formula.

        RT60 = 0.161 * V / A

        where:
        - V = room volume (m³)
        - A = total absorption (m² Sabins) = sum(surface_area * absorption_coeff)

        Args:
            dimensions: (length, width, height) in meters
            absorption_coeff: Average absorption coefficient

        Returns:
            RT60 in seconds
        """
        L, W, H = dimensions

        # Room volume
        V = L * W * H

        # Surface areas
        floor_ceiling = 2 * L * W
        front_back = 2 * W * H
        left_right = 2 * L * H
        total_surface = floor_ceiling + front_back + left_right

        # Total absorption
        A = total_surface * absorption_coeff

        # Sabine formula
        if A > 0:
            rt60 = 0.161 * V / A
        else:
            rt60 = 0.5  # Fallback

        # Clamp to reasonable values
        rt60 = np.clip(rt60, 0.1, 3.0)

        return float(rt60)

    def _analyze_segmentation(self, frame: np.ndarray) -> Dict:
        """Analyze semantic segmentation."""
        features = {}

        if self.segmentation_model is None:
            return features

        try:
            import torch
            from torchvision import transforms

            # Preprocess
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
            ])

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(rgb).unsqueeze(0)

            with torch.no_grad():
                output = self.segmentation_model(input_tensor)["out"]
                pred = output.argmax(1).squeeze().numpy()

            # Count categories
            # COCO categories: 0=background, 13=bench, 14=bird, ...
            # Relevant: sky, grass, floor, wall, ceiling, etc.
            unique, counts = np.unique(pred, return_counts=True)
            total = pred.size

            category_ratios = {}
            for u, c in zip(unique, counts):
                category_ratios[int(u)] = float(c / total)

            features["segmentation_categories"] = category_ratios

        except Exception as e:
            print(f"[room_estimator] Segmentation failed: {e}")

        return features

    def _classify_room(self, features: Dict) -> Tuple[RoomType, float]:
        """
        Classify room type based on features.

        Returns (room_type, confidence)
        """
        scores = {rt: 0.0 for rt in RoomType}

        # Outdoor detection
        sky_ratio = features.get("sky_ratio", 0)
        green_ratio = features.get("green_ratio", 0)
        far_depth = features.get("far_depth_ratio", 0)

        outdoor_score = sky_ratio * 0.4 + green_ratio * 0.3 + far_depth * 0.3
        if outdoor_score > 0.3:
            scores[RoomType.OUTDOOR] = outdoor_score

        # Indoor room size from depth
        depth_std = features.get("depth_std", 0.3)
        depth_range = features.get("depth_range", 0.5)

        # High depth variance → larger room
        if depth_std > 0.3:
            scores[RoomType.LARGE_ROOM] += depth_std
        elif depth_std > 0.2:
            scores[RoomType.MEDIUM_ROOM] += 0.5
        else:
            scores[RoomType.SMALL_ROOM] += 0.5 - depth_std

        # Edge density → clutter → smaller/furnished room
        edge_density = features.get("edge_density", 0.1)
        if edge_density > 0.15:
            scores[RoomType.SMALL_ROOM] += 0.2
            scores[RoomType.MEDIUM_ROOM] += 0.1

        # Brightness → studio often well-lit, uniform
        brightness_std = features.get("brightness_std", 50)
        if brightness_std < 30:
            scores[RoomType.STUDIO] += 0.3

        # Get best match
        best_type = max(scores, key=scores.get)
        confidence = min(1.0, scores[best_type])

        # Default to UNKNOWN if low confidence
        if confidence < 0.2:
            best_type = RoomType.UNKNOWN
            confidence = 0.5

        return best_type, confidence

    def _adjust_reverb_ratios(self, estimate: RoomEstimate) -> RoomEstimate:
        """Adjust reverb ratios based on room type."""
        ratios = {
            RoomType.OUTDOOR: (0.9, 0.05, 0.05),  # Almost all direct
            RoomType.STUDIO: (0.85, 0.1, 0.05),  # Very dry
            RoomType.SMALL_ROOM: (0.7, 0.15, 0.15),
            RoomType.MEDIUM_ROOM: (0.6, 0.2, 0.2),
            RoomType.LARGE_ROOM: (0.5, 0.2, 0.3),
            RoomType.CONCERT_HALL: (0.4, 0.2, 0.4),
            RoomType.UNKNOWN: (0.7, 0.1, 0.2),
        }

        direct, early, late = ratios.get(estimate.room_type, (0.7, 0.1, 0.2))
        estimate.direct_ratio = direct
        estimate.early_ratio = early
        estimate.late_ratio = late

        return estimate

    def _aggregate_estimates(self, estimates: List[RoomEstimate]) -> RoomEstimate:
        """Aggregate multiple frame estimates."""
        # Majority vote for room type
        type_counts = {}
        for e in estimates:
            rt = e.room_type
            type_counts[rt] = type_counts.get(rt, 0) + 1

        best_type = max(type_counts, key=type_counts.get)

        # Average RT60 and confidence
        avg_rt60 = np.mean([e.rt60 for e in estimates])
        avg_conf = np.mean([e.confidence for e in estimates])

        # Average ratios
        avg_direct = np.mean([e.direct_ratio for e in estimates])
        avg_early = np.mean([e.early_ratio for e in estimates])
        avg_late = np.mean([e.late_ratio for e in estimates])

        return RoomEstimate(
            room_type=best_type,
            dimensions=self.DIMENSIONS_TABLE[best_type],
            rt60=float(avg_rt60),
            is_outdoor=(best_type == RoomType.OUTDOOR),
            confidence=float(avg_conf),
            direct_ratio=float(avg_direct),
            early_ratio=float(avg_early),
            late_ratio=float(avg_late),
        )


def estimate_room_parameters(
    video_path: str,
    depth_fn: Optional[callable] = None,
    num_samples: int = 5,
) -> Dict:
    """
    Convenience function to estimate room parameters.

    Args:
        video_path: Path to video
        depth_fn: Optional depth estimation function
        num_samples: Number of frames to sample

    Returns:
        Dict with room parameters ready for IR generation
    """
    estimator = VisualRoomEstimator(use_depth=(depth_fn is not None))
    estimate = estimator.estimate_from_video(
        video_path,
        num_samples=num_samples,
        depth_fn=depth_fn,
    )

    return {
        "room_type": estimate.room_type.value,
        "dimensions": estimate.dimensions,
        "rt60": estimate.rt60,
        "is_outdoor": estimate.is_outdoor,
        "confidence": estimate.confidence,
        "ir_params": {
            "direct_ratio": estimate.direct_ratio,
            "early_ratio": estimate.early_ratio,
            "late_ratio": estimate.late_ratio,
        }
    }


__all__ = [
    "RoomType",
    "RoomEstimate",
    "VisualRoomEstimator",
    "estimate_room_parameters",
]
