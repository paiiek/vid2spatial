"""
Text-Guided Video Object Tracking Pipeline.

Complete pipeline for tracking objects in video using natural language:
1. Text prompt -> Object selection (Grounding-DINO + SAM)
2. Initial mask -> Mask propagation (SAM2/XMem/OpticalFlow)
3. Mask sequence -> 3D trajectory (Metric Depth)

Usage:
    tracker = TextGuidedTracker()
    result = tracker.track_object(
        video_path="video.mp4",
        text_prompt="the red ball",
    )
    trajectory_3d = result.to_trajectory_3d()
"""

import os
import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TextGuidedResult:
    """Complete text-guided tracking result."""
    trajectory_2d: List[Dict]  # [{"frame", "cx", "cy", "w", "h", "mask", ...}]
    initial_selection: Any  # SelectionResult from text_object_selector
    video_info: Dict  # {"width", "height", "fps", "total_frames"}
    text_prompt: str

    def to_trajectory_3d(
        self,
        depth_fn: Optional[Any] = None,
        fov_deg: float = 60.0,
        depth_scale_m: Tuple[float, float] = (0.5, 10.0),
    ) -> Dict:
        """
        Convert to 3D trajectory with depth estimation.

        Args:
            depth_fn: Optional custom depth function
            fov_deg: Camera FOV in degrees
            depth_scale_m: (near, far) depth range in meters

        Returns:
            Dict with "intrinsics" and "frames"
        """
        from .vision import CameraIntrinsics, pixel_to_ray, ray_to_angles

        W = self.video_info["width"]
        H = self.video_info["height"]
        K = CameraIntrinsics(width=W, height=H, fov_deg=fov_deg)

        frames_3d = []

        # Load depth estimator if not provided
        if depth_fn is None:
            try:
                from .depth_metric import MetricDepthEstimator
                depth_estimator = MetricDepthEstimator(
                    scene_type="auto",
                    model_size="small",
                    device="cuda"
                )
                depth_fn = depth_estimator.infer
                use_metric = True
            except Exception:
                use_metric = False
                depth_fn = None

        # Process each frame
        cap = None
        for rec in self.trajectory_2d:
            frame_idx = rec["frame"]
            cx, cy = rec["cx"], rec["cy"]

            # Get depth
            if depth_fn is not None:
                # Read frame for depth estimation
                if cap is None:
                    video_path = self.video_info.get("video_path")
                    if video_path:
                        cap = cv2.VideoCapture(video_path)

                if cap is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        depth_map = depth_fn(frame)

                        # Get depth at object location
                        if "mask" in rec and rec["mask"] is not None:
                            mask = rec["mask"]
                            ys, xs = np.nonzero(mask)
                            if len(xs) > 0:
                                depth_val = np.median(depth_map[ys, xs])
                            else:
                                depth_val = depth_map[int(cy), int(cx)]
                        else:
                            depth_val = depth_map[int(cy), int(cx)]

                        # If metric depth, use directly; otherwise scale
                        if hasattr(depth_fn, '__self__') and hasattr(depth_fn.__self__, 'max_depth'):
                            dist_m = float(depth_val)
                        else:
                            # Relative depth [0,1] -> scale
                            near, far = depth_scale_m
                            dist_m = near + (1.0 - depth_val) * (far - near)
                    else:
                        dist_m = (depth_scale_m[0] + depth_scale_m[1]) / 2
                else:
                    dist_m = (depth_scale_m[0] + depth_scale_m[1]) / 2
            else:
                dist_m = (depth_scale_m[0] + depth_scale_m[1]) / 2

            # Compute 3D position
            ray = pixel_to_ray(cx, cy, K)
            az, el = ray_to_angles(ray)

            pos = ray * dist_m
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

            frames_3d.append({
                "frame": frame_idx,
                "az": float(az),
                "el": float(el),
                "dist_m": float(dist_m),
                "x": x,
                "y": y,
                "z": z,
            })

        if cap is not None:
            cap.release()

        return {
            "intrinsics": {"width": W, "height": H, "fov_deg": fov_deg},
            "frames": frames_3d,
        }


class TextGuidedTracker:
    """
    Complete text-guided video object tracking pipeline.

    Combines:
    - TextObjectSelector: Text -> Initial mask
    - VideoObjectTracker: Initial mask -> Mask sequence
    - MetricDepthEstimator: Mask sequence -> 3D trajectory
    """

    def __init__(
        self,
        device: str = "cuda",
        vos_method: str = "auto",
        sam_model_type: str = "vit_b",
        use_metric_depth: bool = True,
    ):
        """
        Initialize Text-Guided Tracker.

        Args:
            device: "cuda" or "cpu"
            vos_method: Video segmentation method ("auto", "sam2", "xmem", "optical_flow")
            sam_model_type: SAM model type ("vit_h", "vit_l", "vit_b")
            use_metric_depth: Use metric depth for 3D estimation
        """
        self.device = device
        self.vos_method = vos_method
        self.sam_model_type = sam_model_type
        self.use_metric_depth = use_metric_depth

        # Lazy initialization
        self._text_selector = None
        self._video_tracker = None
        self._depth_estimator = None

    @property
    def text_selector(self):
        """Lazy load text object selector."""
        if self._text_selector is None:
            from .text_object_selector import TextObjectSelector
            self._text_selector = TextObjectSelector(
                device=self.device,
                sam_model_type=self.sam_model_type,
            )
        return self._text_selector

    @property
    def video_tracker(self):
        """Lazy load video object tracker."""
        if self._video_tracker is None:
            from .video_object_tracker import VideoObjectTracker
            self._video_tracker = VideoObjectTracker(
                device=self.device,
                method=self.vos_method,
                sam_model_type=self.sam_model_type,
            )
        return self._video_tracker

    @property
    def depth_estimator(self):
        """Lazy load depth estimator."""
        if self._depth_estimator is None and self.use_metric_depth:
            try:
                from .depth_metric import MetricDepthEstimator
                self._depth_estimator = MetricDepthEstimator(
                    scene_type="auto",
                    model_size="small",
                    device=self.device,
                )
            except Exception as e:
                print(f"[text_guided] Failed to load metric depth: {e}")
                self._depth_estimator = None
        return self._depth_estimator

    def track_object(
        self,
        video_path: str,
        text_prompt: str,
        sample_stride: int = 1,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> TextGuidedResult:
        """
        Track object in video using text prompt.

        Args:
            video_path: Path to input video
            text_prompt: Natural language description (e.g., "the red ball")
            sample_stride: Process every Nth frame
            start_frame: Starting frame index
            end_frame: Ending frame index (None = end)

        Returns:
            TextGuidedResult with trajectory and masks
        """
        # Get video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Read first frame for object selection
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, first_frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("Failed to read first frame")

        video_info = {
            "width": W,
            "height": H,
            "fps": fps,
            "total_frames": total_frames,
            "video_path": video_path,
        }

        # Step 1: Select object with text prompt
        print(f"[text_guided] Selecting object: '{text_prompt}'")
        selection = self.text_selector.select(text_prompt, first_frame)

        if selection is None:
            raise RuntimeError(f"Could not find object matching: '{text_prompt}'")

        print(f"[text_guided] Found: {selection.label} (conf={selection.confidence:.2f})")

        # Step 2: Track through video
        print(f"[text_guided] Tracking through video...")
        tracking_result = self.video_tracker.track(
            video_path=video_path,
            initial_mask=selection.mask,
            sample_stride=sample_stride,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        # Convert to trajectory format
        trajectory_2d = tracking_result.get_trajectory()

        print(f"[text_guided] Tracked {len(trajectory_2d)} frames")

        return TextGuidedResult(
            trajectory_2d=trajectory_2d,
            initial_selection=selection,
            video_info=video_info,
            text_prompt=text_prompt,
        )

    def track_with_click(
        self,
        video_path: str,
        click_point: Tuple[int, int],
        sample_stride: int = 1,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> TextGuidedResult:
        """
        Track object in video using point click.

        Args:
            video_path: Path to video
            click_point: (x, y) click coordinates on first frame
            sample_stride: Frame sampling stride
            start_frame: Starting frame
            end_frame: Ending frame

        Returns:
            TextGuidedResult
        """
        # Get video info and first frame
        cap = cv2.VideoCapture(video_path)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, first_frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("Failed to read first frame")

        video_info = {
            "width": W,
            "height": H,
            "fps": fps,
            "total_frames": total_frames,
            "video_path": video_path,
        }

        # Select with point click
        selection = self.text_selector.select_with_point(first_frame, click_point)

        if selection is None:
            raise RuntimeError(f"Could not select object at point {click_point}")

        # Track through video
        tracking_result = self.video_tracker.track(
            video_path=video_path,
            initial_mask=selection.mask,
            sample_stride=sample_stride,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        trajectory_2d = tracking_result.get_trajectory()

        return TextGuidedResult(
            trajectory_2d=trajectory_2d,
            initial_selection=selection,
            video_info=video_info,
            text_prompt="click_selection",
        )

    def track_with_bbox(
        self,
        video_path: str,
        bbox: Tuple[int, int, int, int],
        sample_stride: int = 1,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> TextGuidedResult:
        """
        Track object in video using bounding box.

        Args:
            video_path: Path to video
            bbox: (x, y, w, h) bounding box on first frame
            sample_stride: Frame sampling stride
            start_frame: Starting frame
            end_frame: Ending frame

        Returns:
            TextGuidedResult
        """
        # Get video info
        cap = cv2.VideoCapture(video_path)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        video_info = {
            "width": W,
            "height": H,
            "fps": fps,
            "total_frames": total_frames,
            "video_path": video_path,
        }

        # Track with bbox
        tracking_result = self.video_tracker.track_from_bbox(
            video_path=video_path,
            initial_bbox=bbox,
            sample_stride=sample_stride,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        trajectory_2d = tracking_result.get_trajectory()

        # Create dummy selection result
        from .text_object_selector import SelectionResult
        x, y, w, h = bbox
        initial_selection = SelectionResult(
            mask=np.zeros((H, W), dtype=np.uint8),
            bbox=bbox,
            confidence=1.0,
            label="bbox_selection",
            center=(x + w/2, y + h/2),
        )

        return TextGuidedResult(
            trajectory_2d=trajectory_2d,
            initial_selection=initial_selection,
            video_info=video_info,
            text_prompt="bbox_selection",
        )


def track_with_text(
    video_path: str,
    text_prompt: str,
    sample_stride: int = 1,
    device: str = "cuda",
    fov_deg: float = 60.0,
    return_3d: bool = True,
) -> Dict:
    """
    Convenience function for text-guided tracking.

    Args:
        video_path: Path to video
        text_prompt: Object description (e.g., "the person in red")
        sample_stride: Frame sampling stride
        device: "cuda" or "cpu"
        fov_deg: Camera FOV for 3D estimation
        return_3d: Return 3D trajectory

    Returns:
        Dict with trajectory data
    """
    tracker = TextGuidedTracker(device=device)
    result = tracker.track_object(video_path, text_prompt, sample_stride)

    if return_3d:
        return result.to_trajectory_3d(fov_deg=fov_deg)
    else:
        return {
            "intrinsics": {
                "width": result.video_info["width"],
                "height": result.video_info["height"],
                "fov_deg": fov_deg,
            },
            "frames": [
                {
                    "frame": r["frame"],
                    "cx": r["cx"],
                    "cy": r["cy"],
                    "w": r["w"],
                    "h": r["h"],
                }
                for r in result.trajectory_2d
            ],
        }


__all__ = [
    "TextGuidedTracker",
    "TextGuidedResult",
    "track_with_text",
]
