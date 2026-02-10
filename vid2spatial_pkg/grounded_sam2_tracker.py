"""
Grounded SAM2 Video Object Tracker.

State-of-the-art pipeline:
1. Grounding-DINO: Text prompt -> Bounding boxes
2. SAM2: Box -> Initial segmentation mask
3. SAM2 Video Predictor: Mask propagation through video

This is the recommended pipeline for text-guided video object tracking.

Usage:
    tracker = GroundedSAM2Tracker()
    result = tracker.track(
        video_path="video.mp4",
        text_prompt="the person in red shirt"
    )
"""

import os
import sys
import numpy as np
import cv2
import torch
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import shutil


@dataclass
class DetectionResult:
    """Single detection result from Grounding-DINO."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) format
    confidence: float
    label: str


@dataclass
class TrackingFrame:
    """Single frame tracking result."""
    frame_idx: int
    mask: np.ndarray  # Binary mask (H, W)
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[float, float]  # (cx, cy)
    confidence: float
    area: int


@dataclass
class TrackingResult:
    """Complete video tracking result."""
    frames: List[TrackingFrame]
    video_width: int
    video_height: int
    fps: float
    total_frames: int
    text_prompt: str
    initial_detection: Optional[DetectionResult] = None

    def get_trajectory(self) -> List[Dict]:
        """Convert to vid2spatial trajectory format."""
        return [
            {
                "frame": f.frame_idx,
                "cx": f.center[0],
                "cy": f.center[1],
                "w": f.bbox[2],
                "h": f.bbox[3],
                "x": f.bbox[0],
                "y": f.bbox[1],
                "confidence": f.confidence,
            }
            for f in self.frames
        ]

    def get_masks(self) -> Dict[int, np.ndarray]:
        """Get frame_idx -> mask mapping."""
        return {f.frame_idx: f.mask for f in self.frames}


class GroundedSAM2Tracker:
    """
    Grounded SAM2 Video Object Tracker.

    Uses Grounding-DINO for text-based detection and
    SAM2 Video Predictor for mask propagation.
    """

    def __init__(
        self,
        device: str = "cuda",
        grounding_dino_model: str = "IDEA-Research/grounding-dino-tiny",
        sam2_model: str = "facebook/sam2-hiera-large",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ):
        """
        Initialize Grounded SAM2 Tracker.

        Args:
            device: "cuda" or "cpu"
            grounding_dino_model: Grounding-DINO model name/path
            sam2_model: SAM2 model name/path
            box_threshold: Detection confidence threshold
            text_threshold: Text matching threshold
        """
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Model references
        self.grounding_dino = None
        self.grounding_processor = None
        self.sam2_predictor = None
        self.sam2_image_predictor = None

        # Load models
        self._load_grounding_dino(grounding_dino_model)
        self._load_sam2(sam2_model)

    def _load_grounding_dino(self, model_name: str):
        """Load Grounding-DINO model."""
        try:
            from groundingdino.util.inference import load_model, predict
            from groundingdino.util import box_ops

            # Store functions for inference
            self._gd_predict = predict
            self._gd_box_ops = box_ops

            # Try loading from HuggingFace or local path
            # groundingdino-py uses a different loading mechanism
            print(f"[grounded_sam2] Loading Grounding-DINO...")

            # Check for local weights
            weights_dir = os.path.expanduser("~/.cache/groundingdino")
            config_path = os.path.join(weights_dir, "GroundingDINO_SwinT_OGC.py")
            weights_path = os.path.join(weights_dir, "groundingdino_swint_ogc.pth")

            if not os.path.exists(weights_path):
                os.makedirs(weights_dir, exist_ok=True)
                print("[grounded_sam2] Downloading Grounding-DINO weights...")
                import urllib.request
                url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
                urllib.request.urlretrieve(url, weights_path)
                print(f"[grounded_sam2] Saved to {weights_path}")

            if not os.path.exists(config_path):
                # Download config
                config_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
                urllib.request.urlretrieve(config_url, config_path)

            self.grounding_dino = load_model(config_path, weights_path, device=self.device)
            print("[grounded_sam2] Loaded Grounding-DINO (SwinT)")

        except Exception as e:
            print(f"[grounded_sam2] Failed to load Grounding-DINO: {e}")
            import traceback
            traceback.print_exc()
            self.grounding_dino = None

    def _load_sam2(self, model_name: str):
        """Load SAM2 models."""
        try:
            from sam2.sam2_video_predictor import SAM2VideoPredictor
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            print(f"[grounded_sam2] Loading SAM2...")

            # Load video predictor
            self.sam2_predictor = SAM2VideoPredictor.from_pretrained(model_name)

            # Load image predictor for initial segmentation
            self.sam2_image_predictor = SAM2ImagePredictor.from_pretrained(model_name)

            print(f"[grounded_sam2] Loaded SAM2 ({model_name})")

        except Exception as e:
            print(f"[grounded_sam2] Failed to load SAM2: {e}")
            import traceback
            traceback.print_exc()
            self.sam2_predictor = None
            self.sam2_image_predictor = None

    def detect_objects(
        self,
        image: np.ndarray,
        text_prompt: str,
    ) -> List[DetectionResult]:
        """
        Detect objects matching text prompt using Grounding-DINO.

        Args:
            image: BGR image (H, W, 3)
            text_prompt: Text description (e.g., "person . dog . ball")

        Returns:
            List of DetectionResult
        """
        if self.grounding_dino is None:
            return []

        from groundingdino.util.inference import load_image, predict
        import tempfile

        # Save image temporarily
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, image)
            temp_path = f.name

        try:
            # Load for groundingdino
            image_source, image_tensor = load_image(temp_path)

            # Ensure text prompt ends with period
            if not text_prompt.endswith("."):
                text_prompt = text_prompt + "."

            # Predict
            boxes, logits, phrases = predict(
                model=self.grounding_dino,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )

        finally:
            os.unlink(temp_path)

        if len(boxes) == 0:
            return []

        H, W = image.shape[:2]
        results = []

        # Convert normalized boxes to pixel coordinates
        boxes_pixel = boxes.cpu().numpy() * np.array([W, H, W, H])

        for box, conf, phrase in zip(boxes_pixel, logits.cpu().numpy(), phrases):
            x1, y1, x2, y2 = map(int, box)
            results.append(DetectionResult(
                bbox=(x1, y1, x2, y2),
                confidence=float(conf),
                label=phrase,
            ))

        return results

    def segment_with_box(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Segment object using SAM2 with bounding box prompt.

        Args:
            image: BGR image
            bbox: (x1, y1, x2, y2) bounding box

        Returns:
            Binary mask (H, W)
        """
        if self.sam2_image_predictor is None:
            # Fallback to rectangular mask
            H, W = image.shape[:2]
            mask = np.zeros((H, W), dtype=np.uint8)
            x1, y1, x2, y2 = bbox
            mask[y1:y2, x1:x2] = 1
            return mask

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with torch.inference_mode():
            self.sam2_image_predictor.set_image(image_rgb)

            box = np.array([bbox])  # (1, 4)

            masks, scores, _ = self.sam2_image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=True,
            )

        # Return best mask
        best_idx = scores.argmax()
        return masks[best_idx].astype(np.uint8)

    def track(
        self,
        video_path: str,
        text_prompt: str,
        sample_stride: int = 1,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        select_largest: bool = True,
    ) -> TrackingResult:
        """
        Track object in video using text prompt.

        Args:
            video_path: Path to input video
            text_prompt: Object description (e.g., "the person in red")
            sample_stride: Process every Nth frame
            start_frame: Starting frame index
            end_frame: Ending frame index (None = end of video)
            select_largest: If multiple detections, select the largest one

        Returns:
            TrackingResult with mask sequence
        """
        # Get video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if end_frame is None:
            end_frame = total_frames

        # Read first frame for detection
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, first_frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("Failed to read first frame")

        # Step 1: Detect object with Grounding-DINO
        print(f"[grounded_sam2] Detecting: '{text_prompt}'")
        detections = self.detect_objects(first_frame, text_prompt)

        if not detections:
            raise RuntimeError(f"No object found matching: '{text_prompt}'")

        # Select detection (largest or highest confidence)
        if select_largest and len(detections) > 1:
            # Select by area
            def get_area(det):
                x1, y1, x2, y2 = det.bbox
                return (x2 - x1) * (y2 - y1)
            detection = max(detections, key=get_area)
        else:
            detection = max(detections, key=lambda d: d.confidence)

        print(f"[grounded_sam2] Found: {detection.label} (conf={detection.confidence:.2f})")

        # Step 2: Get initial mask with SAM2
        initial_mask = self.segment_with_box(first_frame, detection.bbox)

        # Step 3: Propagate through video with SAM2 Video Predictor
        print(f"[grounded_sam2] Propagating mask through video...")
        frames = self._propagate_mask(
            video_path, initial_mask, start_frame, end_frame, sample_stride
        )

        return TrackingResult(
            frames=frames,
            video_width=W,
            video_height=H,
            fps=fps,
            total_frames=total_frames,
            text_prompt=text_prompt,
            initial_detection=detection,
        )

    def _propagate_mask(
        self,
        video_path: str,
        initial_mask: np.ndarray,
        start_frame: int,
        end_frame: int,
        sample_stride: int,
    ) -> List[TrackingFrame]:
        """Propagate mask through video using SAM2 Video Predictor."""

        if self.sam2_predictor is None:
            # Fallback to optical flow tracking
            return self._propagate_optical_flow(
                video_path, initial_mask, start_frame, end_frame, sample_stride
            )

        # SAM2 Video Predictor needs video as frame directory
        # Create temp directory with frames
        temp_dir = tempfile.mkdtemp(prefix="sam2_video_")

        try:
            # Extract frames
            cap = cv2.VideoCapture(video_path)
            frame_paths = []
            fidx = 0

            while fidx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                if fidx >= start_frame:
                    frame_path = os.path.join(temp_dir, f"{fidx:06d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)

                fidx += 1

            cap.release()

            # Initialize SAM2 video predictor
            with torch.inference_mode():
                inference_state = self.sam2_predictor.init_state(video_path=temp_dir)

                # Add initial mask as prompt (frame 0 relative to extracted frames)
                _, out_obj_ids, out_mask_logits = self.sam2_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=1,
                    mask=initial_mask,
                )

                # Propagate through video
                frames = []

                for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_predictor.propagate_in_video(inference_state):
                    actual_frame_idx = start_frame + out_frame_idx

                    if (out_frame_idx % sample_stride) != 0:
                        continue

                    # Get mask - handle different tensor shapes
                    mask_tensor = out_mask_logits[0]
                    if mask_tensor.dim() == 3:
                        mask_tensor = mask_tensor[0]  # Remove channel dim if present
                    mask = (mask_tensor > 0.0).cpu().numpy().astype(np.uint8)

                    # Calculate bbox and center
                    tracking_frame = self._mask_to_tracking_frame(mask, actual_frame_idx)
                    if tracking_frame is not None:
                        frames.append(tracking_frame)

            return frames

        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _propagate_optical_flow(
        self,
        video_path: str,
        initial_mask: np.ndarray,
        start_frame: int,
        end_frame: int,
        sample_stride: int,
    ) -> List[TrackingFrame]:
        """Fallback: propagate using optical flow."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        fidx = 0

        prev_gray = None
        prev_mask = initial_mask.copy()

        while fidx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if fidx < start_frame:
                fidx += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if fidx == start_frame:
                mask = initial_mask.copy()
                if (fidx - start_frame) % sample_stride == 0:
                    tracking_frame = self._mask_to_tracking_frame(mask, fidx)
                    if tracking_frame is not None:
                        frames.append(tracking_frame)
                prev_gray = gray
                fidx += 1
                continue

            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            # Warp previous mask
            h, w = gray.shape
            flow_map = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).astype(np.float32)
            flow_map += flow
            warped_mask = cv2.remap(
                prev_mask.astype(np.float32), flow_map[:, :, 0], flow_map[:, :, 1],
                interpolation=cv2.INTER_LINEAR
            )
            mask = (warped_mask > 0.5).astype(np.uint8)

            if (fidx - start_frame) % sample_stride == 0:
                tracking_frame = self._mask_to_tracking_frame(mask, fidx)
                if tracking_frame is not None:
                    frames.append(tracking_frame)

            prev_gray = gray
            prev_mask = mask
            fidx += 1

        cap.release()
        return frames

    def _mask_to_tracking_frame(
        self,
        mask: np.ndarray,
        frame_idx: int,
    ) -> Optional[TrackingFrame]:
        """Convert mask to TrackingFrame."""
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None

        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max() - x), int(ys.max() - y)
        cx, cy = float(xs.mean()), float(ys.mean())
        area = len(xs)

        return TrackingFrame(
            frame_idx=frame_idx,
            mask=mask,
            bbox=(x, y, w, h),
            center=(cx, cy),
            confidence=1.0,
            area=area,
        )

    def track_with_bbox(
        self,
        video_path: str,
        bbox: Tuple[int, int, int, int],
        sample_stride: int = 1,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> TrackingResult:
        """
        Track object using bounding box instead of text.

        Args:
            video_path: Path to video
            bbox: (x, y, w, h) or (x1, y1, x2, y2) bounding box
            sample_stride: Frame sampling stride
            start_frame: Starting frame
            end_frame: Ending frame

        Returns:
            TrackingResult
        """
        # Get video info
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

        if end_frame is None:
            end_frame = total_frames

        # Convert to (x1, y1, x2, y2) if needed
        x, y, w, h = bbox
        if w < W // 2 and h < H // 2:  # Likely (x, y, w, h) format
            bbox_xyxy = (x, y, x + w, y + h)
        else:  # Already (x1, y1, x2, y2)
            bbox_xyxy = bbox

        # Get initial mask
        initial_mask = self.segment_with_box(first_frame, bbox_xyxy)

        # Propagate
        frames = self._propagate_mask(
            video_path, initial_mask, start_frame, end_frame, sample_stride
        )

        return TrackingResult(
            frames=frames,
            video_width=W,
            video_height=H,
            fps=fps,
            total_frames=total_frames,
            text_prompt="bbox_selection",
            initial_detection=DetectionResult(
                bbox=bbox_xyxy,
                confidence=1.0,
                label="manual_bbox",
            ),
        )

    def track_with_point(
        self,
        video_path: str,
        point: Tuple[int, int],
        sample_stride: int = 1,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> TrackingResult:
        """
        Track object using point click.

        Args:
            video_path: Path to video
            point: (x, y) click coordinates
            sample_stride: Frame sampling stride
            start_frame: Starting frame
            end_frame: Ending frame

        Returns:
            TrackingResult
        """
        if self.sam2_image_predictor is None:
            raise RuntimeError("SAM2 not available for point selection")

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

        if end_frame is None:
            end_frame = total_frames

        # Get mask from point
        image_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

        with torch.inference_mode():
            self.sam2_image_predictor.set_image(image_rgb)

            point_coords = np.array([[point[0], point[1]]])
            point_labels = np.array([1])  # Foreground

            masks, scores, _ = self.sam2_image_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

        best_idx = scores.argmax()
        initial_mask = masks[best_idx].astype(np.uint8)

        # Propagate
        frames = self._propagate_mask(
            video_path, initial_mask, start_frame, end_frame, sample_stride
        )

        return TrackingResult(
            frames=frames,
            video_width=W,
            video_height=H,
            fps=fps,
            total_frames=total_frames,
            text_prompt="point_selection",
            initial_detection=None,
        )


def create_tracker(
    device: str = "cuda",
    **kwargs,
) -> GroundedSAM2Tracker:
    """Factory function to create GroundedSAM2Tracker."""
    return GroundedSAM2Tracker(device=device, **kwargs)


__all__ = [
    "GroundedSAM2Tracker",
    "TrackingResult",
    "TrackingFrame",
    "DetectionResult",
    "create_tracker",
]
