"""
Video Object Segmentation (VOS) Tracker.

Propagates initial mask through video frames using:
1. SAM2 Video Predictor (if available)
2. XMem-style memory-based tracking (fallback)
3. Simple optical flow + SAM (basic fallback)

This module enables:
- Initial mask -> Video-wide mask sequence
- Robust tracking through occlusions
- Accurate object boundaries

Usage:
    tracker = VideoObjectTracker()
    mask_sequence = tracker.track(video_path, initial_mask)
"""

import os
import sys
import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path


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

    def get_trajectory(self) -> List[Dict]:
        """Convert to trajectory format for vid2spatial pipeline."""
        return [
            {
                "frame": f.frame_idx,
                "cx": f.center[0],
                "cy": f.center[1],
                "w": f.bbox[2],
                "h": f.bbox[3],
                "x": f.bbox[0],
                "y": f.bbox[1],
                "mask": f.mask,
                "confidence": f.confidence,
            }
            for f in self.frames
        ]


class VideoObjectTracker:
    """
    Video Object Segmentation tracker.

    Propagates initial mask through video using best available method.
    """

    def __init__(
        self,
        device: str = "cuda",
        method: str = "auto",
        sam_checkpoint: Optional[str] = None,
        sam_model_type: str = "vit_b",
    ):
        """
        Initialize Video Object Tracker.

        Args:
            device: "cuda" or "cpu"
            method: "auto", "sam2", "xmem", "optical_flow"
            sam_checkpoint: Path to SAM checkpoint (for optical_flow method)
            sam_model_type: SAM model type
        """
        self.device = device
        self.method = method
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type

        # Backend references
        self.sam2_predictor = None
        self.sam_predictor = None
        self.xmem_model = None

        # Initialize based on method
        if method == "auto":
            self._auto_init()
        elif method == "sam2":
            self._init_sam2()
        elif method == "xmem":
            self._init_xmem()
        elif method == "optical_flow":
            self._init_optical_flow()
        else:
            raise ValueError(f"Unknown tracking method: {method}")

    def _auto_init(self):
        """Auto-initialize with best available method."""
        # Try SAM2 Video Predictor first
        if self._init_sam2():
            self.method = "sam2"
            return

        # Try XMem
        if self._init_xmem():
            self.method = "xmem"
            return

        # Fall back to optical flow + SAM
        self._init_optical_flow()
        self.method = "optical_flow"

    def _init_sam2(self) -> bool:
        """Initialize SAM2 Video Predictor."""
        try:
            from sam2.sam2_video_predictor import SAM2VideoPredictor

            # Try from pretrained
            self.sam2_predictor = SAM2VideoPredictor.from_pretrained(
                "facebook/sam2-hiera-large"
            )
            print("[vos_tracker] Loaded SAM2 Video Predictor")
            return True

        except ImportError:
            print("[vos_tracker] SAM2 not available")
            return False
        except Exception as e:
            print(f"[vos_tracker] SAM2 initialization failed: {e}")
            return False

    def _init_xmem(self) -> bool:
        """Initialize XMem model."""
        try:
            # Check if XMem is installed
            xmem_path = os.path.expanduser("~/XMem")
            if os.path.exists(xmem_path):
                sys.path.insert(0, xmem_path)

            from inference.inference_core import InferenceCore
            from model.network import XMem as XMemModel

            # Load checkpoint
            checkpoint_path = os.path.join(xmem_path, "saves/XMem.pth")
            if not os.path.exists(checkpoint_path):
                print("[vos_tracker] XMem checkpoint not found")
                return False

            import torch
            config = {
                "top_k": 30,
                "mem_every": 5,
                "deep_update_every": -1,
                "enable_long_term": True,
                "enable_long_term_count_usage": True,
                "num_prototypes": 128,
                "min_mid_term_frames": 5,
                "max_mid_term_frames": 10,
                "max_long_term_elements": 10000,
            }

            network = XMemModel(config).to(self.device).eval()
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            network.load_state_dict(checkpoint)

            self.xmem_model = InferenceCore(network, config)
            print("[vos_tracker] Loaded XMem")
            return True

        except ImportError:
            print("[vos_tracker] XMem not available")
            return False
        except Exception as e:
            print(f"[vos_tracker] XMem initialization failed: {e}")
            return False

    def _init_optical_flow(self) -> bool:
        """Initialize optical flow + SAM fallback."""
        try:
            from segment_anything import sam_model_registry, SamPredictor

            # Load SAM
            if self.sam_checkpoint is None:
                checkpoint_dir = os.path.expanduser("~/.cache/sam")
                self.sam_checkpoint = os.path.join(
                    checkpoint_dir, f"sam_{self.sam_model_type}.pth"
                )

            if os.path.exists(self.sam_checkpoint):
                import torch
                sam = sam_model_registry[self.sam_model_type](
                    checkpoint=self.sam_checkpoint
                )
                sam.to(self.device)
                self.sam_predictor = SamPredictor(sam)
                print("[vos_tracker] Loaded SAM for optical flow tracking")
                return True
            else:
                print(f"[vos_tracker] SAM checkpoint not found: {self.sam_checkpoint}")
                # Still return True, will use pure optical flow
                return True

        except ImportError:
            print("[vos_tracker] segment_anything not available, using pure optical flow")
            return True
        except Exception as e:
            print(f"[vos_tracker] SAM initialization failed: {e}")
            return True

    def track(
        self,
        video_path: str,
        initial_mask: np.ndarray,
        sample_stride: int = 1,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> TrackingResult:
        """
        Track object through video given initial mask.

        Args:
            video_path: Path to input video
            initial_mask: Binary mask of object in first frame (H, W)
            sample_stride: Process every Nth frame
            start_frame: Starting frame index
            end_frame: Ending frame index (None = end of video)

        Returns:
            TrackingResult with mask sequence
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if end_frame is None:
            end_frame = total_frames

        # Resize initial mask if needed
        if initial_mask.shape != (H, W):
            initial_mask = cv2.resize(
                initial_mask.astype(np.uint8), (W, H),
                interpolation=cv2.INTER_NEAREST
            )

        # Track using appropriate method
        if self.method == "sam2" and self.sam2_predictor is not None:
            frames = self._track_sam2(
                video_path, initial_mask, sample_stride, start_frame, end_frame
            )
        elif self.method == "xmem" and self.xmem_model is not None:
            frames = self._track_xmem(
                video_path, initial_mask, sample_stride, start_frame, end_frame
            )
        else:
            frames = self._track_optical_flow(
                video_path, initial_mask, sample_stride, start_frame, end_frame
            )

        return TrackingResult(
            frames=frames,
            video_width=W,
            video_height=H,
            fps=fps,
            total_frames=total_frames,
        )

    def _track_sam2(
        self,
        video_path: str,
        initial_mask: np.ndarray,
        sample_stride: int,
        start_frame: int,
        end_frame: int,
    ) -> List[TrackingFrame]:
        """Track using SAM2 Video Predictor."""
        import torch

        # Initialize SAM2 with video
        inference_state = self.sam2_predictor.init_state(video_path=video_path)

        # Add initial mask as prompt
        _, out_obj_ids, out_mask_logits = self.sam2_predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=start_frame,
            obj_id=1,
            mask=initial_mask,
        )

        # Propagate through video
        frames = []
        for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_predictor.propagate_in_video(
            inference_state
        ):
            if out_frame_idx < start_frame or out_frame_idx >= end_frame:
                continue
            if (out_frame_idx - start_frame) % sample_stride != 0:
                continue

            # Get mask
            mask = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)

            # Calculate bbox and center
            tracking_frame = self._mask_to_tracking_frame(mask, out_frame_idx)
            if tracking_frame is not None:
                frames.append(tracking_frame)

        return frames

    def _track_xmem(
        self,
        video_path: str,
        initial_mask: np.ndarray,
        sample_stride: int,
        start_frame: int,
        end_frame: int,
    ) -> List[TrackingFrame]:
        """Track using XMem."""
        import torch

        cap = cv2.VideoCapture(video_path)
        frames = []
        fidx = 0

        self.xmem_model.clear_memory()

        while True:
            ret, frame = cap.read()
            if not ret or fidx >= end_frame:
                break

            if fidx < start_frame:
                fidx += 1
                continue

            # Convert frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
            frame_tensor = frame_tensor.unsqueeze(0).to(self.device)

            if fidx == start_frame:
                # Initialize with first frame and mask
                mask_tensor = torch.from_numpy(initial_mask).long().unsqueeze(0).to(self.device)
                output_mask = self.xmem_model.step(frame_tensor, mask_tensor, first_frame_labels=[1])
            else:
                # Propagate
                output_mask = self.xmem_model.step(frame_tensor)

            if (fidx - start_frame) % sample_stride == 0:
                mask = (output_mask[0] == 1).cpu().numpy().astype(np.uint8)
                tracking_frame = self._mask_to_tracking_frame(mask, fidx)
                if tracking_frame is not None:
                    frames.append(tracking_frame)

            fidx += 1

        cap.release()
        return frames

    def _track_optical_flow(
        self,
        video_path: str,
        initial_mask: np.ndarray,
        sample_stride: int,
        start_frame: int,
        end_frame: int,
    ) -> List[TrackingFrame]:
        """Track using optical flow + SAM refinement."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        fidx = 0

        prev_gray = None
        prev_mask = initial_mask.copy()
        prev_bbox = self._mask_to_bbox(initial_mask)

        while True:
            ret, frame = cap.read()
            if not ret or fidx >= end_frame:
                break

            if fidx < start_frame:
                fidx += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if fidx == start_frame:
                # First frame: use initial mask
                mask = initial_mask.copy()
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

            # Warp previous mask using flow
            h, w = gray.shape
            flow_map = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).astype(np.float32)
            flow_map += flow
            warped_mask = cv2.remap(
                prev_mask.astype(np.float32), flow_map[:, :, 0], flow_map[:, :, 1],
                interpolation=cv2.INTER_LINEAR
            )
            warped_mask = (warped_mask > 0.5).astype(np.uint8)

            # Get bbox from warped mask
            current_bbox = self._mask_to_bbox(warped_mask)

            # Refine with SAM if available
            if self.sam_predictor is not None and current_bbox is not None:
                refined_mask = self._refine_with_sam(frame, current_bbox, warped_mask)
                if refined_mask is not None:
                    mask = refined_mask
                else:
                    mask = warped_mask
            else:
                mask = warped_mask

            if (fidx - start_frame) % sample_stride == 0:
                tracking_frame = self._mask_to_tracking_frame(mask, fidx)
                if tracking_frame is not None:
                    frames.append(tracking_frame)

            prev_gray = gray
            prev_mask = mask
            prev_bbox = current_bbox
            fidx += 1

        cap.release()
        return frames

    def _refine_with_sam(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        prev_mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Refine mask using SAM with bbox prompt."""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(frame_rgb)

            x, y, w, h = bbox
            box = np.array([x, y, x + w, y + h])

            # Use center of previous mask as point prompt
            ys, xs = np.nonzero(prev_mask)
            if len(xs) > 0:
                point_coords = np.array([[xs.mean(), ys.mean()]])
                point_labels = np.array([1])
            else:
                point_coords = None
                point_labels = None

            masks, scores, _ = self.sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box[None, :],
                multimask_output=True,
            )

            # Select best mask
            best_idx = scores.argmax()
            return masks[best_idx].astype(np.uint8)

        except Exception as e:
            return None

    def _mask_to_bbox(
        self,
        mask: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Convert mask to bounding box."""
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        x, y = xs.min(), ys.min()
        w, h = xs.max() - x, ys.max() - y
        return (int(x), int(y), int(w), int(h))

    def _mask_to_tracking_frame(
        self,
        mask: np.ndarray,
        frame_idx: int,
    ) -> Optional[TrackingFrame]:
        """Convert mask to TrackingFrame."""
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None

        x, y = xs.min(), ys.min()
        w, h = xs.max() - x, ys.max() - y
        cx, cy = xs.mean(), ys.mean()
        area = len(xs)

        return TrackingFrame(
            frame_idx=frame_idx,
            mask=mask,
            bbox=(int(x), int(y), int(w), int(h)),
            center=(float(cx), float(cy)),
            confidence=1.0,
            area=area,
        )

    def track_from_bbox(
        self,
        video_path: str,
        initial_bbox: Tuple[int, int, int, int],
        sample_stride: int = 1,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> TrackingResult:
        """
        Track object starting from bounding box.

        Creates initial mask from bbox using SAM, then tracks.

        Args:
            video_path: Path to video
            initial_bbox: (x, y, w, h) bounding box
            sample_stride: Frame sampling stride
            start_frame: Starting frame
            end_frame: Ending frame

        Returns:
            TrackingResult
        """
        # Read first frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("Failed to read first frame")

        # Create initial mask from bbox
        if self.sam_predictor is not None:
            mask = self._get_sam_mask_from_bbox(frame, initial_bbox)
        else:
            # Simple rectangular mask
            H, W = frame.shape[:2]
            mask = np.zeros((H, W), dtype=np.uint8)
            x, y, w, h = initial_bbox
            mask[y:y+h, x:x+w] = 1

        return self.track(video_path, mask, sample_stride, start_frame, end_frame)

    def _get_sam_mask_from_bbox(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Get SAM mask from bounding box."""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(frame_rgb)

            x, y, w, h = bbox
            box = np.array([x, y, x + w, y + h])

            masks, _, _ = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],
                multimask_output=False,
            )

            return masks[0].astype(np.uint8)

        except Exception:
            # Fallback to rectangular mask
            H, W = frame.shape[:2]
            mask = np.zeros((H, W), dtype=np.uint8)
            x, y, w, h = bbox
            mask[y:y+h, x:x+w] = 1
            return mask


def create_video_tracker(
    device: str = "cuda",
    method: str = "auto",
    **kwargs,
) -> VideoObjectTracker:
    """
    Factory function to create VideoObjectTracker.

    Args:
        device: "cuda" or "cpu"
        method: "auto", "sam2", "xmem", "optical_flow"

    Returns:
        VideoObjectTracker instance
    """
    return VideoObjectTracker(device=device, method=method, **kwargs)


__all__ = [
    "VideoObjectTracker",
    "TrackingFrame",
    "TrackingResult",
    "create_video_tracker",
]
