"""
Text-based Object Selector using Grounding-DINO + SAM.

Allows selecting objects in video using natural language text prompts.
Falls back to YOLO if Grounding-DINO is not available.

Pipeline:
1. Text prompt -> Grounding-DINO -> Bounding boxes
2. Bounding boxes -> SAM -> Segmentation masks

Usage:
    selector = TextObjectSelector()
    mask, bbox = selector.select("the red ball", frame)
"""

import os
import sys
import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass


@dataclass
class SelectionResult:
    """Object selection result."""
    mask: np.ndarray  # Binary mask (H, W)
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    label: str
    center: Tuple[float, float]  # (cx, cy)


class TextObjectSelector:
    """
    Select objects in images using text prompts.

    Uses Grounding-DINO for text-to-box detection and SAM for segmentation.
    Falls back to YOLO + color/class matching if Grounding-DINO unavailable.
    """

    def __init__(
        self,
        device: str = "cuda",
        grounding_dino_config: Optional[str] = None,
        grounding_dino_checkpoint: Optional[str] = None,
        sam_checkpoint: Optional[str] = None,
        sam_model_type: str = "vit_b",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ):
        """
        Initialize Text Object Selector.

        Args:
            device: "cuda" or "cpu"
            grounding_dino_config: Path to Grounding-DINO config
            grounding_dino_checkpoint: Path to Grounding-DINO checkpoint
            sam_checkpoint: Path to SAM checkpoint
            sam_model_type: SAM model type ("vit_h", "vit_l", "vit_b")
            box_threshold: Grounding-DINO box confidence threshold
            text_threshold: Grounding-DINO text confidence threshold
        """
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Model references
        self.grounding_dino = None
        self.sam_predictor = None
        self.yolo_model = None

        # Try to load Grounding-DINO
        self._load_grounding_dino(grounding_dino_config, grounding_dino_checkpoint)

        # Load SAM
        self._load_sam(sam_checkpoint, sam_model_type)

        # Fallback to YOLO if Grounding-DINO not available
        if self.grounding_dino is None:
            self._load_yolo_fallback()

    def _load_grounding_dino(self, config_path: Optional[str], checkpoint_path: Optional[str]):
        """Load Grounding-DINO model."""
        try:
            # Try importing groundingdino
            from groundingdino.util.inference import load_model, predict
            from groundingdino.util.inference import load_image as gd_load_image

            # Default paths
            if config_path is None:
                config_path = os.path.expanduser(
                    "~/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
                )
            if checkpoint_path is None:
                checkpoint_path = os.path.expanduser(
                    "~/GroundingDINO/weights/groundingdino_swint_ogc.pth"
                )

            if os.path.exists(config_path) and os.path.exists(checkpoint_path):
                self.grounding_dino = load_model(config_path, checkpoint_path)
                self._gd_predict = predict
                self._gd_load_image = gd_load_image
                print("[text_selector] Loaded Grounding-DINO")
            else:
                print(f"[text_selector] Grounding-DINO files not found, using YOLO fallback")
                self.grounding_dino = None

        except ImportError:
            print("[text_selector] Grounding-DINO not installed, using YOLO fallback")
            self.grounding_dino = None
        except Exception as e:
            print(f"[text_selector] Failed to load Grounding-DINO: {e}")
            self.grounding_dino = None

    def _load_sam(self, checkpoint_path: Optional[str], model_type: str):
        """Load SAM model."""
        try:
            from segment_anything import sam_model_registry, SamPredictor

            # Default checkpoint path
            if checkpoint_path is None:
                checkpoint_dir = os.path.expanduser("~/.cache/sam")
                checkpoint_path = os.path.join(checkpoint_dir, f"sam_{model_type}.pth")

                # Download if not exists
                if not os.path.exists(checkpoint_path):
                    self._download_sam_checkpoint(model_type, checkpoint_path)

            if os.path.exists(checkpoint_path):
                import torch
                sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
                sam.to(self.device)
                self.sam_predictor = SamPredictor(sam)
                print(f"[text_selector] Loaded SAM ({model_type})")
            else:
                print(f"[text_selector] SAM checkpoint not found: {checkpoint_path}")
                self.sam_predictor = None

        except ImportError:
            print("[text_selector] segment_anything not installed")
            self.sam_predictor = None
        except Exception as e:
            print(f"[text_selector] Failed to load SAM: {e}")
            self.sam_predictor = None

    def _download_sam_checkpoint(self, model_type: str, output_path: str):
        """Download SAM checkpoint from Meta."""
        import urllib.request

        urls = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        }

        if model_type not in urls:
            print(f"[text_selector] Unknown SAM model type: {model_type}")
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"[text_selector] Downloading SAM {model_type}...")
        urllib.request.urlretrieve(urls[model_type], output_path)
        print(f"[text_selector] Saved to {output_path}")

    def _load_yolo_fallback(self):
        """Load YOLO for fallback detection."""
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolo11n.pt")
            print("[text_selector] Loaded YOLO fallback")
        except Exception as e:
            print(f"[text_selector] Failed to load YOLO: {e}")
            self.yolo_model = None

    def select(
        self,
        text_prompt: str,
        image: np.ndarray,
        return_all: bool = False,
    ) -> Optional[SelectionResult]:
        """
        Select object matching text prompt.

        Args:
            text_prompt: Natural language description (e.g., "the red ball", "person on the left")
            image: BGR image (H, W, 3)
            return_all: Return all matches instead of best one

        Returns:
            SelectionResult or None if no match
        """
        if self.grounding_dino is not None:
            results = self._select_grounding_dino(text_prompt, image)
        else:
            results = self._select_yolo_fallback(text_prompt, image)

        if not results:
            return None

        if return_all:
            return results

        # Return highest confidence result
        return max(results, key=lambda x: x.confidence)

    def _select_grounding_dino(
        self,
        text_prompt: str,
        image: np.ndarray,
    ) -> List[SelectionResult]:
        """Select using Grounding-DINO + SAM."""
        import torch
        from PIL import Image
        import tempfile

        # Save image temporarily for groundingdino
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, image)
            temp_path = f.name

        try:
            # Load for groundingdino
            image_source, image_tensor = self._gd_load_image(temp_path)

            # Predict boxes
            boxes, logits, phrases = self._gd_predict(
                model=self.grounding_dino,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )

        finally:
            os.unlink(temp_path)

        if len(boxes) == 0:
            return []

        results = []
        H, W = image.shape[:2]

        # Convert boxes to pixel coordinates
        boxes_pixel = boxes.cpu().numpy() * np.array([W, H, W, H])

        for i, (box, conf, phrase) in enumerate(zip(boxes_pixel, logits.cpu().numpy(), phrases)):
            x1, y1, x2, y2 = box
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

            # Get SAM mask
            mask = self._get_sam_mask(image, (x, y, w, h))
            if mask is None:
                mask = self._create_bbox_mask(image.shape[:2], (x, y, w, h))

            # Calculate center
            if mask.sum() > 0:
                ys, xs = np.nonzero(mask)
                cx, cy = xs.mean(), ys.mean()
            else:
                cx, cy = x + w / 2, y + h / 2

            results.append(SelectionResult(
                mask=mask,
                bbox=(x, y, w, h),
                confidence=float(conf),
                label=phrase,
                center=(cx, cy),
            ))

        return results

    def _select_yolo_fallback(
        self,
        text_prompt: str,
        image: np.ndarray,
    ) -> List[SelectionResult]:
        """Fallback selection using YOLO + text matching."""
        if self.yolo_model is None:
            return []

        # Parse text prompt for class name
        cls_name = self._parse_class_from_prompt(text_prompt)

        # Run YOLO detection
        results = self.yolo_model(image, verbose=False)

        matches = []
        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls_id = int(box.cls[0])
                detected_name = self.yolo_model.names[cls_id].lower()

                # Check if class matches
                if cls_name.lower() in detected_name or detected_name in cls_name.lower():
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    conf = float(box.conf[0])

                    # Get SAM mask
                    mask = self._get_sam_mask(image, (x, y, w, h))
                    if mask is None:
                        mask = self._create_bbox_mask(image.shape[:2], (x, y, w, h))

                    # Calculate center
                    if mask.sum() > 0:
                        ys, xs = np.nonzero(mask)
                        cx, cy = xs.mean(), ys.mean()
                    else:
                        cx, cy = x + w / 2, y + h / 2

                    matches.append(SelectionResult(
                        mask=mask,
                        bbox=(x, y, w, h),
                        confidence=conf,
                        label=detected_name,
                        center=(cx, cy),
                    ))

        return matches

    def _parse_class_from_prompt(self, text_prompt: str) -> str:
        """Extract class name from natural language prompt."""
        # Common COCO class names
        coco_classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
            "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
            "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
            "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
            "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
            "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush", "ball", "guitar",
            "drum", "piano", "violin", "hand", "face", "head"
        ]

        text_lower = text_prompt.lower()

        # Check for exact class name matches
        for cls in coco_classes:
            if cls in text_lower:
                return cls

        # Default to person
        return "person"

    def _get_sam_mask(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """Get SAM segmentation mask for bounding box."""
        if self.sam_predictor is None:
            return None

        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(image_rgb)

            x, y, w, h = bbox
            box = np.array([x, y, x + w, y + h])

            masks, _, _ = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],
                multimask_output=False,
            )

            return masks[0].astype(np.uint8)

        except Exception as e:
            print(f"[text_selector] SAM inference failed: {e}")
            return None

    def _create_bbox_mask(
        self,
        shape: Tuple[int, int],
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Create simple rectangular mask from bounding box."""
        H, W = shape
        mask = np.zeros((H, W), dtype=np.uint8)
        x, y, w, h = bbox
        mask[y:y+h, x:x+w] = 1
        return mask

    def select_with_point(
        self,
        image: np.ndarray,
        point: Tuple[int, int],
        point_label: int = 1,
    ) -> Optional[SelectionResult]:
        """
        Select object using point click.

        Args:
            image: BGR image
            point: (x, y) click coordinates
            point_label: 1 for foreground, 0 for background

        Returns:
            SelectionResult or None
        """
        if self.sam_predictor is None:
            return None

        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(image_rgb)

            point_coords = np.array([[point[0], point[1]]])
            point_labels = np.array([point_label])

            masks, scores, _ = self.sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

            # Get best mask
            best_idx = scores.argmax()
            mask = masks[best_idx].astype(np.uint8)

            # Calculate bbox from mask
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                return None

            x, y = xs.min(), ys.min()
            w, h = xs.max() - x, ys.max() - y
            cx, cy = xs.mean(), ys.mean()

            return SelectionResult(
                mask=mask,
                bbox=(int(x), int(y), int(w), int(h)),
                confidence=float(scores[best_idx]),
                label="selected",
                center=(cx, cy),
            )

        except Exception as e:
            print(f"[text_selector] Point selection failed: {e}")
            return None


def create_text_selector(
    device: str = "cuda",
    **kwargs,
) -> TextObjectSelector:
    """
    Factory function to create TextObjectSelector.

    Args:
        device: "cuda" or "cpu"
        **kwargs: Additional arguments for TextObjectSelector

    Returns:
        TextObjectSelector instance
    """
    return TextObjectSelector(device=device, **kwargs)


__all__ = [
    "TextObjectSelector",
    "SelectionResult",
    "create_text_selector",
]
