"""
Depth Anything V2 integration for improved metric depth estimation.

Replaces MiDaS with Depth Anything V2 for better depth accuracy.
"""
import sys
import os
from pathlib import Path
import numpy as np
import cv2
from typing import Optional, Tuple


class DepthAnythingV2:
    """Depth Anything V2 wrapper for metric depth estimation."""

    def __init__(self, model_size: str = "small", device: str = "cuda"):
        """
        Initialize Depth Anything V2.

        Args:
            model_size: Model size ('small', 'base', 'large')
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_size = model_size
        self.device = device
        self.model = None
        self.transform = None

        # Add Depth-Anything-V2 to path
        dav2_path = "/home/seung/Depth-Anything-V2"
        if os.path.exists(dav2_path):
            sys.path.insert(0, dav2_path)

        self._load_model()

    def _load_model(self):
        """Load Depth Anything V2 model."""
        try:
            import torch
            from depth_anything_v2.dpt import DepthAnythingV2 as DAV2Model

            # Model configurations
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            }

            # Map model_size to encoder
            size_map = {'small': 'vits', 'base': 'vitb', 'large': 'vitl'}
            encoder = size_map.get(self.model_size, 'vits')

            # Initialize model
            self.model = DAV2Model(**model_configs[encoder])

            # Load pretrained weights
            checkpoint_path = f"/home/seung/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth"

            if not os.path.exists(checkpoint_path):
                print(f"[warn] Checkpoint not found: {checkpoint_path}")
                print("[info] Downloading checkpoint...")
                self._download_checkpoint(encoder)

            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()

            print(f"[info] Loaded Depth Anything V2 ({encoder}) on {self.device}")

        except Exception as e:
            print(f"[error] Failed to load Depth Anything V2: {e}")
            self.model = None

    def _download_checkpoint(self, encoder: str):
        """Download checkpoint from Hugging Face."""
        import urllib.request

        checkpoint_dir = "/home/seung/Depth-Anything-V2/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        urls = {
            'vits': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth',
            'vitb': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth',
            'vitl': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth',
        }

        url = urls[encoder]
        output_path = f"{checkpoint_dir}/depth_anything_v2_{encoder}.pth"

        print(f"[info] Downloading from {url}")
        urllib.request.urlretrieve(url, output_path)
        print(f"[info] Saved to {output_path}")

    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from image.

        Args:
            image: Input image (H, W, 3) in BGR format

        Returns:
            depth: Depth map (H, W) normalized to [0, 1]
        """
        if self.model is None:
            # Fallback to dummy depth
            h, w = image.shape[:2]
            return np.ones((h, w), dtype=np.float32) * 0.5

        try:
            import torch

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Inference
            with torch.no_grad():
                depth = self.model.infer_image(image_rgb)

            # Normalize to [0, 1]
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

            return depth.astype(np.float32)

        except Exception as e:
            print(f"[error] Depth inference failed: {e}")
            h, w = image.shape[:2]
            return np.ones((h, w), dtype=np.float32) * 0.5


def create_depth_anything_v2_backend(
    model_size: str = "small",
    device: str = "cuda"
) -> callable:
    """
    Create Depth Anything V2 depth estimation function.

    Args:
        model_size: Model size ('small', 'base', 'large')
        device: Device to use

    Returns:
        depth_fn: Function that takes image and returns depth map
    """
    dav2 = DepthAnythingV2(model_size=model_size, device=device)

    def depth_fn(image: np.ndarray) -> np.ndarray:
        return dav2.infer(image)

    return depth_fn


__all__ = [
    "DepthAnythingV2",
    "create_depth_anything_v2_backend",
]
