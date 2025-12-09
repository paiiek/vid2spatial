"""
Test Depth Anything V2 integration.

Usage:
    python3 scripts/test_depth_v2.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import matplotlib.pyplot as plt


def test_depth_anything_v2():
    """Test Depth Anything V2 depth estimation."""
    print("="*60)
    print("Testing Depth Anything V2")
    print("="*60)

    # Import depth backend
    from vid2spatial_pkg.depth_anything_v2 import create_depth_anything_v2_backend

    # Create depth estimator
    print("\n[1] Loading Depth Anything V2...")
    depth_fn = create_depth_anything_v2_backend(model_size="small", device="cuda")

    # Create test image
    print("\n[2] Creating test image...")
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Estimate depth
    print("\n[3] Estimating depth...")
    depth_map = depth_fn(test_image)

    print(f"   Input shape: {test_image.shape}")
    print(f"   Depth shape: {depth_map.shape}")
    print(f"   Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
    print(f"   Depth mean: {depth_map.mean():.3f}")

    # Visualize
    print("\n[4] Saving visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    im = axes[1].imshow(depth_map, cmap="magma")
    axes[1].set_title("Depth Map (Depth Anything V2)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1])

    output_path = "results/depth_v2_test.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"   Saved to: {output_path}")

    print("\n" + "="*60)
    print("✅ Depth Anything V2 test PASSED")
    print("="*60)


if __name__ == "__main__":
    test_depth_anything_v2()
