# Vid2Spatial

**Text-guided video object tracking for spatial audio authoring with DINO Adaptive-K re-detection and metric depth estimation.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## Overview

Vid2Spatial extracts 3D spatial trajectories from video using text-guided object detection, then renders First-Order Ambisonics (FOA) spatial audio or streams parameters via OSC to DAW.

**Key Innovations**:
1. **DINO Adaptive-K**: Solves SAM2's motion collapse at >0.5Hz motion
2. **Dual-Layer Depth**: Metric depth + bbox-scale proxy for robust distance estimation
3. **RTS Smoother**: 93-97% jerk reduction while preserving true motion

### Pipeline Architecture

```
Video + Audio Input
        │
        ▼
┌───────────────────────────────────────────────────┐
│  1. Trajectory Authority                          │
│     DINO Adaptive K-frame Detection               │
│     - Fast motion: K=2-3 (frequent detection)     │
│     - Slow motion: K=10-15 (save compute)         │
│     + Linear interpolation between keyframes      │
└───────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────┐
│  2. Robustness Layer                              │
│     - Confidence gating (conf < 0.35 → reject)    │
│     - Jump rejection (velocity > 150 px/f → reject)│
└───────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────┐
│  3. Dual-Layer Depth Estimation                   │
│     Layer 1: Depth Anything V2 Metric (meters)    │
│     Layer 2: BBox-scale proxy (fast response)     │
│     → Confidence-weighted blending                │
└───────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────┐
│  4. 3D Projection                                 │
│     pixel (cx, cy) → ray → (az, el) + depth_m     │
└───────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────┐
│  5. RTS Smoother (offline) / EMA (realtime)       │
│     → 93-97% jerk reduction                       │
└───────────────────────────────────────────────────┘
        │
        ├──→ (A) FOA Renderer → AmbiX 4ch WAV
        │
        └──→ (B) OSC Sender → DAW (az, el, dist, d_rel, velocity)
```

---

## Performance Comparison

| Aspect | SAM2 | Proposed (Adaptive K + RTS) | Improvement |
|--------|------|------------------------------|-------------|
| **Amplitude (0.6Hz)** | 3.4% | **100.0%** | **29x** |
| **MAE** | 142.9px | **16.1px** | **9x** |
| **Velocity correlation** | -0.088 | **0.930** | ✅ Recovered |
| **Jerk (after RTS)** | 0.037* | **0.026** | ✅ Lower |
| **Real video win-rate** | — | **8/13 (62%)** | ✅ Majority |
| **FPS** | 13.5 | **26.4** | **2x faster** |

*SAM2's low jerk is misleading — the trajectory has only 3.4% amplitude (near-stationary).

### Depth Enhancement Results

| Method | Jitter Reduction | Compute Savings |
|--------|-----------------|-----------------|
| BBox-scale proxy blending | **60%** | - |
| Adaptive depth stride | - | **85%** |

---

## Quick Start

### Installation

```bash
git clone https://github.com/paiiek/vid2spatial.git
cd vid2spatial
pip install -r requirements.txt

# Download model weights (not included in repo)
# - Grounding DINO
# - Depth Anything V2 Metric
# - SAM2 (optional, for comparison)
```

### Basic Usage

```python
from vid2spatial_pkg.hybrid_tracker import HybridTracker
from vid2spatial_pkg.trajectory_stabilizer import rts_smooth_trajectory
from vid2spatial_pkg.foa_render import render_foa_from_trajectory
from vid2spatial_pkg.depth_utils import process_trajectory_depth, DepthConfig

# 1. Track object in video (depth always enabled)
tracker = HybridTracker(device="cuda")
result = tracker.track(
    video_path="input.mp4",
    text_prompt="person",
    tracking_method="adaptive_k",  # recommended
    depth_stride=5,
)

# 2. Get 3D trajectory and smooth
traj_3d = result.get_trajectory_3d(smooth=False)
smoothed = rts_smooth_trajectory(traj_3d["frames"])

# 3. (Optional) Enhance depth with bbox proxy
config = DepthConfig(use_bbox_proxy=True, output_d_rel=True)
enhanced = process_trajectory_depth(smoothed, config)

# 4. Render FOA
render_foa_from_trajectory(
    audio_path="input.wav",
    trajectory={"frames": enhanced},
    output_path="output_foa.wav",
)
```

### OSC Streaming (DAW Integration)

```python
from vid2spatial_pkg.osc_sender import OSCSpatialSender

sender = OSCSpatialSender(host="127.0.0.1", port=9000)
sender.connect()
sender.stream_trajectory(trajectory, fps=30, realtime=True)
```

**OSC Addresses:**
| Address | Value | Description |
|---------|-------|-------------|
| `/vid2spatial/azimuth` | -180 to 180 | Degrees |
| `/vid2spatial/elevation` | -90 to 90 | Degrees |
| `/vid2spatial/distance` | meters | Metric depth |
| `/vid2spatial/d_rel` | 0-1 | Relative distance (normalized) |
| `/vid2spatial/velocity` | deg/s | Angular velocity |
| `/vid2spatial/timecode` | seconds | Sync reference |

---

## Project Structure

```
vid2spatial/
├── vid2spatial_pkg/              # Core Python package
│   ├── hybrid_tracker.py         # Main tracker (DINO adaptive-K)
│   ├── trajectory_stabilizer.py  # RTS smoother
│   ├── foa_render.py             # FOA AmbiX rendering
│   ├── osc_sender.py             # OSC streaming for DAW
│   ├── depth_metric.py           # Depth Anything V2 Metric
│   ├── depth_utils.py            # Depth enhancement utilities
│   └── vision.py                 # Camera/geometry utilities
│
├── eval/                         # Evaluation scripts
│   └── comprehensive_results/
│       └── FINAL_EVALUATION_REPORT.md
│
├── docs/
│   └── ARCHITECTURE.md           # Detailed technical documentation
│
├── README.md
└── requirements.txt
```

---

## Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Detailed technical documentation for thesis/papers
- **[FINAL_EVALUATION_REPORT.md](eval/comprehensive_results/FINAL_EVALUATION_REPORT.md)**: Evaluation results

---

## Key Components

### 1. DINO Adaptive-K Re-detection
- Uses Grounding DINO for text-guided object detection
- Adaptive keyframe interval based on motion velocity
- Linear interpolation between keyframes

### 2. Robustness Layer
- **Confidence gating**: Reject detections with conf < 0.35
- **Jump rejection**: Reject velocity > 150 px/frame as outliers

### 3. Dual-Layer Depth Estimation
- **Layer 1 (Metric)**: Depth Anything V2 for absolute distance in meters
- **Layer 2 (Proxy)**: BBox-scale proxy for fast-moving objects
- **Blending**: Confidence-weighted fusion (low conf → more proxy)

### 4. RTS Smoother
- Rauch-Tung-Striebel two-pass optimal smoothing
- 93-97% jerk reduction while preserving true motion
- Recommended for offline/authoring use

### 5. Dual Output
- **FOA Rendering**: AmbiX 4-channel WAV (W, Y, Z, X)
- **OSC Streaming**: Real-time parameter output for DAW automation

---

## Citation

```bibtex
@misc{vid2spatial2026,
  title={Vid2Spatial: Text-Guided Video Tracking for Spatial Audio Authoring},
  author={Seungheon Doh},
  year={2026},
  howpublished={\url{https://github.com/paiiek/vid2spatial}}
}
```

---

## License

MIT License

---

**Last Updated**: 2026-02-05
