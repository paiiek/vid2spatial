# Vid2Spatial

**A deterministic vision-guided control pipeline for spatial audio authoring**

Extracts stable spatial control trajectories from video and outputs FOA / HRTF Binaural / OSC for DAW integration.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## Contributions

1. **C1. Deterministic vision-guided control pipeline** for spatial audio authoring
2. **C2. Adaptive-K tracking** that prevents trajectory collapse in fast motion
3. **C3. RTS-based trajectory smoothing** preserving motion amplitude
4. **C4. Confidence-weighted depth blending** for stable distance control
5. **C5. FOA/OSC dual output** for rendering and DAW authoring workflows

---

## Pipeline

```
Video + Text Prompt + Audio
         │
         ▼
┌─────────────────────────────────────────────────┐
│  1. Detection & Tracking                        │
│     Grounding DINO (text-guided detection)       │
│     + Adaptive-K re-detection interval           │
│       Fast motion → K=2-3 (frequent re-detect)   │
│       Slow motion → K=10-15 (save compute)       │
│     + Linear interpolation between keyframes     │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  2. Robustness Layer                            │
│     - Confidence gating (< 0.35 → reject)        │
│     - Jump rejection (> 150 px/f → outlier)      │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  3. Depth Stability Module                      │
│     Layer 1: Depth Anything V2 Metric (meters)   │
│     Layer 2: BBox-scale proxy (smooth response)  │
│     → Confidence-weighted blending (60% jitter↓) │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  4. 3D Projection                               │
│     pixel (cx, cy) → camera ray → (az, el) + d  │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  5. RTS Smoother (93-97% jerk reduction)        │
│     Rauch-Tung-Striebel optimal smoothing        │
│     Preserves motion amplitude while removing    │
│     high-frequency jitter                        │
└─────────────────────────────────────────────────┘
         │
         ├──→ FOA AmbiX 4ch WAV
         ├──→ HRTF Binaural (KEMAR SOFA) → Stereo WAV
         └──→ OSC → DAW (az, el, dist, d_rel, velocity)
```

---

## Evaluation Results

### Trajectory Reliability (C2)

| Metric | SAM2 | Proposed (Adaptive-K + RTS) | Improvement |
|--------|------|------------------------------|-------------|
| **Amplitude (0.6Hz)** | 3.4% | **100.0%** | 29x |
| **MAE** | 142.9px | **16.1px** | 9x |
| **Velocity correlation** | -0.088 | **0.930** | Recovered |
| **FPS** | 13.5 | **26.4** | 2x faster |

*SAM2's low jerk is misleading — the trajectory has near-zero amplitude (stationary).

### Control Stability (C3)

| Smoothing | Median Jerk | Reduction |
|-----------|-------------|-----------|
| None | 0.0230 | — |
| EMA (α=0.3) | 0.0115 | 50% |
| **RTS** | **0.0018** | **92%** |

### Depth Stability (C4)

| Method | Jitter | Improvement |
|--------|--------|-------------|
| Metric only | 1.00x | — |
| BBox proxy only | 0.65x | 35%↓ |
| **Blended (proposed)** | **0.40x** | **60%↓** |

### Perceptual Evaluation (In Progress)

- 12 video clips × 3 conditions (HRTF binaural / stereo pan / mono anchor)
- 4 MOS dimensions: Spatial Alignment, Motion Smoothness, Depth Perception, Overall Quality
- Web-based listening test, 25 participants target

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/paiiek/vid2spatial.git
cd vid2spatial
pip install -r requirements.txt
```

### 2. Download Model Weights

Weights are not included in the repository (gitignored). Download and place in `weights/`:

- **Grounding DINO** — text-guided object detection
- **Depth Anything V2 Metric** — monocular depth estimation

### 3. Basic Usage

```python
from vid2spatial_pkg.hybrid_tracker import HybridTracker
from vid2spatial_pkg.trajectory_stabilizer import rts_smooth_trajectory
from vid2spatial_pkg.foa_render import render_foa_from_trajectory

# 1. Track object in video
tracker = HybridTracker(device="cuda")
result = tracker.track(
    video_path="input.mp4",
    text_prompt="person",
    tracking_method="adaptive_k",
    depth_stride=5,
)

# 2. Smooth the 3D trajectory (RTS optimal smoother)
traj_3d = result.get_trajectory_3d(smooth=False)
smoothed = rts_smooth_trajectory(traj_3d["frames"])

# 3. Render FOA spatial audio
render_foa_from_trajectory(
    audio_path="input.wav",
    trajectory={"frames": smoothed},
    output_path="output_foa.wav",
)
```

### 4. HRTF Binaural Rendering

```python
from vid2spatial_pkg.foa_render import render_binaural_from_trajectory

render_binaural_from_trajectory(
    audio_path="input.wav",
    trajectory={"frames": smoothed},
    output_path="output_binaural.wav",
    sofa_path="/path/to/kemar.sofa",
)
```

Direct HRTF convolution via KEMAR SOFA with 50ms overlap-add (Hann window), providing full ILD, ITD, and pinna cues.

### 5. OSC Streaming (DAW Integration)

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
| `/vid2spatial/distance` | 0 to 1 | Normalized |
| `/vid2spatial/velocity` | deg/s | Angular velocity |
| `/vid2spatial/spatial` | [az, el, dist, vel, tc] | Bundled message |

### 6. End-to-End Pipeline

```python
from vid2spatial_pkg.pipeline import SpatialAudioPipeline

pipeline = SpatialAudioPipeline()
pipeline.process(
    video_path="input.mp4",
    audio_path="input.wav",
    text_prompt="guitar",
    output_dir="output/",
)
```

---

## Project Structure

```
vid2spatial/
├── vid2spatial_pkg/              # Core Python package (19 modules)
│   ├── pipeline.py                   # End-to-end pipeline orchestration
│   ├── hybrid_tracker.py             # Adaptive-K tracker (DINO + YOLO/ByteTrack)
│   ├── trajectory_stabilizer.py      # RTS smoother + depth stabilization
│   ├── foa_render.py                 # FOA AmbiX + HRTF binaural + stereo pan
│   ├── vision.py                     # Camera geometry (pixel→ray→angles)
│   ├── depth_metric.py               # Depth Anything V2 integration
│   ├── depth_utils.py                # Depth blending + confidence weighting
│   ├── osc_sender.py                 # OSC streaming for DAW
│   └── ...                           # + 11 supporting modules
│
├── experiments/                  # Experiment scripts & results
│   ├── e2e_20_videos/                # 20 real videos, end-to-end
│   ├── gt_eval_synthetic/            # 15 synthetic GT scenes
│   ├── sot_15_videos/                # SOT benchmark + HRTF binaural
│   └── synthetic_render.py           # Synthetic scenario renderer
│
├── evaluation/                   # Evaluation code & results
│   ├── tracking_ablation/            # (A) Trajectory reliability ablation
│   ├── comprehensive_results/        # (B-C) Control + depth stability
│   ├── ablation_output/              # Renderer/baseline ablation
│   ├── listening_test/               # (D) Web-based perceptual evaluation
│   ├── tests/                        # Unit tests
│   └── plots/                        # Evaluation plots
│
├── docs/                         # Documentation (dated versions)
├── archive/                      # Old versions (gitignored)
├── data/                         # Datasets (gitignored)
├── weights/                      # Model weights (gitignored)
│
├── .gitignore
├── STRUCTURE.md
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Documentation

- **[STRUCTURE.md](STRUCTURE.md)** — Detailed directory layout
- **[docs/ARCHITECTURE_20260210.md](docs/ARCHITECTURE_20260210.md)** — System architecture
- **[docs/PROJECT_DOCUMENTATION_20260210.md](docs/PROJECT_DOCUMENTATION_20260210.md)** — Full project documentation (Korean)
- **[docs/PERCEPTUAL_EVALUATION_20260210.md](docs/PERCEPTUAL_EVALUATION_20260210.md)** — Listening test design
- **[docs/FINAL_EVALUATION_REPORT_20260210.md](docs/FINAL_EVALUATION_REPORT_20260210.md)** — Comprehensive evaluation results
- **[docs/OSC_INTERFACE_SPEC_20260210.md](docs/OSC_INTERFACE_SPEC_20260210.md)** — OSC protocol specification

---

## Citation

```bibtex
@misc{vid2spatial2026,
  title={Vid2Spatial: Vision-Guided Spatial Control Trajectory Extraction for Audio Authoring},
  author={Seungheon Doh},
  year={2026},
  howpublished={\url{https://github.com/paiiek/vid2spatial}}
}
```

---

## License

MIT License

---

**Last Updated**: 2026-02-10
