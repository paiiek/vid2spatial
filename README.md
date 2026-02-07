# Vid2Spatial

**Text-guided video object tracking → 3D spatial trajectory → First-Order Ambisonics (FOA) / HRTF Binaural / OSC for DAW**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## What Is This?

Vid2Spatial takes a **video** and a **text prompt** (e.g., "person", "guitar"), detects and tracks the object across frames, estimates its 3D position (azimuth, elevation, distance), and generates spatial audio that follows the object's trajectory.

**Outputs**:
- **FOA (AmbiX)**: 4-channel First-Order Ambisonics WAV (W, Y, Z, X — ACN/SN3D)
- **HRTF Binaural**: Stereo WAV rendered via KEMAR HRTF (SOFA) for headphone listening
- **OSC Stream**: Real-time spatial parameters to DAW (Reaper, Max/MSP, etc.)

---

## Pipeline Overview

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
│  3. Dual-Layer Depth Estimation                 │
│     Layer 1: Depth Anything V2 Metric (meters)   │
│     Layer 2: BBox-scale proxy (fast response)    │
│     → Confidence-weighted blending               │
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
│  5. RTS Smoother (offline) / EMA (realtime)     │
│     Rauch-Tung-Striebel optimal smoothing        │
│     → 93-97% jerk reduction                      │
└─────────────────────────────────────────────────┘
         │
         ├──→ FOA AmbiX 4ch WAV
         ├──→ HRTF Binaural (KEMAR SOFA) → Stereo WAV
         └──→ OSC → DAW (az, el, dist, d_rel, velocity)
```

---

## Performance

| Metric | SAM2-only | Proposed (Adaptive-K + RTS) | Improvement |
|--------|-----------|------------------------------|-------------|
| **Amplitude (0.6Hz)** | 3.4% | **100.0%** | 29x |
| **MAE** | 142.9px | **16.1px** | 9x |
| **Velocity correlation** | -0.088 | **0.930** | Recovered |
| **Jerk (after RTS)** | 0.037* | **0.026** | Lower |
| **Real video win-rate** | — | **8/13 (62%)** | Majority |
| **FPS** | 13.5 | **26.4** | 2x faster |

*SAM2's low jerk is misleading — the trajectory has near-zero amplitude (stationary).

| Depth Enhancement | Result |
|-------------------|--------|
| BBox-proxy blending | 60% jitter reduction |
| Adaptive depth stride | 85% compute savings |
| RTS depth smoothing | 80.9x jerk reduction |

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
- **SAM2** (optional) — segmentation for comparison

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

For headphone-optimized output using a KEMAR HRTF (SOFA file):

```python
render_foa_from_trajectory(
    audio_path="input.wav",
    trajectory={"frames": smoothed},
    output_path="output_binaural.wav",
    sofa_path="/path/to/kemar.sofa",  # HRTF file
)
```

The HRTF renderer uses 8-speaker virtual decode with nearest-neighbor HRIR convolution.
Compared to the default crossfeed method, HRTF provides:
- 3.6x stronger high-frequency spectral differentiation (pinna cues)
- More natural ILD (Interaural Level Difference) patterns
- Lower interaural coherence (better spatial separation)

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
| `/vid2spatial/distance` | 0 to 1 | Normalized (1=near, 0=far) |
| `/vid2spatial/distance_m` | meters | Raw metric distance (alt mode) |
| `/vid2spatial/velocity` | deg/s | Angular velocity |
| `/vid2spatial/timecode` | seconds | Sync reference |
| `/vid2spatial/spatial` | [az, el, dist, vel, tc] | Bundled atomic message |

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
├── vid2spatial_pkg/              # Core Python package
│   ├── hybrid_tracker.py             # Adaptive-K tracker (DINO + SAM2 + YOLO/ByteTrack)
│   ├── trajectory_stabilizer.py      # RTS smoother + Kalman + 1-Euro filter
│   ├── foa_render.py                 # FOA AmbiX encoding + HRTF binaural + crossfeed
│   ├── pipeline.py                   # End-to-end pipeline orchestration
│   ├── osc_sender.py                 # OSC streaming for DAW
│   ├── vision.py                     # Camera geometry (pixel→ray→angles)
│   ├── depth_metric.py               # Depth Anything V2 integration
│   ├── video_utils.py                # Video I/O, zoom detection
│   ├── config.py                     # Configuration management
│   └── multi_source.py               # Multi-source FOA mixing
│
├── experiments/                  # Experiment scripts & results
│   ├── e2e_20_videos/                # 20 real videos, end-to-end
│   ├── gt_eval_synthetic/            # 15 synthetic GT scenes (Az MAE 0.68°, El MAE 0.18°)
│   ├── sot_15_videos/                # SOT benchmark + HRTF binaural renders
│   └── synthetic_render.py           # Synthetic scenario renderer
│
├── evaluation/                   # Evaluation code & results
│   ├── tracking_ablation/            # SAM2 vs DINO vs Hybrid ablation
│   ├── ablation_output/              # Renderer/baseline ablation
│   ├── comprehensive_results/        # Final evaluation report
│   ├── tests/                        # Unit tests
│   └── plots/                        # Evaluation plots
│
├── docs/                         # Documentation
│   ├── PROJECT_DOCUMENTATION.md      # Full project documentation
│   ├── ARCHITECTURE.md               # System architecture
│   ├── OSC_INTERFACE_SPEC.md         # OSC protocol specification
│   └── FINAL_EVALUATION_REPORT.md    # Evaluation results
│
├── weights/                      # Model weights (gitignored)
├── data/                         # Datasets (gitignored)
├── archive/                      # Old versions (gitignored)
│
├── .gitignore
├── STRUCTURE.md                  # Detailed directory layout
├── pytest.ini
├── requirements.txt
└── README.md
```

See [STRUCTURE.md](STRUCTURE.md) for detailed file descriptions.

---

## Key Technical Details

### Adaptive-K Re-detection
The hybrid tracker uses Grounding DINO for text-guided detection with an adaptive re-detection interval (K). High object motion triggers frequent re-detection (K=2-3), while slow/stationary objects use longer intervals (K=10-15) to save compute. Between keyframes, YOLO/ByteTrack provides frame-to-frame tracking with linear interpolation.

### Dual-Layer Depth
- **Layer 1 (Metric)**: Depth Anything V2 provides absolute depth in meters (run every `depth_stride` frames, default 5)
- **Layer 2 (Proxy)**: BBox-scale inverse proxy for fast-moving objects
- **Blending**: Confidence-weighted fusion — low detection confidence increases proxy weight

### RTS Smoothing
Rauch-Tung-Striebel two-pass optimal smoother (forward Kalman + backward smoother). Applied to azimuth, elevation, and distance jointly. The `depth_render` value used for FOA encoding is the RTS-smoothed metric distance, ensuring stutter-free spatial audio.

### FOA Encoding
AmbiX format (ACN/SN3D): W = mono, Y = sin(az)·cos(el), Z = sin(el), X = cos(az)·cos(el). Distance-dependent gain (inverse-square law) and low-pass filtering simulate natural attenuation.

### HRTF Binaural
8-speaker virtual cube decode → nearest-neighbor HRIR lookup from SOFA file → per-channel convolution. Tested with MIT KEMAR (64,800 measurements, 48kHz, 384-tap FIR).

---

## Documentation

- **[STRUCTURE.md](STRUCTURE.md)** — Detailed directory layout
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — System architecture
- **[docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md)** — Full project documentation
- **[docs/OSC_INTERFACE_SPEC.md](docs/OSC_INTERFACE_SPEC.md)** — OSC protocol specification
- **[docs/FINAL_EVALUATION_REPORT.md](docs/FINAL_EVALUATION_REPORT.md)** — Evaluation results

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

**Last Updated**: 2026-02-07
