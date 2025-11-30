# Vid2Spatial: Video-to-Spatial Audio Generation

**Real-time geometric approach for video-to-spatial audio using depth estimation and object tracking.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## 🎯 Overview

Vid2Spatial generates First-Order Ambisonics (FOA) spatial audio from monaural audio and video input.

**Pipeline**:
- Input: Video (MP4) + Mono Audio (WAV)
- Vision: Depth (MiDaS) + Tracking (YOLO/KCF/SAM2)
- Output: FOA spatial audio (AmbiX: ACN/SN3D, channel order [W,Y,Z,X])

**Key Features**:
- ✅ **Near real-time** (0.91× RTF on RTX 2080 Ti)
- ✅ **Interpretable pipeline** (geometric, no black-box)
- ✅ **Modular design** (easy to extend)
- ✅ **Strong spatial accuracy** (ILD error: 1.91 dB)

---

## 📊 Performance (FAIR-Play Dataset)

Evaluated on 20 samples:

| Metric | Score | Quality |
|--------|-------|---------|
| **Correlation** | 0.72 ± 0.11 | Good |
| **ILD Error** | 1.91 ± 1.14 dB | Excellent |
| **SI-SDR** | +0.7 ± 3.1 dB | Positive |
| **RTF** | 0.91× | Near real-time |

See [docs/FAIR_PLAY_EVALUATION_REPORT.md](docs/FAIR_PLAY_EVALUATION_REPORT.md) for details.

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/vid2spatial.git
cd vid2spatial

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from pipeline import SpatialAudioPipeline
from config import PipelineConfig, OutputConfig

# Configure pipeline
config = PipelineConfig(
    video_path="input.mp4",
    audio_path="input.wav",
    output=OutputConfig(foa_path="output.foa.wav")
)

# Run pipeline
pipeline = SpatialAudioPipeline(config)
pipeline.process()
```

### Command Line

```bash
python scripts/run_demo.py \
    --video input.mp4 \
    --audio input.wav \
    --out_foa output.foa.wav \
    --method kcf \
    --ir_backend none
```

**Options**:
- `--method`: Tracking method (`yolo`, `kcf`, `sam2`)
- `--cls`: YOLO class filter (e.g., `person`)
- `--ir_backend`: IR backend (`none`, `schroeder`, `pra`)
- `--depth_backend`: Depth backend (`auto`, `midas`, `none`)
- `--smooth_alpha`: Temporal smoothing (0.0-1.0, default: 0.2)

---

## 📁 Project Structure

```
vid2spatial/
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── config.py                 # Configuration
├── pipeline.py               # Main pipeline
├── vision.py                 # Vision (depth + tracking)
├── foa_render.py            # FOA encoding
├── irgen.py                 # IR generation
├── utils.py                 # Utilities
│
├── evaluation/              # Evaluation code
│   ├── fairplay_loader.py   # Dataset loader
│   ├── metrics.py           # Metrics
│   ├── baseline_systems.py  # Baselines
│   ├── ablation_study.py    # Ablation study
│   ├── learned_ir.py        # IR learning
│   └── improved_ir.py       # GT-matched IR
│
├── scripts/                 # Scripts
│   ├── run_demo.py          # Demo
│   └── train_ir_predictor.py # Train IR
│
├── docs/                    # Documentation
│   ├── ABLATION_STUDY_REPORT.md
│   ├── FAIR_PLAY_EVALUATION_REPORT.md
│   └── CRITICAL_ACADEMIC_EVALUATION.md
│
├── tests/                   # Tests
│   ├── test_pipeline.py
│   └── test_vision.py
│
└── results/                 # Outputs (gitignored)
```

---

## 🔬 Evaluation

### Run Evaluation

```bash
# Ablation study (5 configurations)
python evaluation/ablation_study.py --num_samples 5

# Baseline comparison
python evaluation/baseline_systems.py --num_samples 20
```

### Key Findings (Ablation Study)

| Configuration | Correlation | SI-SDR | Notes |
|---------------|-------------|--------|-------|
| **No IR (recommended)** | **0.72** | **+0.7 dB** | ✅ Best |
| With IR (Schroeder) | 0.37 | -8.6 dB | ❌ Degrades |
| No depth | 0.38 | -8.1 dB | 4× slower |
| No smoothing | 0.37 | -8.6 dB | Negligible |

**Critical finding**: IR convolution degrades performance by 50% because FAIR-Play uses dry acoustics (direct=73%, early=7%, late=20%).

See [docs/ABLATION_STUDY_REPORT.md](docs/ABLATION_STUDY_REPORT.md).

---

## 🛠️ Advanced Usage

### Custom Configuration

```python
from config import (
    PipelineConfig, VisionConfig, DepthConfig,
    TrackingConfig, RoomConfig, OutputConfig
)

config = PipelineConfig(
    video_path="input.mp4",
    audio_path="input.wav",
    vision=VisionConfig(
        depth=DepthConfig(backend='midas'),
        tracking=TrackingConfig(
            method='kcf',
            smooth_alpha=0.2
        )
    ),
    room=RoomConfig(
        disabled=True  # Disable IR (recommended)
    ),
    output=OutputConfig(foa_path="output.foa.wav")
)
```

### Train IR Predictor

```bash
python scripts/train_ir_predictor.py \
    --dataset results/ir_dataset_50.json \
    --epochs 100 \
    --device cuda
```

---

## 📈 Roadmap

### Completed ✅
- [x] Real-time geometric pipeline
- [x] FAIR-Play evaluation (20 samples)
- [x] Ablation study
- [x] Learned IR module
- [x] Baseline comparisons

### In Progress 🔄
- [ ] Multi-source support (2-3 sources)
- [ ] Modern tracker (OSTrack)
- [ ] Metric depth (Depth Anything V2)
- [ ] 100+ sample evaluation

### Future Work 📝
- [ ] Neural refiner
- [ ] Cross-dataset validation
- [ ] User study
- [ ] Real-time demo app

---

## 📚 Citation

```bibtex
@misc{vid2spatial2024,
  title={Vid2Spatial: Video-to-Spatial Audio with Geometric Encoding},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/YOUR_USERNAME/vid2spatial}}
}
```

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push and open a Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE).

---

## 🙏 Acknowledgments

- **MiDaS** for depth estimation
- **FAIR-Play** dataset
- **OpenCV** for tracking

---

**Last Updated**: 2024-11-30
