# Vid2Spatial Project Structure

Last updated: 2026-02-04

## Core Directories

```
vid2spatial/
├── vid2spatial_pkg/     # Core Python package
│   ├── hybrid_tracker.py    # Main tracker (DINO adaptive-K + robustness)
│   ├── trajectory_stabilizer.py  # RTS smoother
│   ├── foa_render.py        # FOA AmbiX rendering
│   ├── osc_sender.py        # OSC streaming for DAW
│   ├── depth_metric.py      # Depth estimation
│   ├── vision.py            # Camera/geometry utilities
│   └── ...
│
├── eval/                # Active evaluation scripts
│   ├── comprehensive_results/   # Final evaluation data
│   │   ├── FINAL_EVALUATION_REPORT.md
│   │   ├── demos/               # Audio demos (FOA, stereo)
│   │   └── *.json               # Metrics data
│   ├── test_adaptive_k_and_rts.py
│   ├── test_robustness_layer.py
│   ├── test_osc_sender.py
│   └── compare_hybrid_vs_redetect.py
│
├── paper/               # ISMAR paper materials
│   └── FINAL_EVALUATION_REPORT.md
│
├── test_videos/         # Test video assets
├── weights/             # Model weights (YOLO, etc.)
├── data/                # Dataset files
│
├── README.md
├── requirements.txt
└── archive/             # Non-essential files (old docs, logs, scripts)
```

## Key Files for ISMAR

1. **Core Implementation**
   - `vid2spatial_pkg/hybrid_tracker.py` - Adaptive K-frame + robustness
   - `vid2spatial_pkg/trajectory_stabilizer.py` - RTS smoother
   - `vid2spatial_pkg/foa_render.py` - FOA rendering
   - `vid2spatial_pkg/osc_sender.py` - DAW integration

2. **Evaluation**
   - `eval/comprehensive_results/FINAL_EVALUATION_REPORT.md` - Main results
   - `eval/comprehensive_results/demos/` - Audio comparisons

3. **Paper**
   - `paper/FINAL_EVALUATION_REPORT.md` - Paper-ready report

## Archive Contents

All non-essential files moved to `archive/`:
- `docs_old/` - Old documentation
- `logs/` - Training and debug logs
- `scripts_old/` - Deprecated scripts
- `eval_old/` - Old evaluation scripts
- `results/`, `wandb/` - Old experiment results
