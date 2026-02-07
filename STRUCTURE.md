# Vid2Spatial Project Structure

Last updated: 2026-02-07

## Directory Layout

```
vid2spatial/
├── vid2spatial_pkg/          # Core Python package
│   ├── hybrid_tracker.py         # Adaptive-K hybrid tracker (DINO + SAM2)
│   ├── trajectory_stabilizer.py  # RTS smoother + depth_render fix
│   ├── foa_render.py             # FOA encoding + binaural (crossfeed / HRTF SOFA)
│   ├── pipeline.py               # End-to-end pipeline orchestration
│   ├── config.py                 # Configuration management
│   ├── vision.py                 # Camera geometry (pixel_to_ray, ray_to_angles)
│   ├── video_utils.py            # Video I/O utilities
│   ├── depth_metric.py           # Depth estimation
│   └── osc_sender.py             # OSC streaming for DAW
│
├── experiments/              # Experiment scripts + results
│   ├── e2e_20_videos/            # 20 diverse real videos, E2E pipeline
│   ├── gt_eval_synthetic/        # 15 synthetic GT scenes, param accuracy
│   ├── sot_15_videos/            # 15 SOT benchmark videos + 30 re-renders
│   │   ├── render_A_instrument_hrtf/  # HRTF binaural (instrument audio)
│   │   ├── render_B_foley_hrtf/       # HRTF binaural (foley audio)
│   │   ├── render_orig_hrtf/          # HRTF binaural (original audio)
│   │   └── hrtf_quality_analysis.json # HRTF vs crossfeed comparison
│   └── synthetic_render.py       # 15 synthetic scenario renderer
│
├── evaluation/               # Evaluation code + results
│   ├── tracking_ablation/        # Tracker ablation study (SAM2 vs DINO vs Hybrid)
│   ├── ablation_output/          # Renderer/baseline ablation results
│   ├── comprehensive_results/    # Final comprehensive evaluation
│   ├── tests/                    # Unit tests (tracker, OSC, robustness)
│   ├── plots/                    # Evaluation plots (PNG)
│   ├── baseline_ablation.py
│   ├── foa_distance_validation.py
│   └── FOA_DISTANCE_REPORT.md
│
├── docs/                     # Documentation
│   ├── PROJECT_DOCUMENTATION.md  # Full project documentation (thesis)
│   ├── ARCHITECTURE.md           # System architecture
│   ├── OSC_INTERFACE_SPEC.md     # OSC protocol spec
│   └── FINAL_EVALUATION_REPORT.md
│
├── archive/                  # Old versions (gitignored, 4.3GB)
├── data/                     # Datasets (gitignored)
├── weights/                  # Model weights (gitignored)
│
├── .gitignore
├── README.md
├── STRUCTURE.md
├── pytest.ini
└── requirements.txt
```

## Key Files

### Core Implementation
- `vid2spatial_pkg/hybrid_tracker.py` — Adaptive K-frame detection + robustness layer
- `vid2spatial_pkg/trajectory_stabilizer.py` — RTS smoother, depth_render smoothing
- `vid2spatial_pkg/foa_render.py` — FOA AmbiX encoding + HRTF binaural via KEMAR SOFA
- `vid2spatial_pkg/pipeline.py` — Full pipeline orchestration

### Experiments
- `experiments/sot_15_videos/` — 45 HRTF binaural files (15 orig + 15 instrument + 15 foley)
- `experiments/gt_eval_synthetic/` — Parameter accuracy (Az MAE 0.68°, El MAE 0.18°)
- `experiments/e2e_20_videos/` — 19/20 diverse real videos end-to-end

### Evaluation
- `evaluation/tracking_ablation/` — SAM2 3.4% → Adaptive-K 100% amplitude
- `evaluation/ablation_output/` — Renderer/baseline comparisons
