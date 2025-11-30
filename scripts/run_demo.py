"""
Refactored end-to-end demo runner using SpatialAudioPipeline.

Usage example:
  python -m mmhoa.vid2spatial.run_demo \
      --video path/to/video.mp4 \
      --audio path/to/mono.wav \
      --out_foa out.foa.wav

Or with config file:
  python -m mmhoa.vid2spatial.run_demo \
      --config config.yaml
"""
import argparse
import sys

from .config import PipelineConfig
from .pipeline import SpatialAudioPipeline


def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser with all pipeline options."""
    ap = argparse.ArgumentParser(
        description="Spatial audio pipeline: video + mono audio → FOA/stereo/binaural"
    )

    # Config file option
    ap.add_argument("--config", type=str, help="YAML config file (overrides other args)")

    # Required inputs
    ap.add_argument("--video", type=str, help="Input video file")
    ap.add_argument("--audio", type=str, help="Input mono audio file")

    # Required output
    ap.add_argument("--out_foa", type=str, help="Output FOA wav file (AmbiX)")

    # Optional outputs
    ap.add_argument("--out_st", type=str, default=None, help="Output stereo wav")
    ap.add_argument("--out_bin", type=str, default=None, help="Output binaural wav")
    ap.add_argument("--save_traj", type=str, default=None, help="Save trajectory JSON")

    # Camera/Video
    ap.add_argument("--fov_deg", type=float, default=60.0, help="Horizontal FOV (degrees)")
    ap.add_argument("--stride", type=int, default=1, help="Video frame sampling stride")

    # Tracking
    ap.add_argument("--method", type=str, default="yolo", choices=["yolo", "kcf", "sam2"],
                    help="Tracking method")
    ap.add_argument("--cls", type=str, default="person", help="Object class name for YOLO")
    ap.add_argument("--select_track_id", type=int, default=None, help="Specific track ID to follow")
    ap.add_argument("--init_bbox", type=str, default=None, help="Initial bbox for KCF: x,y,w,h")
    ap.add_argument("--fallback_center_box", action="store_true",
                    help="Use center box if auto bbox fails")
    ap.add_argument("--smooth_alpha", type=float, default=0.2, help="Tracking smoothing alpha")

    # Depth
    ap.add_argument("--depth_backend", type=str, default="auto", choices=["auto", "midas", "none"],
                    help="Depth estimation backend")
    ap.add_argument("--use_depth_adapter", action="store_true",
                    help="Use depth_anything_adapter for depth estimation")

    # Refinement
    ap.add_argument("--refine_center", action="store_true", help="Refine object center with segmentation")
    ap.add_argument("--refine_center_method", type=str, default="grabcut", choices=["grabcut", "sam2"],
                    help="Center refinement method")
    ap.add_argument("--sam_ckpt", type=str, default=None, help="SAM v1 checkpoint path (fallback)")
    ap.add_argument("--sam2_model_id", type=str, default="facebook/sam2.1-hiera-base-plus",
                    help="SAM2 HuggingFace model ID")
    ap.add_argument("--sam2_cfg", type=str, default=None, help="SAM2 config YAML")
    ap.add_argument("--sam2_ckpt", type=str, default=None, help="SAM2 checkpoint path")

    # Room IR
    ap.add_argument("--room", type=str, default="6,5,3", help="Room dimensions: Lx,Ly,Lz (meters)")
    ap.add_argument("--mic", type=str, default="3,2.5,1.5", help="Mic position: mx,my,mz (meters)")
    ap.add_argument("--rt60", type=float, default=0.6, help="Room reverberation time (seconds)")
    ap.add_argument("--no_ir", action="store_true", help="Disable room IR")
    ap.add_argument("--ir_backend", type=str, default="auto",
                    choices=["auto", "pra", "schroeder", "none", "visual", "brir"],
                    help="IR synthesis backend")
    ap.add_argument("--brir_L", type=str, default=None, help="Left BRIR file")
    ap.add_argument("--brir_R", type=str, default=None, help="Right BRIR file")
    ap.add_argument("--air_foa", type=str, default=None,
                    help="4-channel FOA AIR wav (AmbiX) for per-channel convolution")

    # Spatial rendering
    ap.add_argument("--ang_smooth_ms", type=float, default=50.0,
                    help="Angle smoothing window (milliseconds)")
    ap.add_argument("--max_deg_per_s", type=float, default=None,
                    help="Max angle change rate (degrees/second)")
    ap.add_argument("--dist_gain_k", type=float, default=1.0,
                    help="Distance attenuation factor (0=disabled)")
    ap.add_argument("--dist_lpf_min_hz", type=float, default=800.0,
                    help="Distance LPF minimum cutoff (Hz)")
    ap.add_argument("--dist_lpf_max_hz", type=float, default=8000.0,
                    help="Distance LPF maximum cutoff (Hz)")

    # Trajectory
    ap.add_argument("--traj_json", type=str, default=None,
                    help="Use precomputed trajectory JSON")

    # Occlusion
    ap.add_argument("--occ_json", type=str, default=None,
                    help="Occlusion timeline JSON: {frames:[{frame,occ},...]}")
    ap.add_argument("--estimate_occ", action="store_true",
                    help="Estimate occlusion from depth/bbox")

    # Reverb
    ap.add_argument("--reverb_on", action="store_true", help="Enable time-varying reverb")
    ap.add_argument("--rev_rt60", type=float, default=0.6, help="Reverb RT60")
    ap.add_argument("--rev_wet_min", type=float, default=0.05, help="Reverb wet min")
    ap.add_argument("--rev_wet_max", type=float, default=0.35, help="Reverb wet max")
    ap.add_argument("--rev_wet_occ_boost", type=float, default=0.10, help="Reverb wet occlusion boost")

    # Binaural
    ap.add_argument("--binaural_mode", type=str, default="crossfeed", choices=["crossfeed", "sofa"],
                    help="Binaural rendering mode")
    ap.add_argument("--sofa", type=str, default=None, help="SOFA HRTF file path (.sofa)")

    return ap


def main():
    """Main entry point."""
    ap = create_arg_parser()
    args = ap.parse_args()

    # Handle config file
    if args.config:
        print(f'[info] Loading config from {args.config}')
        try:
            import yaml
            with open(args.config, 'r') as f:
                config_dict = yaml.safe_load(f)
            config = PipelineConfig.from_dict(config_dict)
        except ImportError:
            print('[error] PyYAML not installed. Install with: pip install pyyaml')
            sys.exit(1)
        except Exception as e:
            print(f'[error] Failed to load config: {e}')
            sys.exit(1)
    else:
        # Validate required args
        if not args.video or not args.audio or not args.out_foa:
            print('[error] --video, --audio, and --out_foa are required (or use --config)')
            ap.print_help()
            sys.exit(1)

        # Create config from args
        config = PipelineConfig.from_args(args)

    # Create and run pipeline
    try:
        pipeline = SpatialAudioPipeline(config)
        result = pipeline.run()

        print('\n[success] Pipeline completed!')
        print(f'  Duration: {result["duration_sec"]:.2f}s')
        print(f'  Frames: {result["num_frames"]}')
        print(f'  Sample rate: {result["sample_rate"]} Hz')
        print(f'\nOutputs:')
        for key, path in result["outputs"].items():
            if path:
                print(f'  {key}: {path}')

    except KeyboardInterrupt:
        print('\n[info] Pipeline interrupted by user')
        sys.exit(130)
    except Exception as e:
        print(f'\n[error] Pipeline failed: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
