"""
Ablation Study: Evaluate impact of each component

Components to ablate:
1. Depth estimation (MiDaS) vs No depth (constant distance)
2. Temporal smoothing vs No smoothing
3. IR convolution vs Direct (no room acoustics)
4. Refactored vision vs Original vision

For ICASSP-level analysis.
"""
import sys
sys.path.insert(0, '/home/seung')

import json
import time
import numpy as np
from pathlib import Path
import soundfile as sf
import librosa

from fairplay_loader import FairPlayDataset
from evaluation_v2 import evaluate_spatial_audio_v2
from mmhoa.vid2spatial.config import (
    PipelineConfig, VisionConfig, TrackingConfig, DepthConfig, RoomConfig, OutputConfig, SpatialConfig
)
from mmhoa.vid2spatial.pipeline import SpatialAudioPipeline


def run_ablation_configs(sample, output_dir: Path):
    """
    Run multiple pipeline configurations for ablation study.

    Configurations:
    1. Full (baseline)
    2. No depth
    3. No smoothing
    4. No IR
    5. No depth + No smoothing
    """
    sample_id = sample['sample_id']
    video_path = sample['video_path']
    mono_audio = sample['mono_audio']
    sr = sample['sample_rate']

    # Save mono temporarily
    mono_path = output_dir / f"{sample_id}_mono.wav"
    sf.write(str(mono_path), mono_audio, sr)

    # Get video dimensions
    import cv2
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    init_bbox = (width//2 - 40, height//2 - 60, 80, 120)

    configs = {
        '1_full': {
            'use_depth': True,
            'smooth_alpha': 0.2,
            'use_ir': True,
        },
        '2_no_depth': {
            'use_depth': False,
            'smooth_alpha': 0.2,
            'use_ir': True,
        },
        '3_no_smoothing': {
            'use_depth': True,
            'smooth_alpha': 0.0,  # No smoothing
            'use_ir': True,
        },
        '4_no_ir': {
            'use_depth': True,
            'smooth_alpha': 0.2,
            'use_ir': False,
        },
        '5_minimal': {
            'use_depth': False,
            'smooth_alpha': 0.0,
            'use_ir': False,
        },
    }

    results = {}

    for config_name, config_params in configs.items():
        print(f"\n  [{config_name}] ", end='')

        foa_path = output_dir / f"{sample_id}_{config_name}.foa.wav"
        traj_path = output_dir / f"{sample_id}_{config_name}_traj.json"

        # Build pipeline config
        pipeline_config = PipelineConfig(
            video_path=video_path,
            audio_path=str(mono_path),
            vision=VisionConfig(
                tracking=TrackingConfig(
                    method='kcf',
                    init_bbox=init_bbox,
                    smooth_alpha=config_params['smooth_alpha'],
                ),
                depth=DepthConfig(backend='none') if not config_params['use_depth'] else DepthConfig(),
            ),
            room=RoomConfig(
                disabled=not config_params['use_ir'],
            ),
            spatial=SpatialConfig(),
            output=OutputConfig(
                foa_path=str(foa_path),
                stereo_path=None,
                trajectory_path=str(traj_path),
            )
        )

        # Run pipeline
        start_time = time.time()
        try:
            pipeline = SpatialAudioPipeline(pipeline_config, use_refactored_vision=True)
            result = pipeline.run()
            processing_time = time.time() - start_time
            success = True
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"FAILED: {e}")
            success = False
            results[config_name] = {'success': False, 'error': str(e)}
            continue

        # Load FOA
        foa, foa_sr = sf.read(str(foa_path))
        foa = foa.T

        # Compute metrics
        try:
            metrics = evaluate_spatial_audio_v2(foa, sample['gt_binaural'], foa_sr)
        except Exception as e:
            print(f"Metrics failed: {e}")
            metrics = {}

        results[config_name] = {
            'success': True,
            'processing_time': processing_time,
            'rtf': result['duration_sec'] / processing_time,
            'metrics': metrics,
        }

        print(f"✓ RTF={results[config_name]['rtf']:.2f}x, Corr={metrics.get('correlation_L', 0):.3f}")

    # Clean up
    if mono_path.exists():
        mono_path.unlink()

    return results


def run_ablation_study(num_samples: int = 5, output_dir: str = "ablation_study"):
    """
    Run ablation study on subset of FAIR-Play.

    Args:
        num_samples: Number of samples to test
        output_dir: Output directory
    """
    print("\n" + "="*70)
    print("ABLATION STUDY")
    print("="*70)
    print(f"\nTesting {num_samples} samples with 5 configurations")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Load dataset
    dataset = FairPlayDataset()
    samples = dataset.get_subset(num_samples=num_samples)

    all_results = []

    for i, sample in enumerate(samples):
        sample_id = sample['sample_id']
        print(f"\n[{i+1}/{num_samples}] Sample {sample_id}")

        results = run_ablation_configs(sample, output_path)
        all_results.append({
            'sample_id': sample_id,
            'configs': results,
        })

    # Aggregate results
    print("\n" + "="*70)
    print("AGGREGATED RESULTS")
    print("="*70)

    config_names = ['1_full', '2_no_depth', '3_no_smoothing', '4_no_ir', '5_minimal']

    summary = {}

    for config_name in config_names:
        # Collect metrics from all samples
        metrics_list = []
        rtfs = []

        for sample_result in all_results:
            if config_name in sample_result['configs'] and sample_result['configs'][config_name]['success']:
                metrics_list.append(sample_result['configs'][config_name]['metrics'])
                rtfs.append(sample_result['configs'][config_name]['rtf'])

        if not metrics_list:
            continue

        # Aggregate
        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
            }

        summary[config_name] = {
            'num_samples': len(metrics_list),
            'rtf': {
                'mean': float(np.mean(rtfs)),
                'std': float(np.std(rtfs)),
            },
            'metrics': aggregated,
        }

    # Print comparison
    print("\n Configuration Comparison:")
    print("-" * 70)
    print(f"{'Config':<20} {'RTF':>8} {'Corr_L':>8} {'ILD_err':>8} {'SI-SDR':>8}")
    print("-" * 70)

    for config_name in config_names:
        if config_name not in summary:
            continue

        s = summary[config_name]
        rtf = s['rtf']['mean']
        corr = s['metrics']['correlation_L']['mean']
        ild = s['metrics']['ild_error_db']['mean']
        sdr = s['metrics']['si_sdr_L']['mean']

        print(f"{config_name:<20} {rtf:>8.2f} {corr:>8.3f} {ild:>8.2f} {sdr:>8.2f}")

    # Save results
    results_path = output_path / "ablation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'summary': summary,
            'all_results': all_results,
        }, f, indent=2)

    print(f"\n✓ Results saved to: {results_path}")

    # Analysis
    print("\n" + "="*70)
    print("ABLATION ANALYSIS")
    print("="*70)

    if '1_full' in summary and '2_no_depth' in summary:
        full_corr = summary['1_full']['metrics']['correlation_L']['mean']
        no_depth_corr = summary['2_no_depth']['metrics']['correlation_L']['mean']
        depth_impact = (full_corr - no_depth_corr) / full_corr * 100

        full_rtf = summary['1_full']['rtf']['mean']
        no_depth_rtf = summary['2_no_depth']['rtf']['mean']
        depth_speedup = no_depth_rtf / full_rtf

        print(f"\n📊 Depth Estimation Impact:")
        print(f"  Quality change: {depth_impact:+.1f}% (Correlation)")
        print(f"  Speed change: {depth_speedup:.2f}x faster without depth")

    if '1_full' in summary and '3_no_smoothing' in summary:
        full_corr = summary['1_full']['metrics']['correlation_L']['mean']
        no_smooth_corr = summary['3_no_smoothing']['metrics']['correlation_L']['mean']
        smooth_impact = (full_corr - no_smooth_corr) / full_corr * 100

        print(f"\n📊 Temporal Smoothing Impact:")
        print(f"  Quality change: {smooth_impact:+.1f}% (Correlation)")

    if '1_full' in summary and '4_no_ir' in summary:
        full_corr = summary['1_full']['metrics']['correlation_L']['mean']
        no_ir_corr = summary['4_no_ir']['metrics']['correlation_L']['mean']
        ir_impact = (full_corr - no_ir_corr) / full_corr * 100

        print(f"\n📊 IR Convolution Impact:")
        print(f"  Quality change: {ir_impact:+.1f}% (Correlation)")

    return summary


if __name__ == "__main__":
    import argparse
    import warnings
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples")
    args = parser.parse_args()

    summary = run_ablation_study(num_samples=args.num_samples)

    print("\n" + "="*70)
    print("✓ Ablation study complete!")
    print("="*70)
