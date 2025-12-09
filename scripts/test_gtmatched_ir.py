"""
Test script for GT-matched IR integration.

This script tests three IR configurations on a small sample:
1. No IR (current best baseline)
2. Schroeder IR (known to degrade performance)
3. GT-matched FAIR-Play IR (expected to improve)
"""
import sys
import os
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vid2spatial_pkg.pipeline import SpatialAudioPipeline
from vid2spatial_pkg.config import PipelineConfig, RoomConfig, OutputConfig
from evaluation.fairplay_loader import FairPlayDataset
from evaluation.metrics import compute_spatial_metrics
import soundfile as sf
import numpy as np


def test_single_sample(sample, ir_backend="none", ir_disabled=True, output_suffix=""):
    """Test pipeline on a single sample with specified IR config."""
    print(f"\n{'='*60}")
    print(f"Testing: IR backend={ir_backend}, disabled={ir_disabled}")
    print(f"{'='*60}")

    # Create output path
    out_dir = Path("results/ir_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_id = Path(sample['video_path']).stem
    foa_path = out_dir / f"{sample_id}_{output_suffix}.foa.wav"

    # Configure pipeline
    config = PipelineConfig(
        video_path=sample['video_path'],
        audio_path=sample['source_path'],
        room=RoomConfig(
            backend=ir_backend,
            disabled=ir_disabled,
            rt60=0.5  # GT-matched value
        ),
        output=OutputConfig(
            foa_path=str(foa_path)
        )
    )

    # Run pipeline
    try:
        pipeline = SpatialAudioPipeline(config)
        result = pipeline.run()

        # Load outputs
        foa_pred, sr = sf.read(foa_path, always_2d=True)
        foa_pred = foa_pred.T  # [4, T]

        gt_bin, _ = sf.read(sample['binaural_path'], always_2d=True)
        gt_bin = gt_bin.T  # [2, T]

        # Compute metrics
        metrics = compute_spatial_metrics(foa_pred, gt_bin, sr)

        print(f"\n[Results for {ir_backend}, disabled={ir_disabled}]")
        print(f"  Correlation: {metrics['correlation']:.3f}")
        print(f"  ILD Error:   {metrics['ild_error_db']:.2f} dB")
        print(f"  SI-SDR:      {metrics['si_sdr']:.2f} dB")

        return metrics

    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run comparative IR test on 3 samples."""
    print("GT-matched IR Integration Test")
    print("="*60)

    # Load FAIR-Play dataset
    dataset = FairPlayDataset("/home/seung/external/FAIR-Play/")
    samples = dataset.load_samples(num_samples=3)

    print(f"Loaded {len(samples)} test samples")

    # Test configurations
    configs = [
        ("no_ir", "none", True),           # Current best baseline
        ("schroeder", "schroeder", False),  # Known to degrade
        ("gtmatched", "fairplay", False),   # GT-matched (expected to improve)
    ]

    all_results = {name: [] for name, _, _ in configs}

    # Run tests
    for i, sample in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"Sample {i+1}/{len(samples)}: {Path(sample['video_path']).name}")
        print(f"{'='*60}")

        for name, backend, disabled in configs:
            metrics = test_single_sample(
                sample,
                ir_backend=backend,
                ir_disabled=disabled,
                output_suffix=name
            )
            if metrics:
                all_results[name].append(metrics)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Average Metrics")
    print("="*60)

    for name, _ , _ in configs:
        results = all_results[name]
        if results:
            avg_corr = np.mean([r['correlation'] for r in results])
            avg_ild = np.mean([r['ild_error_db'] for r in results])
            avg_sisdr = np.mean([r['si_sdr'] for r in results])

            print(f"\n{name.upper()}:")
            print(f"  Correlation: {avg_corr:.3f}")
            print(f"  ILD Error:   {avg_ild:.2f} dB")
            print(f"  SI-SDR:      {avg_sisdr:.2f} dB")

    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    corr_no_ir = np.mean([r['correlation'] for r in all_results['no_ir']])
    corr_schroeder = np.mean([r['correlation'] for r in all_results['schroeder']])
    corr_gtmatched = np.mean([r['correlation'] for r in all_results['gtmatched']])

    print(f"\nSchroeder IR impact: {corr_schroeder - corr_no_ir:+.3f}")
    print(f"GT-matched IR impact: {corr_gtmatched - corr_no_ir:+.3f}")

    if corr_gtmatched > corr_no_ir:
        print("\n✅ GT-matched IR IMPROVES performance!")
    elif corr_gtmatched > corr_schroeder:
        print("\n⚠️  GT-matched IR better than Schroeder, but still worse than no IR")
    else:
        print("\n❌ GT-matched IR failed to improve over Schroeder")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
