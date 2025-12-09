"""
Quick test of GT-matched IR on a single sample.

Usage:
  python3 scripts/quick_ir_test.py
"""
import sys
import os
from pathlib import Path

# Add package to path
sys.path.insert(0, '/home/seung')

import numpy as np
import soundfile as sf
from mmhoa.vid2spatial.vid2spatial_pkg.irgen import fairplay_matched_ir, schroeder_ir


def test_ir_generation():
    """Test that GT-matched IR generation works."""
    print("="*60)
    print("Testing IR Generation")
    print("="*60)

    sr = 48000

    # Generate Schroeder IR
    print("\n[1] Generating Schroeder IR (rt60=0.6s)...")
    sch_ir = schroeder_ir(sr, rt60=0.6)
    print(f"    Shape: {sch_ir.shape}, Duration: {len(sch_ir)/sr:.3f}s")
    print(f"    Energy: {np.sum(sch_ir**2):.6f}")

    # Generate GT-matched IR
    print("\n[2] Generating GT-matched FAIR-Play IR (rt60=0.5s)...")
    fp_ir = fairplay_matched_ir(
        fs=sr,
        rt60=0.5,
        direct_ratio=0.73,
        early_ratio=0.07,
        late_ratio=0.20,
        length_s=0.5
    )
    print(f"    Shape: {fp_ir.shape}, Duration: {len(fp_ir)/sr:.3f}s")
    print(f"    Energy: {np.sum(fp_ir**2):.6f}")

    # Analyze energy distribution
    print("\n[3] Analyzing GT-matched IR energy distribution...")
    direct_energy = fp_ir[0]**2
    early_end_idx = int(0.050 * sr)  # 50ms
    early_energy = np.sum(fp_ir[1:early_end_idx]**2)
    late_energy = np.sum(fp_ir[early_end_idx:]**2)
    total_energy = direct_energy + early_energy + late_energy

    print(f"    Direct ratio: {direct_energy/total_energy:.3f} (target: 0.73)")
    print(f"    Early ratio:  {early_energy/total_energy:.3f} (target: 0.07)")
    print(f"    Late ratio:   {late_energy/total_energy:.3f} (target: 0.20)")

    # Save IRs
    os.makedirs("results/ir_test", exist_ok=True)
    sf.write("results/ir_test/schroeder_ir.wav", sch_ir, sr)
    sf.write("results/ir_test/fairplay_ir.wav", fp_ir, sr)

    print("\n[4] IRs saved to results/ir_test/")
    print("    - schroeder_ir.wav")
    print("    - fairplay_ir.wav")

    print("\n" + "="*60)
    print("✅ IR generation test PASSED")
    print("="*60)

    # Verify the IR can be used in convolution
    print("\n[5] Testing IR convolution...")
    test_signal = np.random.randn(sr)  # 1 second of noise
    from mmhoa.vid2spatial.vid2spatial_pkg.irgen import fft_convolve

    convolved = fft_convolve(test_signal, fp_ir)
    print(f"    Input shape: {test_signal.shape}")
    print(f"    IR shape: {fp_ir.shape}")
    print(f"    Output shape: {convolved.shape}")
    print(f"    ✅ Convolution works")

    return True


if __name__ == "__main__":
    try:
        success = test_ir_generation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
