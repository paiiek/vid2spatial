"""
Simple GT-matched IR integration test.

Tests three IR configurations on FAIR-Play samples using existing evaluation infrastructure.
"""
import sys
from pathlib import Path

# Run from vid2spatial directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Use existing ablation study code
from evaluation.ablation_study import run_ablation_study


def main():
    """Run simple IR comparison test."""
    print("="*80)
    print("GT-matched IR Integration Test")
    print("="*80)
    print("\nComparing 3 IR configurations on 3 FAIR-Play samples:")
    print("  1. No IR (current best baseline)")
    print("  2. Schroeder IR (known to degrade performance)")
    print("  3. GT-matched FAIR-Play IR (expected to improve)")
    print("="*80)

    # Use existing ablation study with custom configurations
    configs_to_test = [
        {
            "name": "no_ir",
            "room_disabled": True,
            "ir_backend": "none"
        },
        {
            "name": "schroeder",
            "room_disabled": False,
            "ir_backend": "schroeder",
            "rt60": 0.6
        },
        {
            "name": "gtmatched_fairplay",
            "room_disabled": False,
            "ir_backend": "fairplay",
            "rt60": 0.5
        }
    ]

    # Run with 3 samples
    print("\n[TASK] Running ablation study with 3 samples...")
    print("=" *80)

    try:
        results = run_ablation_study(
            num_samples=3,
            fairplay_root="/home/seung/external/FAIR-Play/",
            output_dir="results/ir_comparison",
            custom_configs=configs_to_test
        )

        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\nResults saved to: results/ir_comparison/")
        print(f"Summary: {results}")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
