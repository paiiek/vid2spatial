"""
Tracking Ablation Configurations.

Defines all ablation conditions for systematic evaluation.
Each configuration isolates a single variable change.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class TrackerBackend(Enum):
    """Tracker backend options."""
    SAM2 = "sam2"                    # SAM2 mask propagation only
    DINO_K1 = "dino_k1"              # DINO detection every frame
    DINO_K5 = "dino_k5"              # DINO detection every 5 frames
    DINO_K10 = "dino_k10"            # DINO detection every 10 frames
    ADAPTIVE_K = "adaptive_k"        # Motion-aware detection interval
    YOLO = "yolo"                    # YOLO + ByteTrack


class InterpolationMethod(Enum):
    """Interpolation between keyframes."""
    NONE = "none"                    # Hold last keyframe value
    LINEAR = "linear"                # Linear interpolation
    CUBIC = "cubic"                  # Cubic spline (smoother)


class SmoothingMethod(Enum):
    """Post-extraction smoothing."""
    NONE = "none"                    # No smoothing
    EMA = "ema"                      # Exponential moving average
    RTS = "rts"                      # Rauch-Tung-Striebel smoother


@dataclass
class RobustnessConfig:
    """Robustness layer configuration."""
    enabled: bool = True
    confidence_gating: bool = True   # Reject low-confidence detections
    confidence_threshold: float = 0.35
    jump_rejection: bool = True      # Reject large velocity jumps
    jump_threshold: float = 150.0    # pixels/frame


@dataclass
class AblationConfig:
    """Complete configuration for one ablation condition."""
    name: str
    description: str

    # Tracker backend
    tracker: TrackerBackend = TrackerBackend.ADAPTIVE_K

    # K-frame detection (for DINO variants)
    k_frame: int = 5

    # Interpolation
    interpolation: InterpolationMethod = InterpolationMethod.LINEAR

    # Smoothing
    smoothing: SmoothingMethod = SmoothingMethod.RTS
    ema_alpha: float = 0.3           # For EMA smoothing

    # Robustness
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tracker": self.tracker.value,
            "k_frame": self.k_frame,
            "interpolation": self.interpolation.value,
            "smoothing": self.smoothing.value,
            "ema_alpha": self.ema_alpha,
            "robustness": {
                "enabled": self.robustness.enabled,
                "confidence_gating": self.robustness.confidence_gating,
                "confidence_threshold": self.robustness.confidence_threshold,
                "jump_rejection": self.robustness.jump_rejection,
                "jump_threshold": self.robustness.jump_threshold,
            }
        }


# =============================================================================
# A. Tracker Backend Ablation
# =============================================================================

TRACKER_ABLATION_CONFIGS = [
    AblationConfig(
        name="tracker_sam2",
        description="SAM2 mask propagation only (baseline)",
        tracker=TrackerBackend.SAM2,
        interpolation=InterpolationMethod.NONE,
        smoothing=SmoothingMethod.NONE,
        robustness=RobustnessConfig(enabled=False),
    ),
    AblationConfig(
        name="tracker_dino_k1",
        description="DINO detection every frame",
        tracker=TrackerBackend.DINO_K1,
        k_frame=1,
        interpolation=InterpolationMethod.NONE,
        smoothing=SmoothingMethod.NONE,
        robustness=RobustnessConfig(enabled=False),
    ),
    AblationConfig(
        name="tracker_dino_k5",
        description="DINO detection every 5 frames",
        tracker=TrackerBackend.DINO_K5,
        k_frame=5,
        interpolation=InterpolationMethod.LINEAR,
        smoothing=SmoothingMethod.NONE,
        robustness=RobustnessConfig(enabled=False),
    ),
    AblationConfig(
        name="tracker_dino_k10",
        description="DINO detection every 10 frames",
        tracker=TrackerBackend.DINO_K10,
        k_frame=10,
        interpolation=InterpolationMethod.LINEAR,
        smoothing=SmoothingMethod.NONE,
        robustness=RobustnessConfig(enabled=False),
    ),
    AblationConfig(
        name="tracker_adaptive_k",
        description="Motion-aware detection interval",
        tracker=TrackerBackend.ADAPTIVE_K,
        interpolation=InterpolationMethod.LINEAR,
        smoothing=SmoothingMethod.NONE,
        robustness=RobustnessConfig(enabled=False),
    ),
]


# =============================================================================
# B. Interpolation Ablation (using DINO_K5 as base)
# =============================================================================

INTERPOLATION_ABLATION_CONFIGS = [
    AblationConfig(
        name="interp_none",
        description="No interpolation (hold last keyframe)",
        tracker=TrackerBackend.DINO_K5,
        k_frame=5,
        interpolation=InterpolationMethod.NONE,
        smoothing=SmoothingMethod.NONE,
        robustness=RobustnessConfig(enabled=False),
    ),
    AblationConfig(
        name="interp_linear",
        description="Linear interpolation between keyframes",
        tracker=TrackerBackend.DINO_K5,
        k_frame=5,
        interpolation=InterpolationMethod.LINEAR,
        smoothing=SmoothingMethod.NONE,
        robustness=RobustnessConfig(enabled=False),
    ),
]


# =============================================================================
# C. Smoothing Ablation (using Adaptive-K + Linear interp as base)
# =============================================================================

SMOOTHING_ABLATION_CONFIGS = [
    AblationConfig(
        name="smooth_none",
        description="No smoothing",
        tracker=TrackerBackend.ADAPTIVE_K,
        interpolation=InterpolationMethod.LINEAR,
        smoothing=SmoothingMethod.NONE,
        robustness=RobustnessConfig(enabled=False),
    ),
    AblationConfig(
        name="smooth_ema",
        description="EMA smoothing (alpha=0.3)",
        tracker=TrackerBackend.ADAPTIVE_K,
        interpolation=InterpolationMethod.LINEAR,
        smoothing=SmoothingMethod.EMA,
        ema_alpha=0.3,
        robustness=RobustnessConfig(enabled=False),
    ),
    AblationConfig(
        name="smooth_rts",
        description="RTS smoothing (optimal offline)",
        tracker=TrackerBackend.ADAPTIVE_K,
        interpolation=InterpolationMethod.LINEAR,
        smoothing=SmoothingMethod.RTS,
        robustness=RobustnessConfig(enabled=False),
    ),
]


# =============================================================================
# D. Robustness Layer Ablation (using Adaptive-K + Linear + RTS as base)
# =============================================================================

ROBUSTNESS_ABLATION_CONFIGS = [
    AblationConfig(
        name="robust_off",
        description="No robustness layer",
        tracker=TrackerBackend.ADAPTIVE_K,
        interpolation=InterpolationMethod.LINEAR,
        smoothing=SmoothingMethod.RTS,
        robustness=RobustnessConfig(enabled=False),
    ),
    AblationConfig(
        name="robust_conf_only",
        description="Confidence gating only",
        tracker=TrackerBackend.ADAPTIVE_K,
        interpolation=InterpolationMethod.LINEAR,
        smoothing=SmoothingMethod.RTS,
        robustness=RobustnessConfig(
            enabled=True,
            confidence_gating=True,
            jump_rejection=False,
        ),
    ),
    AblationConfig(
        name="robust_jump_only",
        description="Jump rejection only",
        tracker=TrackerBackend.ADAPTIVE_K,
        interpolation=InterpolationMethod.LINEAR,
        smoothing=SmoothingMethod.RTS,
        robustness=RobustnessConfig(
            enabled=True,
            confidence_gating=False,
            jump_rejection=True,
        ),
    ),
    AblationConfig(
        name="robust_full",
        description="Full robustness (conf + jump)",
        tracker=TrackerBackend.ADAPTIVE_K,
        interpolation=InterpolationMethod.LINEAR,
        smoothing=SmoothingMethod.RTS,
        robustness=RobustnessConfig(
            enabled=True,
            confidence_gating=True,
            jump_rejection=True,
        ),
    ),
]


# =============================================================================
# Full System Configuration (Paper's Proposed Method)
# =============================================================================

FULL_SYSTEM_CONFIG = AblationConfig(
    name="full_system",
    description="Complete proposed system (Adaptive-K + Linear + RTS + Full Robustness)",
    tracker=TrackerBackend.ADAPTIVE_K,
    interpolation=InterpolationMethod.LINEAR,
    smoothing=SmoothingMethod.RTS,
    robustness=RobustnessConfig(
        enabled=True,
        confidence_gating=True,
        jump_rejection=True,
    ),
)


# =============================================================================
# All Configs for Complete Ablation
# =============================================================================

def get_all_ablation_configs() -> List[AblationConfig]:
    """Get all unique ablation configurations."""
    all_configs = (
        TRACKER_ABLATION_CONFIGS +
        INTERPOLATION_ABLATION_CONFIGS +
        SMOOTHING_ABLATION_CONFIGS +
        ROBUSTNESS_ABLATION_CONFIGS +
        [FULL_SYSTEM_CONFIG]
    )

    # Remove duplicates by name
    seen = set()
    unique = []
    for cfg in all_configs:
        if cfg.name not in seen:
            seen.add(cfg.name)
            unique.append(cfg)

    return unique


def get_configs_by_category() -> Dict[str, List[AblationConfig]]:
    """Get configs organized by ablation category."""
    return {
        "A_tracker_backend": TRACKER_ABLATION_CONFIGS,
        "B_interpolation": INTERPOLATION_ABLATION_CONFIGS,
        "C_smoothing": SMOOTHING_ABLATION_CONFIGS,
        "D_robustness": ROBUSTNESS_ABLATION_CONFIGS,
    }


__all__ = [
    "TrackerBackend",
    "InterpolationMethod",
    "SmoothingMethod",
    "RobustnessConfig",
    "AblationConfig",
    "TRACKER_ABLATION_CONFIGS",
    "INTERPOLATION_ABLATION_CONFIGS",
    "SMOOTHING_ABLATION_CONFIGS",
    "ROBUSTNESS_ABLATION_CONFIGS",
    "FULL_SYSTEM_CONFIG",
    "get_all_ablation_configs",
    "get_configs_by_category",
]
