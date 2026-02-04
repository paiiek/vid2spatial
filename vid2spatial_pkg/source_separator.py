"""
Audio Source Separation for Multi-Source Spatial Audio

Mono mix → separated sources using neural network models

지원 모델:
1. Demucs (music source separation)
2. SepFormer (speech separation)
3. Conv-TasNet (general separation)

설치:
    pip install demucs
    pip install asteroid-filterbanks
    pip install speechbrain
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import warnings


class SourceSeparator:
    """
    Audio source separation wrapper.

    Usage:
        separator = SourceSeparator(model="demucs")
        sources = separator.separate(mono_mix, num_sources=2)
    """

    def __init__(
        self,
        model: str = "demucs",
        device: str = "cuda",
        sample_rate: int = 44100,
    ):
        """
        Initialize source separator.

        Args:
            model: Separation model
                - "demucs": Best for music (vocals, drums, bass, other)
                - "sepformer": Best for speech
                - "convtasnet": General purpose
                - "none": No separation (just duplicate mono)
            device: cuda or cpu
            sample_rate: Expected sample rate
        """
        self.model_name = model
        self.device = device
        self.sample_rate = sample_rate
        self.model = None

        if model != "none":
            self._load_model()

    def _load_model(self):
        """Load separation model."""
        if self.model_name == "demucs":
            self._load_demucs()
        elif self.model_name == "sepformer":
            self._load_sepformer()
        elif self.model_name == "convtasnet":
            self._load_convtasnet()
        else:
            warnings.warn(f"Unknown model {self.model_name}, using no separation")

    def _load_demucs(self):
        """Load Demucs model."""
        try:
            import torch
            from demucs import pretrained
            from demucs.apply import apply_model

            self.model = pretrained.get_model("htdemucs")
            self.model.to(self.device)
            self.model.eval()
            self._apply_fn = apply_model
            print(f"[separator] Loaded Demucs on {self.device}")

        except ImportError:
            warnings.warn("Demucs not installed. Run: pip install demucs")
            self.model = None
        except Exception as e:
            warnings.warn(f"Failed to load Demucs: {e}")
            self.model = None

    def _load_sepformer(self):
        """Load SepFormer model."""
        try:
            from speechbrain.inference.separation import SepformerSeparation

            self.model = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-wsj02mix",
                savedir="pretrained_models/sepformer-wsj02mix",
                run_opts={"device": self.device}
            )
            print(f"[separator] Loaded SepFormer on {self.device}")

        except ImportError:
            warnings.warn("SpeechBrain not installed. Run: pip install speechbrain")
            self.model = None
        except Exception as e:
            warnings.warn(f"Failed to load SepFormer: {e}")
            self.model = None

    def _load_convtasnet(self):
        """Load Conv-TasNet model."""
        try:
            from asteroid.models import ConvTasNet

            self.model = ConvTasNet.from_pretrained("mpariente/ConvTasNet_WHAM!_sepclean")
            self.model.to(self.device)
            self.model.eval()
            print(f"[separator] Loaded Conv-TasNet on {self.device}")

        except ImportError:
            warnings.warn("Asteroid not installed. Run: pip install asteroid-filterbanks")
            self.model = None
        except Exception as e:
            warnings.warn(f"Failed to load Conv-TasNet: {e}")
            self.model = None

    def separate(
        self,
        mono_mix: np.ndarray,
        num_sources: int = 2,
        sr: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Separate mono mix into individual sources.

        Args:
            mono_mix: Mono audio signal [T]
            num_sources: Number of sources to extract
            sr: Sample rate (if different from init)

        Returns:
            List of separated sources [source1, source2, ...]
        """
        if sr is None:
            sr = self.sample_rate

        if self.model is None:
            # Fallback: just return copies
            return self._fallback_separate(mono_mix, num_sources)

        if self.model_name == "demucs":
            return self._separate_demucs(mono_mix, num_sources, sr)
        elif self.model_name == "sepformer":
            return self._separate_sepformer(mono_mix, num_sources, sr)
        elif self.model_name == "convtasnet":
            return self._separate_convtasnet(mono_mix, num_sources, sr)
        else:
            return self._fallback_separate(mono_mix, num_sources)

    def _fallback_separate(
        self,
        mono_mix: np.ndarray,
        num_sources: int,
    ) -> List[np.ndarray]:
        """Fallback: frequency band splitting."""
        print("[separator] Using fallback frequency band splitting")

        from scipy.signal import butter, filtfilt

        sources = []
        nyq = self.sample_rate / 2

        # Split into frequency bands
        if num_sources == 2:
            # Low + High split at 2kHz
            cutoff = 2000 / nyq
            b, a = butter(4, cutoff, btype='low')
            low = filtfilt(b, a, mono_mix).astype(np.float32)

            b, a = butter(4, cutoff, btype='high')
            high = filtfilt(b, a, mono_mix).astype(np.float32)

            sources = [low, high]

        elif num_sources == 3:
            # Low / Mid / High
            low_cut = 500 / nyq
            high_cut = 4000 / nyq

            b, a = butter(4, low_cut, btype='low')
            low = filtfilt(b, a, mono_mix).astype(np.float32)

            b, a = butter(4, [low_cut, high_cut], btype='band')
            mid = filtfilt(b, a, mono_mix).astype(np.float32)

            b, a = butter(4, high_cut, btype='high')
            high = filtfilt(b, a, mono_mix).astype(np.float32)

            sources = [low, mid, high]

        else:
            # Just duplicate
            sources = [mono_mix.copy() for _ in range(num_sources)]

        return sources

    def _separate_demucs(
        self,
        mono_mix: np.ndarray,
        num_sources: int,
        sr: int,
    ) -> List[np.ndarray]:
        """Separate using Demucs."""
        import torch
        from scipy.signal import resample_poly

        # Demucs expects 44100Hz
        if sr != 44100:
            mono_resampled = resample_poly(mono_mix, 44100, sr).astype(np.float32)
        else:
            mono_resampled = mono_mix.astype(np.float32)

        # Demucs expects [batch, channels, samples]
        # Mono → stereo (duplicate)
        audio_tensor = torch.from_numpy(mono_resampled).unsqueeze(0).unsqueeze(0)
        audio_tensor = audio_tensor.repeat(1, 2, 1)  # [1, 2, T]
        audio_tensor = audio_tensor.to(self.device)

        # Separate
        with torch.no_grad():
            sources = self._apply_fn(
                self.model,
                audio_tensor,
                device=self.device,
                progress=False,
            )  # [1, num_sources, 2, T]

        # Extract and convert back
        sources = sources[0].cpu().numpy()  # [4, 2, T] for htdemucs (drums, bass, other, vocals)

        # Take left channel and resample back
        result = []
        for i in range(min(num_sources, sources.shape[0])):
            src = sources[i, 0]  # Left channel

            if sr != 44100:
                src = resample_poly(src, sr, 44100).astype(np.float32)

            result.append(src.astype(np.float32))

        # Pad if needed
        while len(result) < num_sources:
            result.append(mono_mix.copy())

        return result

    def _separate_sepformer(
        self,
        mono_mix: np.ndarray,
        num_sources: int,
        sr: int,
    ) -> List[np.ndarray]:
        """Separate using SepFormer."""
        import torch
        from scipy.signal import resample_poly

        # SepFormer expects 8000Hz
        if sr != 8000:
            mono_resampled = resample_poly(mono_mix, 8000, sr).astype(np.float32)
        else:
            mono_resampled = mono_mix.astype(np.float32)

        # Separate
        audio_tensor = torch.from_numpy(mono_resampled).unsqueeze(0).to(self.device)
        sources = self.model.separate_batch(audio_tensor)  # [1, num_sources, T]

        sources = sources[0].cpu().numpy()  # [2, T]

        # Resample back
        result = []
        for i in range(min(num_sources, sources.shape[0])):
            src = sources[i]
            if sr != 8000:
                src = resample_poly(src, sr, 8000).astype(np.float32)
            result.append(src.astype(np.float32))

        while len(result) < num_sources:
            result.append(mono_mix.copy())

        return result

    def _separate_convtasnet(
        self,
        mono_mix: np.ndarray,
        num_sources: int,
        sr: int,
    ) -> List[np.ndarray]:
        """Separate using Conv-TasNet."""
        import torch
        from scipy.signal import resample_poly

        # Conv-TasNet expects 8000Hz
        if sr != 8000:
            mono_resampled = resample_poly(mono_mix, 8000, sr).astype(np.float32)
        else:
            mono_resampled = mono_mix.astype(np.float32)

        audio_tensor = torch.from_numpy(mono_resampled).unsqueeze(0).to(self.device)

        with torch.no_grad():
            sources = self.model(audio_tensor)  # [1, num_sources, T]

        sources = sources[0].cpu().numpy()

        result = []
        for i in range(min(num_sources, sources.shape[0])):
            src = sources[i]
            if sr != 8000:
                src = resample_poly(src, sr, 8000).astype(np.float32)
            result.append(src.astype(np.float32))

        while len(result) < num_sources:
            result.append(mono_mix.copy())

        return result


def separate_and_spatialize(
    mono_mix: np.ndarray,
    video_path: str,
    sr: int = 44100,
    num_sources: int = 2,
    separator_model: str = "demucs",
    fov_deg: float = 60.0,
) -> Tuple[np.ndarray, Dict]:
    """
    End-to-end: separate mono → track multiple objects → spatialize.

    Args:
        mono_mix: Mono audio mix
        video_path: Video file path
        sr: Sample rate
        num_sources: Number of sources
        separator_model: "demucs", "sepformer", "convtasnet", or "none"
        fov_deg: Camera FOV

    Returns:
        foa: Spatialized FOA [4, T]
        info: Metadata dict
    """
    from .multi_source import track_multiple_sources, encode_multi_source_foa

    print("="*60)
    print("Source Separation + Multi-Source Spatialization")
    print("="*60)

    # 1. Separate audio
    print(f"\n[1/3] Separating audio with {separator_model}...")
    separator = SourceSeparator(model=separator_model, sample_rate=sr)
    sources = separator.separate(mono_mix, num_sources=num_sources, sr=sr)
    print(f"  → Extracted {len(sources)} sources")

    # 2. Track multiple objects
    print(f"\n[2/3] Tracking {num_sources} objects in video...")
    trajectories = track_multiple_sources(
        video_path=video_path,
        num_sources=num_sources,
        fov_deg=fov_deg,
    )
    print(f"  → Got {len(trajectories)} trajectories")

    # 3. Encode to FOA
    print(f"\n[3/3] Encoding to spatial audio...")

    # Match sources to trajectories (simple: by order)
    # TODO: smarter matching based on audio characteristics
    sources_matched = sources[:len(trajectories)]
    while len(sources_matched) < len(trajectories):
        sources_matched.append(sources[-1].copy())

    foa = encode_multi_source_foa(
        audio_sources=sources_matched,
        trajectories=trajectories,
        sr=sr,
    )

    print(f"  → Generated FOA: {foa.shape}")
    print("="*60)

    return foa, {
        "num_sources": len(sources),
        "trajectories": trajectories,
        "separator_model": separator_model,
    }


__all__ = [
    "SourceSeparator",
    "separate_and_spatialize",
]
