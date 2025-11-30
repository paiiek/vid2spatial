"""
FAIR-Play Dataset Loader

FAIR-Play contains:
- Videos: RGB videos with moving objects
- Binaural Audio: Stereo audio (ground truth)

We need to:
1. Extract mono from binaural
2. Run our pipeline to generate FOA
3. Convert our FOA to binaural for comparison
4. Compute angular localization error
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import librosa
import soundfile as sf
import cv2


class FairPlayDataset:
    """FAIR-Play dataset loader and preprocessor"""

    def __init__(self, root_dir: str = "/home/seung/data/fairplay"):
        self.root_dir = Path(root_dir)
        self.videos_dir = self.root_dir / "videos"
        self.audios_dir = self.root_dir / "binaural_audios"

        # Check if data exists
        if not self.videos_dir.exists():
            raise FileNotFoundError(f"Videos directory not found: {self.videos_dir}")
        if not self.audios_dir.exists():
            raise FileNotFoundError(f"Audios directory not found: {self.audios_dir}")

        # List all samples
        self.video_files = sorted(list(self.videos_dir.glob("*.mp4")))
        self.audio_files = sorted(list(self.audios_dir.glob("*.wav")))

        print(f"Found {len(self.video_files)} videos and {len(self.audio_files)} audios")

    def __len__(self):
        return min(len(self.video_files), len(self.audio_files))

    def get_sample(self, idx: int) -> Dict:
        """
        Get a single sample from FAIR-Play dataset.

        Returns:
            dict with keys:
                - video_path: Path to video file
                - audio_path: Path to binaural audio file
                - mono_audio: Mono audio extracted from binaural
                - gt_binaural: Ground truth binaural audio
                - sample_id: Sample ID (e.g., "000001")
        """
        video_path = self.video_files[idx]
        audio_path = self.audio_files[idx]

        sample_id = video_path.stem

        # Load binaural audio
        gt_binaural, sr = librosa.load(str(audio_path), sr=None, mono=False)

        # Extract mono (average L+R)
        if gt_binaural.ndim == 2:
            mono_audio = np.mean(gt_binaural, axis=0)
        else:
            mono_audio = gt_binaural

        return {
            'video_path': str(video_path),
            'audio_path': str(audio_path),
            'mono_audio': mono_audio,
            'gt_binaural': gt_binaural,
            'sample_rate': sr,
            'sample_id': sample_id,
        }

    def get_video_info(self, video_path: str) -> Dict:
        """Get video metadata"""
        cap = cv2.VideoCapture(video_path)

        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'num_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
        }

        cap.release()
        return info

    def get_subset(self, num_samples: int = 10, start_idx: int = 0) -> List[Dict]:
        """Get a subset of samples for quick evaluation"""
        subset = []
        for i in range(start_idx, min(start_idx + num_samples, len(self))):
            subset.append(self.get_sample(i))
        return subset


def extract_mono_from_binaural(binaural: np.ndarray) -> np.ndarray:
    """
    Extract mono from binaural audio.

    Args:
        binaural: [2, T] stereo audio

    Returns:
        mono: [T] mono audio
    """
    if binaural.ndim == 2:
        return np.mean(binaural, axis=0)
    return binaural


def save_mono_audio(mono: np.ndarray, output_path: str, sr: int = 48000):
    """Save mono audio to file"""
    sf.write(output_path, mono, sr)


if __name__ == "__main__":
    # Test dataset loader
    print("Testing FAIR-Play Dataset Loader")
    print("=" * 60)

    dataset = FairPlayDataset()
    print(f"\nDataset size: {len(dataset)} samples")

    # Load first sample
    print("\nLoading first sample...")
    sample = dataset.get_sample(0)

    print(f"\nSample ID: {sample['sample_id']}")
    print(f"Video: {sample['video_path']}")
    print(f"Audio: {sample['audio_path']}")
    print(f"Sample rate: {sample['sample_rate']} Hz")
    print(f"Mono audio shape: {sample['mono_audio'].shape}")
    print(f"GT binaural shape: {sample['gt_binaural'].shape}")

    # Get video info
    video_info = dataset.get_video_info(sample['video_path'])
    print(f"\nVideo info:")
    for key, val in video_info.items():
        print(f"  {key}: {val}")

    # Get subset
    print(f"\nLoading subset (10 samples)...")
    subset = dataset.get_subset(num_samples=10)
    print(f"Loaded {len(subset)} samples")

    print("\n" + "=" * 60)
    print("Dataset loader test completed!")
