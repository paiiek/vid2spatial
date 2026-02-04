"""
Extended IR generation - Load external IR files.
"""
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional


def load_external_ir(ir_path: str, target_sr: int = 48000) -> np.ndarray:
    """
    Load external IR file.

    Supports WAV files with auto-resampling and mono conversion.

    Args:
        ir_path: Path to IR WAV file
        target_sr: Target sample rate

    Returns:
        IR as float32 array
    """
    ir, sr = sf.read(ir_path)

    # Convert to mono
    if ir.ndim > 1:
        ir = ir.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        from scipy.signal import resample
        num_samples = int(len(ir) * target_sr / sr)
        ir = resample(ir, num_samples)

    # Normalize
    ir = ir / (np.max(np.abs(ir)) + 1e-9)

    return ir.astype(np.float32)


def get_bundled_ir(room_type: str = 'living_room', ir_dir: Optional[str] = None) -> np.ndarray:
    """
    Get bundled IR from library.

    Args:
        room_type: 'living_room', 'small_hall', 'church'
        ir_dir: Custom IR directory (default: ./ir_library)

    Returns:
        IR as float32 array
    """
    if ir_dir is None:
        # Default to ir_library in project root
        ir_dir = Path(__file__).parent.parent / "ir_library"
    else:
        ir_dir = Path(ir_dir)

    ir_filename = f"{room_type}_ir.wav"
    ir_path = ir_dir / ir_filename

    if not ir_path.exists():
        raise FileNotFoundError(f"IR not found: {ir_path}")

    return load_external_ir(str(ir_path))
