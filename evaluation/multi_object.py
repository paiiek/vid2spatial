"""
Multi-object spatial audio rendering.

Extends the single-object pipeline to handle multiple sound sources
simultaneously, each tracked independently and spatialized to FOA.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .vision import compute_trajectory_3d
from .foa_render import (
    interpolate_angles_distance,
    smooth_limit_angles,
    apply_distance_gain_lpf,
    dir_to_foa_acn_sn3d_gains,
)


def compute_multi_object_trajectories(
    video_path: str,
    object_specs: List[Dict[str, Any]],
    **common_kwargs
) -> Dict[int, Dict[str, Any]]:
    """
    Compute trajectories for multiple objects in the same video.

    Args:
        video_path: Path to input video
        object_specs: List of object specifications, each dict with:
            - track_id: int (required if method='yolo')
            - cls_name: str (for YOLO, default='person')
            - init_bbox: Tuple[int,int,int,int] (for KCF)
            - method: str (optional, overrides common)
        common_kwargs: Common args passed to compute_trajectory_3d
            (fov_deg, sample_stride, depth_backend, etc.)

    Returns:
        Dict mapping object_id to trajectory dict
    """
    trajectories = {}

    for i, spec in enumerate(object_specs):
        object_id = spec.get('object_id', i)
        print(f'[info] Computing trajectory for object {object_id}...')

        # Merge spec with common kwargs
        kwargs = {**common_kwargs}
        if 'track_id' in spec:
            kwargs['select_track_id'] = spec['track_id']
        if 'cls_name' in spec:
            kwargs['cls_name'] = spec['cls_name']
        if 'init_bbox' in spec:
            kwargs['init_bbox'] = spec['init_bbox']
        if 'method' in spec:
            kwargs['method'] = spec['method']

        traj = compute_trajectory_3d(video_path, **kwargs)
        trajectories[object_id] = traj

    return trajectories


def encode_multi_source_to_foa(
    audio_sources: Dict[int, np.ndarray],
    trajectories: Dict[int, Dict[str, Any]],
    sr: int,
    spatial_config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Encode multiple mono sources to a single FOA mix.

    Args:
        audio_sources: Dict mapping object_id to mono audio array
        trajectories: Dict mapping object_id to trajectory dict
        sr: Sample rate
        spatial_config: Optional dict with:
            - angle_smooth_ms: float
            - max_deg_per_s: float
            - dist_gain_k: float
            - dist_lpf_min_hz: float
            - dist_lpf_max_hz: float

    Returns:
        Mixed FOA audio [4, T] where T is max length of all sources
    """
    if spatial_config is None:
        spatial_config = {}

    # Get max length
    max_T = max(len(audio) for audio in audio_sources.values())

    # Initialize output FOA
    foa_mix = np.zeros((4, max_T), dtype=np.float32)

    # Process each source
    for object_id, audio in audio_sources.items():
        if object_id not in trajectories:
            print(f'[warn] No trajectory for object {object_id}, skipping')
            continue

        traj = trajectories[object_id]
        T = len(audio)

        print(f'[info] Processing object {object_id}: {T} samples')

        # Interpolate trajectory
        az_s, el_s, dist_s = interpolate_angles_distance(traj["frames"], T=T, sr=sr)

        # Smooth angles
        az_s, el_s = smooth_limit_angles(
            az_s, el_s, sr,
            smooth_ms=spatial_config.get('angle_smooth_ms', 50.0),
            max_deg_per_s=spatial_config.get('max_deg_per_s', None)
        )

        # Apply distance effects
        audio_dist = apply_distance_gain_lpf(
            audio, sr, dist_s,
            gain_k=spatial_config.get('dist_gain_k', 1.0),
            lpf_min_hz=spatial_config.get('dist_lpf_min_hz', 800.0),
            lpf_max_hz=spatial_config.get('dist_lpf_max_hz', 8000.0)
        )

        # Encode to FOA
        gains = dir_to_foa_acn_sn3d_gains(az_s, el_s)  # [4, T]
        foa_source = gains * audio_dist[None, :]  # [4, T]

        # Add to mix (pad if needed)
        foa_mix[:, :T] += foa_source

    # Normalize to prevent clipping
    peak = np.max(np.abs(foa_mix))
    if peak > 1.0:
        foa_mix /= (peak * 1.01)

    return foa_mix.astype(np.float32)


def spatialize_multi_source(
    video_path: str,
    audio_sources: Dict[int, np.ndarray],
    object_specs: List[Dict[str, Any]],
    sr: int,
    fov_deg: float = 60.0,
    sample_stride: int = 1,
    spatial_config: Optional[Dict[str, Any]] = None,
    **vision_kwargs
) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
    """
    High-level API for multi-source spatialization.

    Args:
        video_path: Input video path
        audio_sources: Dict mapping object_id to mono audio
        object_specs: List of object tracking specs
        sr: Audio sample rate
        fov_deg: Camera horizontal FOV
        sample_stride: Video frame stride
        spatial_config: Spatial rendering config
        **vision_kwargs: Additional vision args

    Returns:
        Tuple of (foa_audio [4,T], trajectories dict)

    Example:
        >>> audio_sources = {
        ...     0: guitar_mono,  # First source
        ...     1: vocals_mono,  # Second source
        ... }
        >>> object_specs = [
        ...     {'object_id': 0, 'track_id': 5, 'cls_name': 'person'},
        ...     {'object_id': 1, 'track_id': 12, 'cls_name': 'person'},
        ... ]
        >>> foa, trajs = spatialize_multi_source(
        ...     'video.mp4', audio_sources, object_specs, sr=48000
        ... )
    """
    print('='*60)
    print('Multi-Object Spatial Audio Pipeline')
    print('='*60)
    print(f'  Sources: {len(audio_sources)}')
    print(f'  Objects: {len(object_specs)}')
    print()

    # Compute all trajectories
    print('[1/2] Computing object trajectories...')
    trajectories = compute_multi_object_trajectories(
        video_path,
        object_specs,
        fov_deg=fov_deg,
        sample_stride=sample_stride,
        **vision_kwargs
    )

    # Encode all sources to FOA
    print('\n[2/2] Encoding sources to FOA...')
    foa = encode_multi_source_to_foa(
        audio_sources,
        trajectories,
        sr,
        spatial_config=spatial_config
    )

    print('\n' + '='*60)
    print('Multi-object pipeline completed!')
    print('='*60)

    return foa, trajectories


class MultiObjectPipeline:
    """
    Pipeline for handling multiple audio sources with independent trajectories.

    This extends the single-object SpatialAudioPipeline to handle:
    - Multiple tracked objects in the same video
    - Multiple audio sources (e.g., separated stems)
    - Automatic mixing to single FOA output
    """

    def __init__(
        self,
        video_path: str,
        fov_deg: float = 60.0,
        sample_stride: int = 1,
        **vision_kwargs
    ):
        """
        Initialize multi-object pipeline.

        Args:
            video_path: Input video path
            fov_deg: Camera horizontal FOV
            sample_stride: Video frame stride
            **vision_kwargs: Additional vision processing args
        """
        self.video_path = video_path
        self.fov_deg = fov_deg
        self.sample_stride = sample_stride
        self.vision_kwargs = vision_kwargs

        self.object_specs: List[Dict[str, Any]] = []
        self.audio_sources: Dict[int, np.ndarray] = {}
        self.trajectories: Optional[Dict[int, Dict[str, Any]]] = None

    def add_object(
        self,
        object_id: int,
        audio: np.ndarray,
        track_id: Optional[int] = None,
        cls_name: str = "person",
        init_bbox: Optional[Tuple[int, int, int, int]] = None,
        method: str = "yolo",
    ):
        """
        Add an object to track and spatialize.

        Args:
            object_id: Unique identifier for this object
            audio: Mono audio signal for this object
            track_id: YOLO track ID (if using YOLO)
            cls_name: Object class name
            init_bbox: Initial bounding box for KCF
            method: Tracking method (yolo, kcf, sam2)
        """
        spec = {
            'object_id': object_id,
            'cls_name': cls_name,
            'method': method,
        }

        if track_id is not None:
            spec['track_id'] = track_id
        if init_bbox is not None:
            spec['init_bbox'] = init_bbox

        self.object_specs.append(spec)
        self.audio_sources[object_id] = audio

        print(f'[info] Added object {object_id}: method={method}, cls={cls_name}')

    def compute_trajectories(self):
        """Compute trajectories for all added objects."""
        if not self.object_specs:
            raise ValueError("No objects added. Call add_object() first.")

        self.trajectories = compute_multi_object_trajectories(
            self.video_path,
            self.object_specs,
            fov_deg=self.fov_deg,
            sample_stride=self.sample_stride,
            **self.vision_kwargs
        )

    def render(
        self,
        sr: int,
        spatial_config: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Render FOA audio from all sources.

        Args:
            sr: Sample rate
            spatial_config: Spatial rendering configuration

        Returns:
            FOA audio [4, T]
        """
        if self.trajectories is None:
            print('[info] Computing trajectories...')
            self.compute_trajectories()

        return encode_multi_source_to_foa(
            self.audio_sources,
            self.trajectories,
            sr,
            spatial_config=spatial_config
        )

    def run(
        self,
        sr: int,
        output_path: str,
        spatial_config: Optional[Dict[str, Any]] = None
    ):
        """
        Complete pipeline: compute trajectories and render FOA.

        Args:
            sr: Sample rate
            output_path: Output FOA wav path
            spatial_config: Spatial rendering configuration
        """
        print('='*60)
        print('Multi-Object Spatial Audio Pipeline')
        print('='*60)
        print(f'  Objects: {len(self.object_specs)}')
        print(f'  Video: {self.video_path}')
        print()

        # Render FOA
        foa = self.render(sr, spatial_config)

        # Write output
        from .foa_render import write_foa_wav
        print(f'\n[info] Writing FOA to {output_path}')
        write_foa_wav(output_path, foa, sr)

        print('\n' + '='*60)
        print('Pipeline completed successfully!')
        print('='*60)

        return foa


__all__ = [
    'compute_multi_object_trajectories',
    'encode_multi_source_to_foa',
    'spatialize_multi_source',
    'MultiObjectPipeline',
]
