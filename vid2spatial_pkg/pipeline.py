"""
Spatial audio pipeline orchestration.
"""
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, Callable

import librosa
import numpy as np

from .config import PipelineConfig
from .vision import compute_trajectory_3d

# Try to import refactored version if available
try:
    from .vision_refactored import compute_trajectory_3d_refactored
    USE_REFACTORED_VISION = True
except ImportError:
    compute_trajectory_3d_refactored = None
    USE_REFACTORED_VISION = False

from .foa_render import (
    interpolate_angles_distance,
    smooth_limit_angles,
    apply_distance_gain_lpf,
    encode_mono_to_foa,
    write_foa_wav,
    foa_to_stereo,
    foa_to_binaural,
    foa_to_binaural_sofa,
)


class SpatialAudioPipeline:
    """
    End-to-end pipeline for creating spatial audio from video and mono audio.

    This class orchestrates the complete workflow:
    1. Vision processing: object tracking + depth estimation → 3D trajectory
    2. Audio preprocessing: room IR convolution
    3. Spatial rendering: FOA encoding with distance effects
    4. Output: FOA, stereo, binaural formats
    """

    def __init__(self, config: PipelineConfig, use_refactored_vision: bool = True):
        """
        Initialize pipeline with configuration.

        Args:
            config: Complete pipeline configuration
            use_refactored_vision: Whether to use refactored vision module (default: True)
        """
        self.config = config
        self.use_refactored_vision = use_refactored_vision and USE_REFACTORED_VISION
        self._sam2_mask_fn: Optional[Callable] = None
        self._depth_fn: Optional[Callable] = None
        self._trajectory: Optional[Dict[str, Any]] = None

    def _build_sam2_predictor(self) -> Optional[Callable]:
        """Build SAM2 predictor if refinement is enabled."""
        if not self.config.vision.refinement.enabled:
            return None

        if self.config.vision.refinement.method != 'sam2':
            return None

        try:
            from .sam2_adapter import build_sam2_predictor
            predictor = build_sam2_predictor(
                checkpoint_path=self.config.vision.refinement.sam_ckpt,
                config_path=None,
                model_id=self.config.vision.refinement.sam2_model_id
            )
            return predictor
        except Exception as e:
            print(f'[warn] SAM2 integration failed: {e}')
            print('[warn] falling back to grabcut center refinement')
            self.config.vision.refinement.method = 'grabcut'
            return None

    def _build_depth_predictor(self) -> Optional[Callable]:
        """Build depth predictor if adapter is enabled."""
        if not self.config.vision.depth.use_adapter:
            return None

        try:
            from .depth_anything_adapter import build_depth_predictor
            return build_depth_predictor()
        except Exception as e:
            print(f'[warn] depth adapter init failed: {e}')
            return None

    def _compute_trajectory(self) -> Dict[str, Any]:
        """
        Compute or load 3D trajectory from video.

        Returns:
            Trajectory dict with 'intrinsics' and 'frames' keys
        """
        # Check for precomputed trajectory
        if self.config.trajectory_json and os.path.exists(self.config.trajectory_json):
            print(f'[info] Loading precomputed trajectory from {self.config.trajectory_json}')
            with open(self.config.trajectory_json, "r") as f:
                return json.load(f)

        # Build vision components
        print('[info] Computing trajectory from video...')
        if self._sam2_mask_fn is None:
            self._sam2_mask_fn = self._build_sam2_predictor()

        if self._depth_fn is None:
            self._depth_fn = self._build_depth_predictor()

        # Choose vision implementation
        compute_fn = compute_trajectory_3d_refactored if self.use_refactored_vision else compute_trajectory_3d

        if self.use_refactored_vision:
            print('[info] Using refactored vision module')

        # Compute trajectory
        traj = compute_fn(
            self.config.video_path,
            init_bbox=self.config.vision.tracking.init_bbox,
            fov_deg=self.config.vision.camera.fov_deg,
            sample_stride=self.config.vision.camera.sample_stride,
            method=self.config.vision.tracking.method,
            cls_name=self.config.vision.tracking.class_name,
            refine_center=self.config.vision.refinement.enabled,
            refine_center_method=self.config.vision.refinement.method,
            depth_backend=self.config.vision.depth.backend,
            sam2_mask_fn=self._sam2_mask_fn,
            depth_fn=self._depth_fn,
            sam2_model_id=self.config.vision.refinement.sam2_model_id,
            sam2_cfg=self.config.vision.refinement.sam2_cfg,
            sam2_ckpt=self.config.vision.refinement.sam2_ckpt,
            select_track_id=self.config.vision.tracking.select_track_id,
            smooth_alpha=self.config.vision.tracking.smooth_alpha,
            fallback_center_if_no_bbox=self.config.vision.tracking.fallback_center_if_no_bbox,
            target_color=self.config.vision.tracking.target_color,
            color_tolerance=self.config.vision.tracking.color_tolerance,
            color_min_area=self.config.vision.tracking.color_min_area,
            point_method=self.config.vision.tracking.point_method,
            point_min_brightness=self.config.vision.tracking.point_min_brightness,
            point_template_path=self.config.vision.tracking.point_template_path,
            point_template_threshold=self.config.vision.tracking.point_template_threshold,
            point_use_optical_flow=self.config.vision.tracking.point_use_optical_flow,
            skeleton_joint=self.config.vision.tracking.skeleton_joint,
            skeleton_backend=self.config.vision.tracking.skeleton_backend,
            skeleton_min_visibility=self.config.vision.tracking.skeleton_min_visibility,
            skeleton_smooth_alpha=self.config.vision.tracking.skeleton_smooth_alpha,
        )

        # Apply Kalman filter temporal smoothing
        if getattr(self.config.spatial, 'use_kalman_smoothing', True):
            print('[info] Applying Kalman filter temporal smoothing...')
            from .temporal_smoother import smooth_trajectory_batch

            # Get FPS from video
            import cv2
            cap = cv2.VideoCapture(self.config.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            if fps <= 0:
                fps = 30.0  # Default fallback

            traj['frames'] = smooth_trajectory_batch(
                traj['frames'],
                fps=fps,
                process_noise=0.01,
                measurement_noise=0.1
            )

        # Save if requested
        if self.config.output.trajectory_path:
            print(f'[info] Saving trajectory to {self.config.output.trajectory_path}')
            with open(self.config.output.trajectory_path, "w") as f:
                json.dump(traj, f, indent=2)

        return traj

    def _apply_room_ir(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply room impulse response to audio.

        Args:
            audio: Input audio signal
            sr: Sample rate

        Returns:
            Audio with room IR applied
        """
        if self.config.room.disabled or self.config.room.backend in ("none", "visual", "brir"):
            return audio.astype(np.float32)

        Lx, Ly, Lz = self.config.room.dimensions
        mx, my, mz = self.config.room.mic_position

        # FAIR-Play matched IR (recommended based on ablation study)
        if self.config.room.backend == "fairplay":
            from .irgen import fairplay_matched_ir
            print('[info] Using FAIR-Play matched IR (dry acoustics)...')
            rir = fairplay_matched_ir(fs=sr, rt60=self.config.room.rt60)
            return self._convolve_ir(audio, rir)

        # Try PRA backend
        if self.config.room.backend in ("auto", "pra"):
            try:
                from .irgen import synthesize_mono_rir, fft_convolve
                print('[info] Synthesizing room IR with pyroomacoustics...')
                rir = synthesize_mono_rir(
                    (Lx, Ly, Lz),
                    (mx - 1.0, my, mz),  # Source offset
                    (mx, my, mz),  # Mic position
                    fs=sr,
                    rt60=self.config.room.rt60
                )
                return self._convolve_ir(audio, rir)
            except Exception as e:
                print(f'[warn] PRA backend failed: {e}, falling back to FAIR-Play matched IR')
                # Fallback to FAIR-Play matched instead of Schroeder
                from .irgen import fairplay_matched_ir
                print('[info] Using FAIR-Play matched IR (fallback)...')
                rir = fairplay_matched_ir(fs=sr, rt60=self.config.room.rt60)
                return self._convolve_ir(audio, rir)

        # Schroeder IR (legacy, not recommended)
        from .irgen import schroeder_ir
        print('[info] Using Schroeder IR...')
        rir = schroeder_ir(sr, rt60=self.config.room.rt60)
        return self._convolve_ir(audio, rir)

    def _convolve_ir(self, audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
        """Convolve audio with impulse response and trim to original length."""
        from .irgen import fft_convolve
        convolved = fft_convolve(audio.astype(np.float32), rir)

        # Trim or pad to original length
        if convolved.shape[0] >= audio.shape[0]:
            return convolved[: audio.shape[0]]
        else:
            return np.pad(convolved, (0, audio.shape[0] - convolved.shape[0]))

    def _load_occlusion_timeline(self, T: int) -> Optional[np.ndarray]:
        """
        Load or estimate occlusion timeline.

        Args:
            T: Number of audio samples

        Returns:
            Occlusion values [0, 1] per sample, or None
        """
        if not self.config.occlusion.enabled:
            return None

        # Load from JSON
        if self.config.occlusion.json_path and os.path.exists(self.config.occlusion.json_path):
            print(f'[info] Loading occlusion from {self.config.occlusion.json_path}')
            with open(self.config.occlusion.json_path, "r") as f:
                occ_data = json.load(f)

            if isinstance(occ_data, dict) and "frames" in occ_data:
                idx = np.array([it.get("frame", 0) for it in occ_data["frames"]], np.float32)
                occ = np.array([float(it.get("occ", 0.0)) for it in occ_data["frames"]], np.float32)
                s = np.linspace(idx[0], idx[-1], T, dtype=np.float32)
                return np.interp(s, idx, occ).astype(np.float32)

        # Estimate from video
        if self.config.occlusion.estimate:
            try:
                from .occlusion import estimate_occlusion_timeline
                print('[info] Estimating occlusion from video...')
                occ_tl = estimate_occlusion_timeline(
                    self.config.video_path,
                    self._trajectory["frames"],
                    use_depth=True,
                    stride=self.config.vision.camera.sample_stride
                )
                idx = np.array([it.get("frame", 0) for it in occ_tl["frames"]], np.float32)
                occ = np.array([float(it.get("occ", 0.0)) for it in occ_tl["frames"]], np.float32)
                s = np.linspace(idx[0], idx[-1], T, dtype=np.float32)
                return np.interp(s, idx, occ).astype(np.float32)
            except Exception as e:
                print(f"[warn] occlusion estimation failed: {e}")

        return None

    def _render_spatial_audio(
        self,
        audio: np.ndarray,
        sr: int,
        trajectory: Dict[str, Any]
    ) -> np.ndarray:
        """
        Render spatial FOA audio from mono audio and trajectory.

        Args:
            audio: Mono audio signal
            sr: Sample rate
            trajectory: 3D trajectory dict

        Returns:
            FOA audio [4, T]
        """
        T = audio.shape[0]

        # Interpolate angles and distance
        print('[info] Interpolating trajectory to audio timeline...')
        az_s, el_s, dist_s = interpolate_angles_distance(trajectory["frames"], T=T, sr=sr)

        # Smooth angles
        print('[info] Smoothing angles...')
        az_s, el_s = smooth_limit_angles(
            az_s, el_s, sr,
            smooth_ms=self.config.spatial.angle_smooth_ms,
            max_deg_per_s=self.config.spatial.max_deg_per_s
        )

        # Apply distance effects
        print('[info] Applying distance-based gain and filtering...')
        audio_dist = apply_distance_gain_lpf(
            audio, sr, dist_s,
            gain_k=self.config.spatial.dist_gain_k,
            lpf_min_hz=self.config.spatial.dist_lpf_min_hz,
            lpf_max_hz=self.config.spatial.dist_lpf_max_hz
        )

        # Load occlusion if needed
        occ_s = self._load_occlusion_timeline(T)

        # Apply reverb coupling if enabled
        if self.config.reverb.enabled:
            print('[info] Applying time-varying reverb...')
            from .foa_render import build_wet_curve_from_dist_occ, apply_timevarying_reverb_mono
            wet = build_wet_curve_from_dist_occ(
                dist_s, occ_s,
                wet_min=self.config.reverb.wet_min,
                wet_max=self.config.reverb.wet_max,
                occ_boost=self.config.reverb.wet_occ_boost
            )
            audio_dist = apply_timevarying_reverb_mono(audio_dist, sr, wet, rt60=self.config.reverb.rt60)

        # Encode to FOA
        print('[info] Encoding to first-order ambisonics...')
        foa = encode_mono_to_foa(audio_dist, az_s, el_s)

        # Optional FOA AIR convolution
        if self.config.air_foa_path:
            print(f'[info] Applying FOA AIR from {self.config.air_foa_path}')
            import soundfile as sf
            air, sra = sf.read(self.config.air_foa_path, always_2d=True)
            air = air.T.astype(np.float32)

            if air.shape[0] != 4:
                raise ValueError("--air_foa must be 4-channel AmbiX wav")

            if self.config.reverb.enabled:
                from .foa_render import build_wet_curve_from_dist_occ, apply_timevarying_reverb_foa
                wet = build_wet_curve_from_dist_occ(
                    dist_s, occ_s,
                    wet_min=self.config.reverb.wet_min,
                    wet_max=self.config.reverb.wet_max,
                    occ_boost=self.config.reverb.wet_occ_boost
                )
                foa = apply_timevarying_reverb_foa(foa, sr, wet, air_foa=air, rt60=self.config.reverb.rt60)
            else:
                from .irgen import fft_convolve
                foa_conv = np.zeros_like(foa)
                for ch in range(4):
                    foa_conv[ch] = fft_convolve(foa[ch], air[ch])[:T]
                foa = foa_conv

        return foa

    def _write_outputs(self, foa: np.ndarray, sr: int):
        """
        Write output files in various formats.

        Args:
            foa: FOA audio [4, T]
            sr: Sample rate
        """
        # Write FOA
        print(f'[info] Writing FOA to {self.config.output.foa_path}')
        write_foa_wav(self.config.output.foa_path, foa, sr)

        # Write stereo if requested
        if self.config.output.stereo_path:
            print(f'[info] Writing stereo to {self.config.output.stereo_path}')
            stereo = foa_to_stereo(foa, sr)
            import soundfile as sf
            sf.write(self.config.output.stereo_path, stereo.T, sr)

        # Write binaural if requested
        if self.config.output.binaural_path:
            print(f'[info] Writing binaural to {self.config.output.binaural_path}')

            if self.config.output.binaural_config.mode == "sofa":
                if not self.config.output.binaural_config.sofa_path:
                    print('[warn] SOFA mode requires --sofa path, falling back to crossfeed')
                    binaural = foa_to_binaural(foa, sr)
                else:
                    binaural = foa_to_binaural_sofa(
                        foa, sr,
                        self.config.output.binaural_config.sofa_path
                    )
            else:
                binaural = foa_to_binaural(foa, sr)

            import soundfile as sf
            sf.write(self.config.output.binaural_path, binaural.T, sr)

    def run(self) -> Dict[str, Any]:
        """
        Run the complete spatial audio pipeline.

        Returns:
            Dictionary with pipeline outputs and metadata
        """
        print('='*60)
        print('Spatial Audio Pipeline')
        print('='*60)

        # Step 1: Compute trajectory
        print('\n[1/4] Computing 3D trajectory from video...')
        self._trajectory = self._compute_trajectory()
        print(f'      → Found {len(self._trajectory["frames"])} trajectory frames')

        # Step 2: Load and process audio
        print('\n[2/4] Loading and processing audio...')
        audio, sr = librosa.load(self.config.audio_path, sr=None, mono=True)
        print(f'      → Loaded {len(audio)} samples at {sr} Hz ({len(audio)/sr:.2f}s)')

        audio_processed = self._apply_room_ir(audio, sr)

        # Step 3: Render spatial audio
        print('\n[3/4] Rendering spatial audio...')
        foa = self._render_spatial_audio(audio_processed, sr, self._trajectory)
        print(f'      → Generated FOA audio: {foa.shape}')

        # Step 4: Write outputs
        print('\n[4/4] Writing outputs...')
        self._write_outputs(foa, sr)

        print('\n' + '='*60)
        print('Pipeline completed successfully!')
        print('='*60)

        return {
            "trajectory": self._trajectory,
            "sample_rate": sr,
            "duration_sec": len(audio) / sr,
            "num_frames": len(self._trajectory["frames"]),
            "outputs": {
                "foa": self.config.output.foa_path,
                "stereo": self.config.output.stereo_path,
                "binaural": self.config.output.binaural_path,
            }
        }
