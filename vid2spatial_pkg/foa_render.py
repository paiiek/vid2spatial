import math
import numpy as np
import soundfile as sf
from typing import List, Dict, Tuple
from typing import Optional


SQ2 = math.sqrt(2.0)
SQ3_2 = math.sqrt(3.0 / 2.0)  # SN3D scaling for first order


def dir_to_foa_acn_sn3d_gains(az: np.ndarray, el: np.ndarray) -> np.ndarray:
    """Compute FOA gains in AmbiX (ACN/SN3D) channel order [W, Y, Z, X].

    x = cos(az)cos(el), y = sin(az)cos(el), z = sin(el)
    W = 1/sqrt(2)
    X = sqrt(3/2)*x, Y = sqrt(3/2)*y, Z = sqrt(3/2)*z
    Returned gains: [W, Y, Z, X] with shape [4, T]
    """
    x = np.cos(az) * np.cos(el)
    y = np.sin(az) * np.cos(el)
    z = np.sin(el)
    W = np.full_like(x, 1.0 / SQ2)
    X = SQ3_2 * x
    Y = SQ3_2 * y
    Z = SQ3_2 * z
    return np.stack([W, Y, Z, X], axis=0).astype(np.float32)


def interpolate_angles(frames: List[Dict], T: int, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of (az, el) over sample grid [0..T-1].
    frames: list of {frame, az, el}
    Uses frame indices as time with fps inferred from sr if available.
    Assumes video fps unknown → uniform spacing by frame index.
    """
    if not frames:
        raise ValueError("Empty frames for interpolation")
    idx = np.array([f["frame"] for f in frames], dtype=np.float32)
    az = np.array([f["az"] for f in frames], dtype=np.float32)
    el = np.array([f["el"] for f in frames], dtype=np.float32)
    if len(idx) == 1:
        az_s = np.full((T,), float(az[0]), np.float32)
        el_s = np.full((T,), float(el[0]), np.float32)
        return az_s, el_s
    # map sample positions to fractional frame indices
    s = np.linspace(idx[0], idx[-1], T, dtype=np.float32)
    az_s = np.interp(s, idx, az)
    el_s = np.interp(s, idx, el)
    return az_s.astype(np.float32), el_s.astype(np.float32)


def interpolate_angles_distance(frames: List[Dict], T: int, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate (az, el, dist, d_rel) per sample.

    Priority for distance:
    1. depth_render (explicit render value from RTS smoothing)
    2. depth_blended (from depth enhancement)
    3. dist_m_raw (raw metric depth)
    4. dist_m (may be smoothed)
    5. 1.0 (fallback)

    Returns (az_s, el_s, dist_s, d_rel_s) arrays of length T.
    d_rel_s is the normalized distance [0,1] for perceptual mapping.
    """
    az_s, el_s = interpolate_angles(frames, T, sr)
    idx = np.array([f["frame"] for f in frames], dtype=np.float32)

    # Determine which depth field to use
    if any("depth_render" in f for f in frames):
        # Best: explicit render value
        dist = np.array([float(f.get("depth_render", f.get("dist_m", 1.0))) for f in frames], dtype=np.float32)
    elif any("depth_blended" in f for f in frames):
        # Fallback: depth_blended (enhanced)
        dist = np.array([float(f.get("depth_blended", f.get("dist_m", 1.0))) for f in frames], dtype=np.float32)
    elif any("dist_m_raw" in f for f in frames):
        # Fallback: raw metric depth
        dist = np.array([float(f.get("dist_m_raw", 1.0)) for f in frames], dtype=np.float32)
    elif any("dist_m" in f for f in frames):
        # Fallback: dist_m (may be smoothed)
        dist = np.array([float(f.get("dist_m", 1.0)) for f in frames], dtype=np.float32)
    else:
        dist = np.ones((len(frames),), np.float32)

    # Get d_rel (normalized distance 0-1) - use pre-computed if available
    if any("d_rel" in f for f in frames):
        d_rel = np.array([float(f.get("d_rel", 0.5)) for f in frames], dtype=np.float32)
    else:
        # Compute d_rel from dist with global range [0.5m, 10m]
        d_rel = np.clip((dist - 0.5) / (10.0 - 0.5), 0.0, 1.0)

    s = np.linspace(idx[0], idx[-1], T, dtype=np.float32)
    dist_s = np.interp(s, idx, dist).astype(np.float32)
    d_rel_s = np.interp(s, idx, d_rel).astype(np.float32)
    return az_s, el_s, dist_s, d_rel_s


def smooth_limit_angles(
    az_s: np.ndarray,
    el_s: np.ndarray,
    sr: int,
    *,
    smooth_ms: float = 50.0,
    max_deg_per_s: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply moving-average smoothing and optional per-second delta limiting to az/el series.
    - smooth_ms: moving average window (milliseconds)
    - max_deg_per_s: if set, clamp per-sample delta to this rate
    """
    T = az_s.shape[0]
    az = az_s.astype(np.float32).copy()
    el = el_s.astype(np.float32).copy()
    # moving average smoothing
    win = max(int(sr * (float(smooth_ms) / 1000.0)), 1)
    if win > 1:
        def movavg(x: np.ndarray, w: int) -> np.ndarray:
            k = np.ones((w,), dtype=np.float32) / float(w)
            y = np.convolve(x, k, mode='same')
            return y.astype(np.float32)
        az = movavg(az, win)
        el = movavg(el, win)
    # delta limit (radians per sample)
    if max_deg_per_s is not None and float(max_deg_per_s) > 0:
        max_rad_per_s = float(max_deg_per_s) * (np.pi / 180.0)
        thr = max_rad_per_s / float(sr)
        def clamp_delta(x: np.ndarray, thr: float) -> np.ndarray:
            y = x.copy()
            for i in range(1, y.shape[0]):
                d = y[i] - y[i-1]
                if d > thr:
                    y[i] = y[i-1] + thr
                elif d < -thr:
                    y[i] = y[i-1] - thr
            return y
        az = clamp_delta(az, thr)
        el = clamp_delta(el, thr)
    return az.astype(np.float32), el.astype(np.float32)


def apply_distance_gain_lpf(x: np.ndarray, sr: int, dist_s: np.ndarray,
                            d_rel_s: np.ndarray = None,
                            *, gain_k: float = 1.0,
                            lpf_min_hz: float = 800.0,
                            lpf_max_hz: float = 8000.0) -> np.ndarray:
    """Apply distance-based gain and 1st-order low-pass filter to mono signal.

    Uses d_rel (normalized 0-1) for consistent perceptual mapping across tracks.
    - gain: 1/d law with floor, using dist_s (meters)
    - LPF cutoff: log-scaled from d_rel (0=near=high cutoff, 1=far=low cutoff)

    Args:
        x: mono audio signal
        sr: sample rate
        dist_s: distance in meters (for gain calculation)
        d_rel_s: normalized distance 0-1 (for LPF). If None, computed from dist_s.
        gain_k: gain exponent (1.0 = 1/r law)
        lpf_min_hz: LPF cutoff for far objects
        lpf_max_hz: LPF cutoff for near objects
    """
    T = x.shape[0]
    d = dist_s[:T].astype(np.float32)

    # Gain mapping: 1/r law with floor
    g = 1.0 / np.maximum(d, 1.0)
    if float(gain_k) != 1.0:
        g = g ** float(gain_k)
    g = np.clip(g, 0.2, 1.0)
    y = (x.astype(np.float32) * g).astype(np.float32)

    # LPF: use d_rel for consistent mapping across tracks
    lp_min = max(50.0, float(lpf_min_hz))
    lp_max = max(lp_min + 10.0, float(lpf_max_hz))

    if d_rel_s is not None:
        # Use pre-computed d_rel (global normalization)
        nd = np.clip(d_rel_s[:T].astype(np.float32), 0.0, 1.0)
    else:
        # Fallback: compute from dist with global range [0.5m, 10m]
        nd = np.clip((d - 0.5) / (10.0 - 0.5), 0.0, 1.0)

    # Log-scale cutoff: near (nd=0) → lp_max, far (nd=1) → lp_min
    log_min = math.log(lp_min)
    log_max = math.log(lp_max)
    log_fc = log_max - (log_max - log_min) * nd
    fc = np.exp(log_fc)

    # One-pole LPF with time-varying alpha
    two_pi = 2.0 * np.pi
    y_lp = np.zeros_like(y)
    prev = 0.0
    for i in range(T):
        a = (two_pi * fc[i]) / (two_pi * fc[i] + sr)
        prev = prev + a * (y[i] - prev)
        y_lp[i] = prev

    # Soft normalize
    peak = float(np.max(np.abs(y_lp)) + 1e-9)
    if peak > 1.0:
        y_lp = y_lp / peak
    return y_lp.astype(np.float32)


def build_wet_curve_from_dist_occ(d_rel_s: np.ndarray,
                                  occ_s: np.ndarray | None = None,
                                  *,
                                  wet_min: float = 0.05,
                                  wet_max: float = 0.35,
                                  occ_boost: float = 0.10) -> np.ndarray:
    """Map d_rel/occlusion to reverb wetness curve in [0,1].

    Uses d_rel (pre-normalized 0-1) for consistent reverb mapping across tracks.
    - d_rel=0 (near) → wet_min (less reverb)
    - d_rel=1 (far) → wet_max (more reverb)
    - occlusion adds additional wetness

    Args:
        d_rel_s: normalized distance [0,1] from depth_utils
        occ_s: occlusion values [0,1], optional
        wet_min: reverb wetness for near objects
        wet_max: reverb wetness for far objects
        occ_boost: additional wetness for occluded objects
    """
    d_norm = np.clip(d_rel_s.astype(np.float32), 0.0, 1.0)
    wet = wet_min + (wet_max - wet_min) * d_norm
    if occ_s is not None:
        occ = np.clip(occ_s.astype(np.float32), 0.0, 1.0)
        wet = wet + occ_boost * occ
    return np.clip(wet, 0.0, 1.0).astype(np.float32)


def apply_timevarying_reverb_mono(x: np.ndarray, sr: int, wet_curve: np.ndarray, rt60: float = 0.6) -> np.ndarray:
    """Apply time-varying wet mix using a single mono Schroeder IR.
    y = (1-wet)*x + wet*conv(x, ir)
    """
    from .irgen import schroeder_ir, fft_convolve
    T = x.shape[0]
    wet = wet_curve[:T].astype(np.float32)
    ir = schroeder_ir(sr, rt60=rt60).astype(np.float32)
    y_rev = fft_convolve(x.astype(np.float32), ir)[:T]
    y = (1.0 - wet) * x.astype(np.float32) + wet * y_rev
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 1.0:
        y = y / (peak * 1.01)
    return y.astype(np.float32)


def apply_timevarying_reverb_foa(foa: np.ndarray, sr: int, wet_curve: np.ndarray, air_foa: np.ndarray | None = None, rt60: float = 0.6) -> np.ndarray:
    """Apply time-varying wet mix to FOA.
    - If air_foa(4ch) provided: wet * (foa * air) + (1-wet)*foa
    - Else: per-channel Schroeder IR convolution
    """
    T = foa.shape[1]
    wet = wet_curve[:T].astype(np.float32)
    if air_foa is not None:
        from .irgen import convolve_foa_with_air
        wet_foa = convolve_foa_with_air(foa, air_foa)
    else:
        from .irgen import schroeder_ir, fft_convolve
        ir = schroeder_ir(sr, rt60=rt60).astype(np.float32)
        wet_foa = np.vstack([fft_convolve(foa[c], ir)[:T] for c in range(4)]).astype(np.float32)
    y = (1.0 - wet)[None, :] * foa.astype(np.float32) + wet[None, :] * wet_foa
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 1.0:
        y = y / (peak * 1.01)
    return y.astype(np.float32)


def encode_mono_to_foa(mono: np.ndarray,
                       az_series: np.ndarray,
                       el_series: np.ndarray) -> np.ndarray:
    """Time-varying FOA encoding of mono signal using per-sample az/el.
    Returns FOA array [4, T] in AmbiX (ACN/SN3D) [W, Y, Z, X]."""
    assert mono.ndim == 1
    T = mono.shape[0]
    assert az_series.shape[0] == T and el_series.shape[0] == T
    gains = dir_to_foa_acn_sn3d_gains(az_series, el_series)  # [4,T]
    foa = gains * mono[None, :]
    # peak normalization to avoid clipping
    peak = float(np.max(np.abs(foa)))
    if peak > 1.0:
        foa /= (peak * 1.01)
    return foa.astype(np.float32)


def write_foa_wav(path: str, foa: np.ndarray, sr: int) -> None:
    if foa.shape[0] != 4:
        raise ValueError("FOA must have 4 channels [W,Y,Z,X] in AmbiX order")
    # apply gentle soft limiter to prevent inter-sample clipping
    y = foa.astype(np.float32)
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 0.98:
        # soft clipper: y = tanh(a*y)/tanh(a)
        a = 1.5
        y = np.tanh(a * y) / math.tanh(a)
    sf.write(path, y.T, sr, subtype="FLOAT")


def encode_many_to_foa(monolist: List[np.ndarray], az_list: List[np.ndarray], el_list: List[np.ndarray]) -> np.ndarray:
    """Sum multiple mono sources into one FOA.
    All sequences must be length-matched; returns [4, T]."""
    assert len(monolist) == len(az_list) == len(el_list)
    if len(monolist) == 0:
        raise ValueError("empty sources")
    T = monolist[0].shape[0]
    acc = np.zeros((4, T), np.float32)
    for x, az, el in zip(monolist, az_list, el_list):
        assert x.shape[0] == T and az.shape[0] == T and el.shape[0] == T
        acc += encode_mono_to_foa(x, az, el)
    peak = float(np.max(np.abs(acc)))
    if peak > 1.0:
        acc /= (peak * 1.01)
    return acc.astype(np.float32)


def render_foa_from_trajectory(
    audio_path: str,
    trajectory: Dict,
    output_path: str,
    *,
    smooth_ms: float = 50.0,
    dist_gain_k: float = 1.0,
    dist_lpf_min_hz: float = 800.0,
    dist_lpf_max_hz: float = 8000.0,
    apply_reverb: bool = False,
    rt60: float = 0.5,
    output_stereo: bool = True,
    sofa_path: str = None,
) -> Dict:
    """
    Render mono audio to FOA using trajectory from tracking.

    Args:
        audio_path: Path to mono or stereo audio file (will be mixed to mono)
        trajectory: Dict with 'frames' list containing {frame, az, el, dist_m} per frame
        output_path: Output FOA wav path (4-channel)
        smooth_ms: Smoothing window for angle interpolation
        dist_gain_k: Distance gain exponent
        dist_lpf_min_hz: Min LPF cutoff for far objects
        dist_lpf_max_hz: Max LPF cutoff for near objects
        apply_reverb: Whether to apply distance-based reverb
        rt60: Reverb time if apply_reverb=True
        output_stereo: Also output stereo/binaural version

    Returns:
        Dict with output file paths and metadata
    """
    import os

    # Load audio
    audio, sr = sf.read(audio_path, dtype='float32')
    if audio.ndim == 2:
        # Mix to mono
        audio = audio.mean(axis=1)
    T = audio.shape[0]

    # Get trajectory frames
    frames = trajectory.get("frames", [])
    if not frames:
        raise ValueError("Empty trajectory frames")

    # Interpolate angles, distance, and d_rel to audio sample rate
    az_s, el_s, dist_s, d_rel_s = interpolate_angles_distance(frames, T, sr)

    # Apply smoothing
    az_s, el_s = smooth_limit_angles(az_s, el_s, sr, smooth_ms=smooth_ms)

    # Apply distance-based gain and LPF (using d_rel for consistent perceptual mapping)
    audio_proc = apply_distance_gain_lpf(
        audio, sr, dist_s, d_rel_s,
        gain_k=dist_gain_k,
        lpf_min_hz=dist_lpf_min_hz,
        lpf_max_hz=dist_lpf_max_hz,
    )

    # Encode to FOA
    foa = encode_mono_to_foa(audio_proc, az_s, el_s)

    # Apply reverb if requested (using d_rel for consistent reverb mapping)
    if apply_reverb:
        wet_curve = build_wet_curve_from_dist_occ(d_rel_s)
        foa = apply_timevarying_reverb_foa(foa, sr, wet_curve, rt60=rt60)

    # Write FOA
    write_foa_wav(output_path, foa, sr)

    result = {
        "foa_path": output_path,
        "sample_rate": sr,
        "duration_sec": T / sr,
        "num_frames": len(frames),
    }

    # Output stereo/binaural version
    if output_stereo:
        stereo_path = output_path.replace(".wav", "_stereo.wav").replace(".foa", "")
        if stereo_path == output_path:
            stereo_path = output_path.replace(".wav", "_stereo.wav")

        if sofa_path and os.path.exists(sofa_path):
            stereo = foa_to_binaural_sofa(foa, sr, sofa_path)
            result["binaural_method"] = "hrtf_sofa"
        else:
            stereo = foa_to_binaural(foa, sr)
            result["binaural_method"] = "crossfeed"
        sf.write(stereo_path, stereo.T, sr, subtype="FLOAT")
        result["stereo_path"] = stereo_path

    return result


def foa_to_stereo(foa_sn3d: np.ndarray, sr: int, az_deg_L: float = +30.0, az_deg_R: float = -30.0) -> np.ndarray:
    """Decode FOA (AmbiX ACN/SN3D [W,Y,Z,X]) to stereo at ±az degrees on horizon.
    Prefer spaudiopy; fall back to a pure‑NumPy first‑order decoder if unavailable.
    Returns [2, T] float32.
    """
    if foa_sn3d.shape[0] != 4:
        raise ValueError("FOA must be [4,T] in AmbiX [W,Y,Z,X]")
    try:
        import spaudiopy as spa  # type: ignore
        foa_n3d = foa_sn3d.copy()
        foa_n3d[1:4, :] *= np.sqrt(3.0)
        az = np.deg2rad(np.array([az_deg_L, az_deg_R], np.float32))
        zen = np.deg2rad(np.array([90.0, 90.0], np.float32))
        Y = spa.sph.sh_matrix(1, az, zen, sh_type='real')  # [2, 4]
        stereo = (Y @ foa_n3d).astype(np.float32)
    except Exception:
        # Pure‑NumPy fallback (ACN/N3D, real, order‑1, elevation 0)
        foa_n3d = foa_sn3d.copy()
        foa_n3d[1:4, :] *= np.sqrt(3.0)
        az = np.deg2rad(np.array([az_deg_L, az_deg_R], np.float32))
        # For el=0: Y00=1, Y1-1=sqrt(3)sin(az), Y10=0, Y11=sqrt(3)cos(az)
        Y = np.stack([
            np.array([1.0, np.sqrt(3.0)*np.sin(az[0]), 0.0, np.sqrt(3.0)*np.cos(az[0])], dtype=np.float32),
            np.array([1.0, np.sqrt(3.0)*np.sin(az[1]), 0.0, np.sqrt(3.0)*np.cos(az[1])], dtype=np.float32),
        ], axis=0)
        stereo = (Y @ foa_n3d).astype(np.float32)
    peak = float(np.max(np.abs(stereo)))
    if peak > 1.0:
        stereo /= (peak * 1.01)
    return stereo


def foa_to_binaural(foa_sn3d: np.ndarray, sr: int) -> np.ndarray:
    """Convenience: FOA -> stereo(±30°) -> simple crossfeed binaural.
    Returns [2, T] float32.
    """
    st = foa_to_stereo(foa_sn3d, sr)
    # simple crossfeed
    L, R = st[0], st[1]
    d = max(1, int(0.0003 * sr))
    Lp = np.pad(L, (0, 0))
    Rp = np.pad(R, (0, 0))
    Ld = np.pad(Lp, (d, 0))[: L.shape[0]]
    Rd = np.pad(Rp, (d, 0))[: R.shape[0]]
    Lo = L + 0.22 * Rd
    Ro = R + 0.22 * Ld
    peak = float(max(np.max(np.abs(Lo)), np.max(np.abs(Ro))))
    if peak > 1.0:
        Lo /= (peak * 1.01)
        Ro /= (peak * 1.01)
    return np.stack([Lo.astype(np.float32), Ro.astype(np.float32)], 0)


def foa_to_binaural_sofa(foa_sn3d: np.ndarray, sr: int, sofa_path: str) -> np.ndarray:
    """Decode FOA (AmbiX ACN/SN3D [W,Y,Z,X]) to binaural using SOFA HRTF.

    Uses direct HRIR convolution via h5py (no spaudiopy dependency).
    FOA is decoded to virtual speakers, each convolved with the nearest HRIR.
    Returns [2, T] float32. Falls back to simple crossfeed on error.
    """
    if foa_sn3d.shape[0] != 4:
        raise ValueError("FOA must be [4,T] in AmbiX [W,Y,Z,X]")
    try:
        import h5py
        from scipy.signal import fftconvolve, resample_poly

        # --- Load SOFA HRTF ---
        with h5py.File(sofa_path, 'r') as sofa:
            ir_data = sofa['Data.IR'][:]          # (M, 2, N) — M measurements, 2 ears, N taps
            fs_hrir = float(sofa['Data.SamplingRate'][0])
            src_pos = sofa['SourcePosition'][:]   # (M, 3) — [az_deg, el_deg, dist]

        src_az = np.radians(src_pos[:, 0])   # azimuth in radians
        src_el = np.radians(src_pos[:, 1])   # elevation in radians

        # Precompute unit vectors for fast nearest-neighbor lookup
        src_x = np.cos(src_el) * np.cos(src_az)
        src_y = np.cos(src_el) * np.sin(src_az)
        src_z = np.sin(src_el)
        src_cart = np.stack([src_x, src_y, src_z], axis=1)  # (M, 3)

        # Resample HRIRs if sample rates differ
        if int(fs_hrir) != int(sr):
            gcd = np.gcd(int(sr), int(fs_hrir))
            up, down = int(sr) // gcd, int(fs_hrir) // gcd
            M, _, L = ir_data.shape
            L_out = int(np.ceil(L * sr / fs_hrir))
            ir_resampled = np.zeros((M, 2, L_out), dtype=np.float32)
            for i in range(M):
                for ch in range(2):
                    ir_resampled[i, ch] = resample_poly(
                        ir_data[i, ch].astype(np.float32), up, down
                    )[:L_out]
            ir_data = ir_resampled

        def find_nearest_hrir(az_rad: float, el_rad: float) -> np.ndarray:
            """Find closest HRIR by angular distance. Returns (2, L)."""
            qx = np.cos(el_rad) * np.cos(az_rad)
            qy = np.cos(el_rad) * np.sin(az_rad)
            qz = np.sin(el_rad)
            dots = src_cart[:, 0] * qx + src_cart[:, 1] * qy + src_cart[:, 2] * qz
            idx = int(np.argmax(dots))
            return ir_data[idx]  # (2, L)

        # --- Virtual speaker decode (8 speakers on cube for order-1) ---
        # Speaker directions: front, back, left, right, up-front, up-back, down-front, down-back
        vspk_dirs = [
            (0.0, 0.0),                    # front
            (np.pi, 0.0),                  # back
            (np.pi/2, 0.0),                # left
            (-np.pi/2, 0.0),               # right
            (np.pi/4, np.pi/4),            # upper-front-left
            (-np.pi/4, np.pi/4),           # upper-front-right
            (np.pi/4, -np.pi/4),           # lower-front-left
            (-np.pi/4, -np.pi/4),          # lower-front-right
        ]

        T = foa_sn3d.shape[1]
        binlr = np.zeros((2, T + ir_data.shape[2] - 1), dtype=np.float64)

        for az_spk, el_spk in vspk_dirs:
            # AmbiX ACN/SN3D decode weights for order-1
            # W=1, Y=sin(az)*cos(el), Z=sin(el), X=cos(az)*cos(el)
            w_W = 1.0
            w_Y = np.sin(az_spk) * np.cos(el_spk)
            w_Z = np.sin(el_spk)
            w_X = np.cos(az_spk) * np.cos(el_spk)
            weights = np.array([w_W, w_Y, w_Z, w_X], dtype=np.float32)

            # Decode: speaker signal = sum of weighted FOA channels
            spk_signal = weights @ foa_sn3d  # (T,)

            # Get HRIR for this speaker direction
            # SOFA convention: az=0 is front, positive=counterclockwise
            hrir = find_nearest_hrir(az_spk, el_spk)  # (2, L)

            # Convolve
            for ch in range(2):
                binlr[ch] += fftconvolve(spk_signal, hrir[ch], mode='full')[:binlr.shape[1]]

        # Normalize by number of speakers
        binlr /= len(vspk_dirs)

        # Trim to input length
        binlr = binlr[:, :T]

        # Soft normalize
        peak = float(np.max(np.abs(binlr)) + 1e-9)
        if peak > 1.0:
            binlr = binlr / (peak * 1.01)
        return binlr.astype(np.float32)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return foa_to_binaural(foa_sn3d, sr)


# =============================================================================
# BASELINE RENDERERS FOR COMPARISON
# =============================================================================

def render_stereo_pan_baseline(
    audio: np.ndarray,
    sr: int,
    az_series: np.ndarray,
    dist_s: np.ndarray = None,
    d_rel_s: np.ndarray = None,
    apply_gain: bool = True,
) -> np.ndarray:
    """
    Simple stereo panning baseline (no FOA).

    This is the simplest spatial audio approach:
    pan = sin(azimuth)
    L = audio * (1 - pan) / 2
    R = audio * (1 + pan) / 2

    Args:
        audio: Mono audio signal
        sr: Sample rate
        az_series: Azimuth in radians per sample
        dist_s: Distance in meters (optional, for gain)
        d_rel_s: Normalized distance 0-1 (optional, for gain)
        apply_gain: Apply distance-based gain attenuation

    Returns:
        Stereo [2, T] array
    """
    T = len(audio)
    audio = audio.astype(np.float32)

    # Apply distance gain if available
    if apply_gain and dist_s is not None:
        d = dist_s[:T].astype(np.float32)
        gain = 1.0 / np.maximum(d, 1.0)
        gain = np.clip(gain, 0.2, 1.0)
        audio = audio * gain

    # Simple sine-law panning
    # pan: -1 = full left, +1 = full right
    # az: 0 = front, +pi/2 = left, -pi/2 = right
    az = az_series[:T].astype(np.float32)
    pan = np.sin(az)  # -1 to +1

    # Constant power panning (more natural than linear)
    # L = cos((pan + 1) * pi/4), R = sin((pan + 1) * pi/4)
    pan_angle = (pan + 1) * (np.pi / 4)  # 0 to pi/2
    L_gain = np.cos(pan_angle)
    R_gain = np.sin(pan_angle)

    L = audio * L_gain
    R = audio * R_gain

    stereo = np.stack([L, R], axis=0).astype(np.float32)

    # Normalize
    peak = float(np.max(np.abs(stereo)) + 1e-9)
    if peak > 1.0:
        stereo = stereo / (peak * 1.01)

    return stereo


def render_stereo_pan_reverb_baseline(
    audio: np.ndarray,
    sr: int,
    az_series: np.ndarray,
    dist_s: np.ndarray = None,
    d_rel_s: np.ndarray = None,
    apply_lpf: bool = True,
    rt60: float = 0.5,
) -> np.ndarray:
    """
    Stereo panning + distance-based reverb baseline.

    This is a more sophisticated baseline that adds:
    1. Simple stereo panning (sin law)
    2. Distance-based gain (1/r)
    3. Distance-based LPF
    4. Distance-based reverb

    Args:
        audio: Mono audio signal
        sr: Sample rate
        az_series: Azimuth in radians per sample
        dist_s: Distance in meters
        d_rel_s: Normalized distance 0-1
        apply_lpf: Apply distance-based low-pass filter
        rt60: Reverb time

    Returns:
        Stereo [2, T] array
    """
    T = len(audio)
    audio = audio.astype(np.float32)

    # Apply distance processing (gain + LPF)
    if dist_s is not None:
        if d_rel_s is None:
            d_rel_s = np.clip((dist_s - 0.5) / 9.5, 0.0, 1.0)

        if apply_lpf:
            audio = apply_distance_gain_lpf(audio, sr, dist_s, d_rel_s)
        else:
            # Just gain
            d = dist_s[:T].astype(np.float32)
            gain = 1.0 / np.maximum(d, 1.0)
            gain = np.clip(gain, 0.2, 1.0)
            audio = audio * gain

    # Apply panning
    az = az_series[:T].astype(np.float32)
    pan = np.sin(az)
    pan_angle = (pan + 1) * (np.pi / 4)
    L_gain = np.cos(pan_angle)
    R_gain = np.sin(pan_angle)

    L = audio * L_gain
    R = audio * R_gain

    stereo = np.stack([L, R], axis=0).astype(np.float32)

    # Apply reverb based on distance
    if d_rel_s is not None:
        wet_curve = build_wet_curve_from_dist_occ(d_rel_s[:T])
        stereo = _apply_stereo_reverb(stereo, sr, wet_curve, rt60)

    # Normalize
    peak = float(np.max(np.abs(stereo)) + 1e-9)
    if peak > 1.0:
        stereo = stereo / (peak * 1.01)

    return stereo


def _apply_stereo_reverb(
    stereo: np.ndarray,
    sr: int,
    wet_curve: np.ndarray,
    rt60: float = 0.5,
) -> np.ndarray:
    """Apply time-varying reverb to stereo signal."""
    from .irgen import schroeder_ir, fft_convolve

    T = stereo.shape[1]
    wet = wet_curve[:T].astype(np.float32)

    ir = schroeder_ir(sr, rt60=rt60).astype(np.float32)

    # Convolve each channel
    L_rev = fft_convolve(stereo[0], ir)[:T]
    R_rev = fft_convolve(stereo[1], ir)[:T]

    # Time-varying mix
    L_out = (1.0 - wet) * stereo[0] + wet * L_rev
    R_out = (1.0 - wet) * stereo[1] + wet * R_rev

    result = np.stack([L_out, R_out], axis=0).astype(np.float32)

    # Normalize
    peak = float(np.max(np.abs(result)) + 1e-9)
    if peak > 1.0:
        result = result / (peak * 1.01)

    return result


def render_baselines_from_trajectory(
    audio_path: str,
    trajectory: Dict,
    output_dir: str,
    prefix: str = "baseline",
) -> Dict[str, str]:
    """
    Render all baseline versions from a trajectory for comparison.

    Generates:
    1. stereo_pan: Simple stereo panning only
    2. stereo_pan_reverb: Stereo panning + distance reverb
    3. foa: Full FOA rendering (for reference)

    Args:
        audio_path: Path to mono audio file
        trajectory: Trajectory dict with 'frames'
        output_dir: Output directory
        prefix: Output filename prefix

    Returns:
        Dict mapping baseline name to output path
    """
    import os
    audio, sr = sf.read(audio_path, dtype='float32')
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    T = audio.shape[0]

    frames = trajectory.get("frames", [])
    if not frames:
        raise ValueError("Empty trajectory frames")

    # Interpolate to audio sample rate
    az_s, el_s, dist_s, d_rel_s = interpolate_angles_distance(frames, T, sr)

    os.makedirs(output_dir, exist_ok=True)
    outputs = {}

    # 1. Simple stereo panning
    stereo_pan = render_stereo_pan_baseline(audio, sr, az_s, dist_s, d_rel_s)
    path_pan = os.path.join(output_dir, f"{prefix}_stereo_pan.wav")
    sf.write(path_pan, stereo_pan.T, sr, subtype="FLOAT")
    outputs["stereo_pan"] = path_pan

    # 2. Stereo panning + reverb
    stereo_reverb = render_stereo_pan_reverb_baseline(audio, sr, az_s, dist_s, d_rel_s)
    path_reverb = os.path.join(output_dir, f"{prefix}_stereo_reverb.wav")
    sf.write(path_reverb, stereo_reverb.T, sr, subtype="FLOAT")
    outputs["stereo_pan_reverb"] = path_reverb

    # 3. FOA reference (using existing function)
    foa_path = os.path.join(output_dir, f"{prefix}_foa.wav")
    render_foa_from_trajectory(
        audio_path, trajectory, foa_path,
        apply_reverb=True,
        output_stereo=False,
    )
    outputs["foa"] = foa_path

    # 4. FOA decoded to stereo (for direct A/B comparison)
    foa_data, _ = sf.read(foa_path, dtype='float32')
    foa_data = foa_data.T  # [4, T]
    foa_stereo = foa_to_stereo(foa_data, sr)
    path_foa_stereo = os.path.join(output_dir, f"{prefix}_foa_stereo.wav")
    sf.write(path_foa_stereo, foa_stereo.T, sr, subtype="FLOAT")
    outputs["foa_stereo"] = path_foa_stereo

    print(f"[baseline] Generated {len(outputs)} baseline renders in {output_dir}")
    return outputs


# Update __all__
__all__ = [
    "dir_to_foa_acn_sn3d_gains",
    "interpolate_angles",
    "interpolate_angles_distance",
    "encode_mono_to_foa",
    "write_foa_wav",
    "encode_many_to_foa",
    "render_foa_from_trajectory",
    "foa_to_stereo",
    "foa_to_binaural",
    "foa_to_binaural_sofa",
    # Baselines
    "render_stereo_pan_baseline",
    "render_stereo_pan_reverb_baseline",
    "render_baselines_from_trajectory",
    # Distance processing
    "apply_distance_gain_lpf",
    "build_wet_curve_from_dist_occ",
]
