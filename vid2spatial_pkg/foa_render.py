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


def interpolate_angles_distance(frames: List[Dict], T: int, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate (az, el, dist_m) per sample. If dist_m missing, uses 1.0.
    Returns (az_s, el_s, dist_s) arrays of length T.
    """
    az_s, el_s = interpolate_angles(frames, T, sr)
    idx = np.array([f["frame"] for f in frames], dtype=np.float32)
    if any("dist_m" in f for f in frames):
        dist = np.array([float(f.get("dist_m", 1.0)) for f in frames], dtype=np.float32)
        s = np.linspace(idx[0], idx[-1], T, dtype=np.float32)
        dist_s = np.interp(s, idx, dist).astype(np.float32)
    else:
        dist_s = np.ones((T,), np.float32)
    return az_s, el_s, dist_s


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
                            *, gain_k: float = 1.0,
                            lpf_min_hz: float = 800.0,
                            lpf_max_hz: float = 8000.0) -> np.ndarray:
    """Apply simple distance-based gain and 1st-order low-pass filter to mono signal.
    - gain ~ 1 / (1 + k*(d-1)) clipped to [0.3, 1.0]
    - cutoff fc(d) between [lpf_min, lpf_max] decreases with distance
    """
    T = x.shape[0]
    d = dist_s[:T].astype(np.float32)
    # gain mapping (approx. 1/d with floor)
    g = 1.0 / np.maximum(d, 1.0)
    if float(gain_k) != 1.0:
        g = g ** float(gain_k)
    g = np.clip(g, 0.2, 1.0)
    y = (x.astype(np.float32) * g).astype(np.float32)
    # low-pass per-sample (time-varying, simple one-pole)
    lp_min = float(lpf_min_hz); lp_max = float(lpf_max_hz)
    lp_min = max(50.0, lp_min); lp_max = max(lp_min + 10.0, lp_max)
    # map distance to cutoff (log scale): nearer → lp_max, farther → lp_min
    nd = (d - d.min()) / (max(d.max() - d.min(), 1e-6))
    log_min = math.log(lp_min)
    log_max = math.log(lp_max)
    log_fc = log_max - (log_max - log_min) * nd
    fc = np.exp(log_fc)
    # one-pole with time-varying alpha
    two_pi = 2.0 * np.pi
    y_lp = np.zeros_like(y)
    prev = 0.0
    for i in range(T):
        a = (two_pi * fc[i]) / (two_pi * fc[i] + sr)
        prev = prev + a * (y[i] - prev)
        y_lp[i] = prev
    # normalize a bit
    peak = float(np.max(np.abs(y_lp)) + 1e-9)
    if peak > 1.0:
        y_lp = y_lp / peak
    return y_lp.astype(np.float32)


def build_wet_curve_from_dist_occ(dist_s: np.ndarray,
                                  occ_s: np.ndarray | None = None,
                                  *,
                                  wet_min: float = 0.05,
                                  wet_max: float = 0.35,
                                  occ_boost: float = 0.10) -> np.ndarray:
    """Map distance/occlusion to wetness curve in [0,1].
    - dist: nearer → wet_min, farther → wet_max
    - occ: increases wetness up to +occ_boost
    """
    d = dist_s.astype(np.float32)
    d_norm = (d - d.min()) / (max(d.max() - d.min(), 1e-6))
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


__all__ = [
    "dir_to_foa_acn_sn3d_gains",
    "interpolate_angles",
    "encode_mono_to_foa",
    "write_foa_wav",
    "encode_many_to_foa",
]


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
    """Decode FOA (AmbiX ACN/SN3D [W,Y,Z,X]) to binaural using SOFA HRTF via spaudiopy.
    Returns [2, T] float32. Falls back to simple crossfeed if spaudiopy/SOFA load fails.
    """
    if foa_sn3d.shape[0] != 4:
        raise ValueError("FOA must be [4,T] in AmbiX [W,Y,Z,X]")
    try:
        import spaudiopy as spa  # type: ignore
        from scipy.signal import resample_poly  # type: ignore
        # Convert FOA SN3D to N3D (ACN), order-1: multiply 1st-order by sqrt(3)
        foa_n3d = foa_sn3d.copy().astype(np.float32)
        foa_n3d[1:4, :] *= np.sqrt(3.0)
        # Load SH-HRIRs from SOFA for order=1
        N_sph = 1
        hrirs_nm, fs_hrir = spa.io.sofa_to_sh(sofa_path, N_sph, sh_type='real')  # (2,(N+1)^2,L)
        # Resample HRIRs to match sr if needed
        if fs_hrir != sr:
            L_out = int(np.ceil(hrirs_nm.shape[-1] * sr / fs_hrir))
            y = np.zeros((2, hrirs_nm.shape[1], L_out), dtype=np.float32)
            for ch in range(2):
                for n in range(hrirs_nm.shape[1]):
                    y[ch, n] = resample_poly(hrirs_nm[ch, n].astype(np.float32), sr, fs_hrir)
            hrirs_nm = y
        # Decode SH signals to binaural
        binlr = spa.decoder.sh2bin(foa_n3d, hrirs_nm).astype(np.float32)  # (2, T+L-1)
        # Trim to input length
        T = foa_sn3d.shape[1]
        if binlr.shape[1] >= T:
            binlr = binlr[:, :T]
        else:
            pad = T - binlr.shape[1]
            binlr = np.pad(binlr, ((0, 0), (0, pad)))
        # Normalize softly
        peak = float(np.max(np.abs(binlr)) + 1e-9)
        if peak > 1.0:
            binlr = binlr / (peak * 1.01)
        return binlr.astype(np.float32)
    except Exception:
        # fallback
        return foa_to_binaural(foa_sn3d, sr)
