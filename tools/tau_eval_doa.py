import argparse
import csv
import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf


def foa_sn3d_to_vec_xyz(foa: np.ndarray) -> np.ndarray:
    """Return per-sample pseudo-intensity vector from FOA AmbiX SN3D [W,Y,Z,X].
    I ~ [X*W, Y*W, Z*W], a more robust DOA proxy than raw coefficients.
    foa: [T,4] or [4,T]; returns [T,3].
    """
    if foa.ndim != 2:
        raise ValueError("foa must be 2D")
    if foa.shape[0] == 4:
        foa = foa.T  # [T,4]
    W = foa[:, 0].astype(np.float32)
    Yc = foa[:, 1].astype(np.float32)
    Zc = foa[:, 2].astype(np.float32)
    Xc = foa[:, 3].astype(np.float32)
    Ix = Xc * W
    Iy = Yc * W
    Iz = Zc * W
    vec = np.stack([Ix, Iy, Iz], axis=-1)
    # avoid NaNs from all-zeros
    n = np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-8
    return (vec / n).astype(np.float32)


def vec_to_angles(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = xyz[:, 0].astype(np.float32)
    y = xyz[:, 1].astype(np.float32)
    z = xyz[:, 2].astype(np.float32)
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.sqrt(x * x + y * y))
    return az, el


def angular_error_deg(az1: float, el1: float, az2: float, el2: float) -> float:
    x1 = math.cos(az1) * math.cos(el1); y1 = math.sin(az1) * math.cos(el1); z1 = math.sin(el1)
    x2 = math.cos(az2) * math.cos(el2); y2 = math.sin(az2) * math.cos(el2); z2 = math.sin(el2)
    dot = max(-1.0, min(1.0, x1 * x2 + y1 * y2 + z1 * z2))
    return float(math.degrees(math.acos(dot)))


def load_meta_csv(path: Path) -> List[Tuple[int, float, float]]:
    """Load TAU dev-train CSV. Assumes last two columns are az, el in degrees; first column is frame index.
    Returns list of (frame_idx, az_rad, el_rad).
    """
    rows = []
    with open(path, 'r') as f:
        rdr = csv.reader(f)
        for r in rdr:
            if not r or len(r) < 5:
                continue
            try:
                fi = int(float(r[0]))
                az = float(r[-2]); el = float(r[-1])
                rows.append((fi, math.radians(az), math.radians(el)))
            except Exception:
                continue
    return rows


def eval_file(foa_path: Path, meta_csv: Path, frame_hop_s: float = 0.1) -> dict:
    sig, sr = sf.read(str(foa_path), always_2d=True)
    T = sig.shape[0]
    xyz = foa_sn3d_to_vec_xyz(sig)
    az, el = vec_to_angles(xyz)
    metas = load_meta_csv(meta_csv)
    errs = []
    for fi, maz, mel in metas:
        t = max(0, min(T - 1, int(round((fi - 1) * frame_hop_s * sr))))
        e = angular_error_deg(az[t], el[t], maz, mel)
        errs.append(e)
    out = {
        'file': str(foa_path),
        'meta': str(meta_csv),
        'count': len(errs),
        'mae_deg': float(np.mean(errs)) if errs else None,
        'p50_deg': float(np.percentile(errs, 50)) if errs else None,
        'p75_deg': float(np.percentile(errs, 75)) if errs else None,
        'p90_deg': float(np.percentile(errs, 90)) if errs else None,
    }
    return out


# --- FOA scan-based DOA (simple beamformer) ---
SQ2 = math.sqrt(2.0)
SQ3_2 = math.sqrt(3.0 / 2.0)


def foa_sn3d_steer_gains(az: float, el: float) -> np.ndarray:
    """Unit-amplitude FOA (AmbiX ACN/SN3D [W,Y,Z,X]) encoding gains for a direction.
    W=1/sqrt(2); X=sqrt(3/2)*cos(az)cos(el); Y=sqrt(3/2)*sin(az)cos(el); Z=sqrt(3/2)*sin(el)
    Returned shape [4]."""
    x = math.cos(az) * math.cos(el)
    y = math.sin(az) * math.cos(el)
    z = math.sin(el)
    W = 1.0 / SQ2
    X = SQ3_2 * x
    Y = SQ3_2 * y
    Z = SQ3_2 * z
    return np.array([W, Y, Z, X], dtype=np.float32)


def scan_doa_for_window(foa_ty: np.ndarray, grid_deg: int = 10) -> Tuple[float, float]:
    """Brute-force steerable power scan over az∈[-180,180), el∈[-60,60].
    foa_ty: [T,4] float32 segment; returns (az, el) in radians.
    """
    if foa_ty.ndim != 2 or foa_ty.shape[1] != 4:
        raise ValueError("foa_ty must be [N,4] in [W,Y,Z,X]")
    # precompute covariance in FOA domain for stability
    Y = foa_ty.astype(np.float32)
    # energy by projection (delay-and-sum equivalent in SH domain)
    az_best, el_best, p_best = 0.0, 0.0, -1.0
    for az_deg in range(-180, 180, grid_deg):
        az = math.radians(az_deg)
        for el_deg in range(-60, 61, grid_deg):
            el = math.radians(el_deg)
            g = foa_sn3d_steer_gains(az, el)  # [4]
            s = Y @ g  # [N]
            p = float(np.mean(s * s))
            if p > p_best:
                p_best, az_best, el_best = p, az, el
    return az_best, el_best


def eval_file_scan(foa_path: Path, meta_csv: Path, frame_hop_s: float = 0.1,
                   win_s: float = 0.1, grid_deg: int = 10) -> dict:
    sig, sr = sf.read(str(foa_path), always_2d=True)
    T = sig.shape[0]
    metas = load_meta_csv(meta_csv)
    errs = []
    win = max(1, int(round(win_s * sr)))
    half = win // 2
    for fi, maz, mel in metas:
        t = max(0, min(T - 1, int(round((fi - 1) * frame_hop_s * sr))))
        t0 = max(0, t - half)
        t1 = min(T, t0 + win)
        seg = sig[t0:t1, :]  # [N,4]
        # estimate DOA by scan
        az_hat, el_hat = scan_doa_for_window(seg, grid_deg=grid_deg)
        errs.append(angular_error_deg(az_hat, el_hat, maz, mel))
    out = {
        'file': str(foa_path),
        'meta': str(meta_csv),
        'count': len(errs),
        'mae_deg': float(np.mean(errs)) if errs else None,
        'p50_deg': float(np.percentile(errs, 50)) if errs else None,
        'p75_deg': float(np.percentile(errs, 75)) if errs else None,
        'p90_deg': float(np.percentile(errs, 90)) if errs else None,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--limit', type=int, default=50)
    ap.add_argument('--out', required=True)
    ap.add_argument('--frame_hop_s', type=float, default=0.1)
    ap.add_argument('--method', choices=['piv','scan'], default='piv', help='piv=pseudo-intensity vector (fast), scan=FOA steerable power scan')
    ap.add_argument('--scan_win_s', type=float, default=0.1)
    ap.add_argument('--scan_grid_deg', type=int, default=10)
    args = ap.parse_args()

    items = []
    with open(args.manifest, 'r') as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            rec = json.loads(line)
            items.append(rec)
    results = []
    for rec in items:
        foa_rel = rec['audio']  # e.g., TAU2021/foa_dev/fold1_room1_mix001.wav
        meta_csv = rec['meta_csv']
        foa_path = Path(args.root) / foa_rel
        meta_path = Path(meta_csv)
        try:
            if args.method == 'piv':
                res = eval_file(foa_path, meta_path, frame_hop_s=args.frame_hop_s)
            else:
                res = eval_file_scan(foa_path, meta_path, frame_hop_s=args.frame_hop_s,
                                     win_s=args.scan_win_s, grid_deg=args.scan_grid_deg)
            results.append(res)
        except Exception as e:
            results.append({'file': str(foa_path), 'meta': str(meta_path), 'error': str(e)})

    maes = [r['mae_deg'] for r in results if r.get('mae_deg') is not None]
    summ = {
        'count_files': len(results),
        'mae_mean_deg': float(np.mean(maes)) if maes else None,
        'mae_median_deg': float(np.median(maes)) if maes else None,
        'mae_p75_deg': float(np.percentile(maes, 75)) if maes else None,
        'frame_hop_s': args.frame_hop_s,
    }
    out = {'summary': summ, 'items': results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, 'w'), indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
