import argparse
import json
import numpy as np
import soundfile as sf


def foa_to_pseudo_intensity_doa(foa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if foa.shape[0] != 4:
        foa = foa.T
    W, Y, Z, X = foa[0], foa[1], foa[2], foa[3]
    Ix = X * W; Iy = Y * W; Iz = Z * W
    n = np.sqrt(Ix*Ix + Iy*Iy + Iz*Iz) + 1e-8
    Ix /= n; Iy /= n; Iz /= n
    az = np.arctan2(Iy, Ix)
    el = np.arctan2(Iz, np.sqrt(Ix*Ix + Iy*Iy))
    return az.astype(np.float32), el.astype(np.float32)


def interp_traj_angles(traj_frames: list[dict], T: int, intrinsics: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    if not traj_frames:
        raise ValueError('empty traj')
    idx = np.array([f['frame'] for f in traj_frames], dtype=np.float32)
    # If az/el not present, derive from pixel centers using camera intrinsics
    if 'az' not in traj_frames[0] or 'el' not in traj_frames[0]:
        if not intrinsics:
            raise ValueError('traj has no az/el and intrinsics missing')
        W = intrinsics.get('width'); H = intrinsics.get('height'); fov_deg = intrinsics.get('fov_deg', 60.0)
        fx = 0.5 * W / np.tan(np.deg2rad(fov_deg) / 2.0)
        fy = fx
        cx0 = W * 0.5; cy0 = H * 0.5
        az_list=[]; el_list=[]
        for f in traj_frames:
            u=f.get('cx', cx0); v=f.get('cy', cy0)
            x=(u-cx0)/fx; y=(v-cy0)/fy; z=1.0
            n=np.sqrt(x*x+y*y+z*z)+1e-8; x/=n; y/=n; z/=n
            az_list.append(np.arctan2(y,x)); el_list.append(np.arctan2(z, np.sqrt(x*x+y*y)))
        az = np.array(az_list, dtype=np.float32)
        el = np.array(el_list, dtype=np.float32)
    else:
        az = np.array([f['az'] for f in traj_frames], dtype=np.float32)
        el = np.array([f['el'] for f in traj_frames], dtype=np.float32)
    s = np.linspace(idx[0], idx[-1], T, dtype=np.float32)
    az_s = np.interp(s, idx, az)
    el_s = np.interp(s, idx, el)
    return az_s.astype(np.float32), el_s.astype(np.float32)


def ang_err_deg(a1, e1, a2, e2):
    x1 = np.cos(a1) * np.cos(e1); y1 = np.sin(a1) * np.cos(e1); z1 = np.sin(e1)
    x2 = np.cos(a2) * np.cos(e2); y2 = np.sin(a2) * np.cos(e2); z2 = np.sin(e2)
    dot = np.clip(x1*x2 + y1*y2 + z1*z2, -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--foa', required=True)
    ap.add_argument('--traj', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    y, sr = sf.read(args.foa, always_2d=True)
    foa = y.T.astype(np.float32)
    az_f, el_f = foa_to_pseudo_intensity_doa(foa)

    traj = json.load(open(args.traj))
    az_t, el_t = interp_traj_angles(traj['frames'], T=len(az_f), intrinsics=traj.get('intrinsics'))

    e = ang_err_deg(az_f, el_f, az_t, el_t)
    out = {
        'foa': args.foa,
        'traj': args.traj,
        'count': int(len(e)),
        'mae_deg': float(np.mean(e)),
        'p50_deg': float(np.percentile(e, 50)),
        'p75_deg': float(np.percentile(e, 75)),
        'p90_deg': float(np.percentile(e, 90)),
    }
    json.dump(out, open(args.out, 'w'), indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
