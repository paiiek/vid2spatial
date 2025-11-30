import argparse
import json
import numpy as np


def compute_metrics(frames):
    if not frames:
        return {}
    t = np.array([f['frame'] for f in frames], dtype=np.float32)
    az = np.array([f['az'] for f in frames], dtype=np.float32)
    el = np.array([f['el'] for f in frames], dtype=np.float32)
    # unwrap azimuth to reduce artificial jumps
    az_u = np.unwrap(az)
    dt = np.gradient(t)
    vaz = np.gradient(az_u) / (dt + 1e-6)
    vel = np.gradient(el) / (dt + 1e-6)
    aaz = np.gradient(vaz) / (dt + 1e-6)
    ael = np.gradient(vel) / (dt + 1e-6)
    jerk = np.sqrt((np.gradient(aaz) / (dt + 1e-6)) ** 2 + (np.gradient(ael) / (dt + 1e-6)) ** 2).mean()
    speed = np.sqrt(vaz ** 2 + vel ** 2)
    smoothness = 1.0 / (1.0 + float(np.mean(np.abs(aaz)) + np.mean(np.abs(ael))))
    return {
        'n_frames': int(len(frames)),
        'speed_mean': float(speed.mean()),
        'speed_std': float(speed.std()),
        'jerk_mean': float(jerk),
        'smoothness_score': float(smoothness),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traj', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    with open(args.traj, 'r') as f:
        traj = json.load(f)
    m = compute_metrics(traj.get('frames', []))
    with open(args.out, 'w') as f:
        json.dump(m, f, indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()

