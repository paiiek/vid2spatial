import argparse
import json
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--traj', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sample', type=int, default=12)
    args = ap.parse_args()

    with open(args.traj, 'r') as f:
        traj = json.load(f)
    frames = traj.get('frames', [])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError('failed to open video')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(frames)
    idxs = np.linspace(0, max(0, total-1), args.sample, dtype=int)
    grid = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        # draw center if available near index
        # find nearest record
        if frames:
            nearest = min(frames, key=lambda r: abs(r['frame']-int(i)))
            cx, cy = int(nearest['cx']) if 'cx' in nearest else W//2, int(nearest['cy']) if 'cy' in nearest else H//2
            cv2.circle(frame, (cx, cy), 6, (0,255,0), -1)
        grid.append(frame)
    cap.release()
    if not grid:
        raise RuntimeError('no frames sampled')
    out = np.concatenate(grid, axis=0)
    cv2.imwrite(args.out, out)
    print('wrote', args.out)


if __name__ == '__main__':
    main()

