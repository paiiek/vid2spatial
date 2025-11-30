import argparse
import json
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--traj', required=True)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--sec', type=float, default=5.0)
    ap.add_argument('--W', type=int, default=640)
    ap.add_argument('--H', type=int, default=360)
    ap.add_argument('--radius', type=int, default=20)
    args = ap.parse_args()

    T = int(args.fps * args.sec)
    xs = np.linspace(0.1, 0.9, T)
    ys = 0.5 + 0.2 * np.sin(np.linspace(0, 2*np.pi, T))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(args.out, fourcc, args.fps, (args.W, args.H))
    frames=[]
    for i in range(T):
        frame = np.zeros((args.H, args.W, 3), np.uint8)
        cx = int(xs[i] * args.W)
        cy = int(ys[i] * args.H)
        cv2.circle(frame, (cx, cy), args.radius, (0,255,0), -1)
        vw.write(frame)
        frames.append({'frame': i, 'cx': float(cx), 'cy': float(cy)})
    vw.release()
    json.dump({'frames': frames, 'intrinsics': {'width': args.W, 'height': args.H, 'fov_deg': 60.0}}, open(args.traj,'w'), indent=2)
    print('wrote', args.out, 'and', args.traj)


if __name__ == '__main__':
    main()

