import argparse
import time
from mmhoa.vid2spatial.vision import compute_trajectory_3d


def run_once(video, method, stride, cls):
    t0 = time.time()
    traj = compute_trajectory_3d(
        video,
        method=method,
        cls_name=cls,
        sample_stride=stride,
        depth_backend='none',
        refine_center=False,
        fallback_center_if_no_bbox=True,
    )
    dt = time.time() - t0
    frames = len(traj.get('frames', []))
    fps = frames / dt if dt > 0 else 0.0
    return dt, frames, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--methods', nargs='+', default=['kcf','yolo','sam2'])
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--cls', type=str, default='none')
    args = ap.parse_args()

    for m in args.methods:
        try:
            dt, frames, fps = run_once(args.video, m, args.stride, args.cls)
            print(f"{m}: {dt:.2f}s, frames={frames}, eff_fps={fps:.2f}")
        except Exception as e:
            print(f"{m}: ERROR {e}")


if __name__ == '__main__':
    main()

