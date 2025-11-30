"""Multi-object spatialization runner.

Example:
  python -m mmhoa.vid2spatial.run_multi \
    --video path/to/video.mp4 \
    --audios a.wav b.wav \
    --track_ids 1 3 \
    --out_foa out_multi.foa.wav
"""
import argparse
import json
import librosa
import numpy as np

from .vision import yolo_bytetrack_all, compute_trajectory_3d
from .foa_render import interpolate_angles, encode_many_to_foa, write_foa_wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--audios', nargs='+', required=True)
    ap.add_argument('--track_ids', type=int, nargs='+', default=None, help='IDs to use; if missing, pick top-K by area')
    ap.add_argument('--cls', type=str, default=None)
    ap.add_argument('--out_foa', required=True)
    ap.add_argument('--stride', type=int, default=1)
    args = ap.parse_args()

    # Track all
    tracks = yolo_bytetrack_all(args.video, cls_name=args.cls, conf=0.25)
    if not tracks:
        raise RuntimeError('no tracks found')

    # choose track_ids
    if args.track_ids is None:
        # choose top-K by median area
        def med_area(seq):
            return float(np.median([r['w']*r['h'] for r in seq])) if seq else 0.0
        tids = sorted(tracks.keys(), key=lambda k: med_area(tracks[k]), reverse=True)[:len(args.audios)]
    else:
        tids = args.track_ids

    # compute per-id trajectories (re-using compute_trajectory_3d for az/el)
    # feed precomputed 2D centers into angle computation by mocking method='kcf' path
    az_list, el_list = [], []
    sr_ref, T_ref = None, None
    monos = []
    for i, (tid, a_path) in enumerate(zip(tids, args.audios)):
        # audio
        x, sr = librosa.load(a_path, sr=None, mono=True)
        if sr_ref is None:
            sr_ref = sr
            T_ref = x.shape[0]
        else:
            if sr != sr_ref:
                import librosa as _lb
                x = _lb.resample(x, orig_sr=sr, target_sr=sr_ref)
            x = x[:T_ref] if x.shape[0] >= T_ref else np.pad(x, (0, T_ref - x.shape[0]))
        monos.append(x.astype(np.float32))

        # angle interpolation
        frames = tracks[tid]
        # build minimal structure matching compute_trajectory_3d output
        # reuse intrinsics via a helper call
        # For simplicity, we estimate angles using vision functions directly
        # by mapping pixel centers to rays with default fov.
        from .vision import CameraIntrinsics, pixel_to_ray, ray_to_angles
        import cv2
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError('failed to open video')
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        K = CameraIntrinsics(W, H, 60.0)
        az_frames = []
        el_frames = []
        for r in frames:
            vec = pixel_to_ray(r['cx'], r['cy'], K)
            az, el = ray_to_angles(vec)
            az_frames.append(az)
            el_frames.append(el)
        idx = np.array([r['frame'] for r in frames], dtype=np.float32)
        if len(idx) == 1:
            az_s = np.full((T_ref,), float(az_frames[0]), np.float32)
            el_s = np.full((T_ref,), float(el_frames[0]), np.float32)
        else:
            s = np.linspace(idx[0], idx[-1], T_ref, dtype=np.float32)
            az_s = np.interp(s, idx, np.array(az_frames, np.float32))
            el_s = np.interp(s, idx, np.array(el_frames, np.float32))
        az_list.append(az_s.astype(np.float32))
        el_list.append(el_s.astype(np.float32))

    foa = encode_many_to_foa(monos, az_list, el_list)
    write_foa_wav(args.out_foa, foa, sr_ref)
    print('[done] wrote', args.out_foa)


if __name__ == '__main__':
    main()

