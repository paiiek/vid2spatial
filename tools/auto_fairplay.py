import argparse
import json
import os
from typing import List, Dict

import librosa
import numpy as np

from mmhoa.vid2spatial.vision import compute_trajectory_3d
from mmhoa.vid2spatial.foa_render import interpolate_angles, encode_mono_to_foa, write_foa_wav
from mmhoa.vid2spatial.evaluate import pesq_nb


def read_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--root', required=True, help='Root where FAIR-Play/{audios,videos} are located')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--limit', type=int, default=10)
    ap.add_argument('--method', type=str, default='yolo', choices=['yolo','kcf','sam2'])
    ap.add_argument('--sam2_model_id', type=str, default='facebook/sam2.1-hiera-base-plus')
    ap.add_argument('--stride', type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    items = read_jsonl(args.manifest)[: args.limit]
    results = []
    for rec in items:
        a_rel = rec['audio']; v_rel = rec['video']
        a_path = os.path.join(args.root, a_rel)
        v_path = os.path.join(args.root, v_rel)
        if not (os.path.isfile(a_path) and os.path.isfile(v_path)):
            results.append({'id': rec.get('id'), 'status': 'skip_missing'})
            continue
        # audio to mono
        y, sr = librosa.load(a_path, sr=None, mono=False)
        if y.ndim == 2:
            mono = np.mean(y, axis=0)
        else:
            mono = y
        # trajectory
        traj = compute_trajectory_3d(
            v_path,
            method=args.method,
            cls_name='person',
            sample_stride=args.stride,
            depth_backend='none',
            refine_center=False,
            sam2_model_id=args.sam2_model_id,
            fallback_center_if_no_bbox=True,
        )
        az, el = interpolate_angles(traj['frames'], T=mono.shape[0], sr=sr)
        foa = encode_mono_to_foa(mono.astype(np.float32), az, el)
        base = os.path.splitext(os.path.basename(v_rel))[0]
        out_foa = os.path.join(args.out_dir, f'{base}.foa.wav')
        write_foa_wav(out_foa, foa, sr)
        # simple PESQ NB between mono (reference) and W channel as a sanity score
        try:
            ref = librosa.resample(mono, orig_sr=sr, target_sr=16000)
            deg = librosa.resample(foa[0], orig_sr=sr, target_sr=16000)
            pesq = pesq_nb(ref.astype(np.float32), deg.astype(np.float32), 16000)
        except Exception:
            pesq = float('nan')
        results.append({'id': rec.get('id'), 'status': 'ok', 'out_foa': out_foa, 'pesq_nb_w_vs_mono': pesq})

    report = {
        'manifest': args.manifest,
        'processed': len(results),
        'results': results,
    }
    out_report = os.path.join(args.out_dir, 'report.json')
    with open(out_report, 'w') as f:
        json.dump(report, f, indent=2)
    print('wrote', out_report)


if __name__ == '__main__':
    main()

