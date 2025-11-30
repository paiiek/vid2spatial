import argparse
import json
import os
import numpy as np
import soundfile as sf
import librosa
from mmhoa.vid2spatial.evaluate import log_spectral_distance, seg_snr


def w_channel(path):
    y, sr = sf.read(path, always_2d=True)
    return y[:,0].astype(np.float32), sr


def ref_mono(root, vid_id):
    for cand in [
        os.path.join(root, 'FAIR-Play', 'audios', f'{vid_id}.wav'),
        os.path.join(root, 'fairplay', 'binaural_audios', f'{vid_id}.wav'),
        os.path.join(root, 'fairplay', 'audios', f'{vid_id}.wav'),
    ]:
        if os.path.isfile(cand):
            y, sr = librosa.load(cand, sr=None, mono=False)
            if y.ndim == 2:
                y = np.mean(y, axis=0)
            return y.astype(np.float32), sr
    raise FileNotFoundError(f'ref audio for {vid_id} not found under {root}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ablate', required=True, help='ir_ablate.json')
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    data = json.load(open(args.ablate))
    rows = []
    for it in data.get('items', []):
        vid_id = it['id']
        none_p = it.get('none_foa')
        vis_p = it.get('visual_foa')
        air_p = it.get('air_foa')
        try:
            ref, sr = ref_mono(args.root, vid_id)
            metas = []
            if none_p and os.path.exists(none_p):
                wn, srn = w_channel(none_p)
                metas.append(('none', wn, srn))
            if vis_p and os.path.exists(vis_p):
                wv, srv = w_channel(vis_p)
                metas.append(('visual', wv, srv))
            if air_p and os.path.exists(air_p):
                wa, sra = w_channel(air_p)
                metas.append(('air', wa, sra))
            if not metas:
                raise FileNotFoundError('no foa variants found')
            # length match at 16k to stabilize metrics
            ref16 = librosa.resample(y=ref, orig_sr=sr, target_sr=16000)
            metrics = {}
            T = min([len(ref16)] + [len(librosa.resample(y=m[1], orig_sr=m[2], target_sr=16000)) for m in metas])
            ref16 = ref16[:T]
            for name, sig, srs in metas:
                s16 = librosa.resample(y=sig, orig_sr=srs, target_sr=16000)[:T]
                metrics[f'seg_{name}'] = float(seg_snr(ref16, s16, sr=16000))
                metrics[f'lsd_{name}'] = float(log_spectral_distance(ref16, s16))
            # gains if both present
            if 'seg_none' in metrics and 'seg_visual' in metrics:
                metrics['seg_gain'] = metrics['seg_visual'] - metrics['seg_none']
            if 'lsd_none' in metrics and 'lsd_visual' in metrics:
                metrics['lsd_gain'] = metrics['lsd_none'] - metrics['lsd_visual']
            metrics['id'] = vid_id
            rows.append(metrics)
        except Exception as e:
            rows.append({'id': vid_id, 'error': str(e)})
    # summary
    seg_gain = [r['seg_gain'] for r in rows if 'seg_gain' in r and np.isfinite(r['seg_gain'])]
    lsd_gain = [r['lsd_gain'] for r in rows if 'lsd_gain' in r and np.isfinite(r['lsd_gain'])]
    summ = {
        'count': len(rows),
        'seg_gain_mean': float(np.mean(seg_gain)) if seg_gain else None,
        'lsd_gain_mean': float(np.mean(lsd_gain)) if lsd_gain else None,
    }
    out = {'summary': summ, 'items': rows}
    json.dump(out, open(args.out, 'w'), indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
