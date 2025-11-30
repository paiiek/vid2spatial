import argparse
import json
import os
import math
import numpy as np
import soundfile as sf
import librosa
from mmhoa.vid2spatial.evaluate import fw_snrseg_mel, seg_snr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    rep = json.load(open(args.report))
    updated = 0
    for r in rep.get('results', []):
        if r.get('status') != 'ok':
            continue
        foa = r.get('out_foa')
        if not foa or not os.path.isfile(foa):
            continue
        vid_id = os.path.splitext(os.path.basename(foa))[0].split('.')[0]
        # reference mono
        a = None
        for cand in [
            os.path.join(args.root, 'FAIR-Play', 'audios', f'{vid_id}.wav'),
            os.path.join(args.root, 'fairplay', 'binaural_audios', f'{vid_id}.wav'),
            os.path.join(args.root, 'fairplay', 'audios', f'{vid_id}.wav'),
        ]:
            if os.path.isfile(cand):
                a = cand; break
        if a is None:
            continue
        y, sr = librosa.load(a, sr=None, mono=False)
        if y.ndim == 2:
            mono = np.mean(y, axis=0)
        else:
            mono = y
        foa_sig, sr2 = sf.read(foa, always_2d=True)
        W = foa_sig[:, 0].astype(np.float32)
        # resample to 16k
        try:
            ref16 = librosa.resample(y=mono, orig_sr=sr, target_sr=16000)
            deg16 = librosa.resample(y=W, orig_sr=sr2, target_sr=16000)
        except Exception:
            ref16 = mono.astype(np.float32); deg16 = W.astype(np.float32)
        T = min(len(ref16), len(deg16))
        ref16 = ref16[:T]; deg16 = deg16[:T]
        # recompute
        fw = fw_snrseg_mel(ref16, deg16, sr=16000)
        ss = seg_snr(ref16, deg16, sr=16000)
        if not (isinstance(r.get('fw_snrseg_mel'), (int,float)) and not math.isnan(r.get('fw_snrseg_mel'))):
            r['fw_snrseg_mel'] = fw; updated += 1
        if not (isinstance(r.get('seg_snr'), (int,float)) and not math.isnan(r.get('seg_snr'))):
            r['seg_snr'] = ss; updated += 1
    if args.overwrite:
        json.dump(rep, open(args.report,'w'), indent=2)
    else:
        out = os.path.splitext(args.report)[0] + '.fwfix.json'
        json.dump(rep, open(out,'w'), indent=2)
        print('wrote', out)
    print('updated', updated)


if __name__ == '__main__':
    main()

