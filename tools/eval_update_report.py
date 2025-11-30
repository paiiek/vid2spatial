import argparse
import json
import os
import numpy as np
import soundfile as sf
import librosa

from mmhoa.vid2spatial.evaluate import si_sdr, log_spectral_distance, stoi_wb, seg_snr, fw_snrseg_mel, visqol_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', required=True, help='auto_fairplay report.json')
    ap.add_argument('--root', required=True, help='dataset root that contains FAIR-Play/{audios,videos}')
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
        # load mono reference: FAIR-Play binaural W-like proxy is not available; use original mono (downmix)
        # auto_fairplay derived mono directly from dataset audio; reconstruct that path
        vid_base = os.path.splitext(os.path.basename(foa))[0].split('.')[0]
        audio_path = os.path.join(args.root, 'FAIR-Play', 'audios', f'{vid_base}.wav')
        if not os.path.isfile(audio_path):
            # try binaural_audios alias
            audio_path = os.path.join(args.root, 'fairplay', 'binaural_audios', f'{vid_base}.wav')
        if not os.path.isfile(audio_path):
            continue
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        if y.ndim == 2:
            mono = np.mean(y, axis=0)
        else:
            mono = y
        foa_sig, sr2 = sf.read(foa, always_2d=True)
        W = foa_sig[:, 0].astype(np.float32)
        # match lengths
        T = min(len(W), len(mono))
        W = W[:T]
        mono = mono[:T]
        # metrics
        try:
            sdr = si_sdr(W, mono)
        except Exception:
            sdr = float('nan')
        try:
            lsd = log_spectral_distance(W, mono)
        except Exception:
            lsd = float('nan')
        # resample to 16k for STOI/segSNR if needed
        try:
            ref16 = librosa.resample(mono, orig_sr=sr, target_sr=16000)
            deg16 = librosa.resample(W, orig_sr=sr, target_sr=16000)
        except Exception:
            ref16, deg16 = mono.astype(np.float32), W.astype(np.float32)
        st = stoi_wb(ref16, deg16, 16000)
        ss = seg_snr(ref16, deg16, sr=16000)
        fw = fw_snrseg_mel(ref16, deg16, sr=16000)
        vq = visqol_score(ref16, deg16, 16000)
        r['si_sdr_w_vs_mono'] = sdr
        r['lsd_w_vs_mono'] = lsd
        r['stoi_wb'] = st
        r['seg_snr'] = ss
        r['fw_snrseg_mel'] = fw
        r['visqol'] = vq
        updated += 1
    if args.overwrite:
        json.dump(rep, open(args.report,'w'), indent=2)
    else:
        out = os.path.splitext(args.report)[0] + '.metrics.json'
        json.dump(rep, open(out,'w'), indent=2)
        print('wrote', out)
    print('updated', updated, 'entries')


if __name__ == '__main__':
    main()
