import argparse
import json
import os
from pathlib import Path
import numpy as np
import soundfile as sf
from mmhoa.vid2spatial.foa_render import foa_to_stereo
from mmhoa.vid2spatial.tools.itd_ild_stats import gcc_phat_itd, ild_stats


def w_channel(path: str):
    y, sr = sf.read(path, always_2d=True)
    return y.T.astype(np.float32), sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ablate', required=True, help='ir_ablate.json')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    data = json.load(open(args.ablate))
    rows = []
    for it in data.get('items', []):
        rec = {'id': it.get('id')}
        for key in ['none_foa', 'visual_foa']:
            p = it.get(key)
            if not p or not os.path.isfile(p):
                rec[f'{key}_error'] = 'missing'
                continue
            foa, sr = w_channel(p)
            if foa.shape[0] != 4:
                rec[f'{key}_error'] = 'not_4ch'
                continue
            # FOA -> stereo (fallback decoder OK)
            st = foa_to_stereo(foa, sr)
            itd_mu, itd_std = gcc_phat_itd(st[0], st[1], sr)
            ild_mu, ild_std = ild_stats(st[0], st[1])
            rec[f'{key}_itd_mean_s'] = float(itd_mu)
            rec[f'{key}_itd_std_s'] = float(itd_std)
            rec[f'{key}_ild_mean_db'] = float(ild_mu)
            rec[f'{key}_ild_std_db'] = float(ild_std)
        rows.append(rec)
    out = {'items': rows}
    json.dump(out, open(args.out, 'w'), indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()

