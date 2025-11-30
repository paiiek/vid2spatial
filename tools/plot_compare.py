import argparse
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_vals(p):
    rep = json.load(open(p))
    if 'results' in rep:
        sdr = [r.get('si_sdr_w_vs_mono') for r in rep['results'] if r.get('status')=='ok' and isinstance(r.get('si_sdr_w_vs_mono'), (int,float))]
        lsd = [r.get('lsd_w_vs_mono') for r in rep['results'] if r.get('status')=='ok' and isinstance(r.get('lsd_w_vs_mono'), (int,float))]
    else:
        # summary only
        sdr = []; lsd = []
    return np.array(sdr, dtype=float), np.array(lsd, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sdr_a, lsd_a = load_vals(args.a)
    sdr_b, lsd_b = load_vals(args.b)
    # histograms
    plt.figure(figsize=(8,4));
    bins = 20
    if sdr_a.size and sdr_b.size:
        plt.hist(sdr_a, bins=bins, alpha=0.5, label='A SI-SDR')
        plt.hist(sdr_b, bins=bins, alpha=0.5, label='B SI-SDR')
        plt.legend(); plt.title('SI-SDR distribution'); plt.tight_layout()
        plt.savefig(str(Path(args.out_dir)/'si_sdr_hist.png')); plt.close()
    if lsd_a.size and lsd_b.size:
        plt.figure(figsize=(8,4));
        plt.hist(lsd_a, bins=bins, alpha=0.5, label='A LSD')
        plt.hist(lsd_b, bins=bins, alpha=0.5, label='B LSD')
        plt.legend(); plt.title('LSD distribution'); plt.tight_layout()
        plt.savefig(str(Path(args.out_dir)/'lsd_hist.png')); plt.close()
    print('wrote plots to', args.out_dir)


if __name__ == '__main__':
    main()

