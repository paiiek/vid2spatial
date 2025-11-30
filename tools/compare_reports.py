import argparse
import json
from pathlib import Path


def load_stats(path):
    p = Path(path)
    if p.suffix == '.json' and p.name.endswith('report.summary.json'):
        return json.load(open(p))
    # fallback: aggregate from report.json if summary not given
    rep = json.load(open(p))
    vals = [r.get('si_sdr_w_vs_mono') for r in rep.get('results', []) if r.get('status')=='ok' and isinstance(r.get('si_sdr_w_vs_mono'), (int,float))]
    lsds = [r.get('lsd_w_vs_mono') for r in rep.get('results', []) if r.get('status')=='ok' and isinstance(r.get('lsd_w_vs_mono'), (int,float))]
    return {
        'count': len(vals),
        'si_sdr_mean': float(sum(vals)/len(vals)) if vals else None,
        'si_sdr_std': None,
        'lsd_mean': float(sum(lsds)/len(lsds)) if lsds else None,
        'lsd_std': None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='path to report.summary.json or report.json')
    ap.add_argument('--b', required=True, help='path to report.summary.json or report.json')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    A = load_stats(args.a)
    B = load_stats(args.b)
    comp = {
        'A_path': args.a,
        'B_path': args.b,
        'A': A,
        'B': B,
        'delta': {
            'si_sdr_mean': (A.get('si_sdr_mean') - B.get('si_sdr_mean')) if A.get('si_sdr_mean') is not None and B.get('si_sdr_mean') is not None else None,
            'lsd_mean': (A.get('lsd_mean') - B.get('lsd_mean')) if A.get('lsd_mean') is not None and B.get('lsd_mean') is not None else None,
        }
    }
    json.dump(comp, open(args.out, 'w'), indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()

