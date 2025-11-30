import argparse
import json
import numpy as np
import csv
from pathlib import Path


def aggregate(report_path: str):
    rep = json.load(open(report_path))
    rows = []
    for r in rep.get('results', []):
        if r.get('status') != 'ok':
            continue
        rows.append({
            'id': r.get('id'),
            'si_sdr': r.get('si_sdr_w_vs_mono'),
            'lsd': r.get('lsd_w_vs_mono'),
        })
    # stats
    sdrs = [x['si_sdr'] for x in rows if isinstance(x['si_sdr'], (int,float))]
    lsds = [x['lsd'] for x in rows if isinstance(x['lsd'], (int,float))]
    stats = {
        'count': len(rows),
        'si_sdr_mean': float(np.mean(sdrs)) if sdrs else None,
        'si_sdr_std': float(np.std(sdrs)) if sdrs else None,
        'lsd_mean': float(np.mean(lsds)) if lsds else None,
        'lsd_std': float(np.std(lsds)) if lsds else None,
    }
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', required=True)
    ap.add_argument('--out_base', default=None, help='output base path (without extension)')
    args = ap.parse_args()
    base = args.out_base or str(Path(args.report).with_suffix(''))
    rows, stats = aggregate(args.report)
    # write CSV
    csv_path = base + '.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id','si_sdr','lsd'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # write JSON stats
    json_path = base + '.summary.json'
    json.dump(stats, open(json_path, 'w'), indent=2)
    print('wrote', csv_path, 'and', json_path)


if __name__ == '__main__':
    main()

