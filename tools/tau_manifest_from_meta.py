import argparse
import os
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta_dir', required=True, help='metadata_dev directory (contains dev-train/dev-val/dev-test)')
    ap.add_argument('--split', required=True, choices=['dev-train','dev-val','dev-test'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--audio_prefix', default='TAU2021/foa_dev', help='prefix for FOA wavs')
    args = ap.parse_args()

    split_dir = os.path.join(args.meta_dir, args.split)
    files = [f for f in os.listdir(split_dir) if f.endswith('.csv')]
    files.sort()
    with open(args.out, 'w') as f:
        for fn in files:
            stem = os.path.splitext(fn)[0]
            rec = {
                'audio': os.path.join(args.audio_prefix, stem + '.wav'),
                'meta_csv': os.path.join(args.meta_dir, args.split, fn),
                'id': stem,
            }
            f.write(json.dumps(rec) + '\n')
    print('wrote', args.out, 'count=', len(files))


if __name__ == '__main__':
    main()

