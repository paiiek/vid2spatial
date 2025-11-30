import argparse
import os
import tarfile
import tempfile
import h5py
import json


def extract_ids(h5_path):
    with h5py.File(h5_path, 'r') as f:
        a = f['audio'][:]
        ids = []
        for b in a:
            s = b.decode('utf-8')
            # expects .../binaural16k/000234.wav
            base = os.path.basename(s)
            stem, _ = os.path.splitext(base)
            ids.append(stem)
        return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', type=int, default=1)
    ap.add_argument('--out_dir', type=str, default='mmhoa/vid2spatial/data/fairplay')
    args = ap.parse_args()

    url = 'http://dl.fbaipublicfiles.com/FAIR-Play/splits.tar.gz'
    os.makedirs(args.out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tgz = os.path.join(td, 'splits.tar.gz')
        import urllib.request
        urllib.request.urlretrieve(url, tgz)
        with tarfile.open(tgz, 'r:gz') as tf:
            tf.extractall(td)
        sp = os.path.join(td, 'splits', f'split{args.split}')
        train_ids = extract_ids(os.path.join(sp, 'train.h5'))
        val_ids = extract_ids(os.path.join(sp, 'val.h5'))
        test_ids = extract_ids(os.path.join(sp, 'test.h5'))

    def write_jsonl(name, ids):
        path = os.path.join(args.out_dir, name)
        with open(path, 'w') as f:
            for i in ids:
                rec = {
                    'audio': f'FAIR-Play/audios/{i}.wav',
                    'video': f'FAIR-Play/videos/{i}.mp4',
                    'id': i,
                }
                f.write(json.dumps(rec) + '\n')
        return path

    p_train = write_jsonl(f'split{args.split}_train.jsonl', train_ids)
    p_val = write_jsonl(f'split{args.split}_val.jsonl', val_ids)
    p_test = write_jsonl(f'split{args.split}_test.jsonl', test_ids)

    summary = {
        'split': args.split,
        'counts': {
            'train': len(train_ids),
            'val': len(val_ids),
            'test': len(test_ids),
        },
        'manifests': {
            'train': p_train,
            'val': p_val,
            'test': p_test,
        },
        'note': 'Paths are placeholders; download FAIR-Play videos.tar.gz and audios.tar.gz and extract under FAIR-Play/{videos,audios}/',
    }
    with open(os.path.join(args.out_dir, f'split{args.split}_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

