import argparse
import csv
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta_csv', type=str, required=False, help='TAU metadata CSV path (dev or eval)')
    ap.add_argument('--audio_root', type=str, required=False, help='Root dir of TAU audio')
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    items = []
    if args.meta_csv and os.path.isfile(args.meta_csv):
        # Expect columns with filename or relative path; we write minimal manifest
        with open(args.meta_csv, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                # try common schemas
                # If CSV has 'filename' or first column is filename
                fn = None
                if header and 'filename' in header:
                    fn = row[header.index('filename')]
                else:
                    fn = row[0]
                audio_path = fn if os.path.isabs(fn) else os.path.join(args.audio_root or '', fn)
                items.append({'audio': audio_path})
    else:
        # Fallback: create placeholder note-only file
        with open(args.out, 'w') as f:
            f.write('')
        print('wrote empty manifest (no meta_csv). Provide --meta_csv to populate.')
        return

    with open(args.out, 'w') as f:
        for it in items:
            f.write(json.dumps(it) + '\n')
    print('wrote', args.out, 'count=', len(items))


if __name__ == '__main__':
    main()

