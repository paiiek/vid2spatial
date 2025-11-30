import argparse
import json
import os
import time


def read_jsonl(path):
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--mode', type=str, default='av', choices=['av','audio_only','video_only'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--sample', type=int, default=50)
    args = ap.parse_args()

    t0 = time.time()
    total = 0
    exist_audio = 0
    exist_video = 0
    entries = []

    for i, rec in enumerate(read_jsonl(args.manifest)):
        if args.sample and i >= args.sample:
            break
        total += 1
        a = rec.get('audio'); v = rec.get('video')
        ea = os.path.isfile(a) if a else False
        ev = os.path.isfile(v) if v else False
        exist_audio += int(ea)
        exist_video += int(ev)
        entries.append({'audio': a, 'video': v, 'audio_exists': ea, 'video_exists': ev})

    report = {
        'manifest': args.manifest,
        'mode': args.mode,
        'sample': args.sample,
        'counts': {
            'total_read': total,
            'audio_available': exist_audio,
            'video_available': exist_video,
        },
        'entries': entries,
        'elapsed_sec': time.time() - t0,
        'note': 'Availability check only. For metrics, provide trajectory JSON and use evaluate.py.',
    }

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)
    print('wrote report', args.out)


if __name__ == '__main__':
    main()

