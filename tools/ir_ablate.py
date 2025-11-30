import argparse
import json
import os
from pathlib import Path
from subprocess import run


def run_once(video, audio, out_dir, *, ir_backend: str, air_foa: str | None = None,
             brir_L: str | None = None, brir_R: str | None = None) -> dict:
    stem = Path(video).stem
    out: dict = {}
    # FOA always written
    out_foa = str(Path(out_dir)/f"{stem}.{ir_backend}.foa.wav")
    cmd = [
        'python3','-m','mmhoa.vid2spatial.run_demo',
        '--video', video,
        '--audio', audio,
        '--out_foa', out_foa,
        '--ir_backend', ir_backend,
        '--stride','8',
        '--method','yolo',
        '--cls','none',
    ]
    if air_foa and ir_backend in ('auto','pra','schroeder','none'):
        # FOA AIR convolution happens in FOA domain; independent of stereo backends
        cmd += ['--air_foa', air_foa]
    if ir_backend == 'brir' and brir_L and brir_R:
        # request stereo output for BRIR evaluation
        out_st = str(Path(out_dir)/f"{stem}.brir.stereo.wav")
        cmd += ['--out_st', out_st, '--brir_L', brir_L, '--brir_R', brir_R]
        out['stereo'] = out_st
    run(cmd, check=False)
    out['foa'] = out_foa
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--limit', type=int, default=5)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--air_foa', type=str, default=None, help='Optional 4ch FOA AIR wav (AmbiX)')
    ap.add_argument('--brir_L', type=str, default=None, help='Optional BRIR left wav')
    ap.add_argument('--brir_R', type=str, default=None, help='Optional BRIR right wav')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    items = []
    with open(args.manifest,'r') as f:
        for i, line in enumerate(f):
            if i >= args.limit: break
            items.append(json.loads(line))
    results = []
    for rec in items:
        a = os.path.join(args.root, rec['audio'])
        v = os.path.join(args.root, rec['video'])
        base = Path(v).stem
        out_rec = {'id': base}
        # none
        none_out = run_once(v, a, args.out_dir, ir_backend='none', air_foa=args.air_foa)
        out_rec['none_foa'] = none_out.get('foa')
        # visual stereo IR
        vis_out = run_once(v, a, args.out_dir, ir_backend='visual', air_foa=args.air_foa)
        out_rec['visual_foa'] = vis_out.get('foa')
        # optional FOA AIR
        if args.air_foa:
            air_out = run_once(v, a, args.out_dir, ir_backend='none', air_foa=args.air_foa)
            out_rec['air_foa'] = air_out.get('foa')
        # optional BRIR stereo (record stereo path for ITD/ILD later)
        if args.brir_L and args.brir_R:
            brir_out = run_once(v, a, args.out_dir, ir_backend='brir', brir_L=args.brir_L, brir_R=args.brir_R)
            if 'stereo' in brir_out:
                out_rec['brir_stereo'] = brir_out['stereo']
        results.append(out_rec)
    out = Path(args.out_dir)/'ir_ablate.json'
    json.dump({'items': results}, open(out,'w'), indent=2)
    print('wrote', out)


if __name__ == '__main__':
    main()
