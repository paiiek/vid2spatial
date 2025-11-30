import argparse
import soundfile as sf
import numpy as np

from mmhoa.vid2spatial.foa_render import foa_to_stereo, foa_to_binaural


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--foa', required=True, help='FOA wav (AmbiX ACN/SN3D [W,Y,Z,X])')
    ap.add_argument('--out', required=True, help='output wav (stereo or binaural)')
    ap.add_argument('--mode', type=str, default='stereo', choices=['stereo','binaural'])
    args = ap.parse_args()

    y, sr = sf.read(args.foa, always_2d=True)
    y = y.T.astype(np.float32)
    if y.shape[0] != 4:
        raise ValueError('Input must be 4-channel FOA [W,Y,Z,X]')
    if args.mode == 'stereo':
        out = foa_to_stereo(y, sr)
    else:
        out = foa_to_binaural(y, sr)
    sf.write(args.out, out.T, sr, subtype='FLOAT')
    print('wrote', args.out)


if __name__ == '__main__':
    main()

