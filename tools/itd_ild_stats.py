import argparse
import json
import numpy as np
import soundfile as sf
import librosa


def gcc_phat_itd(x, y, sr, n_fft=1024, hop=256):
    X = librosa.stft(x, n_fft=n_fft, hop_length=hop)
    Y = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    R = X * np.conj(Y)
    R /= np.abs(R) + 1e-8
    r = np.fft.irfft(R, axis=0)
    # pick max lag
    lags = np.argmax(np.abs(r), axis=0)
    # convert to signed lag around zero
    N = r.shape[0]
    lags = np.where(lags > N//2, lags - N, lags)
    itd = lags / float(sr)
    return float(np.mean(itd)), float(np.std(itd))


def ild_stats(x, y, n_fft=1024, hop=256):
    X = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop)) + 1e-8
    Y = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) + 1e-8
    ild = 20.0 * np.log10(X / Y)
    return float(np.mean(ild)), float(np.std(ild))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stereo', required=True, help='stereo wav path')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    y, sr = sf.read(args.stereo, always_2d=True)
    if y.shape[1] < 2:
        raise ValueError('need stereo')
    L = y[:,0].astype(np.float32); R = y[:,1].astype(np.float32)
    itd_mu, itd_std = gcc_phat_itd(L, R, sr)
    ild_mu, ild_std = ild_stats(L, R)
    out = {'file': args.stereo, 'sr': sr, 'itd_mean_s': itd_mu, 'itd_std_s': itd_std,
           'ild_mean_db': ild_mu, 'ild_std_db': ild_std}
    json.dump(out, open(args.out, 'w'), indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()

