#!/usr/bin/env python3
"""
Quantitative analysis: HRTF vs Crossfeed binaural quality.

Metrics:
1. ILD (Interaural Level Difference) — spatial separation per frame
2. ITD (Interaural Time Difference) — via cross-correlation
3. IC (Interaural Coherence) — similarity between L/R (lower = more spatial)
4. Spectral difference — frequency-dependent spatial cues (HRTF should show pinna effects)
5. Dynamic range / crest factor
6. Temporal smoothness of spatial cues
7. Frequency-dependent ILD (low vs high freq)

Date: 2026-02-07
"""

import sys, os, json
sys.path.insert(0, "/home/seung/mmhoa/vid2spatial")

import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import stft, correlate, butter, filtfilt

BASE_DIR = Path("/home/seung/mmhoa/vid2spatial/sot_e2e_20260207")

# ============================================================================
# Metric Functions
# ============================================================================

def compute_ild_series(L, R, sr, frame_ms=50):
    """Compute ILD (dB) in sliding windows. Returns array of ILD per frame."""
    frame_len = int(sr * frame_ms / 1000)
    hop = frame_len // 2
    n_frames = (len(L) - frame_len) // hop + 1
    ilds = []
    for i in range(n_frames):
        s = i * hop
        e = s + frame_len
        rms_l = np.sqrt(np.mean(L[s:e]**2) + 1e-12)
        rms_r = np.sqrt(np.mean(R[s:e]**2) + 1e-12)
        ilds.append(20 * np.log10(rms_l / rms_r))
    return np.array(ilds)


def compute_itd_series(L, R, sr, frame_ms=50, max_lag_ms=1.0):
    """Compute ITD (ms) via cross-correlation in sliding windows."""
    frame_len = int(sr * frame_ms / 1000)
    hop = frame_len // 2
    max_lag = int(sr * max_lag_ms / 1000)
    n_frames = (len(L) - frame_len) // hop + 1
    itds = []
    for i in range(n_frames):
        s = i * hop
        e = s + frame_len
        l_seg = L[s:e]
        r_seg = R[s:e]
        cc = correlate(l_seg, r_seg, mode='full')
        mid = len(cc) // 2
        region = cc[mid - max_lag:mid + max_lag + 1]
        if len(region) == 0:
            itds.append(0.0)
            continue
        peak = np.argmax(region) - max_lag
        itds.append(peak / sr * 1000)  # ms
    return np.array(itds)


def compute_ic(L, R, sr, frame_ms=50):
    """Interaural Coherence (IC). 1.0 = identical, 0.0 = uncorrelated."""
    frame_len = int(sr * frame_ms / 1000)
    hop = frame_len // 2
    n_frames = (len(L) - frame_len) // hop + 1
    ics = []
    for i in range(n_frames):
        s = i * hop
        e = s + frame_len
        l_seg = L[s:e] - np.mean(L[s:e])
        r_seg = R[s:e] - np.mean(R[s:e])
        norm = np.sqrt(np.sum(l_seg**2) * np.sum(r_seg**2) + 1e-12)
        ic = np.sum(l_seg * r_seg) / norm
        ics.append(ic)
    return np.array(ics)


def compute_freq_ild(L, R, sr):
    """Frequency-dependent ILD in bands: low (<500Hz), mid (500-4kHz), high (>4kHz)."""
    bands = {
        'low': (20, 500),
        'mid': (500, 4000),
        'high': (4000, min(sr//2 - 1, 20000)),
    }
    result = {}
    for name, (lo, hi) in bands.items():
        b, a = butter(3, [lo/(sr/2), hi/(sr/2)], btype='band')
        Lf = filtfilt(b, a, L)
        Rf = filtfilt(b, a, R)
        rms_l = np.sqrt(np.mean(Lf**2) + 1e-12)
        rms_r = np.sqrt(np.mean(Rf**2) + 1e-12)
        result[name] = 20 * np.log10(rms_l / rms_r)
    return result


def compute_spectral_complexity(L, R, sr):
    """Measure spectral difference between L and R (HRTF should show more difference at high freq)."""
    n_fft = 2048
    _, _, Zl = stft(L, sr, nperseg=n_fft)
    _, _, Zr = stft(R, sr, nperseg=n_fft)

    mag_l = np.abs(Zl)
    mag_r = np.abs(Zr)

    # Spectral difference per frequency bin, averaged over time
    diff = np.mean(np.abs(mag_l - mag_r) / (mag_l + mag_r + 1e-12), axis=1)

    # Split into bands
    freqs = np.linspace(0, sr/2, len(diff))
    low_mask = freqs < 500
    mid_mask = (freqs >= 500) & (freqs < 4000)
    high_mask = freqs >= 4000

    return {
        'low_spectral_diff': float(np.mean(diff[low_mask])),
        'mid_spectral_diff': float(np.mean(diff[mid_mask])),
        'high_spectral_diff': float(np.mean(diff[high_mask])),
        'total_spectral_diff': float(np.mean(diff)),
    }


def analyze_file(path):
    """Compute all metrics for a binaural file."""
    audio, sr = sf.read(path, dtype='float32')
    if audio.ndim == 1:
        return None
    L, R = audio[:, 0], audio[:, 1]

    ild = compute_ild_series(L, R, sr)
    itd = compute_itd_series(L, R, sr)
    ic = compute_ic(L, R, sr)
    freq_ild = compute_freq_ild(L, R, sr)
    spectral = compute_spectral_complexity(L, R, sr)

    # Crest factor
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))
    crest = 20 * np.log10(peak / (rms + 1e-12))

    return {
        'sr': sr,
        'duration': len(L) / sr,
        # ILD
        'ild_mean': float(np.mean(np.abs(ild))),
        'ild_max': float(np.max(np.abs(ild))),
        'ild_std': float(np.std(ild)),
        'ild_range': float(np.max(ild) - np.min(ild)),
        # ITD
        'itd_mean': float(np.mean(np.abs(itd))),
        'itd_max': float(np.max(np.abs(itd))),
        'itd_std': float(np.std(itd)),
        # IC
        'ic_mean': float(np.mean(ic)),
        'ic_min': float(np.min(ic)),
        'ic_std': float(np.std(ic)),
        # Frequency ILD
        'ild_low': freq_ild['low'],
        'ild_mid': freq_ild['mid'],
        'ild_high': freq_ild['high'],
        # Spectral
        **spectral,
        # Dynamics
        'crest_factor_dB': float(crest),
        'rms_dB': float(20 * np.log10(rms + 1e-12)),
        'peak_dB': float(20 * np.log10(peak + 1e-12)),
        # Spatial cue smoothness (jerk of ILD)
        'ild_jerk': float(np.mean(np.abs(np.diff(np.diff(np.diff(ild)))))) if len(ild) > 3 else 0.0,
    }


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("=" * 80)
    print("  Quantitative Binaural Quality Analysis: HRTF vs Crossfeed")
    print("=" * 80)

    # Collect pairs: crossfeed vs HRTF
    pairs = []

    # 1) Original pipeline audio
    cf_orig_dir = BASE_DIR / "outputs"
    hrtf_orig_dir = BASE_DIR / "render_orig_hrtf"
    for vid_dir in sorted(cf_orig_dir.iterdir()):
        if not vid_dir.is_dir():
            continue
        vid_id = vid_dir.name
        cf_path = vid_dir / f"{vid_id}_binaural.wav"
        hrtf_path = hrtf_orig_dir / f"{vid_id}_hrtf_binaural.wav"
        if cf_path.exists() and hrtf_path.exists():
            pairs.append({
                'vid_id': vid_id,
                'category': 'original',
                'cf_path': str(cf_path),
                'hrtf_path': str(hrtf_path),
            })

    # 2) Instrument audio
    cf_inst_dir = BASE_DIR / "render_A_instrument"
    hrtf_inst_dir = BASE_DIR / "render_A_instrument_hrtf"
    for cf_file in sorted(cf_inst_dir.glob("*_binaural.wav")):
        vid_id = cf_file.name.split("_binaural")[0]
        # Match HRTF file
        hrtf_files = list(hrtf_inst_dir.glob(f"{vid_id.rsplit('_', 0)[0]}*_hrtf_binaural.wav"))
        # Simpler: match by vid_id prefix (first 2 parts)
        prefix = "_".join(vid_id.split("_")[:2])
        hrtf_files = list(hrtf_inst_dir.glob(f"{prefix}_*_hrtf_binaural.wav"))
        if hrtf_files:
            pairs.append({
                'vid_id': prefix,
                'category': 'instrument',
                'cf_path': str(cf_file),
                'hrtf_path': str(hrtf_files[0]),
            })

    # 3) Foley audio
    cf_foley_dir = BASE_DIR / "render_B_foley"
    hrtf_foley_dir = BASE_DIR / "render_B_foley_hrtf"
    for cf_file in sorted(cf_foley_dir.glob("*_binaural.wav")):
        vid_id = cf_file.name.split("_binaural")[0]
        prefix = "_".join(vid_id.split("_")[:2])
        hrtf_files = list(hrtf_foley_dir.glob(f"{prefix}_*_hrtf_binaural.wav"))
        if hrtf_files:
            pairs.append({
                'vid_id': prefix,
                'category': 'foley',
                'cf_path': str(cf_file),
                'hrtf_path': str(hrtf_files[0]),
            })

    print(f"\n  Found {len(pairs)} crossfeed/HRTF pairs to compare\n")

    # Analyze all
    results = []
    for i, pair in enumerate(pairs):
        print(f"  [{i+1}/{len(pairs)}] {pair['category']:12s} {pair['vid_id']}...", end=" ", flush=True)
        cf_metrics = analyze_file(pair['cf_path'])
        hrtf_metrics = analyze_file(pair['hrtf_path'])
        if cf_metrics and hrtf_metrics:
            results.append({
                'vid_id': pair['vid_id'],
                'category': pair['category'],
                'crossfeed': cf_metrics,
                'hrtf': hrtf_metrics,
            })
            print("OK")
        else:
            print("SKIP")

    # ============================================================================
    # Aggregate Statistics
    # ============================================================================

    cf_all = [r['crossfeed'] for r in results]
    hrtf_all = [r['hrtf'] for r in results]

    def avg(lst, key):
        vals = [d[key] for d in lst]
        return np.mean(vals), np.std(vals)

    print("\n" + "=" * 80)
    print("  AGGREGATE RESULTS")
    print("=" * 80)

    metrics_table = [
        ("ILD (mean abs, dB)", "ild_mean", "higher = more spatial separation"),
        ("ILD (max abs, dB)", "ild_max", "peak spatial separation"),
        ("ILD (range, dB)", "ild_range", "dynamic range of spatial cues"),
        ("ILD (std, dB)", "ild_std", "variation in spatial position"),
        ("ITD (mean abs, ms)", "itd_mean", "temporal delay between ears"),
        ("ITD (max, ms)", "itd_max", "peak temporal delay"),
        ("IC (mean)", "ic_mean", "lower = more decorrelated = more spatial"),
        ("IC (min)", "ic_min", "minimum coherence"),
        ("ILD low (<500Hz, dB)", "ild_low", "low-freq spatial cue"),
        ("ILD mid (500-4k, dB)", "ild_mid", "mid-freq spatial cue"),
        ("ILD high (>4kHz, dB)", "ild_high", "high-freq spatial cue (pinna)"),
        ("Spectral diff (low)", "low_spectral_diff", "L/R spectral difference <500Hz"),
        ("Spectral diff (mid)", "mid_spectral_diff", "L/R spectral difference 500-4kHz"),
        ("Spectral diff (high)", "high_spectral_diff", "L/R spectral difference >4kHz"),
        ("Spectral diff (total)", "total_spectral_diff", "overall L/R spectral difference"),
        ("Crest factor (dB)", "crest_factor_dB", "dynamic range measure"),
        ("RMS level (dB)", "rms_dB", "average loudness"),
        ("ILD jerk", "ild_jerk", "smoothness of spatial cues (lower = smoother)"),
    ]

    print(f"\n  {'Metric':40s} {'Crossfeed':>16s} {'HRTF':>16s} {'Δ':>10s}  Note")
    print("  " + "-" * 100)

    for label, key, note in metrics_table:
        cf_m, cf_s = avg(cf_all, key)
        h_m, h_s = avg(hrtf_all, key)
        delta = h_m - cf_m
        print(f"  {label:40s} {cf_m:>8.4f}±{cf_s:.3f} {h_m:>8.4f}±{h_s:.3f} {delta:>+8.4f}  {note}")

    # ============================================================================
    # Per-Category Breakdown
    # ============================================================================

    print("\n\n" + "=" * 80)
    print("  PER-CATEGORY BREAKDOWN")
    print("=" * 80)

    categories = ['original', 'instrument', 'foley']
    key_metrics = ['ild_mean', 'itd_mean', 'ic_mean', 'high_spectral_diff', 'ild_jerk']

    for cat in categories:
        cat_results = [r for r in results if r['category'] == cat]
        if not cat_results:
            continue
        cf_cat = [r['crossfeed'] for r in cat_results]
        hrtf_cat = [r['hrtf'] for r in cat_results]

        print(f"\n  --- {cat.upper()} (n={len(cat_results)}) ---")
        print(f"  {'Metric':30s} {'Crossfeed':>12s} {'HRTF':>12s} {'Δ':>10s}")
        print("  " + "-" * 70)

        for key in key_metrics:
            cf_m = np.mean([d[key] for d in cf_cat])
            h_m = np.mean([d[key] for d in hrtf_cat])
            delta = h_m - cf_m
            print(f"  {key:30s} {cf_m:>12.4f} {h_m:>12.4f} {delta:>+10.4f}")

    # ============================================================================
    # Per-Video Detail (original only)
    # ============================================================================

    print("\n\n" + "=" * 80)
    print("  PER-VIDEO DETAIL (Original Pipeline Audio)")
    print("=" * 80)

    orig_results = [r for r in results if r['category'] == 'original']

    print(f"\n  {'Video':25s} {'CF_ILD':>8s} {'H_ILD':>8s} {'CF_IC':>7s} {'H_IC':>7s} {'CF_SD':>7s} {'H_SD':>7s} {'CF_Jrk':>8s} {'H_Jrk':>8s}")
    print("  " + "-" * 95)

    for r in orig_results:
        cf = r['crossfeed']
        h = r['hrtf']
        print(f"  {r['vid_id']:25s} "
              f"{cf['ild_mean']:>8.3f} {h['ild_mean']:>8.3f} "
              f"{cf['ic_mean']:>7.4f} {h['ic_mean']:>7.4f} "
              f"{cf['total_spectral_diff']:>7.4f} {h['total_spectral_diff']:>7.4f} "
              f"{cf['ild_jerk']:>8.4f} {h['ild_jerk']:>8.4f}")

    # ============================================================================
    # Quality Score Summary
    # ============================================================================

    print("\n\n" + "=" * 80)
    print("  QUALITY ASSESSMENT SUMMARY")
    print("=" * 80)

    # Compute relative improvements
    ild_mean_cf = np.mean([r['crossfeed']['ild_mean'] for r in results])
    ild_mean_h = np.mean([r['hrtf']['ild_mean'] for r in results])

    ic_mean_cf = np.mean([r['crossfeed']['ic_mean'] for r in results])
    ic_mean_h = np.mean([r['hrtf']['ic_mean'] for r in results])

    sd_hi_cf = np.mean([r['crossfeed']['high_spectral_diff'] for r in results])
    sd_hi_h = np.mean([r['hrtf']['high_spectral_diff'] for r in results])

    sd_total_cf = np.mean([r['crossfeed']['total_spectral_diff'] for r in results])
    sd_total_h = np.mean([r['hrtf']['total_spectral_diff'] for r in results])

    jerk_cf = np.mean([r['crossfeed']['ild_jerk'] for r in results])
    jerk_h = np.mean([r['hrtf']['ild_jerk'] for r in results])

    itd_cf = np.mean([r['crossfeed']['itd_mean'] for r in results])
    itd_h = np.mean([r['hrtf']['itd_mean'] for r in results])

    print(f"""
  1. Spatial Separation (ILD):
     Crossfeed: {ild_mean_cf:.3f} dB  |  HRTF: {ild_mean_h:.3f} dB
     → {'HRTF has more separation' if ild_mean_h > ild_mean_cf else 'Crossfeed has more (possibly exaggerated) separation'}

  2. Interaural Coherence (IC):
     Crossfeed: {ic_mean_cf:.4f}  |  HRTF: {ic_mean_h:.4f}
     → {'HRTF more decorrelated (more spatial)' if ic_mean_h < ic_mean_cf else 'Crossfeed more decorrelated'}

  3. High-frequency spectral difference (pinna cues):
     Crossfeed: {sd_hi_cf:.4f}  |  HRTF: {sd_hi_h:.4f}
     → {'HRTF shows more high-freq L/R difference (expected from pinna filtering)' if sd_hi_h > sd_hi_cf else 'Similar high-freq differences'}

  4. Overall spectral difference:
     Crossfeed: {sd_total_cf:.4f}  |  HRTF: {sd_total_h:.4f}
     → {'HRTF shows richer spectral differences' if sd_total_h > sd_total_cf else 'Similar spectral profiles'}

  5. ITD (temporal cues):
     Crossfeed: {itd_cf:.4f} ms  |  HRTF: {itd_h:.4f} ms
     → {'HRTF shows more ITD variation' if itd_h > itd_cf else 'Similar ITD'}

  6. Spatial cue smoothness (ILD jerk):
     Crossfeed: {jerk_cf:.4f}  |  HRTF: {jerk_h:.4f}
     → {'HRTF smoother' if jerk_h < jerk_cf else 'Crossfeed smoother'} (lower = smoother)
""")

    # Save full results
    out_path = str(BASE_DIR / "hrtf_quality_analysis.json")
    with open(out_path, 'w') as f:
        json.dump({
            'n_pairs': len(results),
            'aggregate': {
                'crossfeed': {key: float(np.mean([r['crossfeed'][key] for r in results])) for _, key, _ in metrics_table},
                'hrtf': {key: float(np.mean([r['hrtf'][key] for r in results])) for _, key, _ in metrics_table},
            },
            'per_video': results,
        }, f, indent=2)
    print(f"  Full results saved: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
