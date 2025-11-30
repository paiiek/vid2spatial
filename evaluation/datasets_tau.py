import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Iterator, Optional
from torch.utils.data import IterableDataset  # type: ignore

import numpy as np
import soundfile as sf


def read_jsonl(path: str) -> List[Dict]:
    items: List[Dict] = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_tau_meta_csv(meta_csv: str) -> List[Tuple[int, float, float]]:
    """Load TAU metadata CSV → list of (frame_idx, az_rad, el_rad)."""
    import csv, math
    rows: List[Tuple[int, float, float]] = []
    with open(meta_csv, 'r') as f:
        rdr = csv.reader(f)
        for r in rdr:
            if not r or len(r) < 5:
                continue
            try:
                fi = int(float(r[0]))
                az = float(r[-2]); el = float(r[-1])
                rows.append((fi, math.radians(az), math.radians(el)))
            except Exception:
                continue
    return rows


def sample_windows(sig: np.ndarray, sr: int, meta: List[Tuple[int, float, float]],
                   frame_hop_s: float = 0.1, win_s: float = 1.0) -> List[Dict]:
    """Create training windows aligned to metadata frames.
    Returns list of dicts with keys: {'start','end','y'(WXYZ [T,4]), 'az', 'el'}
    """
    T = sig.shape[0]
    W = win = max(1, int(round(win_s * sr)))
    half = W // 2
    out: List[Dict] = []
    for fi, az, el in meta:
        t = max(0, min(T - 1, int(round((fi - 1) * frame_hop_s * sr))))
        t0 = max(0, t - half)
        t1 = min(T, t0 + W)
        y = sig[t0:t1, :]
        if y.shape[0] < W:
            pad = np.zeros((W - y.shape[0], y.shape[1]), np.float32)
            y = np.vstack([y, pad])
        out.append({'start': int(t0), 'end': int(t1), 'y': y.astype(np.float32), 'az': float(az), 'el': float(el)})
    return out


class TAUFOALoader:
    """Minimal TAU FOA loader yielding 1-second windows with az/el labels."""
    def __init__(self, manifest: str, root: str, limit: int | None = None,
                 frame_hop_s: float = 0.1, win_s: float = 1.0) -> None:
        items = read_jsonl(manifest)
        if limit is not None:
            items = items[: int(limit)]
        self.records: List[Dict] = []
        for rec in items:
            foa_path = os.path.join(root, rec['audio'])
            meta_csv = rec['meta_csv']
            if not (os.path.isfile(foa_path) and os.path.isfile(meta_csv)):
                continue
            sig, sr = sf.read(foa_path, always_2d=True)
            meta = load_tau_meta_csv(meta_csv)
            wins = sample_windows(sig, sr, meta, frame_hop_s=frame_hop_s, win_s=win_s)
            for w in wins:
                self.records.append({'sr': sr, **w})

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        return self.records[idx]


# Iterable version for large-scale training with multi-worker sharding
class TAUFOAIterable(IterableDataset):
    """IterableDataset-style loader that yields 1s FOA windows lazily.
    Designed for use with torch DataLoader(num_workers>0).
    """
    def __init__(self, manifest: str, root: str,
                 frame_hop_s: float = 0.1, win_s: float = 1.0,
                 limit_files: Optional[int] = None) -> None:
        self.manifest = manifest
        self.root = root
        self.frame_hop_s = frame_hop_s
        self.win_s = win_s
        self.limit_files = limit_files

    def __iter__(self) -> Iterator[Dict]:
        try:
            import torch
            info = torch.utils.data.get_worker_info()
            wid = info.id if info else 0
            wnum = info.num_workers if info else 1
        except Exception:
            wid, wnum = 0, 1

        items = read_jsonl(self.manifest)
        if self.limit_files is not None:
            items = items[: int(self.limit_files)]
        for i, rec in enumerate(items):
            if (i % wnum) != wid:
                continue
            foa_path = os.path.join(self.root, rec['audio'])
            meta_csv = rec.get('meta_csv')
            if not (os.path.isfile(foa_path) and meta_csv and os.path.isfile(meta_csv)):
                continue
            try:
                sig, sr = sf.read(foa_path, always_2d=True)
                meta = load_tau_meta_csv(meta_csv)
                for w in sample_windows(sig, sr, meta, frame_hop_s=self.frame_hop_s, win_s=self.win_s):
                    yield {'sr': sr, **w}
            except Exception:
                continue
