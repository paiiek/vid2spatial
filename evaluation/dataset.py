import json
import os
from typing import List, Dict, Optional, Tuple

import numpy as np
import librosa

from .foa_render import interpolate_angles


class TrajAudioDataset:
    """Dataset that pairs mono audio with precomputed trajectory JSON.

    Manifest formats supported:
    - JSONL: each line has {"audio": path, "traj": path}
    - CSV: audio,traj
    """

    def __init__(self, manifest: str, sr: Optional[int] = None):
        self.items = self._read_manifest(manifest)
        self.sr = sr

    @staticmethod
    def _read_manifest(path: str) -> List[Dict]:
        ext = os.path.splitext(path)[1].lower()
        items: List[Dict] = []
        if ext == ".jsonl":
            with open(path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    items.append(json.loads(line))
        else:
            # CSV fallback, expects headerless: audio,traj
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    a, t = line.split(",")
                    items.append({"audio": a.strip(), "traj": t.strip()})
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict:
        rec = self.items[i]
        x, sr = librosa.load(rec["audio"], sr=self.sr, mono=True)
        with open(rec["traj"], "r") as f:
            traj = json.load(f)
        T = x.shape[0]
        az, el = interpolate_angles(traj["frames"], T=T, sr=sr)
        return {"audio": x.astype(np.float32), "sr": sr, "az": az, "el": el}


def write_manifest_from_glob(audios: List[str], trajs: List[str], out_path: str) -> None:
    assert len(audios) == len(trajs)
    with open(out_path, "w") as f:
        for a, t in zip(audios, trajs):
            f.write(json.dumps({"audio": a, "traj": t}) + "\n")


__all__ = ["TrajAudioDataset", "write_manifest_from_glob"]

