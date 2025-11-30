from __future__ import annotations

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple

from .vision import estimate_depth


def _occ_from_depth(frame_bgr: np.ndarray, cx: float, cy: float, w: float, h: float, depth: np.ndarray) -> float:
    """Heuristic occlusion estimate in [0,1] using depth.
    - depth: normalized [0,1], larger = farther (vision.estimate_depth)
    - object depth ~ mean depth in small window around center
    - potential occluder: min depth in a thin vertical strip from top to cy near cx
    - occ = clip((obj_depth - strip_min)/max(obj_depth,eps), 0..1)
    """
    H, W = depth.shape[:2]
    cx_i = int(np.clip(cx, 0, W - 1))
    cy_i = int(np.clip(cy, 0, H - 1))
    ww = max(3, int(max(3, w * 0.1)))
    hh = max(3, int(max(3, h * 0.1)))
    x0 = max(0, cx_i - ww // 2)
    y0 = max(0, cy_i - hh // 2)
    x1 = min(W, x0 + ww)
    y1 = min(H, y0 + hh)
    obj_d = float(np.mean(depth[y0:y1, x0:x1])) if (y1 > y0 and x1 > x0) else float(depth[cy_i, cx_i])
    # strip from top to cy around cx
    sw = max(3, ww // 2)
    sx0 = max(0, cx_i - sw // 2)
    sx1 = min(W, sx0 + sw)
    sy0 = 0
    sy1 = max(1, cy_i)
    strip_min = float(np.min(depth[sy0:sy1, sx0:sx1])) if (sy1 > sy0 and sx1 > sx0) else obj_d
    if obj_d <= 1e-6:
        return 0.0
    occ = float(np.clip((obj_d - strip_min) / max(obj_d, 1e-6), 0.0, 1.0))
    return occ


def _occ_from_area(baseline_area: float, w: float, h: float) -> float:
    """Fallback occlusion estimate using bbox area shrinkage.
    occ = clip(1 - current_area / (baseline_area+eps), 0..1) with smoothing.
    """
    cur = max(1.0, float(w) * float(h))
    den = max(baseline_area, 1.0)
    r = 1.0 - (cur / den)
    return float(np.clip(r, 0.0, 1.0))


def estimate_occlusion_timeline(video_path: str, frames: List[Dict], *, use_depth: bool = True, stride: int = 1) -> Dict[str, List[Dict]]:
    """Return occlusion timeline JSON-like dict: {frames:[{frame,occ},...]}
    - If use_depth: run MiDaS depth per sampled frame and compute depth-based occ
    - Else: area-based fallback using median area over first 10 frames as baseline
    """
    import cv2
    caps = cv2.VideoCapture(video_path)
    if not caps.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    occ_list: List[Dict] = []
    # Area baseline for fallback
    areas = [float(f.get("w", 0.0)) * float(f.get("h", 0.0)) for f in frames[:10] if f.get("w") and f.get("h")]
    baseline_area = float(np.median(areas)) if areas else 1.0
    # Depth predictor bundle cache
    midas_bundle = None
    idx_set = set(int(f.get("frame", 0)) for f in frames)
    fidx = 0
    while True:
        ok, frame = caps.read()
        if not ok:
            break
        if fidx not in idx_set:
            fidx += 1
            continue
        f = next((ff for ff in frames if int(ff.get("frame", -1)) == fidx), None)
        if f is None:
            fidx += 1
            continue
        cx = float(f.get("cx", 0.0)); cy = float(f.get("cy", 0.0)); w = float(f.get("w", 0.0)); h = float(f.get("h", 0.0))
        if use_depth:
            try:
                depth = estimate_depth(frame, midas_bundle)
                occ = _occ_from_depth(frame, cx, cy, w, h, depth)
            except Exception:
                occ = _occ_from_area(baseline_area, w, h)
        else:
            occ = _occ_from_area(baseline_area, w, h)
        occ_list.append({"frame": int(fidx), "occ": float(np.clip(occ, 0.0, 1.0))})
        fidx += 1
    caps.release()
    return {"frames": occ_list}

