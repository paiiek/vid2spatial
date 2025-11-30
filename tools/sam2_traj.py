import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple


def _load_frames(video_path: str, stride: int = 1) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames = []
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if (f % stride) == 0:
            frames.append(frame)
        f += 1
    cap.release()
    return frames


def sam2_video_traj(video_path: str,
                    init_bbox_xywh: Tuple[int, int, int, int],
                    model_id: Optional[str] = None,
                    config_path: Optional[str] = None,
                    checkpoint_path: Optional[str] = None,
                    stride: int = 1,
                    use_bfloat16: bool = True) -> List[Dict]:
    """Compute trajectory centers via SAM2 video predictor given an init bbox.

    Returns list of dicts with keys: frame, cx, cy, w, h (frame indices reflect subsampling by stride).
    """
    frames = _load_frames(video_path, stride=stride)
    if not frames:
        return []

    predictor = None
    # Try HF pretrained image/video predictor
    try:
        if model_id is not None:
            from sam2.sam2_video_predictor import SAM2VideoPredictor  # type: ignore
            predictor = SAM2VideoPredictor.from_pretrained(model_id)
    except Exception:
        predictor = None

    # Fallback: build from config + checkpoint
    if predictor is None and (config_path and checkpoint_path):
        try:
            from sam2.build_sam import build_sam2_video_predictor  # type: ignore
            predictor = build_sam2_video_predictor(config_path, checkpoint_path)
        except Exception:
            predictor = None

    if predictor is None:
        raise RuntimeError("SAM2VideoPredictor unavailable. Install SAM2 and provide model_id or cfg+ckpt.")

    import torch
    # choose autocast dtype: BF16 on Ampere+ ; else FP16 for Turing (e.g., RTX 2080)
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
        except Exception:
            major, minor = (7, 5)
        if use_bfloat16 and major >= 8:
            ac_dtype = torch.bfloat16
        else:
            ac_dtype = torch.float16
        autocast_ctx = torch.autocast("cuda", dtype=ac_dtype)
    else:
        autocast_ctx = nullcontext()
    with torch.inference_mode(), autocast_ctx:
        state = predictor.init_state(frames)
        x, y, w, h = init_bbox_xywh
        # SAM2 API expects prompts; use box prompt
        prompts = {
            "points": None,
            "labels": None,
            "boxes": np.array([[x, y, x + w, y + h]], dtype=np.float32),
        }
        predictor.add_new_points_or_box(state, prompts)
        traj = []
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
            # masks: [N_obj, H, W] boolean/uint8; pick first object
            if masks is None or len(masks) == 0:
                continue
            m = masks[0]
            ys, xs = np.nonzero(m)
            if len(xs) == 0:
                continue
            cx = float(xs.mean()); cy = float(ys.mean())
            # approximate bbox size from mask area (square root heuristic)
            area = float(m.sum())
            side = float(np.sqrt(area))
            traj.append({"frame": int(frame_idx)*stride, "cx": cx, "cy": cy, "w": side, "h": side})
    return traj


class nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


__all__ = ["sam2_video_traj"]
