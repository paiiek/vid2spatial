"""
Refactored vision module with smaller, testable functions.

This module breaks down the monolithic compute_trajectory_3d function into
logical, composable pieces for better maintainability and testing.
"""
import math
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Callable

# Basic helper classes and functions
# (Previously these were imported from a separate module, now defined inline)

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters for 3D projection."""
    width: int
    height: int
    fov_deg: float

    @property
    def focal_length(self) -> float:
        """Compute focal length from FOV."""
        return (self.width / 2.0) / np.tan(np.radians(self.fov_deg / 2.0))

    @property
    def cx(self) -> float:
        """Principal point x-coordinate."""
        return self.width / 2.0

    @property
    def cy(self) -> float:
        """Principal point y-coordinate."""
        return self.height / 2.0


def pixel_to_ray(px: float, py: float, K: CameraIntrinsics) -> np.ndarray:
    """Convert pixel coordinates to normalized 3D ray."""
    x = (px - K.cx) / K.focal_length
    y = (py - K.cy) / K.focal_length
    z = 1.0
    ray = np.array([x, y, z], dtype=np.float32)
    return ray / np.linalg.norm(ray)


def ray_to_angles(ray: np.ndarray) -> Tuple[float, float]:
    """Convert 3D ray to azimuth and elevation angles."""
    x, y, z = ray
    az = np.arctan2(x, z)  # Azimuth
    el = np.arcsin(y)  # Elevation
    return float(az), float(el)


def smooth_series(series: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Apply exponential moving average smoothing."""
    if len(series) == 0:
        return series
    smoothed = np.zeros_like(series)
    smoothed[0] = series[0]
    for i in range(1, len(series)):
        smoothed[i] = alpha * series[i] + (1 - alpha) * smoothed[i-1]
    return smoothed


def _load_midas():
    """Load MiDaS depth estimation model."""
    try:
        import torch
        model_type = "DPT_Large"
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.dpt_transform

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        midas.to(device)

        return midas, transform, device
    except Exception as e:
        print(f"[warn] MiDaS load failed: {e}")
        return None, None, None


def estimate_depth(frame: np.ndarray, midas_bundle: Optional[tuple] = None) -> np.ndarray:
    """Estimate depth map from frame."""
    if midas_bundle is None:
        # Return dummy depth map
        h, w = frame.shape[:2]
        return np.ones((h, w), dtype=np.float32) * 0.5

    model, transform, device = midas_bundle
    if model is None:
        h, w = frame.shape[:2]
        return np.ones((h, w), dtype=np.float32) * 0.5

    import torch
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    # Normalize to [0, 1]
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth


def auto_init_bbox(frame: np.ndarray, cls_name: str = "person", fallback_center: bool = False) -> Optional[Tuple[int, int, int, int]]:
    """Auto-initialize bounding box using YOLO detection."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolo11n.pt")
        results = model(frame, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name_detected = model.names[cls_id]
                if cls_name_detected.lower() == cls_name.lower():
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                    return (x, y, w, h)
    except Exception as e:
        print(f"[warn] YOLO auto-init failed: {e}")

    if fallback_center:
        h, w = frame.shape[:2]
        box_w, box_h = w // 4, h // 4
        return (w//2 - box_w//2, h//2 - box_h//2, box_w, box_h)

    return None


def tm_track(video_path: str, init_bbox: Tuple[int, int, int, int], sample_stride: int = 1) -> List[Dict]:
    """Track object using KCF tracker."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    tracker = cv2.TrackerKCF_create()
    traj = []

    fidx = 0
    initialized = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if (fidx % sample_stride) != 0:
            fidx += 1
            continue

        if not initialized:
            tracker.init(frame, init_bbox)
            x, y, w, h = init_bbox
            traj.append({"frame": fidx, "x": x, "y": y, "w": w, "h": h})
            initialized = True
        else:
            ok, bbox = tracker.update(frame)
            if ok:
                x, y, w, h = [int(v) for v in bbox]
                traj.append({"frame": fidx, "x": x, "y": y, "w": w, "h": h})
            else:
                # Tracking failed, use last known bbox
                if traj:
                    last = traj[-1]
                    traj.append({"frame": fidx, "x": last["x"], "y": last["y"], "w": last["w"], "h": last["h"]})

        fidx += 1

    cap.release()
    return traj


def yolo_bytetrack_traj(video_path: str, cls_name: str = "person",
                        select_track_id: Optional[int] = None,
                        sample_stride: int = 1,
                        conf: float = 0.25) -> List[Dict]:
    """Track object using YOLO + ByteTrack."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolo11n.pt")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        traj = []
        fidx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if (fidx % sample_stride) != 0:
                fidx += 1
                continue

            results = model.track(frame, persist=True, verbose=False, conf=conf)

            for r in results:
                if r.boxes is None or r.boxes.id is None:
                    continue

                boxes = r.boxes
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    cls_name_detected = model.names[cls_id]

                    if cls_name_detected.lower() != cls_name.lower():
                        continue

                    track_id = int(box.id[0])

                    if select_track_id is not None and track_id != select_track_id:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)

                    traj.append({
                        "frame": fidx,
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "track_id": track_id
                    })

                    if select_track_id is not None:
                        break

            fidx += 1

        cap.release()
        return traj

    except Exception as e:
        print(f"[error] YOLO tracking failed: {e}")
        raise


def refine_center_grabcut(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Refine object center using GrabCut segmentation."""
    x, y, w, h = bbox

    # Ensure bbox is within frame
    h_frame, w_frame = frame.shape[:2]
    x = max(0, min(x, w_frame - 1))
    y = max(0, min(y, h_frame - 1))
    w = max(1, min(w, w_frame - x))
    h = max(1, min(h, h_frame - y))

    try:
        mask = np.zeros(frame.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        rect = (x, y, w, h)
        cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # Find center of mass
        M = cv2.moments(mask2)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            return float(cx), float(cy)
    except Exception as e:
        pass

    # Fallback to bbox center
    return float(x + w/2), float(y + h/2)


# Placeholder for compute_trajectory_3d (basic version)
def compute_trajectory_3d(
    video_path: str,
    init_bbox: Optional[Tuple[int, int, int, int]] = None,
    fov_deg: float = 60.0,
    depth_scale_m: Tuple[float, float] = (1.0, 3.0),
    use_midas: bool = True,
    sample_stride: int = 1,
    method: str = "yolo",
    cls_name: str = "person",
    refine_center: bool = False,
    refine_center_method: str = "grabcut",
    depth_backend: str = "auto",
    sam2_mask_fn: Optional[Callable] = None,
    depth_fn: Optional[Callable] = None,
    fallback_center_if_no_bbox: bool = False,
    select_track_id: Optional[int] = None,
    smooth_alpha: float = 0.2,
    sam2_model_id: Optional[str] = None,
    sam2_cfg: Optional[str] = None,
    sam2_ckpt: Optional[str] = None,
) -> Dict:
    """
    Basic implementation of compute_trajectory_3d.
    Uses compute_trajectory_3d_refactored for actual processing.
    """
    return compute_trajectory_3d_refactored(
        video_path=video_path,
        init_bbox=init_bbox,
        fov_deg=fov_deg,
        depth_scale_m=depth_scale_m,
        use_midas=use_midas,
        sample_stride=sample_stride,
        method=method,
        cls_name=cls_name,
        refine_center=refine_center,
        refine_center_method=refine_center_method,
        depth_backend=depth_backend,
        sam2_mask_fn=sam2_mask_fn,
        depth_fn=depth_fn,
        fallback_center_if_no_bbox=fallback_center_if_no_bbox,
        select_track_id=select_track_id,
        smooth_alpha=smooth_alpha,
        sam2_model_id=sam2_model_id,
        sam2_cfg=sam2_cfg,
        sam2_ckpt=sam2_ckpt,
    )


# ============================================================================
# Tracking Initialization
# ============================================================================

def initialize_tracking(
    video_path: str,
    method: str,
    init_bbox: Optional[Tuple[int, int, int, int]] = None,
    cls_name: str = "person",
    select_track_id: Optional[int] = None,
    fallback_center_if_no_bbox: bool = False,
    sample_stride: int = 1,
    sam2_model_id: Optional[str] = None,
    sam2_cfg: Optional[str] = None,
    sam2_ckpt: Optional[str] = None,
) -> List[Dict]:
    """
    Initialize tracking and return 2D trajectory.

    Args:
        video_path: Path to input video
        method: Tracking method ('yolo', 'kcf', 'sam2')
        init_bbox: Initial bounding box (x, y, w, h)
        cls_name: Object class name for YOLO
        select_track_id: Specific track ID to select
        fallback_center_if_no_bbox: Use center bbox if detection fails
        sample_stride: Frame sampling stride
        sam2_model_id: SAM2 model identifier
        sam2_cfg: SAM2 config path
        sam2_ckpt: SAM2 checkpoint path

    Returns:
        List of tracking records: [{"frame": int, "cx": float, "cy": float, "w": float, "h": float}, ...]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    ok, first = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to read first frame")

    H0, W0 = first.shape[:2]

    # Helper to create center fallback bbox
    def get_fallback_bbox():
        if not fallback_center_if_no_bbox:
            raise RuntimeError("Could not auto-initialize bbox; please provide init_bbox")
        cw, ch = int(W0 * 0.2), int(H0 * 0.2)
        cx0, cy0 = int(W0 * 0.5 - cw * 0.5), int(H0 * 0.5 - ch * 0.5)
        return (cx0, cy0, cw, ch)

    if method == "kcf":
        return _initialize_kcf_tracking(
            video_path, first, init_bbox, get_fallback_bbox
        )
    elif method == "yolo":
        return _initialize_yolo_tracking(
            video_path, first, cls_name, select_track_id, init_bbox, get_fallback_bbox
        )
    elif method == "sam2":
        return _initialize_sam2_tracking(
            video_path, first, init_bbox, cls_name, select_track_id,
            sample_stride, sam2_model_id, sam2_cfg, sam2_ckpt, get_fallback_bbox
        )
    else:
        raise ValueError(f"Unknown tracking method: {method}")


def _initialize_kcf_tracking(
    video_path: str,
    first_frame: np.ndarray,
    init_bbox: Optional[Tuple[int, int, int, int]],
    get_fallback_bbox: Callable,
) -> List[Dict]:
    """Initialize KCF tracking."""
    if init_bbox is None:
        init_bbox = auto_init_bbox(first_frame)
        if init_bbox is None:
            init_bbox = get_fallback_bbox()

    try:
        # Try contrib KCF if available, else use template matching
        tracker = getattr(cv2, 'TrackerKCF_create', None)
        if tracker is not None:
            # Validate tracker works
            tr = tracker()
            tr.init(first_frame, init_bbox)
        # Use template matching for full sequence (deterministic)
        return tm_track(video_path, init_bbox)
    except Exception:
        return tm_track(video_path, init_bbox)


def _initialize_yolo_tracking(
    video_path: str,
    first_frame: np.ndarray,
    cls_name: str,
    select_track_id: Optional[int],
    init_bbox: Optional[Tuple[int, int, int, int]],
    get_fallback_bbox: Callable,
) -> List[Dict]:
    """Initialize YOLO + ByteTrack tracking."""
    traj_2d = yolo_bytetrack_traj(
        video_path, cls_name=cls_name, conf=0.25, select_track_id=select_track_id
    )

    if not traj_2d:
        # Fallback to KCF/template matching
        if init_bbox is None:
            init_bbox = auto_init_bbox(first_frame)
            if init_bbox is None:
                raise RuntimeError("Detection/Tracking failed and no bbox available")
        traj_2d = tm_track(video_path, init_bbox)

    return traj_2d


def _initialize_sam2_tracking(
    video_path: str,
    first_frame: np.ndarray,
    init_bbox: Optional[Tuple[int, int, int, int]],
    cls_name: str,
    select_track_id: Optional[int],
    sample_stride: int,
    sam2_model_id: Optional[str],
    sam2_cfg: Optional[str],
    sam2_ckpt: Optional[str],
    get_fallback_bbox: Callable,
) -> List[Dict]:
    """Initialize SAM2 tracking."""
    # Get initial bbox if not provided
    if init_bbox is None:
        tmp = yolo_bytetrack_traj(
            video_path, cls_name=cls_name, conf=0.25, select_track_id=select_track_id
        )
        if tmp:
            rec0 = tmp[0]
            init_bbox = (
                int(rec0["cx"] - rec0["w"] * 0.5),
                int(rec0["cy"] - rec0["h"] * 0.5),
                int(rec0["w"]),
                int(rec0["h"])
            )
        else:
            init_bbox = get_fallback_bbox()

    try:
        from .sam2_traj import sam2_video_traj
        return sam2_video_traj(
            video_path, init_bbox,
            model_id=sam2_model_id,
            config_path=sam2_cfg,
            checkpoint_path=sam2_ckpt,
            stride=sample_stride
        )
    except Exception:
        # Fallback to YOLO/template matching
        traj_2d = yolo_bytetrack_traj(
            video_path, cls_name=cls_name, conf=0.25, select_track_id=select_track_id
        )
        if not traj_2d:
            traj_2d = tm_track(video_path, init_bbox)
        return traj_2d


# ============================================================================
# Depth Estimation
# ============================================================================

def initialize_depth_backend(
    depth_backend: str,
    use_midas: bool = True,
    depth_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[Optional[Callable], Optional[tuple], Optional[object]]:
    """
    Initialize depth estimation backend.

    Args:
        depth_backend: Backend choice ('auto', 'midas', 'depth_anything_v2')
        use_midas: Whether to use MiDaS
        depth_fn: Custom depth function

    Returns:
        Tuple of (depth_fn, midas_bundle, depth_anything)
    """
    if depth_fn is not None:
        # User provided custom depth function
        return depth_fn, None, None

    # Try to load depth backends
    midas_bundle = None
    depth_anything = None

    # Try Depth Anything V2
    if depth_backend in ("auto", "depth_anything_v2"):
        try:
            from .depth_anything_adapter import build_depth_predictor
            depth_fn = build_depth_predictor(
                backend=depth_backend,
                model_size="small"
            )
            return depth_fn, None, None
        except Exception:
            # Continue to MiDaS fallback
            pass

    # Load MiDaS
    if depth_backend in ("auto", "midas") and use_midas:
        try:
            device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
            midas_bundle = _load_midas(device)
        except Exception:
            midas_bundle = None

    return None, midas_bundle, None


def estimate_depth_at_bbox(
    frame: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    depth_fn: Optional[Callable],
    midas_bundle: Optional[tuple],
) -> float:
    """
    Estimate relative depth at bounding box center.

    Args:
        frame: Input frame (BGR)
        cx, cy: Center coordinates
        w, h: Bounding box size
        depth_fn: Custom depth function
        midas_bundle: MiDaS model bundle

    Returns:
        Relative depth value in [0, 1]
    """
    H, W = frame.shape[:2]

    # Use custom depth function if provided
    if depth_fn is not None:
        try:
            dmap = depth_fn(frame)
            dmap = dmap.astype(np.float32)
            dmap -= dmap.min()
            if dmap.max() > 1e-8:
                dmap /= dmap.max()
            return _extract_depth_from_bbox(dmap, cx, cy, w, h, W, H)
        except Exception:
            return 0.5

    # Use MiDaS if available
    if midas_bundle is not None:
        dmap = estimate_depth(frame, midas_bundle)
        return _extract_depth_from_bbox(dmap, cx, cy, w, h, W, H)

    # Default fallback
    return 0.5


def _extract_depth_from_bbox(
    dmap: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    W: int,
    H: int,
) -> float:
    """Extract median depth from bounding box region."""
    x0 = max(0, int(cx - w * 0.25))
    y0 = max(0, int(cy - h * 0.25))
    x1 = min(W, int(cx + w * 0.25))
    y1 = min(H, int(cy + h * 0.25))

    box = dmap[y0:y1, x0:x1]
    if box.size > 0:
        return float(np.median(box))
    else:
        return float(dmap[min(H-1, int(cy)), min(W-1, int(cx))])


# ============================================================================
# Center Refinement
# ============================================================================

def refine_object_center(
    frame: np.ndarray,
    rec: Dict,
    refine_center: bool,
    refine_center_method: str,
    sam2_mask_fn: Optional[Callable],
) -> Tuple[float, float]:
    """
    Refine object center using GrabCut or SAM2.

    Args:
        frame: Input frame (BGR)
        rec: Tracking record {"cx", "cy", "w", "h"}
        refine_center: Whether to refine
        refine_center_method: Method ('grabcut', 'sam2')
        sam2_mask_fn: SAM2 mask function

    Returns:
        Tuple of (refined_cx, refined_cy)
    """
    cx, cy = rec["cx"], rec["cy"]

    if not refine_center:
        return cx, cy

    if refine_center_method == "grabcut":
        bbox = (
            rec["cx"] - rec["w"] * 0.5,
            rec["cy"] - rec["h"] * 0.5,
            rec["w"],
            rec["h"]
        )
        return refine_center_grabcut(frame, bbox)

    elif refine_center_method == "sam2":
        if sam2_mask_fn is None:
            raise RuntimeError(
                "refine_center_method='sam2' requires sam2_mask_fn(frame_bgr, bbox)->mask"
            )
        bbox = (
            rec["cx"] - rec["w"] * 0.5,
            rec["cy"] - rec["h"] * 0.5,
            rec["w"],
            rec["h"]
        )
        mask = sam2_mask_fn(frame, bbox)
        ys, xs = np.nonzero(mask.astype(np.uint8))
        if len(xs) > 0:
            return float(xs.mean()), float(ys.mean())
        return cx, cy

    else:
        raise ValueError(f"Unknown refine_center_method: {refine_center_method}")


# ============================================================================
# 3D Position Computation
# ============================================================================

def compute_3d_position(
    cx: float,
    cy: float,
    depth_rel: float,
    K: CameraIntrinsics,
    depth_scale_m: Tuple[float, float],
) -> Tuple[float, float, float, float, float, float]:
    """
    Compute 3D position from 2D center and depth.

    Args:
        cx, cy: Center coordinates in pixels
        depth_rel: Relative depth [0, 1]
        K: Camera intrinsics
        depth_scale_m: (near, far) metric depth range

    Returns:
        Tuple of (az, el, dist_m, x, y, z)
    """
    # Convert pixel to ray
    ray = pixel_to_ray(cx, cy, K)
    az, el = ray_to_angles(ray)

    # Map relative depth to metric distance
    near, far = depth_scale_m
    dist_m = near + (1.0 - depth_rel) * (far - near)

    # Compute 3D position
    pos = ray * dist_m
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

    return float(az), float(el), float(dist_m), x, y, z


# ============================================================================
# Frame Processing
# ============================================================================

def process_trajectory_frames(
    video_path: str,
    traj_2d: List[Dict],
    K: CameraIntrinsics,
    sample_stride: int,
    depth_fn: Optional[Callable],
    midas_bundle: Optional[tuple],
    depth_scale_m: Tuple[float, float],
    refine_center: bool,
    refine_center_method: str,
    sam2_mask_fn: Optional[Callable],
) -> List[Dict]:
    """
    Process video frames to compute 3D trajectory.

    Args:
        video_path: Path to input video
        traj_2d: 2D tracking trajectory
        K: Camera intrinsics
        sample_stride: Frame sampling stride
        depth_fn: Depth estimation function
        midas_bundle: MiDaS model bundle
        depth_scale_m: Depth scale range
        refine_center: Whether to refine centers
        refine_center_method: Refinement method
        sam2_mask_fn: SAM2 mask function

    Returns:
        List of 3D trajectory frames
    """
    result_frames = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fidx = 0
    i2d = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Skip frames based on stride
        if (fidx % sample_stride) != 0:
            fidx += 1
            continue

        # Check if we have tracking data for this frame
        if i2d >= len(traj_2d):
            break

        rec = traj_2d[i2d]
        if rec["frame"] < fidx:
            i2d += 1
            continue

        # Refine center if needed
        cx, cy = refine_object_center(
            frame, rec, refine_center, refine_center_method, sam2_mask_fn
        )

        # Estimate depth
        depth_rel = estimate_depth_at_bbox(
            frame, cx, cy, rec["w"], rec["h"], depth_fn, midas_bundle
        )

        # Compute 3D position
        az, el, dist_m, x, y, z = compute_3d_position(
            cx, cy, depth_rel, K, depth_scale_m
        )

        result_frames.append({
            "frame": int(fidx),
            "az": az,
            "el": el,
            "dist_m": dist_m,
            "x": x,
            "y": y,
            "z": z,
        })

        fidx += 1

    cap.release()
    return result_frames


# ============================================================================
# Post-Processing
# ============================================================================

def smooth_trajectory(
    frames: List[Dict],
    smooth_alpha: float = 0.2,
) -> List[Dict]:
    """
    Smooth trajectory angles using exponential moving average.

    Args:
        frames: List of trajectory frames
        smooth_alpha: Smoothing factor (0 = no smoothing, 1 = no memory)

    Returns:
        Smoothed trajectory frames
    """
    if not frames:
        return frames

    # Extract angles
    azs = np.array([r["az"] for r in frames], dtype=np.float32)
    els = np.array([r["el"] for r in frames], dtype=np.float32)

    # Smooth
    azs_s = smooth_series(azs, alpha=smooth_alpha)
    els_s = smooth_series(els, alpha=smooth_alpha)

    # Update frames
    for i, r in enumerate(frames):
        r["az"] = float(azs_s[i])
        r["el"] = float(els_s[i])

    return frames


# ============================================================================
# Main Refactored Function
# ============================================================================

def compute_trajectory_3d_refactored(
    video_path: str,
    init_bbox: Optional[Tuple[int, int, int, int]] = None,
    fov_deg: float = 60.0,
    depth_scale_m: Tuple[float, float] = (1.0, 3.0),
    use_midas: bool = True,
    sample_stride: int = 1,
    method: str = "yolo",
    cls_name: str = "person",
    refine_center: bool = True,
    refine_center_method: str = "grabcut",
    depth_backend: str = "auto",
    sam2_mask_fn: Optional[Callable[[np.ndarray, Tuple[float, float, float, float]], np.ndarray]] = None,
    depth_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    fallback_center_if_no_bbox: bool = False,
    select_track_id: Optional[int] = None,
    smooth_alpha: float = 0.2,
    sam2_model_id: Optional[str] = None,
    sam2_cfg: Optional[str] = None,
    sam2_ckpt: Optional[str] = None,
) -> Dict:
    """
    Track a single object and estimate a time-sequenced 3D trajectory.

    This is a refactored version of compute_trajectory_3d with the same interface
    but better internal structure for maintainability and testing.

    Args:
        video_path: Path to input video
        init_bbox: Initial bounding box (x, y, w, h)
        fov_deg: Camera horizontal FOV in degrees
        depth_scale_m: (near, far) metric depth range in meters
        use_midas: Whether to use MiDaS for depth estimation
        sample_stride: Process every Nth frame
        method: Tracking method ('yolo', 'kcf', 'sam2')
        cls_name: Object class name for YOLO
        refine_center: Whether to refine object centers
        refine_center_method: Center refinement method ('grabcut', 'sam2')
        depth_backend: Depth estimation backend ('auto', 'midas', 'depth_anything_v2')
        sam2_mask_fn: SAM2 mask function for center refinement
        depth_fn: Custom depth estimation function
        fallback_center_if_no_bbox: Use center bbox if detection fails
        select_track_id: Specific track ID to select (YOLO)
        smooth_alpha: Smoothing factor for angles
        sam2_model_id: SAM2 model identifier
        sam2_cfg: SAM2 config path
        sam2_ckpt: SAM2 checkpoint path

    Returns:
        Dict with:
            - intrinsics: {"width": int, "height": int, "fov_deg": float}
            - frames: List[{"frame": int, "az": float, "el": float, "dist_m": float, "x": float, "y": float, "z": float}]
    """
    # 1. Get video dimensions and create camera intrinsics
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    K = CameraIntrinsics(width=W, height=H, fov_deg=fov_deg)

    # 2. Initialize tracking
    traj_2d = initialize_tracking(
        video_path=video_path,
        method=method,
        init_bbox=init_bbox,
        cls_name=cls_name,
        select_track_id=select_track_id,
        fallback_center_if_no_bbox=fallback_center_if_no_bbox,
        sample_stride=sample_stride,
        sam2_model_id=sam2_model_id,
        sam2_cfg=sam2_cfg,
        sam2_ckpt=sam2_ckpt,
    )

    # 3. Initialize depth estimation
    depth_fn_init, midas_bundle, _ = initialize_depth_backend(
        depth_backend=depth_backend,
        use_midas=use_midas,
        depth_fn=depth_fn,
    )

    # Use initialized depth function if custom one not provided
    if depth_fn is None:
        depth_fn = depth_fn_init

    # 4. Process frames to compute 3D trajectory
    result_frames = process_trajectory_frames(
        video_path=video_path,
        traj_2d=traj_2d,
        K=K,
        sample_stride=sample_stride,
        depth_fn=depth_fn,
        midas_bundle=midas_bundle,
        depth_scale_m=depth_scale_m,
        refine_center=refine_center,
        refine_center_method=refine_center_method,
        sam2_mask_fn=sam2_mask_fn,
    )

    # 5. Smooth trajectory
    result_frames = smooth_trajectory(result_frames, smooth_alpha=smooth_alpha)

    # 6. Return result
    return {
        "intrinsics": {"width": W, "height": H, "fov_deg": float(fov_deg)},
        "frames": result_frames,
    }


__all__ = [
    # Main function
    'compute_trajectory_3d_refactored',
    # Tracking
    'initialize_tracking',
    # Depth estimation
    'initialize_depth_backend',
    'estimate_depth_at_bbox',
    # Center refinement
    'refine_object_center',
    # 3D computation
    'compute_3d_position',
    # Frame processing
    'process_trajectory_frames',
    # Post-processing
    'smooth_trajectory',
]
