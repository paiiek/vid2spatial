import argparse
import json
from mmhoa.vid2spatial.vision import compute_trajectory_3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--method', type=str, default='yolo', choices=['yolo','kcf','sam2'])
    ap.add_argument('--cls', type=str, default='person')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--sam2_model_id', type=str, default='facebook/sam2.1-hiera-base-plus')
    ap.add_argument('--sam2_cfg', type=str, default=None)
    ap.add_argument('--sam2_ckpt', type=str, default=None)
    ap.add_argument('--fallback_center_box', action='store_true')
    args = ap.parse_args()

    traj = compute_trajectory_3d(
        args.video,
        init_bbox=None,
        fov_deg=60.0,
        sample_stride=args.stride,
        method=args.method,
        cls_name=args.cls,
        refine_center=False,
        refine_center_method='grabcut',
        depth_backend='none',
        sam2_model_id=args.sam2_model_id,
        sam2_cfg=args.sam2_cfg,
        sam2_ckpt=args.sam2_ckpt,
        fallback_center_if_no_bbox=bool(args.fallback_center_box),
    )
    with open(args.out, 'w') as f:
        json.dump(traj, f, indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()

