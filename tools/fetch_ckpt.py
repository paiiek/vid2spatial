import argparse
import os
import sys
from urllib.request import urlretrieve


SAM_URLS = {
    "sam_vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "sam_vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "sam_vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}

YOLO_URLS = {
    # Ultralytics auto-downloads, but provided for completeness
    "yolo11n": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=False, choices=list(SAM_URLS.keys()) + list(YOLO_URLS.keys()))
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--url", required=False, help="direct URL to download (overrides --name)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.url:
        url = args.url
        fname = os.path.join(args.out, os.path.basename(url))
    else:
        if not args.name:
            raise SystemExit("either --url or --name must be provided")
        if args.name in SAM_URLS:
            url = SAM_URLS[args.name]
            fname = os.path.join(args.out, os.path.basename(url))
        else:
            url = YOLO_URLS[args.name]
            fname = os.path.join(args.out, os.path.basename(url))

    print(f"downloading {args.name} from {url}\n -> {fname}")
    urlretrieve(url, fname)
    print("done")


if __name__ == "__main__":
    main()
