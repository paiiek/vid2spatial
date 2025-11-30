"""Dataset-driven training for TrajIRMapper.

Usage:
  python -m mmhoa.vid2spatial.train_dataset \
    --manifest path/to/train.jsonl \
    --epochs 2 --batch 4
"""
import argparse
import math
import numpy as np
import torch
import torch.nn as nn

from .train import TrajIRMapper, rir_to_feats
from .dataset import TrajAudioDataset


def batchify(batch):
    # pad to max length in batch
    T = max(item['audio'].shape[0] for item in batch)
    B = len(batch)
    x = np.zeros((B, T), np.float32)
    az = np.zeros((B, T), np.float32)
    el = np.zeros((B, T), np.float32)
    for i, item in enumerate(batch):
        t = item['audio'].shape[0]
        x[i, :t] = item['audio']
        az[i, :t] = item['az']
        el[i, :t] = item['el']
    return x, az, el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch', type=int, default=4)
    args = ap.parse_args()

    ds = TrajAudioDataset(args.manifest, sr=16000)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TrajIRMapper().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # dummy IR features: zeros (dataset-independent). Extend when IR available
    ir_feat = torch.zeros((args.batch, 1, 16), device=device)

    for ep in range(args.epochs):
        model.train()
        total = 0.0
        n = 0
        # simple batching without shuffling
        for i in range(0, len(ds), args.batch):
            batch = [ds[j] for j in range(i, min(len(ds), i+args.batch))]
            x_np, az_np, el_np = batchify(batch)
            B, T = x_np.shape
            # features: [az, el, dist(=1), ir_feat(zeros)]
            az = torch.from_numpy(az_np).to(device).unsqueeze(-1)
            el = torch.from_numpy(el_np).to(device).unsqueeze(-1)
            dist = torch.ones((B, T, 1), device=device)
            ir = ir_feat[:B].repeat(1, T, 1)
            feats = torch.cat([az, el, dist, ir], dim=-1)
            # targets: direction unit vector
            x_dir = torch.cos(az) * torch.cos(el)
            y_dir = torch.sin(az) * torch.cos(el)
            z_dir = torch.sin(el)
            y_true = torch.cat([x_dir, y_dir, z_dir], dim=-1)
            y_pred = model(feats)
            loss = loss_fn(y_pred, y_true)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()); n += 1
        print(f"epoch {ep+1}/{args.epochs} loss={total/max(1,n):.4f}")


if __name__ == '__main__':
    main()

