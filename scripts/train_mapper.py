#!/usr/bin/env python3
import argparse
import os
import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .datasets_tau import TAUFOALoader


def mono_and_angles(rec) -> Tuple[np.ndarray, np.ndarray]:
    y = rec['y'].astype(np.float32)  # [T,4] W,Y,Z,X
    mono = y[:, 0].copy()  # W channel as mono proxy
    az = float(rec['az']); el = float(rec['el'])
    return mono, np.array([math.cos(az), math.sin(az), math.cos(el), math.sin(el)], np.float32)


def to_feats(mono: np.ndarray, ang4: np.ndarray, n_fft: int = 512, hop: int = 160) -> torch.Tensor:
    import librosa
    S = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=hop))  # [F,T]
    S = np.log1p(S).astype(np.float32)
    # tile angles over time as simple conditioning
    T = S.shape[1]
    A = np.repeat(ang4[:, None], T, axis=1)  # [4,T]
    X = np.concatenate([S, A], axis=0)  # [F+4,T]
    return torch.from_numpy(X)


class MapperNet(nn.Module):
    def __init__(self, f_bins: int):
        super().__init__()
        # Input channels = 1 (spec) + 4 (ang cond) in frequency-like stacking → treat as 1x(F+4)xT
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(2, 2), padding=2), nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=1), nn.ReLU(True),
        )
        self.head = nn.Sequential(nn.Linear(32, 64), nn.ReLU(True), nn.Linear(64, 3))  # predict X,Y,Z gains

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,Fp,T]
        z = self.conv(x)
        z = z.mean(dim=(2, 3))  # [B,32]
        out = self.head(z)  # [B,3]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--limit', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--project', type=str, default='vid2spatial')
    ap.add_argument('--name', type=str, default='mapper_tau_smoke')
    args = ap.parse_args()

    use_wandb = True
    try:
        import wandb  # type: ignore
        if 'WANDB_API_KEY' not in os.environ:
            os.environ['WANDB_MODE'] = 'offline'
        wandb.init(project=args.project, name=args.name,
                   entity=os.environ.get('WANDB_ENTITY'), config=vars(args))
        try:
            print('wandb url:', wandb.run.url)
        except Exception:
            pass
        wandb.define_metric('global_step')
        wandb.define_metric('train/*', step='global_step')
        wandb.log({'status': 'started', 'global_step': 0}, step=0)
    except Exception:
        use_wandb = False

    ds = TAUFOALoader(args.manifest, args.root, limit=args.limit, frame_hop_s=0.1, win_s=1.0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # infer f_bins from a sample
    import librosa
    ex = ds[0]
    S = np.abs(librosa.stft(ex['y'][:, 0], n_fft=512, hop_length=160))
    f_bins = S.shape[0] + 4
    model = MapperNet(f_bins).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    def iterator(bs: int):
        Xs, Ys = [], []
        for rec in ds:
            mono, ang4 = mono_and_angles(rec)
            X = to_feats(mono, ang4)  # [F+4,T]
            target_xyz = torch.from_numpy(rec['y'][:, 1:4].mean(axis=0) if False else rec['y'][:1, 1:4].mean(axis=0))
            # practical simplification: use frame-averaged target gains per window
            tgt = torch.from_numpy(rec['y'][:, 1:4].astype(np.float32)).mean(dim=0) if isinstance(rec['y'], torch.Tensor) else torch.tensor(rec['y'][:, 1:4].mean(axis=0), dtype=torch.float32)
            # We regress to mean X/Y/Z over window to keep smoke test fast
            Xs.append(X.unsqueeze(0))  # [1,Fp,T]
            Ys.append(tgt)
            if len(Xs) == bs:
                Xb = torch.stack(Xs, dim=0)  # [B,1,Fp,T]
                Yb = torch.stack(Ys, dim=0)  # [B,3]
                Xs, Ys = [], []
                yield Xb, Yb
        if Xs:
            Xb = torch.stack(Xs, dim=0)
            Yb = torch.stack(Ys, dim=0)
            yield Xb, Yb

    global_step = 0
    for ep in range(args.epochs):
        model.train()
        ep_loss = 0.0
        nb = 0
        for xb, yb in iterator(args.batch):
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            yhat = model(xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
            global_step += 1
            if use_wandb and (global_step % 10 == 0):
                wandb.log({'train/loss': float(loss.item())}, step=global_step)
        avg = ep_loss / max(1, nb)
        print(f'[ep {ep+1}] loss={avg:.4f}')
        if use_wandb:
            wandb.log({'epoch/loss': avg}, step=global_step)

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
