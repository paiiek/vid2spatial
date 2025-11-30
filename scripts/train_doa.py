#!/usr/bin/env python3
import argparse
import math
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .datasets_tau import TAUFOALoader, TAUFOAIterable


def foa_to_stft_feats(y: np.ndarray, n_fft: int = 512, hop: int = 160) -> torch.Tensor:
    """Compute magnitude STFT for 4-ch FOA and stack as [C,F,T]."""
    import librosa
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    chs = []
    for c in range(y.shape[1]):
        S = np.abs(librosa.stft(y[:, c].astype(np.float32), n_fft=n_fft, hop_length=hop))
        chs.append(S)
    X = np.stack(chs, axis=0)  # [4,F,T]
    X = np.log1p(X).astype(np.float32)
    return torch.from_numpy(X)


class SmallCRNN(nn.Module):
    def __init__(self, n_ch: int = 4, n_fft: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_ch, 16, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(input_size=32, hidden_size=128, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,F,T]
        z = self.conv(x)  # [B,32,F',T']
        # average across frequency, keep time for sequence modeling
        z = z.mean(dim=2)  # [B,32,T']
        z = z.transpose(1, 2).contiguous()  # [B,T',32]
        o, _ = self.gru(z)
        h = o[:, -1, :]
        out = self.head(h)
        # outputs: cos(az), sin(az), cos(el), sin(el)
        return out


def angular_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cosine loss on angle pairs. pred/target: [B,4] = [ca,sa,ce,se].
    Normalize each pair and compute 1 - cos(delta) for az and el, then mean.
    """
    assert pred.shape[-1] == 4 and target.shape[-1] == 4
    def norm_pair(t: torch.Tensor) -> torch.Tensor:
        a = t[:, 0:2]
        e = t[:, 2:4]
        a = a / (a.norm(dim=1, keepdim=True) + 1e-8)
        e = e / (e.norm(dim=1, keepdim=True) + 1e-8)
        return torch.cat([a, e], dim=1)
    p = norm_pair(pred)
    g = norm_pair(target)
    ca, sa, ce, se = p.t()
    cga, sga, cge, sge = g.t()
    cos_daz = ca * cga + sa * sga
    cos_del = ce * cge + se * sge
    loss = (1.0 - cos_daz).mean() + (1.0 - cos_del).mean()
    return loss * 0.5


def ang_to_vec(az: float, el: float) -> Tuple[float, float, float, float]:
    return math.cos(az), math.sin(az), math.cos(el), math.sin(el)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--project', type=str, default='vid2spatial')
    ap.add_argument('--name', type=str, default='doa_crnn_tau')
    ap.add_argument('--save_dir', type=str, default='mmhoa/vid2spatial/out/doa_ckpt')
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--clip_grad', type=float, default=1.0)
    ap.add_argument('--loss_mse_w', type=float, default=0.2)
    ap.add_argument('--loss_cos_w', type=float, default=0.8)
    ap.add_argument('--aug_snr_db', type=float, default=20.0, help='add noise up to ±SNR dB (uniform)')
    ap.add_argument('--aug_yaw_deg', type=float, default=0.0, help='optional angle jitter in degrees (label)')
    # optional TAU DOA eval per-epoch
    ap.add_argument('--tau_manifest', type=str, default=None)
    ap.add_argument('--tau_root', type=str, default=None)
    ap.add_argument('--tau_limit', type=int, default=200)
    args = ap.parse_args()

    # wandb init (offline if no API key)
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
        # define metrics and heartbeat so the run isn't empty
        wandb.define_metric('global_step')
        wandb.define_metric('train/*', step='global_step')
        wandb.define_metric('val/*', step='global_step')
        param_count = sum(p.numel() for p in SmallCRNN().parameters())
        wandb.log({'status': 'started', 'model/params': param_count, 'global_step': 0}, step=0)
    except Exception:
        use_wandb = False

    # Use iterable loader for large scale
    ds_iter = TAUFOAIterable(args.manifest, args.root, frame_hop_s=0.1, win_s=1.0,
                             limit_files=args.limit)
    from torch.utils.data import DataLoader
    # custom collate to keep numpy arrays intact
    def _collate(samples):
        ys = [s['y'] for s in samples]
        az = [float(s['az']) for s in samples]
        el = [float(s['el']) for s in samples]
        return {'y': ys, 'az': az, 'el': el}

    loader = DataLoader(ds_iter, batch_size=args.batch, num_workers=args.num_workers,
                        pin_memory=True, collate_fn=_collate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmallCRNN().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_mse = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    os.makedirs(args.save_dir, exist_ok=True)

    def batch_iter(bs: int):
        Xs, Ys = [], []
        for rec in ds:
            X = foa_to_stft_feats(rec['y'])  # [4,F,T]
            Xs.append(X)
            Ys.append(torch.tensor(ang_to_vec(rec['az'], rec['el']), dtype=torch.float32))
            if len(Xs) == bs:
                Xb = torch.stack(Xs, dim=0)  # [B,4,F,T]
                Yb = torch.stack(Ys, dim=0)
                Xs, Ys = [], []
                yield Xb, Yb
        if Xs:
            Xb = torch.stack(Xs, dim=0)
            Yb = torch.stack(Ys, dim=0)
            yield Xb, Yb

    # simple split indices for val
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    global_step = 0
    for ep in range(args.epochs):
        model.train()
        ep_loss = 0.0
        nb = 0
        for batch in loader:
            # batch['y'] is a list of numpy arrays [T,4]
            X = torch.stack([foa_to_stft_feats(y) for y in batch['y']], dim=0)  # [B,4,F,T]
            # optional SNR augmentation on features (simulate noise)
            if args.aug_snr_db and args.aug_snr_db > 0:
                # add channel-wise Gaussian noise on STFT magnitude proxy
                noise_level = (torch.rand(()) * 2 - 1.0).item() * float(args.aug_snr_db)
                snr_lin = 10.0 ** (-abs(noise_level) / 20.0)
                X = X + snr_lin * torch.randn_like(X)
            # targets with optional yaw jitter
            az_list = list(batch['az'])
            el_list = list(batch['el'])
            if args.aug_yaw_deg and float(args.aug_yaw_deg) != 0.0:
                import math as _m
                jitter = (torch.rand(len(az_list)) * 2 - 1.0) * _m.radians(float(args.aug_yaw_deg))
                az_list = [float(a + da) for a, da in zip(az_list, jitter)]
            Y = torch.stack([torch.tensor(ang_to_vec(a, e), dtype=torch.float32)
                             for a, e in zip(az_list, el_list)], dim=0)
            xb = X.to(device, non_blocking=True)
            yb = Y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                yhat = model(xb)
                loss = float(args.loss_mse_w) * loss_mse(yhat, yb) + float(args.loss_cos_w) * angular_cosine_loss(yhat, yb)
            scaler.scale(loss).backward()
            if args.clip_grad and args.clip_grad > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(opt)
            scaler.update()
            ep_loss += float(loss.item())
            nb += 1
            global_step += 1
            if use_wandb and (global_step % 10 == 0):
                # quick ang MAE proxy
                with torch.no_grad():
                    ca, sa, ce, se = [t.cpu().numpy() for t in yhat.t()]
                    az_hat = np.arctan2(sa, ca)
                    el_hat = np.arctan2(se, ce)
                    ca_t, sa_t, ce_t, se_t = [t.cpu().numpy() for t in yb.t()]
                    az_t = np.arctan2(sa_t, ca_t)
                    el_t = np.arctan2(se_t, ce_t)
                    mae = float(np.mean(np.degrees(np.abs(az_hat - az_t))))
                wandb.log({'train/loss': float(loss.item()), 'train/az_mae_deg': mae}, step=global_step)
        avg = ep_loss / max(1, nb)
        print(f'[ep {ep+1}] loss={avg:.4f}')
        # quick val pass
        model.eval()
        with torch.no_grad():
            val_loss = float('nan'); val_mae = float('nan')
        if use_wandb:
            wandb.log({'epoch/loss': avg, 'val/loss': val_loss, 'val/az_mae_deg': val_mae}, step=global_step)
        # save checkpoint
        ckpt_path = os.path.join(args.save_dir, f'ep{ep+1}.pt')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': ep+1}, ckpt_path)
        print(f'[ckpt] saved {ckpt_path}')
        scheduler.step()
        # optional TAU DOA report
        if args.tau_manifest and args.tau_root:
            try:
                import json, subprocess, tempfile
                with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
                    outp = tf.name
                cmd = [
                    'python3','-m','mmhoa.vid2spatial.tools.tau_eval_doa',
                    '--manifest', args.tau_manifest,
                    '--root', args.tau_root,
                    '--limit', str(int(args.tau_limit)),
                    '--out', outp
                ]
                subprocess.run(cmd, check=True)
                summ = json.load(open(outp)).get('summary', {})
                print('[tau_eval]', summ)
                if use_wandb:
                    wandb.log({
                        'tau/count_files': summ.get('count_files'),
                        'tau/mae_mean_deg': summ.get('mae_mean_deg'),
                        'tau/mae_median_deg': summ.get('mae_median_deg'),
                        'tau/mae_p75_deg': summ.get('mae_p75_deg'),
                    }, step=global_step)
            except Exception as e:
                print('[tau_eval] failed:', e)

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
