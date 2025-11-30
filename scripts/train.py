import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class TrajIRMapper(nn.Module):
    """Neural mapper: trajectory + IR features → FOA directional gains (X,Y,Z).

    Input per time step: [az, el, dist, ir_feat]
    Output per time step: [gx, gy, gz] with SN3D scaling implicit in render.
    """

    def __init__(self, in_dim: int = 1 + 1 + 1 + 16, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True, bidirectional=False)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3), nn.Tanh()
        )

    def forward(self, x):
        h, _ = self.rnn(x)
        out = self.head(h)
        return out  # [B,T,3]


def rir_to_feats(rir: np.ndarray, n_feats: int = 16) -> np.ndarray:
    """Simple IR featurization: log-mel of |STFT| averaged over time (deterministic)."""
    import librosa
    S = np.abs(librosa.stft(rir.astype(np.float32), n_fft=512, hop_length=128))
    mel = librosa.feature.melspectrogram(S=S, sr=16000, n_mels=n_feats)  # S as magnitude
    f = np.log1p(mel).mean(axis=1)
    return f.astype(np.float32)


@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 10
    batch: int = 8
    seq_len: int = 400


def demo_train(cfg: TrainConfig = TrainConfig()):
    """Minimal self-contained trainer with synthetic data to verify pipeline.
    This is a placeholder to validate model wiring without dataset I/O."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TrajIRMapper().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    for ep in range(cfg.epochs):
        model.train()
        # synthetic batch
        B, T = cfg.batch, cfg.seq_len
        az = torch.rand(B, T, 1, device=device) * 2 * math.pi - math.pi
        el = (torch.rand(B, T, 1, device=device) - 0.5) * (math.pi / 2)
        dist = 1.0 + torch.rand(B, T, 1, device=device) * 2.0
        ir = torch.randn(B, 16, device=device)
        ir = ir[:, None, :].repeat(1, T, 1)
        x = torch.cat([az.sin()*0 + az, el, dist, ir], dim=-1)  # [B,T,19]

        # target gains: analytic mapping for supervision (proxy)
        # unit direction vector
        x_dir = torch.cos(az) * torch.cos(el)
        y_dir = torch.sin(az) * torch.cos(el)
        z_dir = torch.sin(el)
        y_true = torch.cat([x_dir, y_dir, z_dir], dim=-1)  # [B,T,3]

        y_pred = model(x)
        loss = loss_fn(y_pred, y_true)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 1 == 0:
            print(f"epoch {ep+1}/{cfg.epochs} loss={loss.item():.4f}")


__all__ = [
    "TrajIRMapper",
    "rir_to_feats",
    "TrainConfig",
    "demo_train",
]

