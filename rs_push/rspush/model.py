"""One-step object-motion model (recursive planner backbone for Gate A).

Inputs are translation-invariant features; the eef propagates kinematically
(commanded velocity tracks well; verified on the arm interface), the network
predicts only the OBJECT pose change:
    feat = [eef_xy - obj_xy, sin(yaw), cos(yaw), u_xy]      (6)
    out  = [d_obj_x, d_obj_y, d_yaw]                        (3)
Angle enters via sin/cos and leaves as a small per-step delta -> no wrap issue.
"""
import numpy as np
import torch
import torch.nn as nn

from .env import DT


def features(eef_xy, obj, u):
    return np.concatenate([eef_xy - obj[..., :2],
                           np.stack([np.sin(obj[..., 2]), np.cos(obj[..., 2])], -1),
                           u], -1)


class OneStep(nn.Module):
    def __init__(self, hidden=(128, 128)):
        super().__init__()
        d = [6] + list(hidden)
        L = []
        for a, b in zip(d[:-1], d[1:]):
            L += [nn.Linear(a, b), nn.SiLU()]
        L += [nn.Linear(d[-1], 3)]
        self.net = nn.Sequential(*L)
        self.register_buffer("mu", torch.zeros(6))
        self.register_buffer("sd", torch.ones(6))
        self.register_buffer("osd", torch.ones(3))

    def forward(self, z):
        return self.net((z - self.mu) / self.sd) * self.osd


def train(F, Y, hidden=(128, 128), epochs=400, bs=512, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    m = OneStep(hidden)
    m.mu.copy_(torch.tensor(F.mean(0), dtype=torch.float32))
    m.sd.copy_(torch.tensor(F.std(0) + 1e-8, dtype=torch.float32))
    m.osd.copy_(torch.tensor(Y.std(0) + 1e-8, dtype=torch.float32))
    tF = torch.tensor(F, dtype=torch.float32)
    tY = torch.tensor(Y, dtype=torch.float32)
    opt = torch.optim.Adam(m.parameters(), lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        p = torch.randperm(len(tF))
        for i in range(0, len(tF), bs):
            j = p[i:i + bs]
            loss = ((m(tF[j]) - tY[j]) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    m.eval()
    return m


@torch.no_grad()
def rollout(model, eef_xy, obj, U):
    """Vectorised recursive rollout. U: (N, H, 2). Returns obj trajectory (N,H,3)."""
    U = np.asarray(U, dtype=np.float32)
    N, H, _ = U.shape
    e = np.repeat(eef_xy[None], N, 0).astype(np.float32)
    s = np.repeat(obj[None], N, 0).astype(np.float32)
    out = np.zeros((N, H, 3), dtype=np.float32)
    for k in range(H):
        z = torch.tensor(features(e, s, U[:, k]))
        d = model(z).numpy()
        s = s + d
        e = e + U[:, k] * DT
        out[:, k] = s
    return out
