import numpy as np
import torch
import torch.nn as nn

from .env import NQ

NX, NU = 2 * NQ, NQ


class ResidualMLP(nn.Module):
    def __init__(self, hidden=(64, 64)):
        super().__init__()
        d = [NX + NU] + list(hidden)
        L = []
        for a, b in zip(d[:-1], d[1:]):
            L += [nn.Linear(a, b), nn.SiLU()]
        L += [nn.Linear(d[-1], NX)]
        self.net = nn.Sequential(*L)
        self.register_buffer("mu", torch.zeros(NX + NU))
        self.register_buffer("sd", torch.ones(NX + NU))
        self.register_buffer("osd", torch.ones(NX))

    def set_norm(self, X, U, R):
        z = np.concatenate([X, U], 1)
        self.mu.copy_(torch.tensor(z.mean(0), dtype=torch.float32))
        self.sd.copy_(torch.tensor(z.std(0) + 1e-6, dtype=torch.float32))
        self.osd.copy_(torch.tensor(R.std(0) + 1e-8, dtype=torch.float32))

    def forward(self, x, u):
        return self.net((torch.cat([x, u], -1) - self.mu) / self.sd) * self.osd


def train(d, hidden=(64, 64), epochs=500, bs=256, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    X, U, R = (d["X"].astype(np.float32), d["U"].astype(np.float32),
               d["R"].astype(np.float32))
    m = ResidualMLP(hidden)
    m.set_norm(X, U, R)
    opt = torch.optim.Adam(m.parameters(), lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tX, tU, tR = map(torch.tensor, (X, U, R))
    n = len(tX)
    for _ in range(epochs):
        p = torch.randperm(n)
        for i in range(0, n, bs):
            j = p[i:i + bs]
            loss = ((m(tX[j], tU[j]) - tR[j]) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return m


@torch.no_grad()
def mse(m, d):
    e = m(torch.tensor(d["X"].astype(np.float32)),
          torch.tensor(d["U"].astype(np.float32))).numpy() - d["R"]
    return float((e ** 2).sum(1).mean())
