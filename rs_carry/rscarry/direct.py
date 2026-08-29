"""Direct multi-step constraint-channel predictor.

One network maps (x_t, u_{t:t+K-1}) -> eef positions p_{t+1:t+K} in a single
forward pass. The constraint value g is then computed ANALYTICALLY from the
predicted positions, so the predictor is obstacle-agnostic: new obstacle, zero
retraining. Supervision comes from the recorded eef trajectory -- no FK needed
at training time.

This is the method under test: it bypasses the recursive state rollout whose
margin error compounds (measured: x5.6 from k=1 to k=10, driven by a residual
whose direction correlates 0.71 across steps -- the unobserved payload).
"""
import numpy as np
import torch
import torch.nn as nn

from .env import NQ

NX = 2 * NQ


class DirectHead(nn.Module):
    def __init__(self, K=15, hidden=(128, 128)):
        super().__init__()
        self.K = K
        d = [NX + K * NQ] + list(hidden)
        L = []
        for a, b in zip(d[:-1], d[1:]):
            L += [nn.Linear(a, b), nn.SiLU()]
        L += [nn.Linear(d[-1], K * 3)]
        self.net = nn.Sequential(*L)
        self.register_buffer("mu", torch.zeros(NX + K * NQ))
        self.register_buffer("sd", torch.ones(NX + K * NQ))
        self.register_buffer("p_mu", torch.zeros(3))
        self.register_buffer("p_sd", torch.ones(3))

    def set_norm(self, Z, P):
        self.mu.copy_(torch.tensor(Z.mean(0), dtype=torch.float32))
        self.sd.copy_(torch.tensor(Z.std(0) + 1e-6, dtype=torch.float32))
        self.p_mu.copy_(torch.tensor(P.reshape(-1, 3).mean(0), dtype=torch.float32))
        self.p_sd.copy_(torch.tensor(P.reshape(-1, 3).std(0) + 1e-6, dtype=torch.float32))

    def forward(self, z):
        out = self.net((z - self.mu) / self.sd).view(-1, self.K, 3)
        return out * self.p_sd + self.p_mu


def windows(trajs, K, stride=2):
    """(x_s, U_s..s+K-1) -> p_{s+1..s+K}, plus per-window obstacle + true margins."""
    Z, P, OBS = [], [], []
    for t in trajs:
        T = len(t["X"])
        for s in range(0, T - K, stride):
            Z.append(np.concatenate([t["X"][s], t["U"][s:s + K].ravel()]))
            P.append(t["eef"][s + 1:s + K + 1])
            OBS.append(t["p_obs"])
    return (np.array(Z, dtype=np.float32), np.array(P, dtype=np.float32),
            np.array(OBS, dtype=np.float32))


def train_direct(trajs, K=15, hidden=(128, 128), epochs=800, bs=256, lr=1e-3,
                 seed=0):
    torch.manual_seed(seed)
    Z, P, _ = windows(trajs, K)
    m = DirectHead(K, hidden)
    m.set_norm(Z, P)
    opt = torch.optim.Adam(m.parameters(), lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tZ = torch.tensor(Z); tP = torch.tensor(P)
    n = len(tZ)
    for _ in range(epochs):
        pm = torch.randperm(n)
        for i in range(0, n, bs):
            j = pm[i:i + bs]
            loss = ((m(tZ[j]) - tP[j]) ** 2).sum(-1).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return m
