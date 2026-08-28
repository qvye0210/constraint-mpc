"""Residual model, constraint-derived weight metric, and the training arms.

Arms, and what each one isolates:

    uniform   plain MSE. Baseline.
    mask      zero weight on decision-irrelevant output dimensions, UNIFORM on
              the rest. Isolates "stop spending capacity where it cannot matter"
              from "allocate by direction". On the planar testbed this arm alone
              explained the entire gain, which is why it must be here.
    prop      full propagated constraint-gradient metric.
              M = sum_k gamma^{k-1} c_k c_k^T,  c_k = (A^{k-1})^T grad g(x_{t+k}).
    random    same eigenvalue spectrum as prop, random orientation. Without it a
              positive result cannot be attributed to direction rather than to
              the loss merely being anisotropic.

Two implementation points that cost weeks on the planar testbed:

  * weights are normalised to mean trace 1, so arms share a loss SCALE and
    "better allocation" cannot be confused with "different effective step size";
  * rank protection is applied PER STATE BLOCK. A single global floor is uniform
    across dimensions while the structural weight is not, and it swamped the
    velocity block by ~9x -- precisely where the residual lives.
"""

import numpy as np
import torch
import torch.nn as nn

from .plant import NQ, NU, NX, MjParams, UR5ePlant, f_nominal


# ---------------------------------------------------------------- model
class ResidualMLP(nn.Module):
    def __init__(self, hidden=(64, 64), act=nn.SiLU):
        super().__init__()
        dims = [NX + NU] + list(hidden)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), act()]
        layers += [nn.Linear(dims[-1], NX)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("in_mu", torch.zeros(NX + NU))
        self.register_buffer("in_sd", torch.ones(NX + NU))
        self.register_buffer("out_sd", torch.ones(NX))

    def set_norm(self, X, U, R):
        z = np.concatenate([X, U], 1)
        self.in_mu.copy_(torch.tensor(z.mean(0), dtype=torch.float32))
        self.in_sd.copy_(torch.tensor(z.std(0) + 1e-6, dtype=torch.float32))
        self.out_sd.copy_(torch.tensor(R.std(0) + 1e-8, dtype=torch.float32))

    def forward(self, x, u):
        return self.net((torch.cat([x, u], -1) - self.in_mu) / self.in_sd) * self.out_sd


# ------------------------------------------------------- constraint geometry
def linearised_A(p=MjParams):
    A = np.eye(NX)
    A[:NQ, NQ:] = p.dt * np.eye(NQ)
    A[NQ:, NQ:] = (1.0 - p.dt / p.tau_nom) * np.eye(NQ)
    return A


def constraint_grad(plant, Q, p_obs):
    """grad_x g for g = r - ||p(q) - p_obs||, using MuJoCo's own Jacobian.

    The analytic DH table disagreed with the official model by ~1 mm in position
    and 1.2e-3 in the Jacobian, so the model's own kinematics is used instead.
    """
    p_obs = np.asarray(p_obs, dtype=float)
    G = np.zeros((len(Q), NX))
    for i, q in enumerate(Q):
        p = plant.tcp(q)
        J = plant.tcp_jacobian()
        d = p - p_obs
        n = d / (np.linalg.norm(d) + 1e-12)
        G[i, :NQ] = -n @ J
    return G


def build_metric(plant, X, p_obs, mode="prop", H=10, gamma=0.95,
                 eps_floor=0.05, clip_q=0.95, seed=0, params=MjParams):
    """Per-sample metric M, shape (N, NX, NX). Normalised to mean trace 1."""
    N = len(X)
    if mode == "uniform":
        return _finalise(np.tile(np.eye(NX), (N, 1, 1)), clip_q)

    if mode == "mask":
        # keep only dimensions the constraint can ever reach, uniformly
        G = constraint_grad(plant, X[:, :NQ], p_obs)
        A = linearised_A(params)
        reach = np.zeros(NX)
        Ak = np.eye(NX)
        for _ in range(H):
            reach += np.abs(G @ Ak).mean(0)
            Ak = Ak @ A
        keep = (reach > 1e-9 * max(reach.max(), 1e-12)).astype(float)
        M = np.tile(np.diag(keep), (N, 1, 1))
        return _finalise(M, clip_q, eps_floor, params)

    G = constraint_grad(plant, X[:, :NQ], p_obs)
    A = linearised_A(params)
    M = np.zeros((N, NX, NX))
    Ak = np.eye(NX)
    for k in range(1, H + 1):
        ck = G @ Ak
        M += (gamma ** (k - 1)) * ck[:, :, None] * ck[:, None, :]
        Ak = Ak @ A

    if mode == "random":
        w, _ = np.linalg.eigh(M)
        rng = np.random.default_rng(seed)
        Q_, _ = np.linalg.qr(rng.normal(size=(N, NX, NX)))
        M = Q_ @ (w[:, :, None] * np.transpose(Q_, (0, 2, 1)))
    elif mode != "prop":
        raise ValueError(mode)
    return _finalise(M, clip_q, eps_floor, params)


def _finalise(M, clip_q, eps_floor=0.0, params=MjParams):
    if eps_floor > 0:
        floor = np.zeros(NX)
        for lo, hi in ((0, NQ), (NQ, NX)):
            blk = np.trace(M[:, lo:hi, lo:hi], axis1=1, axis2=2).mean()
            floor[lo:hi] = eps_floor * blk / (hi - lo)
        M = M + np.diag(floor)
    tr = np.trace(M, axis1=1, axis2=2)
    cap = np.quantile(tr, clip_q)
    M = M * np.minimum(1.0, cap / np.maximum(tr, 1e-12))[:, None, None]
    return M / max(np.trace(M, axis1=1, axis2=2).mean(), 1e-12)


def weight_report(M):
    d = np.einsum("bii->bi", M)
    tot = d.sum(1).mean()
    return dict(w_q=float(d[:, :NQ].sum(1).mean() / tot),
                w_qd=float(d[:, NQ:].sum(1).mean() / tot),
                cond_qd=float(np.median(
                    np.linalg.eigvalsh(M[:, NQ:, NQ:])[:, -1]
                    / np.maximum(np.linalg.eigvalsh(M[:, NQ:, NQ:])[:, 0], 1e-20))))


# ------------------------------------------------------------------ training
def train(d, M, hidden=(64, 64), epochs=200, bs=256, lr=1e-3, seed=0,
          device="cpu"):
    torch.manual_seed(seed)
    X = d["X"].astype(np.float32); U = d["U"].astype(np.float32)
    R = d["R"].astype(np.float32)
    m = ResidualMLP(hidden).to(device)
    m.set_norm(X, U, R)
    opt = torch.optim.Adam(m.parameters(), lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tX = torch.tensor(X, device=device); tU = torch.tensor(U, device=device)
    tR = torch.tensor(R, device=device)
    tM = torch.tensor(M.astype(np.float32), device=device)
    n = len(tX)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            e = m(tX[idx], tU[idx]) - tR[idx]
            loss = torch.einsum("bi,bij,bj->b", e, tM[idx], e).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
            opt.step()
        sch.step()
    return m


@torch.no_grad()
def evaluate(m, d, plant, p_obs, device="cpu", near=0.10):
    X = torch.tensor(d["X"].astype(np.float32), device=device)
    U = torch.tensor(d["U"].astype(np.float32), device=device)
    e = (m(X, U).cpu().numpy() - d["R"])
    G = constraint_grad(plant, d["X"][:, :NQ], p_obs)
    gn = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-12)
    along = (e * gn).sum(1)
    near_m = d["margin"] < near
    out = dict(mse=float((e ** 2).sum(1).mean()),
               mse_q=float((e[:, :NQ] ** 2).sum(1).mean()),
               mse_qd=float((e[:, NQ:] ** 2).sum(1).mean()),
               err_constraint_dir=float(np.sqrt((along ** 2).mean())))
    if near_m.any():
        out["err_constraint_dir_near"] = float(np.sqrt((along[near_m] ** 2).mean()))
        out["mse_near"] = float((e[near_m] ** 2).sum(1).mean())
    return out
