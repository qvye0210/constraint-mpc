"""Residual dynamics model  x_hat_{t+1} = f_nominal(x_t,u_t) + f_theta(x_t,u_t).

Learning the residual on top of the drag-free nominal model mirrors what will be
done on the UR5e (learn the mismatch, not the whole rigid-body model).

The loss can weight the position error decomposed into the constraint NORMAL and
TANGENTIAL directions.  That decomposition is the instrument for Gate A
(capacity trade-off): if up-weighting the normal direction does not degrade the
tangential direction, there is no trade-off to exploit and the whole idea is a
no-op in this regime.
"""

import numpy as np
import torch
import torch.nn as nn

from .env import NU, NX, Params, f_nominal, normal_dir, tangent_dir


class ResidualMLP(nn.Module):
    def __init__(self, hidden=(256, 256, 128), act=nn.SiLU):
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
        z = np.concatenate([X, U], axis=1)
        self.in_mu.copy_(torch.tensor(z.mean(0), dtype=torch.float32))
        self.in_sd.copy_(torch.tensor(z.std(0) + 1e-6, dtype=torch.float32))
        self.out_sd.copy_(torch.tensor(R.std(0) + 1e-8, dtype=torch.float32))

    def residual(self, x, u):
        z = (torch.cat([x, u], dim=-1) - self.in_mu) / self.in_sd
        return self.net(z) * self.out_sd

    def forward(self, x, u):
        return self.residual(x, u)


# ----------------------------------------------------------------------------
def make_dyn_fn(model, params=Params, device="cpu"):
    """Wrap a torch model as a batched numpy dynamics fn usable inside the MPC."""
    model.eval()

    def dyn(x, u):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        u = np.atleast_2d(np.asarray(u, dtype=float))
        base = f_nominal(x, u, params)
        with torch.no_grad():
            r = model(torch.tensor(x, dtype=torch.float32, device=device),
                      torch.tensor(u, dtype=torch.float32, device=device)).cpu().numpy()
        return base + r.astype(float)

    return dyn


def decompose_pos_error(e_pos, x, p_obs):
    """Split a position error into constraint-normal and tangential components."""
    n = normal_dir(x, p_obs)
    t = tangent_dir(x, p_obs)
    return (e_pos * n).sum(-1), (e_pos * t).sum(-1)


# ----------------------------------------------------------------------------
def train_model(data, mode="uniform", w_normal=1.0, w_tangent=1.0, w_vel=1.0,
                hidden=(256, 256, 128), epochs=200, bs=256, lr=1e-3, seed=0,
                device="cpu", params=Params, verbose=False, val=None):
    """mode: 'uniform' | 'dir_weighted'.

    'dir_weighted' up-weights the constraint-normal component of the position
    error.  Everything else (data, architecture, epochs, seed) is held fixed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = data["X"].astype(np.float32)
    U = data["U"].astype(np.float32)
    Xn = data["Xn"].astype(np.float32)
    OB = data["p_obs"].astype(np.float32)
    R = (Xn - f_nominal(X, U, params)).astype(np.float32)   # target residual

    model = ResidualMLP(hidden).to(device)
    model.set_norm(X, U, R)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tX = torch.tensor(X, device=device)
    tU = torch.tensor(U, device=device)
    tR = torch.tensor(R, device=device)
    # constraint normal / tangent at the CURRENT state (position space)
    nrm = torch.tensor(normal_dir(X, OB).astype(np.float32), device=device)
    tan = torch.tensor(tangent_dir(X, OB).astype(np.float32), device=device)

    n = len(X)
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot, nb = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            pred = model(tX[idx], tU[idx])
            err = pred - tR[idx]
            e_pos, e_vel = err[:, :2], err[:, 2:]
            if mode == "uniform":
                loss = (err ** 2).sum(-1).mean()
            elif mode == "dir_weighted":
                en = (e_pos * nrm[idx]).sum(-1)
                et = (e_pos * tan[idx]).sum(-1)
                loss = (w_normal * en ** 2 + w_tangent * et ** 2
                        + w_vel * (e_vel ** 2).sum(-1)).mean()
            else:
                raise ValueError(mode)
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"NaN/Inf loss at epoch {ep}")
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sched.step()
        rec = dict(epoch=ep, train_loss=tot / nb, grad_norm=float(gn))
        if val is not None and (ep % 10 == 0 or ep == epochs - 1):
            rec.update({"val_" + k: v for k, v in eval_errors(model, val, params, device).items()})
        hist.append(rec)
        if verbose and ep % 20 == 0:
            print(f"  ep {ep:4d} loss {tot/nb:.3e}")
    return model, hist


@torch.no_grad()
def eval_errors(model, data, params=Params, device="cpu"):
    """One-step error diagnostics, decomposed by constraint geometry."""
    model.eval()
    X, U, Xn, OB = data["X"], data["U"], data["Xn"], data["p_obs"]
    base = f_nominal(X, U, params)
    r = model(torch.tensor(X, dtype=torch.float32, device=device),
              torch.tensor(U, dtype=torch.float32, device=device)).cpu().numpy()
    pred = base + r
    e = pred - Xn
    e_pos, e_vel = e[:, :2], e[:, 2:]
    en, et = decompose_pos_error(e_pos, X, OB)
    m = -(params.r_safe - np.linalg.norm(X[:, :2] - OB, axis=-1))  # margin
    near = m < 0.5
    out = dict(
        rmse_all=float(np.sqrt((e ** 2).sum(-1).mean())),
        rmse_pos=float(np.sqrt((e_pos ** 2).sum(-1).mean())),
        rmse_vel=float(np.sqrt((e_vel ** 2).sum(-1).mean())),
        rmse_normal=float(np.sqrt((en ** 2).mean())),
        rmse_tangent=float(np.sqrt((et ** 2).mean())),
    )
    if near.any():
        out["rmse_normal_near"] = float(np.sqrt((en[near] ** 2).mean()))
        out["rmse_tangent_near"] = float(np.sqrt((et[near] ** 2).mean()))
    return out
