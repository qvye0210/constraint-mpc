"""Single-shooting nonlinear MPC with a pluggable internal dynamics model.

The internal model is the whole point of the study: swapping `dyn_fn` between
true / nominal / learned / deliberately-perturbed dynamics is how every gate is
run.  `dyn_fn` must be BATCHED:  (B, nx), (B, nu) -> (B, nx).

Jacobians are forward differences, but all (1 + n_var) rollouts are evaluated in
a single batched call and cached, so one solver iteration costs one batched
forward pass rather than hundreds of scalar ones.  This matters a lot when the
internal model is a neural net.
"""

import numpy as np
from scipy.optimize import minimize

from .env import NU, NX, Params


class MPCConfig:
    H = 15
    Q_track = 1.0
    Q_term = 4.0
    R_u = 0.02
    slack_pen = 5e3      # used only in the soft fallback solve
    maxiter = 30
    ftol = 1e-5
    fd_eps = 1e-5


def rollout_batch(dyn_fn, x0, U):
    """x0: (nx,) or (B,nx).  U: (B,H,nu) -> X: (B,H+1,nx)."""
    U = np.asarray(U, dtype=float)
    B, H, _ = U.shape
    x = np.broadcast_to(np.asarray(x0, dtype=float).reshape(-1, NX), (B, NX)).copy()
    X = np.empty((B, H + 1, NX))
    X[:, 0] = x
    for k in range(H):
        x = dyn_fn(x, U[:, k])
        X[:, k + 1] = x
    return X


class MPC:
    def __init__(self, dyn_fn, p_obs, cfg=MPCConfig, params=Params):
        self.dyn = dyn_fn
        self.p_obs = np.asarray(p_obs, dtype=float)
        self.cfg = cfg
        self.p = params
        self.n_var = cfg.H * NU
        self._warm = np.zeros(self.n_var)
        self._cache_key = None
        self._cache = None

    # -- core evaluation ----------------------------------------------------
    def _cost_and_g(self, X, U, ref):
        """X:(B,H+1,nx) U:(B,H,nu) ref:(H+1,2) -> cost (B,), g (B,H)."""
        cfg = self.cfg
        err = X[:, 1:, :2] - ref[None, 1:, :]
        w = np.full(cfg.H, cfg.Q_track)
        w[-1] = cfg.Q_term
        cost = (w[None, :] * (err ** 2).sum(-1)).sum(-1)
        cost = cost + cfg.R_u * (U ** 2).sum((1, 2))
        d = np.linalg.norm(X[:, 1:, :2] - self.p_obs[None, None, :], axis=-1)
        g = self.p.r_safe - d
        return cost, g

    def _fd(self, u_flat, x0, ref):
        key = u_flat.tobytes()
        if key == self._cache_key:
            return self._cache
        cfg = self.cfg
        n = self.n_var
        eps = cfg.fd_eps
        U0 = u_flat.reshape(cfg.H, NU)
        Ub = np.repeat(U0[None], n + 1, axis=0)
        for i in range(n):
            Ub[i + 1].reshape(-1)[i] += eps
        X = rollout_batch(self.dyn, x0, Ub)
        cost, g = self._cost_and_g(X, Ub, ref)
        dcost = (cost[1:] - cost[0]) / eps                      # (n,)
        dg = ((g[1:] - g[0:1]) / eps).T                          # (H, n)
        out = (cost[0], g[0], dcost, dg)
        self._cache_key, self._cache = key, out
        return out

    # -- solve --------------------------------------------------------------
    def solve(self, x0, ref):
        cfg = self.cfg
        umax = self.p.u_max
        bounds = [(-umax, umax)] * self.n_var

        f = lambda u: self._fd(u, x0, ref)[0]
        fj = lambda u: self._fd(u, x0, ref)[2]
        c = lambda u: -self._fd(u, x0, ref)[1]        # margin >= 0
        cj = lambda u: -self._fd(u, x0, ref)[3]

        res = minimize(f, self._warm, jac=fj, bounds=bounds,
                       constraints=[{"type": "ineq", "fun": c, "jac": cj}],
                       method="SLSQP",
                       options={"maxiter": cfg.maxiter, "ftol": cfg.ftol})
        U = res.x.reshape(cfg.H, NU)
        gpred = self._fd(res.x, x0, ref)[1]
        infeasible = (not res.success) or bool(np.any(gpred > 1e-6))

        if infeasible:
            # soft fallback so the closed loop always has a control to apply
            def fs(u):
                cost, g, _, _ = self._fd(u, x0, ref)
                return cost + cfg.slack_pen * np.sum(np.maximum(g, 0.0) ** 2)

            def fsj(u):
                _, g, dcost, dg = self._fd(u, x0, ref)
                act = 2.0 * np.maximum(g, 0.0)
                return dcost + cfg.slack_pen * (act[None, :] @ dg).ravel()

            res2 = minimize(fs, res.x, jac=fsj, bounds=bounds, method="SLSQP",
                            options={"maxiter": cfg.maxiter, "ftol": cfg.ftol})
            U = res2.x.reshape(cfg.H, NU)
            gpred = self._fd(res2.x, x0, ref)[1]

        self._warm = np.concatenate([U[1:].ravel(), U[-1]])
        info = dict(infeasible=bool(infeasible),
                    g_pred=gpred.copy(),
                    margin_pred=(-gpred).copy())
        return U, info

    def reset(self):
        self._warm = np.zeros(self.n_var)
        self._cache_key, self._cache = None, None
