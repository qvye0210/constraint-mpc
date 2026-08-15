"""Trajectory collection with a nominal-model MPC teacher.

Splits are made at TRAJECTORY level (never transition level) so that adjacent,
highly-correlated transitions cannot leak across train/val/test.
"""

import numpy as np

from .env import NU, NX, Params, f_true, f_nominal, g_val, ref_traj, sample_scenario
from .mpc import MPC, MPCConfig


def collect_trajectory(scn, seed, cfg=MPCConfig, params=Params,
                       explore=0.6, proc_noise=0.0, H_store=10):
    rng = np.random.default_rng(seed)
    n = params.ep_len
    ref_full = ref_traj(scn, n + cfg.H, params)
    mpc = MPC(lambda x, u: f_nominal(x, u, params), scn["p_obs"], cfg, params)

    xs, us, margins = [], [], []
    x = scn["x0"].copy()
    for t in range(n):
        u_seq, info = mpc.solve(x, ref_full[t: t + cfg.H + 1])
        u = np.clip(u_seq[0] + rng.normal(0, explore, NU), -params.u_max, params.u_max)
        xs.append(x.copy())
        us.append(u.copy())
        margins.append(info["margin_pred"][:H_store].copy())
        x = f_true(x, u, params)
        if proc_noise > 0:
            x = x + rng.normal(0, proc_noise, NX) * np.array([1, 1, 2, 2])
    xs.append(x.copy())
    return dict(X=np.array(xs), U=np.array(us), margins=np.array(margins),
                p_obs=scn["p_obs"])


def build_dataset(n_traj, seed=0, cfg=MPCConfig, params=Params, H_store=10,
                  explore=0.6, proc_noise=0.0, splits=(0.7, 0.15, 0.15),
                  verbose=False):
    trajs = []
    for i in range(n_traj):
        rng = np.random.default_rng(seed * 1000 + i)
        scn = sample_scenario(rng, params, jitter=True)
        trajs.append(collect_trajectory(scn, seed * 1000 + i, cfg, params,
                                        explore, proc_noise, H_store))
        if verbose and (i + 1) % 5 == 0:
            print(f"  collected {i+1}/{n_traj}")

    idx = np.random.default_rng(seed).permutation(n_traj)
    n_tr = int(splits[0] * n_traj)
    n_va = int(splits[1] * n_traj)
    groups = dict(train=idx[:n_tr], val=idx[n_tr:n_tr + n_va], test=idx[n_tr + n_va:])

    def pack(ids):
        X, U, Xn, OB, MG, WX, WU = [], [], [], [], [], [], []
        for j in ids:
            tr = trajs[j]
            T = len(tr["U"])
            for t in range(T):
                X.append(tr["X"][t]); U.append(tr["U"][t]); Xn.append(tr["X"][t + 1])
                OB.append(tr["p_obs"]); MG.append(tr["margins"][t])
                # future window (truncated near the end) for the Step-3 rollout loss
                k = min(H_store, T - t)
                wx = np.zeros((H_store, NX)); wu = np.zeros((H_store, NU))
                wx[:k] = tr["X"][t + 1: t + 1 + k]
                wu[:k] = tr["U"][t: t + k]
                if k < H_store:
                    wx[k:] = wx[k - 1]; wu[k:] = wu[k - 1]
                WX.append(wx); WU.append(wu)
        d = dict(X=np.array(X), U=np.array(U), Xn=np.array(Xn),
                 p_obs=np.array(OB), margins=np.array(MG),
                 win_X=np.array(WX), win_U=np.array(WU))
        d["margin_now"] = -g_val(d["X"], d["p_obs"], params)
        return d

    data = {k: pack(v) for k, v in groups.items()}
    data["_meta"] = dict(n_traj=n_traj, seed=seed, H_store=H_store,
                         explore=explore, proc_noise=proc_noise)
    return data


def coverage_report(d, params=Params):
    m = d["margin_now"]
    return dict(n=len(m),
                frac_active=float(np.mean(m < 0.1)),
                frac_near=float(np.mean((m >= 0.1) & (m < 0.5))),
                frac_far=float(np.mean(m >= 0.5)),
                margin_min=float(m.min()), margin_max=float(m.max()))
