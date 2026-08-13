"""Closed-loop rollout: plant is always f_true, MPC's internal model is swappable."""

import numpy as np

from .env import NU, Params, f_true, g_val, ref_traj, sample_scenario
from .mpc import MPC, MPCConfig


def plant_step(x, u, p=Params):
    return f_true(x, u, p)


def run_episode(model_dyn, scn, cfg=MPCConfig, params=Params, seed=0,
                proc_noise=0.0, explore=0.0, resolve_every=None):
    """model_dyn: batched (B,nx),(B,nu)->(B,nx) used INSIDE the MPC."""
    rng = np.random.default_rng(seed)
    resolve_every = params.resolve_every if resolve_every is None else resolve_every
    n = params.ep_len
    ref_full = ref_traj(scn, n + cfg.H, params)
    mpc = MPC(model_dyn, scn["p_obs"], cfg, params)

    x = scn["x0"].copy()
    X, U, G, INF, MARG = [], [], [], [], []
    obj = 0.0
    u_seq, info = None, None
    for t in range(n):
        # zero-order hold: re-solve only every `resolve_every` steps, so the
        # MPC must rely on its own multi-step prediction in between.  This is
        # what makes horizon-propagated model error observable in closed loop.
        if t % resolve_every == 0:
            ref_win = ref_full[t: t + cfg.H + 1]
            u_seq, info = mpc.solve(x, ref_win)
        u = u_seq[t % resolve_every]
        if explore > 0:
            u = u + rng.normal(0, explore, size=NU)
        u = np.clip(u, -params.u_max, params.u_max)

        X.append(x.copy())
        U.append(u.copy())
        G.append(float(g_val(x, scn["p_obs"], params)))
        MARG.append(info["margin_pred"].copy())
        INF.append(info["infeasible"])

        obj += float(np.sum((x[:2] - ref_full[t]) ** 2) + cfg.R_u * np.sum(u ** 2))
        x = plant_step(x, u, params)
        if proc_noise > 0:
            x = x + rng.normal(0, proc_noise, size=x.shape) * np.array([1, 1, 2, 2])

    X.append(x.copy())
    G.append(float(g_val(x, scn["p_obs"], params)))
    X = np.array(X)
    U = np.array(U)
    G = np.array(G)
    viol = np.maximum(G, 0.0)

    metrics = dict(
        viol_any=float(np.any(G > params.viol_tol)),
        viol_freq=float(np.mean(G > params.viol_tol)),
        viol_mean=float(viol[viol > 0].mean()) if np.any(viol > 0) else 0.0,
        viol_max=float(viol.max()),
        viol_integral=float(viol.sum()),
        track_rmse=float(np.sqrt(np.mean(np.sum((X[:-1, :2] - ref_full[:len(X) - 1]) ** 2, axis=1)))),
        objective=obj,
        infeas_rate=float(np.mean(INF)),
        min_margin=float(-G.max()),
    )
    traj = dict(X=X, U=U, G=G, margins=np.array(MARG), p_obs=scn["p_obs"])
    return metrics, traj


def eval_paired(model_dyn, n_ep, base_seed=0, cfg=MPCConfig, params=Params,
                proc_noise=0.0, jitter=True, resolve_every=None):
    """Same scenario seeds across methods -> paired comparison."""
    rows, trajs = [], []
    for i in range(n_ep):
        rng = np.random.default_rng(10_000 + base_seed * 1000 + i)
        scn = sample_scenario(rng, params, jitter=jitter)
        m, tr = run_episode(model_dyn, scn, cfg, params, seed=10_000 + i,
                            proc_noise=proc_noise, resolve_every=resolve_every)
        m["episode"] = i
        rows.append(m)
        trajs.append(tr)
    return rows, trajs


def summarize(rows, keys=None):
    keys = keys or [k for k in rows[0] if k != "episode"]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k + "_mean"] = float(v.mean())
        out[k + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return out


def bootstrap_ci(vals, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    bs = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def paired_diff(a, b, n_boot=2000, seed=0):
    """b - a, paired. Returns mean diff, CI, and a paired-bootstrap p-value."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    lo, hi = bootstrap_ci(d, n_boot, seed=seed)
    rng = np.random.default_rng(seed + 1)
    dc = d - d.mean()
    bs = rng.choice(dc, size=(n_boot, len(d)), replace=True).mean(axis=1)
    p = float(np.mean(np.abs(bs) >= abs(d.mean())))
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    return dict(mean_diff=float(d.mean()), ci_lo=lo, ci_hi=hi, p=p,
                cohen_dz=float(d.mean() / sd) if sd > 0 else 0.0)
