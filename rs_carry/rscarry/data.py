"""Collection + packaging with an IDENTIFIED LINEAR NOMINAL model.

Lesson carried over from the MuJoCo free-space testbed: with a hand-guessed
first-order-lag nominal, 99.4% of the residual was LINEAR mismatch, any network
learned it instantly, and every capacity number was meaningless. Here the
nominal is the ridge-regression best linear model fit on the TRAIN split only,
so the learning target is the genuinely nonlinear part (contact, payload
coupling). The linear-explained fraction is reported so this cannot happen
silently again.
"""

import numpy as np

from .env import NQ, Carry, make_env

R_SAFE = 0.10


def collect(n_traj, seed=0, T=120, mass_range=(0.1, 1.0), explore=0.25,
            verbose=True):
    env = make_env(seed=seed)
    c = Carry(env, seed=seed)
    trajs, fails = [], 0
    while len(trajs) < n_traj:
        m = c.rng.uniform(*mass_range)
        if not c.reset_and_grasp(m):
            fails += 1
            if fails > 3 * n_traj:
                raise RuntimeError("grasp keeps failing -- check smoke_test.py")
            continue
        tr, full = c.transport(T, explore=explore)
        tr["mass"] = m
        if len(tr["X"]) >= T // 2:                  # keep partial but real
            trajs.append(tr)
        if verbose and len(trajs) % 10 == 0:
            print(f"  {len(trajs)}/{n_traj} trajectories "
                  f"(grasp failures so far: {fails})", flush=True)
    env.close()
    return trajs


def margins(tr, r_safe=R_SAFE):
    d = np.linalg.norm(tr["eef"] - tr["p_obs"], axis=1)
    return d - r_safe                                # >0 satisfied


def fit_linear_nominal(trajs, lam=1e-6):
    X = np.concatenate([t["X"] for t in trajs])
    U = np.concatenate([t["U"] for t in trajs])
    Y = np.concatenate([t["Xn"] for t in trajs])
    Z = np.concatenate([X, U, np.ones((len(X), 1))], 1)
    W = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ Y)
    return W


def apply_nominal(W, X, U):
    Z = np.concatenate([X, U, np.ones((len(X), 1))], 1)
    return Z @ W


def build_dataset(n_traj=120, seed=0, splits=(0.7, 0.15, 0.15), **kw):
    trajs = collect(n_traj, seed=seed, **kw)
    idx = np.random.default_rng(seed).permutation(len(trajs))
    a = int(splits[0] * len(trajs)); b = a + int(splits[1] * len(trajs))
    parts = dict(train=[trajs[i] for i in idx[:a]],
                 val=[trajs[i] for i in idx[a:b]],
                 test=[trajs[i] for i in idx[b:]])
    W = fit_linear_nominal(parts["train"])

    out = {"_W": W}
    for k, ts in parts.items():
        X = np.concatenate([t["X"] for t in ts])
        U = np.concatenate([t["U"] for t in ts])
        Y = np.concatenate([t["Xn"] for t in ts])
        R = Y - apply_nominal(W, X, U)
        out[k] = dict(X=X, U=U, Xn=Y, R=R,
                      margin=np.concatenate([margins(t) for t in ts]),
                      trajs=ts)
    # how much did the linear nominal absorb? (guards against the old trap)
    Yall = np.concatenate([t["Xn"] for t in trajs])
    raw = Yall - np.concatenate([np.concatenate([t["X"][:, :NQ] for t in trajs]),
                                 np.zeros((len(Yall), NQ))], 1) * 0 - Yall.mean(0)
    out["_linear_explained"] = 1.0 - (out["train"]["R"] ** 2).sum() / \
        max(((np.concatenate([t["Xn"] for t in parts["train"]])
              - np.concatenate([t["Xn"] for t in parts["train"]]).mean(0)) ** 2).sum(), 1e-12)
    return out


def coverage(d, near=0.05):
    m = d["margin"]
    return dict(n=len(m), frac_violating=float(np.mean(m < 0)),
                frac_near=float(np.mean((m >= 0) & (m < near))),
                frac_far=float(np.mean(m >= near)),
                residual_rms=float(np.sqrt((d["R"] ** 2).sum(1).mean())))
