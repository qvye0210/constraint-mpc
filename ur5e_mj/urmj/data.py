"""Data collection on the MuJoCo UR5e.

Splits are made at TRAJECTORY level, never at transition level: consecutive
transitions are highly correlated and mixing them across train/test leaks the
test set.

Trajectories are deliberately steered toward the obstacle so that the constraint
is genuinely near-active for part of the data. A dataset where the constraint is
never approached cannot test anything about constraint-aware learning, and it is
the easiest mistake to make here.
"""

import numpy as np

from .plant import NQ, NU, NX, MjParams, UR5ePlant, f_nominal

# Obstacle placed at the centre of the reachable TCP cloud for the sampled
# start configurations.  Placed anywhere else the constraint is never
# approached: with the obstacle at (0.45, 0.10, 0.35) the margin never fell
# below 0.106 and 100% of samples sat in the "far" bin, which makes every
# constraint-related measurement meaningless.  Check `coverage()` after any
# change to the start distribution.
DEFAULT_OBS = (-0.379, -0.097, 0.487)


# Base configuration; trajectories start near it rather than anywhere in the
# joint ranges.  The original version sampled each joint independently over a
# wide range, which in a 12-dimensional state space leaves a few thousand
# samples hopelessly sparse: held-out trajectories then visited regions the
# training set never saw, and the resulting "error floor" was extrapolation
# rather than scarce capacity.  Diagnostic: with 40 trajectories a transition
# split gave train/test gap 0.7 while a trajectory split gave 18.4 -- same data,
# same model, 26x difference purely from whether the test states had been seen.
Q_BASE = np.array([-0.3, -1.4, 1.3, -1.5, -1.6, 0.0])

START_SIGMA = 0.15      # rad; the knob that trades coverage against difficulty


def sample_start(rng, sigma=None, base=None):
    base = Q_BASE if base is None else np.asarray(base, dtype=float)
    sigma = START_SIGMA if sigma is None else sigma
    return base + rng.normal(0.0, sigma, NQ)


def collect(n_traj, seed=0, T=80, p_obs=DEFAULT_OBS, r_safe=0.15,
            payload_range=(0.0, 1.5), explore=0.35, attract=0.9,
            params=MjParams, verbose=False, sigma=None, base=None):
    """Roll out with a crude obstacle-seeking controller plus exploration noise.

    `attract` drives the TCP toward the obstacle so the constraint margin gets
    small; `explore` keeps the data informative. Neither is an MPC -- this is
    open-loop data collection, deliberately independent of the controller that
    will later be evaluated.
    """
    plant = UR5ePlant(p=params, seed=seed)
    rng = np.random.default_rng(seed)
    p_obs = np.asarray(p_obs, dtype=float)
    trajs = []
    for ep in range(n_traj):
        plant.set_payload(rng.uniform(*payload_range))
        plant.set_state(np.concatenate([sample_start(rng, sigma, base), np.zeros(NQ)]))
        u = rng.uniform(-0.4, 0.4, NU)
        X, U, Xn, G = [], [], [], []
        for k in range(T):
            x = plant.get_state()
            if k % 5 == 0:
                # pull the TCP toward the obstacle, then add exploration
                J = plant.tcp_jacobian()
                e = p_obs - plant.tcp()
                u_att = np.linalg.lstsq(J, e, rcond=None)[0]
                nrm = np.linalg.norm(u_att)
                if nrm > 1e-9:
                    u_att = u_att / nrm
                u = np.clip(attract * u_att + rng.normal(0, explore, NU),
                            -params.u_max, params.u_max)
            xn = plant.step(u)
            X.append(x); U.append(u.copy()); Xn.append(xn)
            G.append(r_safe - np.linalg.norm(plant.tcp() - p_obs))
        trajs.append(dict(X=np.array(X), U=np.array(U), Xn=np.array(Xn),
                          g=np.array(G)))
        if verbose and (ep + 1) % 25 == 0:
            print(f"  collected {ep + 1}/{n_traj}")
    return trajs


def build_dataset(n_traj=150, seed=0, splits=(0.7, 0.15, 0.15), **kw):
    trajs = collect(n_traj, seed=seed, **kw)
    idx = np.random.default_rng(seed).permutation(len(trajs))
    a = int(splits[0] * len(trajs)); b = a + int(splits[1] * len(trajs))
    groups = dict(train=idx[:a], val=idx[a:b], test=idx[b:])

    def pack(ids):
        d = {k: np.concatenate([trajs[i][k] for i in ids])
             for k in ("X", "U", "Xn", "g")}
        d["R"] = d["Xn"] - f_nominal(d["X"], d["U"])
        d["margin"] = -d["g"]
        return d

    out = {k: pack(v) for k, v in groups.items()}
    out["_meta"] = dict(n_traj=n_traj, seed=seed, T=len(trajs[0]["X"]))
    return out


def coverage(d, near=0.10):
    m = d["margin"]
    return dict(n=len(m),
                frac_violating=float(np.mean(m < 0)),
                frac_near=float(np.mean((m >= 0) & (m < near))),
                frac_far=float(np.mean(m >= near)),
                margin_min=float(m.min()), margin_median=float(np.median(m)),
                residual_rms_q=float(np.sqrt((d["R"][:, :NQ] ** 2).sum(1).mean())),
                residual_rms_qd=float(np.sqrt((d["R"][:, NQ:] ** 2).sum(1).mean())))
