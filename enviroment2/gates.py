"""The three Tier-1 gates.

Each gate can KILL the research line.  They are deliberately cheap and are run
before any of the five-arm comparison.

  Gate DIR  -- does a decision-irrelevant direction exist at all?   (no training)
  Gate B    -- is closed-loop violation actually caused by model error?
  Gate A    -- is there a capacity trade-off to exploit?
"""

import numpy as np

from .env import Params, f_true, normal_dir, tangent_dir, sample_scenario
from .eval import eval_paired, paired_diff, summarize
from .mpc import MPCConfig
from .model import eval_errors, make_dyn_fn, train_model


# ---------------------------------------------------------------------------
# Gate DIR : directional sanity  (no model training required)
# ---------------------------------------------------------------------------
def perturbed_dyn(direction, eps, params=Params):
    """f_true plus a synthetic model error of magnitude eps along `direction`.

    direction: 'normal' | 'tangent' | 'random' | 'none'
    The error is injected in POSITION space, evaluated at the current state, so
    'normal' changes g at first order and 'tangent' does not.

    SIGN MATTERS.  normal_dir points along dg/dp, i.e. TOWARDS the obstacle, so
    eps>0 makes the model pessimistic (MPC backs off, violations go DOWN) while
    eps<0 makes it optimistic (violations go UP).  Both signs must be swept;
    only the optimistic branch tests the hypothesis.
    """
    rng = np.random.default_rng(0)

    def dyn(x, u, p_obs=None):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        nxt = f_true(x, u, params)
        if direction == "none" or eps == 0:
            return nxt
        if direction == "normal":
            d = normal_dir(x, p_obs)
        elif direction == "tangent":
            d = tangent_dir(x, p_obs)
        elif direction == "random":
            th = rng.uniform(0, 2 * np.pi, size=len(x))
            d = np.stack([np.cos(th), np.sin(th)], axis=-1)
        else:
            raise ValueError(direction)
        out = nxt.copy()
        out[:, :2] += eps * d
        return out

    return dyn


def run_gate_dir(eps_list=(-0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1), n_ep=20, seed=0,
                 params=Params, cfg=MPCConfig, directions=("normal", "tangent", "random")):
    rows = []
    for direction in directions:
        for eps in eps_list:
            per_ep = []
            for i in range(n_ep):
                rng = np.random.default_rng(10_000 + seed * 1000 + i)
                scn = sample_scenario(rng, params, jitter=True)
                base = perturbed_dyn(direction, eps, params)
                dyn = lambda x, u, _o=scn["p_obs"]: base(x, u, _o)
                from .eval import run_episode
                m, _ = run_episode(dyn, scn, cfg, params, seed=10_000 + i)
                m.update(direction=direction, eps=eps, episode=i)
                per_ep.append(m)
            rows.extend(per_ep)
    return rows


def gate_dir_verdict(rows, key="viol_integral", ratio_thresh=3.0):
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["direction"], r["eps"])].append(r[key])
    def worst(direction):
        """Largest violation over BOTH signs at the largest |eps|."""
        emax = max(abs(e) for (d, e) in agg if d == direction)
        vals = [(e, float(np.mean(agg[(direction, e)])))
                for (d, e) in agg if d == direction and abs(e) == emax]
        e_star, v_star = max(vals, key=lambda z: z[1])
        return emax, e_star, v_star

    emax, e_n, n = worst("normal")
    _, e_t, t = worst("tangent")
    ratio = n / (t + 1e-9)
    return dict(eps=emax, eps_normal_worst=e_n, eps_tangent_worst=e_t,
                normal=float(n), tangent=float(t), ratio=float(ratio),
                passed=bool(ratio > ratio_thresh and n > 1e-3),
                note="normal-direction error must hurt far more than tangential")


# ---------------------------------------------------------------------------
# Gate B : is violation attributable to model error?
# ---------------------------------------------------------------------------
def run_gate_b(model, n_ep=30, seed=0, params=Params, cfg=MPCConfig):
    dyn_true = lambda x, u: f_true(x, u, params)
    dyn_learn = make_dyn_fn(model, params)
    rows_t, _ = eval_paired(dyn_true, n_ep, base_seed=seed, cfg=cfg, params=params)
    rows_l, _ = eval_paired(dyn_learn, n_ep, base_seed=seed, cfg=cfg, params=params)
    return rows_t, rows_l


def gate_b_verdict(rows_t, rows_l, key="viol_integral"):
    a = [r[key] for r in rows_t]
    b = [r[key] for r in rows_l]
    d = paired_diff(a, b)
    passed = bool(np.mean(a) < 1e-3 and np.mean(b) > 10 * (np.mean(a) + 1e-6)
                  and d["ci_lo"] > 0)
    return dict(true_mean=float(np.mean(a)), learned_mean=float(np.mean(b)),
                passed=passed, **d,
                note="true-dynamics MPC must be ~violation-free while learned is not")


# ---------------------------------------------------------------------------
# Gate A : is there a capacity trade-off?
# ---------------------------------------------------------------------------
def run_gate_a(data, hidden=(64, 64), epochs=150, w_normal=10.0, seed=0,
               params=Params, device="cpu"):
    kw = dict(hidden=hidden, epochs=epochs, seed=seed, params=params, device=device)
    m_uni, h_uni = train_model(data["train"], mode="uniform", **kw)
    m_dir, h_dir = train_model(data["train"], mode="dir_weighted",
                               w_normal=w_normal, w_tangent=1.0, w_vel=1.0, **kw)
    e_uni = eval_errors(m_uni, data["test"], params, device)
    e_dir = eval_errors(m_dir, data["test"], params, device)
    return dict(uniform=e_uni, dir_weighted=e_dir,
                models=(m_uni, m_dir), hist=(h_uni, h_dir))


def gate_a_verdict(res, rel=0.05):
    u, d = res["uniform"], res["dir_weighted"]
    dn = (d["rmse_normal"] - u["rmse_normal"]) / u["rmse_normal"]
    dt = (d["rmse_tangent"] - u["rmse_tangent"]) / u["rmse_tangent"]
    return dict(normal_rel_change=float(dn), tangent_rel_change=float(dt),
                passed=bool(dn < -rel and dt > rel),
                note="up-weighting the normal direction must LOWER normal error "
                     "and RAISE tangential error; otherwise re-allocation is a no-op")
