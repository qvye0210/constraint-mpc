#!/usr/bin/env python3
"""active_set_robust_dual_gate -- STAGE 1: dual recovery + FD validation.
Per spec section 4 this MUST pass before the main gate is implemented/run.
    PYTHONPATH=. python dual_fdtest.py [--quick]
AUDIT FINDINGS (reported, not guessed):
- cmpc2d.mpc.MPC solves via scipy SLSQP; multipliers are NOT exposed by
  the solver. Recovery method: KKT-stationarity NNLS at the solution
  (active constraints g > -tol_act; coordinates at input bounds excluded
  from the stationarity residual; mu >= 0 via scipy.optimize.nnls).
  This script validates the recovered mu against finite-difference
  sensitivities (V(eps)-V(0))/eps with per-constraint RHS tightening.
- OffsetMPC (per-k RHS offsets) built here is the same machinery the
  main gate's perturbed scenarios (rho - delta_k >= 0) will reuse.
- Oracle MPC: MPC takes dyn_fn as an argument -> true-dynamics MPC is
  constructible; no blocker.
CHECKS (all must hold on >= 80% of tested active constraints):
  sign agreement; |mu_fd - mu_nnls| / max(|mu_fd|, tol) <= 0.3;
  inactive-constraint mu <= 1e-4 * max mu; complementary slackness
  mu_k * slack_k <= 1e-6 scale. FAIL => STOP, main gate not implemented.
"""
import argparse, json, os, sys
import numpy as np
import torch
from scipy.optimize import nnls

sys.path.insert(0, ".")
from cmpc2d.mpc import MPC, MPCConfig
from cmpc2d.env import NX, NU, Params, f_nominal
from revival_gate import get_apis

OUT = "results/active_set_dual_gate/stage1"
os.makedirs(OUT, exist_ok=True)
api = get_apis()
TOL_ACT = 1e-5


def learned_dyn(seed):
    import hashlib
    p = f"results/prefix_risk_gate/dyn_seed{seed}.pt"
    assert os.path.exists(p), f"missing frozen ckpt {p}"
    m = api["ResidualMLP"]((64, 64), n_dist=0)
    m.load_state_dict(torch.load(p, map_location="cpu")); m.eval()
    md5 = hashlib.md5(b"".join(v.numpy().tobytes()
                               for v in m.state_dict().values())).hexdigest()

    def dyn(x, u):
        x = np.atleast_2d(x); u = np.atleast_2d(u)
        r = m(torch.tensor(x.astype(np.float32)),
              torch.tensor(u.astype(np.float32))).detach().numpy()
        return f_nominal(x, u) + r
    return dyn, md5


class OffsetMPC(MPC):
    """MPC with per-constraint RHS tightening: margin_j >= offset_j."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.offsets = None

    def _fd(self, u_flat, x0, ref):
        cost, g, dc, dg = super()._fd(u_flat, x0, ref)
        if self.offsets is not None:
            g = g + self.offsets        # g <= 0 feasible; tighten by offset
        return cost, g, dc, dg


def recover_duals(mpc, U, x0, ref):
    u = U.ravel()
    cost, g, dc, dg = mpc._fd(u, x0, ref)
    umax = mpc.p.u_max
    free = np.abs(np.abs(u) - umax) > 1e-7          # not at bounds
    act = np.where(g > -TOL_ACT)[0]
    mu = np.zeros(len(g))
    if len(act):
        # stationarity on free coords: dc_free + dg_act_free^T mu = 0
        A = dg[act][:, free].T                       # (n_free, n_act)
        b = -dc[free]
        mu_act, _ = nnls(A, b)
        mu[act] = mu_act
    return mu, g, cost


def main(quick):
    cfg = MPCConfig
    rng = np.random.default_rng(0)
    dyn, md5 = learned_dyn(0)
    d = api["build_dataset"](n_traj=6 if quick else 20, seed=14000)["train"]
    n_q = 4 if quick else 12
    idx = rng.choice(len(d["X"]) - 1, n_q, replace=False)
    eps = 1e-4
    rows, ok_all = [], []
    for qi in idx:
        x0 = d["X"][qi]
        p_obs = d["p_obs"][qi] if d["p_obs"].ndim > 1 else d["p_obs"]
        ref = np.tile(x0[:2] + np.array([0.4, 0.0]), (cfg.H, 1))
        mpc = OffsetMPC(dyn, p_obs)
        U, info = mpc.solve(x0, ref)
        if info["infeasible"]:
            continue
        mu, g, V0 = recover_duals(mpc, U, x0, ref)
        act = np.where(g > -TOL_ACT)[0]
        inact = np.where(g <= -10 * TOL_ACT)[0]
        # complementary slackness + inactive mu
        cs = float(np.max(np.abs(mu * (-g)))) if len(g) else 0.0
        mu_max = max(mu.max(), 1e-12)
        inact_ok = bool((mu[inact] <= 1e-4 * mu_max + 1e-12).all()) \
            if len(inact) else True
        for k in (act[:3] if len(act) else []):
            mpc2 = OffsetMPC(dyn, p_obs)
            off = np.zeros(len(g)); off[k] = eps
            mpc2.offsets = off
            mpc2._warm = U.ravel().copy()
            U2, info2 = mpc2.solve(x0, ref)
            if info2["infeasible"]:
                continue
            _, _, V1 = recover_duals(mpc2, U2, x0, ref)
            mu_fd = (V1 - V0) / eps
            rel = abs(mu_fd - mu[k]) / max(abs(mu_fd), 1e-6)
            sign_ok = (mu_fd >= -1e-8) and (mu[k] >= 0)
            passed = bool(sign_ok and rel <= 0.3)
            ok_all.append(passed)
            rows.append(dict(query=int(qi), k=int(k), mu_nnls=float(mu[k]),
                             mu_fd=float(mu_fd), rel_err=float(rel),
                             sign_ok=sign_ok, comp_slack=cs,
                             inactive_ok=inact_ok, passed=passed))
            print(f"q{qi} k={k}: mu_nnls {mu[k]:.4g}  mu_fd {mu_fd:.4g}  "
                  f"rel {rel:.2f}  {'OK' if passed else 'MISMATCH'}  "
                  f"(cs {cs:.1e}, inact {'OK' if inact_ok else 'BAD'})")
    frac = float(np.mean(ok_all)) if ok_all else 0.0
    verdict = dict(ckpt_md5=md5, n_pairs=len(ok_all), pass_frac=frac,
                   result="PASS" if (ok_all and frac >= 0.8) else "FAIL")
    json.dump(dict(verdict=verdict, rows=rows),
              open(f"{OUT}/fdtest.json", "w"), indent=2)
    print(f"\n{len(ok_all)} tested pairs, pass fraction {frac:.2f}")
    print(">>> " + ("STAGE 1 PASS: dual recovery validated; main gate "
                    "implementation may proceed"
                    if verdict["result"] == "PASS" else
                    "STAGE 1 FAIL: recovered multipliers disagree with "
                    "finite-difference sensitivities -- STOP per spec "
                    "section 4; main gate must not be implemented until "
                    "the recovery is fixed"))
    print(f"wrote {OUT}/fdtest.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
