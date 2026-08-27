#!/usr/bin/env python3
"""Cross-validate the in-house MPC against an independent CasADi/IPOPT solve.

The project MPC is single-shooting SLSQP with finite-difference Jacobians. It has
never been checked against a reference solver, so a systematic bias in it would
propagate silently into every closed-loop number. This rebuilds the identical
OCP in CasADi with analytic derivatives and IPOPT, and compares optimal cost,
the applied control, and constraint satisfaction over random states.

Uses the analytic nominal model on both sides so that any discrepancy is
attributable to the optimiser rather than to the dynamics.

    python mpc_validate.py --n 40
"""

import argparse
import json

import casadi as ca
import numpy as np

from cmpc2d.env import NU, NX, Params, f_nominal, ref_traj, sample_scenario
from cmpc2d.mpc import MPC, MPCConfig


def casadi_solve(x0, ref, p_obs, cfg=MPCConfig, params=Params, verbose=False):
    H, dt = cfg.H, params.dt
    opti = ca.Opti()
    U = opti.variable(H, NU)
    x = ca.DM(np.asarray(x0, dtype=float).reshape(1, NX))
    cost = 0
    for k in range(H):
        pos, vel = x[0, 0:2], x[0, 2:4]
        u = U[k, :]
        pos_n = pos + vel * dt + 0.5 * u * dt * dt
        vel_n = vel + u * dt
        x = ca.horzcat(pos_n, vel_n)
        w = cfg.Q_term if k == H - 1 else cfg.Q_track
        e = x[0, 0:2] - ca.DM(ref[k + 1].reshape(1, 2))
        cost += w * ca.sumsqr(e) + cfg.R_u * ca.sumsqr(u)
        d = x[0, 0:2] - ca.DM(np.asarray(p_obs).reshape(1, 2))
        opti.subject_to(params.r_safe - ca.sqrt(ca.sumsqr(d) + 1e-12) <= 0)
    opti.subject_to(opti.bounded(-params.u_max, ca.vec(U), params.u_max))
    opti.minimize(cost)
    opti.solver("ipopt", {"print_time": False},
                {"print_level": 0, "sb": "yes", "max_iter": 500})
    try:
        sol = opti.solve()
        return np.array(sol.value(U)).reshape(H, NU), float(sol.value(cost)), True
    except RuntimeError:
        return None, np.inf, False


def eval_cost(U, x0, ref, p_obs, cfg=MPCConfig, params=Params):
    """Evaluate the project's own cost / constraint on any control sequence."""
    x = np.asarray(x0, dtype=float).reshape(1, NX)
    cost, gmax = 0.0, -np.inf
    for k in range(cfg.H):
        x = f_nominal(x, U[k].reshape(1, NU), params)
        w = cfg.Q_term if k == cfg.H - 1 else cfg.Q_track
        cost += w * float(np.sum((x[0, :2] - ref[k + 1]) ** 2))
        cost += cfg.R_u * float(np.sum(U[k] ** 2))
        gmax = max(gmax, params.r_safe - float(np.linalg.norm(x[0, :2] - p_obs)))
    return cost, gmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="results/mpc_validate.json")
    a = ap.parse_args()

    rows = []
    for i in range(a.n):
        rng = np.random.default_rng(7000 + i)
        scn = sample_scenario(rng, Params, jitter=True)
        ref = ref_traj(scn, MPCConfig.H + 2, Params)
        # perturb the start so the sample covers far / near / active regions
        x0 = scn["x0"].copy()
        x0[:2] += rng.normal(0, 1.2, 2)
        x0[2:] += rng.normal(0, 0.5, 2)

        mpc = MPC(lambda x, u: f_nominal(x, u, Params), scn["p_obs"])
        U_ours, info = mpc.solve(x0, ref[: MPCConfig.H + 1])
        c_ours, g_ours = eval_cost(U_ours, x0, ref, scn["p_obs"])

        U_ref, _, ok = casadi_solve(x0, ref, scn["p_obs"])
        if not ok:
            rows.append(dict(i=i, casadi_failed=True))
            continue
        c_ref, g_ref = eval_cost(U_ref, x0, ref, scn["p_obs"])

        rows.append(dict(
            i=i,
            cost_ours=c_ours, cost_casadi=c_ref,
            cost_gap_rel=(c_ours - c_ref) / max(abs(c_ref), 1e-9),
            gmax_ours=g_ours, gmax_casadi=g_ref,
            u0_diff=float(np.linalg.norm(U_ours[0] - U_ref[0])),
            u0_norm=float(np.linalg.norm(U_ref[0]) + 1e-9),
            ours_infeasible=bool(info["infeasible"]),
            casadi_failed=False))

    ok = [r for r in rows if not r.get("casadi_failed")]
    gaps = np.array([r["cost_gap_rel"] for r in ok])
    u0 = np.array([r["u0_diff"] / r["u0_norm"] for r in ok])
    viol_ours = np.array([r["gmax_ours"] for r in ok])
    viol_ref = np.array([r["gmax_casadi"] for r in ok])

    print(f"solved {len(ok)}/{len(rows)}  (CasADi failures: {len(rows)-len(ok)})")
    print(f"relative cost gap (ours - ipopt)/ipopt:")
    print(f"   median {np.median(gaps):+.2e}   mean {gaps.mean():+.2e}   "
          f"p95 {np.quantile(gaps, .95):+.2e}   max {gaps.max():+.2e}")
    print(f"   worse than ipopt by >1%: {int((gaps > 0.01).sum())}/{len(ok)}"
          f"   BETTER than ipopt (suspicious) : {int((gaps < -0.01).sum())}")
    print(f"first-control relative difference: median {np.median(u0):.2e}  "
          f"p95 {np.quantile(u0, .95):.2e}")
    print(f"max predicted constraint value  ours {viol_ours.max():+.2e}  "
          f"ipopt {viol_ref.max():+.2e}   (should both be <= ~0)")

    verdict = dict(
        n=len(ok),
        median_cost_gap=float(np.median(gaps)),
        p95_cost_gap=float(np.quantile(gaps, .95)),
        frac_worse_1pct=float((gaps > 0.01).mean()),
        frac_better_1pct=float((gaps < -0.01).mean()),
        max_g_ours=float(viol_ours.max()), max_g_ipopt=float(viol_ref.max()),
        passes=bool(np.median(gaps) < 0.01 and (gaps > 0.05).mean() < 0.1
                    and viol_ours.max() < 1e-3))
    print("\n>>> " + ("MPC solver validated: matches IPOPT within tolerance."
                      if verdict["passes"] else
                      "MPC solver DIVERGES from IPOPT. Closed-loop numbers are "
                      "not trustworthy until this is resolved."))
    with open(a.out, "w") as f:
        json.dump(dict(verdict=verdict, rows=rows), f, indent=2, default=float)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
