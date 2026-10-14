#!/usr/bin/env python3
"""day6_vel_margin.py -- can small pre-failure corrections avoid the
velocity-bound failure? Repo root, env robmpc. Base = v2 model.
  PYTHONPATH=. python day6_vel_margin.py --audit  # margins over full horizon
  PYTHONPATH=. python day6_vel_margin.py --pair   # model vs true vel margins
  PYTHONPATH=. python day6_vel_margin.py --loop   # extended-corrector A/B/C
SPEC REVISION (registered; corridor-only negative result retained):
correction margin vector = [corridor m/rs_k ; (v_lim-v_jk)/v_lim ;
(v_lim+v_jk)/v_lim] over k=0..N, all non-negative = safe, dimensionless;
same QP min 0.5 d'Rd + 0.5 eta ||xi||^2, m + A d + xi >= 0; cap
||d||_inf <= 2.0 and |u0+d| <= u_lim unchanged; B/C identical rules,
margin source only. Timing rule honoured: corrections act BEFORE failure;
no resurrection at the failure snapshot. 1.22 note: analytic chain is
exact for the linear nominal model (verified in --audit), so the 22%
is nominal-vs-true model sensitivity error."""
import argparse, json, os, sys
import numpy as np
import cvxpy as cp

sys.path.insert(0, ".")
from exec_interface import get_ps, nominal_AB, LinearResidual, ZeroResidual
from day2_diagnosis import model_rollout, true_rollout
from day3_solve_diag import margins6
from day5_corrector_check import episode_ctx, v2_res

OUT = "results/day6_vel"; os.makedirs(OUT, exist_ok=True)
CAP = 2.0; ETA = 1e3; SEEDS = (777, 300, 301)


def vel_margins(X, vlim, m):
    V = X[m:]                                    # (m, N+1)
    return (vlim - V) / vlim, (vlim + V) / vlim  # each (m, N+1), safe>=0


class ExtCorrector:
    """Corridor (normalised by rs) + velocity +/- margins, same QP."""
    def __init__(self, ps, Ad, Bd, N):
        m = ps.config_dim; self.m = m; self.vlim = ps.v_amp_nom
        self.u_lim = ps.u_amp_nom
        n_rows = (N + 1) * (1 + 2 * m)
        self.d = cp.Variable(m); self.xi = cp.Variable(n_rows)
        self.mpar = cp.Parameter(n_rows); self.Apar = cp.Parameter((n_rows, m))
        self.u0 = cp.Parameter(m)
        obj = 0.5 * cp.quad_form(self.d, np.eye(m) * 1e-3) \
            + 0.5 * ETA * cp.sum_squares(self.xi)
        con = [self.mpar + self.Apar @ self.d + self.xi >= 0, self.xi >= 0,
               cp.norm_inf(self.d) <= CAP,
               self.u0 + self.d <= self.u_lim, self.u0 + self.d >= -self.u_lim]
        self.prob = cp.Problem(cp.Minimize(obj), con)
        # dx_k/du0 (nominal chain, exact for the linear model)
        self.dx_du0 = [np.zeros((2 * m, m))]
        M = np.eye(2 * m)
        for k in range(1, N + 1):
            self.dx_du0.append(M @ Bd)
            M = Ad @ M

    def stack(self, Xp, cs, rs):
        m = self.m
        mc, gr = margins6(Xp[:m], cs, rs)          # corridor + pos-grad
        mv_p, mv_m = vel_margins(Xp, self.vlim, m)
        rows_m = [mc / rs, mv_p.T.ravel(), mv_m.T.ravel()]
        A_rows = [(gr[k] @ self.dx_du0[k][:m]) / rs[k]
                  for k in range(len(mc))]
        # d(mv+)/du0 = -dv/du0 / vlim ; d(mv-)/du0 = +dv/du0 / vlim
        for sgn in (-1.0, +1.0):
            for k in range(mv_p.shape[1]):
                for j in range(m):
                    A_rows.append(sgn * self.dx_du0[k][m + j] / self.vlim)
        return np.concatenate(rows_m), np.array(A_rows)

    def solve(self, Xp, cs, rs, u0):
        mvec, Amat = self.stack(Xp, cs, rs)
        self.mpar.value = mvec; self.Apar.value = Amat; self.u0.value = u0
        try:
            self.prob.solve(solver=cp.CLARABEL)
            if self.d.value is None:
                return np.zeros(self.m), 0.0, mvec
            return self.d.value, float(np.abs(self.xi.value).sum()), mvec
        except Exception:
            return np.zeros(self.m), 0.0, mvec


def run(ps, seed, arm, audit=False):
    from aux import compute_goal_state
    res = v2_res(); c = episode_ctx(ps, seed, res)
    m = ps.config_dim; A, B = nominal_AB(ps)
    cont, sim, it = c["cont"], c["sim"], c["it"]
    corr = ExtCorrector(ps, A, B, cont.N) if arm in "BC" else None
    x = np.r_[c["path"][0][:m], np.zeros(m)]
    ptraj = x[:m][None].repeat(cont.N + 1, axis=0)
    rows = []
    for ts in range(40):
        cs, rs = sim.get_corridor_balls(ptraj, c["path"], c["radii"], c["p_g"])
        xg, _ = compute_goal_state(cs, rs, c["pt"], c["p_g"], return_index=True)
        cont.set_parameter_values(x, cs.T, rs, xg)
        if not cont.solve():
            return rows, ts
        Upl = cont.U.value; u0 = Upl[:, 0].copy(); d = np.zeros(m)
        Xt = true_rollout(x, Upl, c["dn"], c["de"], ps.dt)
        mc_t, _ = margins6(Xt[:m], cs.T, rs)
        mvp_t, mvm_t = vel_margins(Xt, ps.v_amp_nom, m)
        if audit:
            Xm = cont.X.value
            mvp_m, _ = vel_margins(Xm, ps.v_amp_nom, m)
            rows.append(dict(ts=ts,
                             min_corr_true=float(mc_t.min()),
                             min_vel_true=float(min(mvp_t.min(), mvm_t.min())),
                             min_vel_model=float(mvp_m.min())))
            # analytic-vs-nominal-FD exactness spot check at ts==2
            if ts == 2 and arm == "A":
                eps = 0.5
                Up = Upl.copy(); Up[0, 0] += eps
                Xn0 = model_rollout(x, Upl, A, B, ZeroResidual())
                Xn1 = model_rollout(x, Up, A, B, ZeroResidual())
                fd = (Xn1 - Xn0)[:, 10] / eps
                corr0 = ExtCorrector(ps, A, B, cont.N)
                print(f"  [exactness] |nominal-FD - analytic| dx10/du0[0] = "
                      f"{np.abs(fd - corr0.dx_du0[10][:, 0]).max():.2e}")
        if corr is not None:
            Xp = model_rollout(x, Upl, cont.A, cont.B, ZeroResidual()) \
                if arm == "B" else Xt
            d, sl, mvec = corr.solve(Xp, cs.T, rs, u0)
            u0 = np.clip(u0 + d, -ps.u_amp_nom, ps.u_amp_nom)
            rows.append(dict(ts=ts, dnorm=float(np.linalg.norm(d)),
                             min_m=float(mvec.min())))
        x = np.array(it.solve_time_step(x, u0), float)
        ptraj = sim.get_shifted_trajectory(cont.X.value)[:, :m]
    return rows, None


def main():
    ap = argparse.ArgumentParser()
    for f in ("audit", "pair", "loop"):
        ap.add_argument(f"--{f}", action="store_true")
    a = ap.parse_args(); ps = get_ps()
    if a.audit:
        for sd in SEEDS:
            rows, fail = run(ps, sd, "A", audit=True)
            print(f"seed {sd} fail@{fail}; over episode: "
                  f"min corridor(true,horizon) "
                  f"{min(r['min_corr_true'] for r in rows):+.4f}")
            for r in rows[-4:]:
                print(f"  t={r['ts']:2d} corr_true {r['min_corr_true']:+.4f} "
                      f"vel_true {r['min_vel_true']:+.4f} "
                      f"vel_model {r['min_vel_model']:+.4f}")
        json.dump({}, open(f"{OUT}/audit_done.json", "w"))
    if a.pair:
        # model vs true velocity margins on the same plan, pre-failure steps
        for sd in SEEDS:
            res = v2_res(); c = episode_ctx(ps, sd, res)
            from aux import compute_goal_state
            m = ps.config_dim
            x = np.r_[c["path"][0][:m], np.zeros(m)]
            ptraj = x[:m][None].repeat(c["cont"].N + 1, axis=0)
            for ts in range(40):
                cs, rs = c["sim"].get_corridor_balls(ptraj, c["path"],
                                                     c["radii"], c["p_g"])
                xg, _ = compute_goal_state(cs, rs, c["pt"], c["p_g"],
                                           return_index=True)
                c["cont"].set_parameter_values(x, cs.T, rs, xg)
                if not c["cont"].solve():
                    break
                Upl = c["cont"].U.value
                Xm = model_rollout(x, Upl, c["cont"].A, c["cont"].B,
                                   ZeroResidual())
                Xt = true_rollout(x, Upl, c["dn"], c["de"], ps.dt)
                pm, _ = vel_margins(Xm, ps.v_amp_nom, m)
                pt_, _ = vel_margins(Xt, ps.v_amp_nom, m)
                if ts >= 7:
                    print(f"seed {sd} t={ts}: min vel margin model "
                          f"{pm.min():+.4f} true {pt_.min():+.4f} "
                          f"gap {pm.min()-pt_.min():+.4f}")
                x = np.array(c["it"].solve_time_step(x, Upl[:, 0]), float)
                ptraj = c["sim"].get_shifted_trajectory(
                    c["cont"].X.value)[:, :m]
    if a.loop:
        for arm in "ABC":
            for sd in SEEDS:
                rows, fail = run(ps, sd, arm)
                dn = [r.get("dnorm", 0.0) for r in rows] or [0.0]
                print(f"{arm} seed {sd}: fail@{fail} "
                      f"dnorm med/max {np.median(dn):.3f}/{np.max(dn):.3f}")


if __name__ == "__main__":
    main()
