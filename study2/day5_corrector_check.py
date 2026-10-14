#!/usr/bin/env python3
"""day5_corrector_check.py -- step 3: does the corrector solve the obstacle?
Repo root, env robmpc. Base model = v2 (post-coverage-fix, @11 parity).
  PYTHONPATH=. python day5_corrector_check.py --sens   # sensitivity check
  PYTHONPATH=. python day5_corrector_check.py --resp   # capped-correction response
  PYTHONPATH=. python day5_corrector_check.py --loop   # short A/B/C closed loops
DECLARED before results: dnorm = FIRST-STEP correction norm only (delta
in R^6 acts on u0; source: Corrector.d); margins in rad (config-space
corridor), inputs rad/s^2 (virtual acceleration), identity normalisation.
FROZEN: per-step per-joint cap ||delta||_inf <= 2.0 (=0.1*u_lim,
pre-declared candidate, validity CHECKED by --resp, not assumed);
final input |u0+delta| <= u_lim; B and C share QP, scale, cap — only the
margin source differs. True-rollout margins here are POST-HOC measurement
for all arms (execution rules unchanged); C's margin source remains its
definition. eta=1e3 unchanged until checks say otherwise.
OUT = results/day5_corr/"""
import argparse, json, os, sys
import numpy as np
import cvxpy as cp

sys.path.insert(0, ".")
from exec_interface import get_ps, nominal_AB, LinearResidual, ZeroResidual, \
    make_controller
from day2_diagnosis import model_rollout, true_rollout, margins_from_states
from day3_solve_diag import margins6

OUT = "results/day5_corr"; os.makedirs(OUT, exist_ok=True)
CAP = 2.0; ETA = 1e3; SNAP_T = (2, 5, 8); SEEDS = (777, 300, 301)


def v2_res():
    d = np.load("results/day4_excite/model_ls_v2.npz")
    return LinearResidual(d["dA"], d["dB"])


class CappedCorrector:
    """Same QP as block 2.2 plus the frozen per-joint cap."""
    def __init__(self, ps, Ad, Bd, N):
        m = ps.config_dim
        self.m = m; self.u_lim = ps.u_amp_nom
        self.d = cp.Variable(m); self.xi = cp.Variable(N + 1)
        self.mpar = cp.Parameter(N + 1); self.Apar = cp.Parameter((N + 1, m))
        self.u0 = cp.Parameter(m)
        obj = 0.5 * cp.quad_form(self.d, np.eye(m) * 1e-3) \
            + 0.5 * ETA * cp.sum_squares(self.xi)
        con = [self.mpar + self.Apar @ self.d + self.xi >= 0, self.xi >= 0,
               cp.norm_inf(self.d) <= CAP,
               self.u0 + self.d <= self.u_lim, self.u0 + self.d >= -self.u_lim]
        self.prob = cp.Problem(cp.Minimize(obj), con)
        self.chain = []; M = np.eye(2 * m)
        for k in range(N + 1):
            self.chain.append(M[:m] @ Bd if k > 0 else np.zeros((m, m)))
            M = Ad @ M

    def sens_matrix(self, grads):
        return np.array([grads[k] @ self.chain[k] for k in range(len(grads))])

    def solve(self, margins, grads, u0):
        self.mpar.value = margins
        self.Apar.value = self.sens_matrix(grads)
        self.u0.value = u0
        try:
            self.prob.solve(solver=cp.CLARABEL)
            if self.d.value is None:
                return np.zeros(self.m), 0.0
            return self.d.value, float(np.abs(self.xi.value).sum())
        except Exception:
            return np.zeros(self.m), 0.0


def episode_ctx(ps, seed, res):
    from aux import (TimeStepIntegratorContinuous, interpolate_equidistant,
                     compute_goal_state)
    from corridor_simulators.nom import NomSimulator
    from examples.six_dof.world import DemoWorld
    np.random.seed(seed)
    dn, de = ps.get_nominal_dynamics(), ps.get_err_dyn_random()
    it = TimeStepIntegratorContinuous(dt=ps.dt); it.set_dyns(dn, de)
    w = DemoWorld(); cont = make_controller(ps, res)
    sim = NomSimulator(cont, it, w)
    path = interpolate_equidistant(w.get_demo_path(), delta=0.05)
    return dict(dn=dn, de=de, it=it, w=w, cont=cont, sim=sim, path=path,
                radii=w.sdf(path[:, :3]), p_g=path[-1],
                pt=interpolate_equidistant(path.copy(), delta=0.001))


def run(ps, seed, arm, corr_cls=CappedCorrector, hook=None):
    from aux import compute_goal_state
    res = v2_res(); c = episode_ctx(ps, seed, res)
    m = ps.config_dim; A, B = nominal_AB(ps)
    cont, sim, it = c["cont"], c["sim"], c["it"]
    corr = corr_cls(ps, A, B, cont.N) if arm in "BC" else None
    x = np.r_[c["path"][0][:m], np.zeros(m)]
    ptraj = x[:m][None].repeat(cont.N + 1, axis=0)
    log = []
    for ts in range(40):
        cs, rs = sim.get_corridor_balls(ptraj, c["path"], c["radii"], c["p_g"])
        xg, _ = compute_goal_state(cs, rs, c["pt"], c["p_g"], return_index=True)
        cont.set_parameter_values(x, cs.T, rs, xg)
        ok = cont.solve()
        row = dict(ts=ts, raw=str(getattr(cont.problem, "status", "EXC")),
                   true_m0=float(rs[0] - np.linalg.norm(x[:m] - cs[0])))
        if not ok:
            row["failed"] = True; log.append(row)
            return log, ts
        u_pl = cont.U.value; u0 = u_pl[:, 0].copy(); d = np.zeros(m)
        if corr is not None:
            if arm == "B":
                Xp = model_rollout(x, u_pl, cont.A, cont.B, ZeroResidual())
            else:
                Xp = true_rollout(x, u_pl, c["dn"], c["de"], ps.dt)
            mg, gr = margins6(Xp[:m], cs.T, rs)
            d, sl = corr.solve(mg, gr, u0)
            u0 = np.clip(u0 + d, -ps.u_amp_nom, ps.u_amp_nom)
            row.update(dnorm_first=float(np.linalg.norm(d)),
                       sat=int(np.sum(np.abs(u0) >= ps.u_amp_nom - 1e-9)),
                       slack=sl)
        if hook and ts in SNAP_T:
            hook(dict(ts=ts, x=x.copy(), cs=cs.T.copy(),
                      rs=np.asarray(rs).copy(), U=u_pl.copy(), u0=u0.copy(),
                      ctx=c))
        log.append(row)
        x = np.array(it.solve_time_step(x, u0), float)
        ptraj = sim.get_shifted_trajectory(cont.X.value)[:, :m]
    return log, None


def cmd_sens(ps):
    """FD check of the corrector's sensitivity rows at frozen snapshots:
    sign + magnitude of dm0..m3/du0 vs the nominal-chain analytic A."""
    A, B = nominal_AB(ps); m = ps.config_dim; eps = 0.5
    snaps = []
    run(ps, 777, "A", hook=lambda s: snaps.append(s))
    out = []
    for s in snaps:
        corr = CappedCorrector(ps, A, B, s["U"].shape[1])
        Xt = true_rollout(s["x"], s["U"], s["ctx"]["dn"], s["ctx"]["de"], ps.dt)
        mg0, gr = margins6(Xt[:m], s["cs"], s["rs"])
        Aan = corr.sens_matrix(gr)
        rows = []
        for i in range(m):
            Up = s["U"].copy(); Up[i, 0] += eps
            Xp = true_rollout(s["x"], Up, s["ctx"]["dn"], s["ctx"]["de"], ps.dt)
            mgp, _ = margins6(Xp[:m], s["cs"], s["rs"])
            fd = (mgp - mg0) / eps
            for k in (2, 5, 10):
                rows.append(dict(joint=i, k=k, fd=float(fd[k]),
                                 an=float(Aan[k, i])))
        agree = [r for r in rows if abs(r["fd"]) > 1e-5]
        sgn = np.mean([np.sign(r["fd"]) == np.sign(r["an"]) for r in agree]) \
            if agree else float("nan")
        mag = np.median([abs(r["an"]) / abs(r["fd"]) for r in agree]) \
            if agree else float("nan")
        print(f"snap t={s['ts']}: sign agreement {sgn:.2f} "
              f"(n={len(agree)}), |analytic/FD| median {mag:.2f}")
        out.append(dict(ts=s["ts"], rows=rows))
    json.dump(out, open(f"{OUT}/sens.json", "w"), indent=2)


def cmd_resp(ps):
    """Capped correction at frozen snapshots: predicted vs TRUE margin
    change (direction + magnitude), next-solve feasibility untouched."""
    A, B = nominal_AB(ps); m = ps.config_dim
    snaps = []
    run(ps, 777, "A", hook=lambda s: snaps.append(s))
    for s in snaps:
        corr = CappedCorrector(ps, A, B, s["U"].shape[1])
        for src in ("model", "true"):
            if src == "model":
                cont = s["ctx"]["cont"]
                Xp = model_rollout(s["x"], s["U"], cont.A, cont.B,
                                   ZeroResidual())
            else:
                Xp = true_rollout(s["x"], s["U"], s["ctx"]["dn"],
                                  s["ctx"]["de"], ps.dt)
            mg, gr = margins6(Xp[:m], s["cs"], s["rs"])
            d, sl = corr.solve(mg, gr, s["U"][:, 0])
            pred_dm = corr.sens_matrix(gr) @ d
            Uc = s["U"].copy(); Uc[:, 0] += d
            Xc = true_rollout(s["x"], Uc, s["ctx"]["dn"], s["ctx"]["de"], ps.dt)
            mgc, _ = margins6(Xc[:m], s["cs"], s["rs"])
            Xt = true_rollout(s["x"], s["U"], s["ctx"]["dn"], s["ctx"]["de"],
                              ps.dt)
            mgt, _ = margins6(Xt[:m], s["cs"], s["rs"])
            true_dm = mgc - mgt
            i = int(np.argmax(np.abs(pred_dm)))
            print(f"t={s['ts']} src={src:5s} |d|inf={np.abs(d).max():.3f} "
                  f"slack={sl:.1e}  worst-k pred_dm={pred_dm[i]:+.2e} "
                  f"true_dm={true_dm[i]:+.2e}  min(true m) "
                  f"{mgt.min():+.4f}->{mgc.min():+.4f}")


def cmd_loop(ps):
    res_rows = {}
    for arm in "ABC":
        for sd in SEEDS:
            log, fail = run(ps, sd, arm)
            key = f"{arm}{sd}"
            dn = [r.get("dnorm_first", 0.0) for r in log]
            res_rows[key] = dict(fail=fail, steps=len(log),
                                 dnorm_med=float(np.median(dn)),
                                 dnorm_max=float(np.max(dn)),
                                 sat=sum(r.get("sat", 0) for r in log),
                                 min_true_m0=float(min(r["true_m0"]
                                                       for r in log)))
            print(f"{arm} seed {sd}: fail@{fail}  dnorm_first med/max "
                  f"{res_rows[key]['dnorm_med']:.3f}/"
                  f"{res_rows[key]['dnorm_max']:.3f}  sat {res_rows[key]['sat']}"
                  f"  min true m0 {res_rows[key]['min_true_m0']:+.4f}")
    json.dump(res_rows, open(f"{OUT}/loops.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("sens", "resp", "loop"):
        ap.add_argument(f"--{f}", action="store_true")
    a = ap.parse_args(); ps = get_ps()
    if a.sens: cmd_sens(ps)
    if a.resp: cmd_resp(ps)
    if a.loop: cmd_loop(ps)
