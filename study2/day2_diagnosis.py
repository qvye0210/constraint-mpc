#!/usr/bin/env python3
"""day2_diagnosis.py -- block 2.2 paired improvement-space diagnosis.
Repo root, env robmpc:
  PYTHONPATH=. python day2_diagnosis.py --failcase   # section 2: earliest ls_mse failure record
  PYTHONPATH=. python day2_diagnosis.py --check      # 3-arm difference sanity (2 episodes)
  PYTHONPATH=. python day2_diagnosis.py --eval       # 50 dev episodes, paired (background this)
ARMS (shared plant, task, candidate actions, sensitivity A, R, eta, C,
correction count, failure rule; ONLY the margin source differs):
  A: MSE controller, no corrector
  B: + soft corrector, MODEL-predicted margins (MSE rollout of the plan)
  C: + SAME corrector, TRUE margins (independent rollout, same mismatch
     realisation, same candidate actions; diagnostic reference only —
     true params generate labels, never enter any learner/Jacobian)
Corrector (frozen minimal version, first action only):
  V(m) = min 0.5 d'Rd + 0.5 eta ||xi||^2
         s.t. m + A d + xi >= 0, xi >= 0, |u0+d| <= u_lim
  A_k = g_k' [Ad^{k-1} B]_pos  (nominal, deployment-available)
  R = 1e-3 I, eta = 1e3, one correction per executed step.
Margins = controller corridor: m_k = rs_k - ||x_pos_k - cs_k|| (optimiser
constraint definition; learned-sdf and gt collision recorded separately).
Failure rule: MPC infeasible -> episode ends for every arm (no backup).
Eval seeds 300-349 (cross-parameter draws); test 900-909 stay sealed.
OUT = results/day22_diag/"""
import argparse, json, os, sys, time
import numpy as np
import cvxpy as cp

sys.path.insert(0, ".")
from exec_interface import (get_ps, nominal_AB, LinearResidual,
                            ZeroResidual, make_controller)

OUT = "results/day22_diag"; os.makedirs(OUT, exist_ok=True)
EVAL_SEEDS = list(range(300, 350)); T_MAX = 300; ETA = 1e3


def load_res():
    d = np.load("results/day2_dev_data/model_ls.npz")
    return LinearResidual(d["dA"], d["dB"])


class Corrector:
    def __init__(self, ps, Ad, Bd, N):
        m = ps.config_dim
        self.Ad, self.Bd, self.N, self.m = Ad, Bd, N, m
        self.u_lim = ps.u_amp_nom
        self.d = cp.Variable(m); self.xi = cp.Variable(N + 1)
        self.mpar = cp.Parameter(N + 1); self.Apar = cp.Parameter((N + 1, m))
        self.u0 = cp.Parameter(m)
        obj = 0.5 * cp.quad_form(self.d, np.eye(m) * 1e-3) \
            + 0.5 * ETA * cp.sum_squares(self.xi)
        con = [self.mpar + self.Apar @ self.d + self.xi >= 0, self.xi >= 0,
               self.u0 + self.d <= self.u_lim,
               self.u0 + self.d >= -self.u_lim]
        self.prob = cp.Problem(cp.Minimize(obj), con)
        # sensitivity chain [Ad^{k-1} B]_pos rows filled per-step with g_k
        self.chain = []
        M = np.eye(2 * m)
        for k in range(N + 1):
            self.chain.append(M[:m] @ Bd if k > 0 else np.zeros((m, m)))
            M = Ad @ M

    def solve(self, margins, grads, u0):
        A = np.array([grads[k] @ self.chain[k] for k in range(len(margins))])
        self.mpar.value = margins; self.Apar.value = A; self.u0.value = u0
        try:
            self.prob.solve(solver=cp.CLARABEL)
            if self.d.value is None:
                return np.zeros(self.m), 0.0
            return self.d.value, float(np.abs(self.xi.value).sum())
        except Exception:
            return np.zeros(self.m), 0.0


def margins_from_states(Xpos, cs, rs):
    dif = Xpos - cs
    dist = np.linalg.norm(dif, axis=0)
    g = np.where(dist[None] > 1e-9, -dif / np.maximum(dist[None], 1e-9), 0.0)
    return rs - dist, g.T          # (N+1,), (N+1, m) grad wrt position


def model_rollout(x, U, A, B, res):
    X = [x]
    for k in range(U.shape[1]):
        X.append(A @ X[-1] + B @ U[:, k] + res(X[-1], U[:, k]))
    return np.array(X).T


def true_rollout(x, U, dyn_nom, dyn_err, dt):
    from aux import TimeStepIntegratorContinuous
    it = TimeStepIntegratorContinuous(dt=dt)     # independent instance
    it.set_dyns(dyn_nom, dyn_err)
    X = [np.array(x, float)]
    for k in range(U.shape[1]):
        X.append(np.array(it.solve_time_step(X[-1], U[:, k]), float))
    return np.array(X).T


def run_arm(ps, seed, arm, res, log_first_fail=False):
    from aux import (TimeStepIntegratorContinuous, interpolate_equidistant)
    from corridor_simulators.nom import NomSimulator
    from examples.six_dof.world import DemoWorld
    A, B = nominal_AB(ps); m = ps.config_dim
    np.random.seed(seed)
    dyn_nom = ps.get_nominal_dynamics(); dyn_err = ps.get_err_dyn_random()
    integ = TimeStepIntegratorContinuous(dt=ps.dt)
    integ.set_dyns(dyn_nom, dyn_err)
    world = DemoWorld()
    cont = make_controller(ps, res)
    sim = NomSimulator(cont, integ, world)       # helper methods only
    path = interpolate_equidistant(world.get_demo_path(), delta=0.05)
    radii = world.sdf(path[:, :3])
    p_g = path[-1]; x_g = np.r_[p_g[:m], np.zeros(m)]
    path_track = interpolate_equidistant(path.copy(), delta=0.001)
    corr = Corrector(ps, A, B, cont.N) if arm in "BC" else None
    x = np.r_[path[0][:m], np.zeros(m)]
    path_traj = x[:m][None].repeat(cont.N + 1, axis=0)
    rec = dict(status="timeout", ts=T_MAX, coll=0, slack=0.0,
               dnorm=[], solve_s=0.0, sim_s=0.0)
    from aux import compute_goal_state
    for ts in range(T_MAX):
        cs, rs = sim.get_corridor_balls(path_traj, path, radii, p_g)
        x_g_v, _ = compute_goal_state(cs, rs, path_track, p_g,
                                      return_index=True)
        cont.set_parameter_values(x, cs.T, rs, x_g_v)
        t0 = time.time()
        ok = cont.solve()
        rec["solve_s"] += time.time() - t0
        if not ok:
            rec["status"] = "infeasible"; rec["ts"] = ts
            if log_first_fail:
                rec["fail_detail"] = dict(
                    step=ts, raw=str(getattr(cont.problem, "status", "?")),
                    x=x.tolist(),
                    margins_pred=(rs - np.linalg.norm(
                        path_traj.T[:3] - cs.T[:3], axis=0)).tolist()[:5])
            break
        Upl = cont.U.value; u0 = Upl[:, 0].copy()
        if corr is not None:
            if arm == "B":
                Xp = model_rollout(x, Upl, cont.A, cont.B, ZeroResidual())
            else:
                t1 = time.time()
                Xp = true_rollout(x, Upl, dyn_nom, dyn_err, ps.dt)
                rec["sim_s"] += time.time() - t1
            mg, gr = margins_from_states(Xp[:3], cs.T[:3], rs)
            gfull = np.zeros((len(mg), m)); gfull[:, :3] = gr
            d, sl = corr.solve(mg, gfull, u0)
            rec["slack"] += sl; rec["dnorm"].append(float(np.linalg.norm(d)))
            u0 = np.clip(u0 + d, -ps.u_amp_nom, ps.u_amp_nom)
        if not world.is_collision_free_gt(x[:3]):
            rec["coll"] += 1
        x = np.array(integ.solve_time_step(x, u0), float)
        Xpred = cont.X.value
        path_traj = sim.get_shifted_trajectory(Xpred)[:, :m]
        if np.linalg.norm(x - x_g) < 0.01:
            rec["status"] = "success"; rec["ts"] = ts + 1
            break
    rec["dnorm_med"] = float(np.median(rec["dnorm"])) if rec["dnorm"] else 0.0
    rec.pop("dnorm")
    return rec


def main():
    ap = argparse.ArgumentParser()
    for f in ("failcase", "check", "eval"):
        ap.add_argument(f"--{f}", action="store_true")
    a = ap.parse_args(); ps = get_ps(); res = load_res()
    if a.failcase:
        r = run_arm(ps, 777, "A", res, log_first_fail=True)
        json.dump(r, open(f"{OUT}/failcase_777.json", "w"), indent=2)
        print(json.dumps(r, indent=2))
    if a.check:
        for sd in (300, 301):
            row = {arm: run_arm(ps, sd, arm, res) for arm in "ABC"}
            print(sd, {k: (v["status"], v["ts"], round(v["dnorm_med"], 4))
                       for k, v in row.items()})
    if a.eval:
        rows = []
        for sd in EVAL_SEEDS:
            row = {"seed": sd}
            for arm in "ABC":
                t0 = time.time()
                r = run_arm(ps, sd, arm, res)
                r["wall"] = time.time() - t0
                row[arm] = r
            rows.append(row)
            print(f"seed {sd}: " + "  ".join(
                f"{k}:{row[k]['status'][:4]}@{row[k]['ts']}" for k in "ABC"),
                flush=True)
        json.dump(rows, open(f"{OUT}/eval50.json", "w"), indent=2)
        succ = {k: sum(r[k]["status"] == "success" for r in rows) for k in "ABC"}
        ttg = {k: float(np.mean([r[k]["ts"] if r[k]["status"] == "success"
                                 else T_MAX for r in rows])) for k in "ABC"}
        coll = {k: sum(r[k]["coll"] for r in rows) for k in "ABC"}
        sim_s = float(np.mean([r["C"]["sim_s"] for r in rows]))
        print(f"\nsuccess {succ}  time-to-goal(T_max-imputed) "
              f"{ {k: round(v,1) for k,v in ttg.items()} }  gt_coll {coll}")
        print(f"C-arm true-rollout query cost: {sim_s:.2f} s/episode "
              f"(diagnostic reference, not deployable)")


if __name__ == "__main__":
    main()
