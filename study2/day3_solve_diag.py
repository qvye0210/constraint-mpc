#!/usr/bin/env python3
"""day3_solve_diag.py -- one question: why does the learned model make a
solvable problem infeasible? Repo root, env robmpc.
  PYTHONPATH=. python day3_solve_diag.py --capture   # 0-45 min phase
  PYTHONPATH=. python day3_solve_diag.py --swap      # 45-120 phase part 1
  PYTHONPATH=. python day3_solve_diag.py --relax     # 45-120 phase part 2
  PYTHONPATH=. python day3_solve_diag.py --coverage  # 120-180 phase
Rules honoured: 6-d margins fixed; raw solver status saved (mathematical
infeasibility vs numerical failure distinguished); relaxation is for
LOCALISATION only (dynamics equality always kept; relaxed success is not
a method gain and one relaxed group does not prove a unique root cause).
OUT = results/day3_diag/"""
import argparse, json, os, sys
import numpy as np
import cvxpy as cp

sys.path.insert(0, ".")
from exec_interface import get_ps, nominal_AB, LinearResidual, ZeroResidual, \
    make_controller
from day2_diagnosis import Corrector, model_rollout, true_rollout, load_res

OUT = "results/day3_diag"; os.makedirs(OUT, exist_ok=True)


def margins6(X, cs, rs):
    """CORRECTED: full config-space corridor margins + gradients."""
    dif = X - cs                                   # (m, N+1)
    dist = np.linalg.norm(dif, axis=0)
    g = -(dif / np.maximum(dist[None], 1e-9)).T    # (N+1, m)
    return rs - dist, g


def run_to_failure(ps, seed, arm, res):
    """Learned-model episode with 6-d corrector margins; logs every step."""
    from aux import (TimeStepIntegratorContinuous, interpolate_equidistant,
                     compute_goal_state)
    from corridor_simulators.nom import NomSimulator
    from examples.six_dof.world import DemoWorld
    A, B = nominal_AB(ps); m = ps.config_dim
    np.random.seed(seed)
    dn, de = ps.get_nominal_dynamics(), ps.get_err_dyn_random()
    it = TimeStepIntegratorContinuous(dt=ps.dt); it.set_dyns(dn, de)
    w = DemoWorld(); cont = make_controller(ps, res)
    sim = NomSimulator(cont, it, w)
    path = interpolate_equidistant(w.get_demo_path(), delta=0.05)
    radii = w.sdf(path[:, :3]); p_g = path[-1]
    pt = interpolate_equidistant(path.copy(), delta=0.001)
    x = np.r_[path[0][:m], np.zeros(m)]
    ptraj = x[:m][None].repeat(cont.N + 1, axis=0)
    corr = Corrector(ps, A, B, cont.N) if arm in "BC" else None
    steps = []
    for ts in range(40):
        cs, rs = sim.get_corridor_balls(ptraj, path, radii, p_g)
        xg, _ = compute_goal_state(cs, rs, pt, p_g, return_index=True)
        cont.set_parameter_values(x, cs.T, rs, xg)
        ok = cont.solve()
        raw = str(getattr(cont.problem, "status", "EXC"))
        if not ok:
            snap = dict(step=ts, x=x.tolist(), cs=cs.T.tolist(),
                        rs=np.asarray(rs).tolist(), xg=np.asarray(xg).tolist(),
                        raw_status=raw, arm=arm)
            return steps, snap
        u0 = cont.U.value[:, 0].copy(); d = np.zeros(m); sl = 0.0
        if corr is not None:
            if arm == "B":
                Xp = model_rollout(x, cont.U.value, cont.A, cont.B,
                                   ZeroResidual())
            else:
                Xp = true_rollout(x, cont.U.value, dn, de, ps.dt)
            mg, gr = margins6(Xp[:m], cs.T, rs)      # 6-d, fixed
            d, sl = corr.solve(mg, gr, u0)
            u0 = np.clip(u0 + d, -ps.u_amp_nom, ps.u_amp_nom)
        steps.append(dict(ts=ts, x=x.tolist(), u_exec=u0.tolist(),
                          dnorm=float(np.linalg.norm(d)), slack=sl,
                          raw=raw))
        x = np.array(it.solve_time_step(x, u0), float)
        ptraj = sim.get_shifted_trajectory(cont.X.value)[:, :m]
    return steps, None


def build_problem(ps, A, B, snap, groups):
    """Exact NominalController constraint set, group-selectable.
    groups: subset of {dyn(always on), ubox, pbox, term, corridor, vbox,
    start(always on)}."""
    m = ps.config_dim; N = 20
    Q = np.eye(2 * m) * 10; Q[m:, m:] *= .01
    X = cp.Variable((2 * m, N + 1)); U = cp.Variable((m, N))
    x0 = np.array(snap["x"]); cs = np.array(snap["cs"])
    rs = np.array(snap["rs"]); xg = np.array(snap["xg"])
    obj = sum(cp.quad_form(X[:, i] - X[:, -1], Q)
              + cp.quad_form(U[:, i], np.eye(m) * 1e-3) for i in range(N))
    obj += cp.quad_form(X[:, -1] - xg, np.eye(2 * m) * 1e4)
    con = [X[:, 1:] == A @ X[:, :-1] + B @ U, X[:, 0] == x0]
    if "ubox" in groups:
        con += [U <= ps.u_amp_nom, U >= -ps.u_amp_nom]
    if "pbox" in groups:
        con += [X[:m, :] <= np.pi, X[:m, :] >= -np.pi]
    if "term" in groups:
        con += [X[m:, -1] == 0, U[:, -1] == 0]
    if "corridor" in groups:
        con += [cp.norm(X[:m] - cs, axis=0) <= rs]
    if "vbox" in groups:
        con += [X[m:, :] <= ps.v_amp_nom, X[m:, :] >= -ps.v_amp_nom]
    p = cp.Problem(cp.Minimize(obj), con)
    try:
        p.solve(solver=cp.CLARABEL)
        return p.status
    except Exception as e:
        return f"EXC:{type(e).__name__}"


ALL = {"ubox", "pbox", "term", "corridor", "vbox"}


def main():
    ap = argparse.ArgumentParser()
    for f in ("capture", "swap", "relax", "coverage"):
        ap.add_argument(f"--{f}", action="store_true")
    a = ap.parse_args(); ps = get_ps(); res = load_res()
    A, B = nominal_AB(ps)
    ld = np.load("results/day2_dev_data/model_ls.npz")
    Al, Bl = A + ld["dA"], B + ld["dB"]

    if a.capture:
        allsteps = {}
        for arm in "ABC":
            steps, snap = run_to_failure(ps, 777, arm, res)
            allsteps[arm] = dict(steps=steps, snap=snap)
            print(f"arm {arm}: failed@{snap['step'] if snap else None} "
                  f"raw={snap['raw_status'] if snap else '-'}")
        json.dump(allsteps, open(f"{OUT}/capture_777.json", "w"), indent=2)
        # same-problem check: state trajectories identical up to failure?
        for t in range(min(len(allsteps[k]["steps"]) for k in "ABC")):
            xs = [np.array(allsteps[k]["steps"][t]["x"]) for k in "ABC"]
            us = [np.array(allsteps[k]["steps"][t]["u_exec"]) for k in "ABC"]
            print(f"t={t} max|x_A-x_B|={np.max(np.abs(xs[0]-xs[1])):.2e} "
                  f"|x_A-x_C|={np.max(np.abs(xs[0]-xs[2])):.2e} "
                  f"dnorm B={allsteps['B']['steps'][t]['dnorm']:.4f} "
                  f"C={allsteps['C']['steps'][t]['dnorm']:.4f} "
                  f"max|u_A-u_C|={np.max(np.abs(us[0]-us[2])):.2e}")

    if a.swap:
        snap = json.load(open(f"{OUT}/capture_777.json"))["A"]["snap"]
        for nm, Am, Bm in (("nominal", A, B), ("learned", Al, Bl)):
            st = build_problem(ps, Am, Bm, snap, ALL)
            print(f"swap {nm:8s} (full constraint set): {st}")

    if a.relax:
        snap = json.load(open(f"{OUT}/capture_777.json"))["A"]["snap"]
        print("learned matrices, dynamics equality KEPT, one group "
              "removed at a time (localisation only):")
        for g in sorted(ALL):
            st = build_problem(ps, Al, Bl, snap, ALL - {g})
            print(f"  without {g:8s}: {st}")
        for pair in (("term",), ("term", "vbox"), ("corridor", "term")):
            st = build_problem(ps, Al, Bl, snap, ALL - set(pair))
            print(f"  without {'+'.join(pair):16s}: {st}")

    if a.coverage:
        snap = json.load(open(f"{OUT}/capture_777.json"))["A"]["snap"]
        d = np.load("results/day2_dev_data/train.npz")
        x = np.array(snap["x"]); m = ps.config_dim
        Xtr = d["X"]
        print(f"train |v| max {np.abs(Xtr[:, m:]).max():.3f}  "
              f"failure-state |v| max {np.abs(x[m:]).max():.3f}")
        for i in range(2 * m):
            lo, hi = Xtr[:, i].min(), Xtr[:, i].max()
            inside = lo <= x[i] <= hi
            print(f"  dim{i:2d}: fail {x[i]:+.3f}  train [{lo:+.3f},"
                  f"{hi:+.3f}]  {'IN' if inside else 'OUT'}")
        r_here = LinearResidual(ld["dA"], ld["dB"])(x, np.zeros(m))
        print(f"learned residual at failure state (u=0): |pos| max "
              f"{np.abs(r_here[:m]).max():.2e} |vel| max "
              f"{np.abs(r_here[m:]).max():.2e}")


if __name__ == "__main__":
    main()
