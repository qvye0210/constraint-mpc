"""
day7_fixed_margin.py — planner-internal fixed velocity margin baseline (repo root, PYTHONPATH=.)

Spec frozen 2026-09-22 (Dolly):
  |v_0| <= v_lim (author vbox kept for all k)  ;  |v_k| <= v_lim - b for k = 1..H (added)
  b = Q0.99{ s_t },  s_t = max_{1<=k<=H} || v_true_{t+k} - v_nom_{t+k|t} ||_inf   on DEV windows only
  b common to all arms, frozen after --calib.  Engineering baseline, no safety-guarantee claim.
  Solve failures classified: F1 executed |v0| > v_lim ; F3 v_lim-b < |v0| <= v_lim ; F2 |v0| <= v_lim-b.

Stages:
  --calib   b from dev data (day2 dev.npz [ep/step] + day4 dev.npz [continuity rule]); writes b.json
  --verify  b=0 regression: added constraints inert, nom reproduces fail@10-11 on 777/300/301
  --test    seeds 900-909 (READ ONCE), arms {nom, nom+b, ls, ls+b}; success / t2g / |v| overshoot
  --slack   deterministic replay of day6 C arm (ExtCorrector) with per-step post-correction margins
  --gt      author is_collision_free_gt on executed states of all traj_*.npz
"""
import argparse, json, os, sys, time
import numpy as np
import cvxpy as cp
sys.path.insert(0, ".")

from exec_interface import get_ps, ZeroResidual
from day5_corrector_check import episode_ctx, v2_res
from aux import compute_goal_state, get_linear_double_integrator_discrete_dynamics

OUT = "results/day7_margin"
os.makedirs(OUT, exist_ok=True)
GOAL_TOL, MAX_STEPS = 1e-2, 500
TEST_SEEDS = list(range(900, 910))
VERIFY_SEEDS = [777, 300, 301]


# ---------------------------------------------------------------- helpers
def tighten(cont, b, m):
    """Add |v_k| <= v_lim-b for k>=1 only. Author vbox (all k) untouched. Returns cont."""
    cons = list(cont.problem.constraints)
    n0 = len(cons)
    if b > 0:
        v_lim = cont._v_lim
        cons += [cont.X[m:, 1:] <= v_lim - b, cont.X[m:, 1:] >= -(v_lim - b)]
        # assert column 0 excluded from the added constraints
        for c_ in cons[n0:]:
            assert c_.shape == (m, cont.N), f"tightening shape {c_.shape} != ({m},{cont.N})"
    cont.problem = cp.Problem(cont.problem.objective, cons)
    cont._n_added = len(cons) - n0
    return cont


def make_ctx(ps, seed, res, b):
    c = episode_ctx(ps, seed, res)
    cont = c["cont"]
    cont._v_lim = float(ps.v_amp_nom)
    tighten(cont, b, ps.config_dim)
    return c


def classify_fail(x, m, v_lim, b):
    a = np.abs(x[m:]).max()
    if a > v_lim:
        return "F1_executed_v_over_limit"
    if b > 0 and a > v_lim - b:
        return "F3_v0_in_buffer_plan_infeasible"
    return "F2_v0_inside_tightened_plan_infeasible"


def run_episode(ps, seed, res, b, log_cb=None):
    """Closed loop identical to Dolly's day6 snippet; returns dict + executed trajectory."""
    m = ps.config_dim
    v_lim = float(ps.v_amp_nom)
    c = make_ctx(ps, seed, res, b)
    cont, sim, it = c["cont"], c["sim"], c["it"]
    x = np.r_[c["path"][0][:m], np.zeros(m)]
    x_g = np.r_[c["p_g"], np.zeros(m)]
    ptraj = x[:m][None].repeat(cont.N + 1, axis=0)
    xs, us, statuses = [x.copy()], [], []
    max_over = 0.0  # executed |v| - v_lim, positive = violation
    rec = dict(seed=seed, b=b, status="truncated", t=None, fail_class=None)
    for ts in range(MAX_STEPS):
        cs, rs = sim.get_corridor_balls(ptraj, c["path"], c["radii"], c["p_g"])
        xg, _ = compute_goal_state(cs, rs, c["pt"], c["p_g"], return_index=True)
        cont.set_parameter_values(x, cs.T, rs, xg)
        ok = cont.solve()
        statuses.append(getattr(cont, "status", "exception"))
        if not ok:
            rec.update(status="infeasible", t=ts, raw_status=statuses[-1],
                       fail_class=classify_fail(x, m, v_lim, b),
                       v0_abs_max=float(np.abs(x[m:]).max()))
            break
        Xm, Um = cont.X.value, cont.U.value
        u = Um[:, 0].copy()
        if log_cb is not None:
            u = log_cb(ts, x, Xm, Um, cont, c)  # slack stage may replace u
        x = np.array(it.solve_time_step(x, u), float)
        xs.append(x.copy()); us.append(u.copy())
        max_over = max(max_over, float(np.abs(x[m:]).max() - v_lim))
        ptraj = sim.get_shifted_trajectory(cont.X.value)[:, :m]
        if np.linalg.norm(x - x_g) < GOAL_TOL:
            rec.update(status="success", t=ts + 1)
            break
    rec.update(max_exec_v_over_limit=max_over, n_added_cons=cont._n_added)
    return rec, np.array(xs), np.array(us), statuses


# ---------------------------------------------------------------- --calib
def windows_from(X, U, Xn, m, H, A, B, ep=None, tol=1e-12):
    """Yield s_t per H-step window. Episode boundary: ep[] if given, else Xn[i]==X[i+1]."""
    n = len(X)
    if ep is None:
        same = np.array([np.allclose(Xn[i], X[i + 1], atol=tol, rtol=0) for i in range(n - 1)] + [False])
    else:
        same = np.array([ep[i] == ep[i + 1] for i in range(n - 1)] + [False])
    s = []
    for t in range(n):
        if t + H - 1 >= n or not same[t:t + H - 1].all():
            continue
        xk = X[t].copy(); smax = 0.0
        for k in range(1, H + 1):
            xk = A @ xk + B @ U[t + k - 1]
            v_true = Xn[t + k - 1][m:]
            smax = max(smax, float(np.abs(v_true - xk[m:]).max()))
        s.append(smax)
    return np.array(s)


def stage_calib(ps, H):
    m = ps.config_dim
    A, B = get_linear_double_integrator_discrete_dynamics(m, dt=ps.dt, method="zoh")
    d2 = np.load("results/day2_dev_data/dev.npz")
    # lock A,B against recorded one-step nominal prediction
    pred1 = (A @ d2["X"].T + B @ d2["U"].T).T
    err_ab = float(np.abs(pred1 - d2["Xn_nom"]).max())
    assert err_ab < 1e-9, f"A,B mismatch vs Xn_nom: {err_ab:.2e}"
    s2 = windows_from(d2["X"], d2["U"], d2["Xn"], m, H, A, B, ep=d2["ep"])
    d4 = np.load("results/day4_excite/dev.npz")
    s4 = windows_from(d4["X"], d4["U"], d4["Xn"], m, H, A, B, ep=None)
    s_all = np.r_[s2, s4]
    out = dict(H=H, v_lim=float(ps.v_amp_nom), AB_check=err_ab,
               n_windows=dict(day2=int(len(s2)), day4=int(len(s4)), merged=int(len(s_all))),
               q99=dict(day2=float(np.quantile(s2, .99)), day4=float(np.quantile(s4, .99)) if len(s4) else None,
                        merged=float(np.quantile(s_all, .99))),
               median=dict(day2=float(np.median(s2)), day4=float(np.median(s4)) if len(s4) else None),
               b=float(np.quantile(s_all, .99)), rule="Q0.99 of merged dev windows (day2 ft + day4 excite); FROZEN")
    json.dump(out, open(f"{OUT}/b.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    print(f"b / v_lim = {out['b'] / out['v_lim']:.4%}")


# ---------------------------------------------------------------- --verify
def stage_verify(ps):
    res = ZeroResidual()
    for seed in VERIFY_SEEDS:
        r0, x0, _, st0 = run_episode(ps, seed, res, 0.0)
        r1, x1, _, st1 = run_episode(ps, seed, res, 1e-12)  # near-zero b: constraints present but inert
        d = float(np.abs(x0[:len(x1)] - x1[:len(x0)]).max()) if len(x0) == len(x1) else np.inf
        print(f"seed {seed}: b=0 {r0['status']}@{r0['t']} [{r0['raw_status'] if 't' in r0 and r0['fail_class'] else ''}] "
              f"| b~0 {r1['status']}@{r1['t']} | max|dx| {d:.2e} | added cons {r1['n_added_cons']}")
        assert r0["status"] == "infeasible" and r0["t"] in (10, 11), "nom@10-11 not reproduced"
        assert r0["t"] == r1["t"] and d < 1e-6, "inert tightening changed the trajectory"
    print("VERIFY PASS")


# ---------------------------------------------------------------- --test
def stage_test(ps, b):
    assert not os.path.exists(f"{OUT}/test_done.flag"), "test seeds already consumed — read once rule"
    arms = {"nom": (ZeroResidual, 0.0), "nom+b": (ZeroResidual, b),
            "ls": (v2_res, 0.0), "ls+b": (v2_res, b)}
    table = {}
    for name, (mk, bb) in arms.items():
        rows = []
        for seed in TEST_SEEDS:
            r, xs, us, st = run_episode(ps, seed, mk(), bb)
            np.savez(f"{OUT}/traj_{name.replace('+', 'p')}_{seed}.npz", X=xs, U=us, status=np.array(st))
            rows.append(r)
            print(f"{name:6s} seed {seed}: {r['status']}@{r['t']} {r.get('fail_class') or ''} "
                  f"max_over {r['max_exec_v_over_limit']:+.2e}")
        succ = [x for x in rows if x["status"] == "success"]
        table[name] = dict(b=bb, success_rate=len(succ) / len(rows),
                           t2g_median=float(np.median([x["t"] for x in succ])) if succ else None,
                           t2g_all=[x["t"] if x["status"] == "success" else None for x in rows],
                           fail_classes={k: sum(x.get("fail_class") == k for x in rows)
                                         for k in ("F1_executed_v_over_limit", "F2_v0_inside_tightened_plan_infeasible",
                                                   "F3_v0_in_buffer_plan_infeasible")},
                           max_exec_v_over_limit=float(max(x["max_exec_v_over_limit"] for x in rows)),
                           truncated=sum(x["status"] == "truncated" for x in rows))
    json.dump(dict(b=b, seeds=TEST_SEEDS, arms=table,
                   note="gt collision check: run existing tool on traj_*.npz; author ft/rt not rerun"),
              open(f"{OUT}/test.json", "w"), indent=1)
    open(f"{OUT}/test_done.flag", "w").write(time.ctime())
    print(json.dumps(table, indent=1))


# ---------------------------------------------------------------- --slack
def stage_slack(ps):
    """Deterministic replay of day6 C arm (true-rollout margins) with per-step logging. No design change."""
    from day6_vel_margin import ExtCorrector
    from exec_interface import nominal_AB
    from day2_diagnosis import true_rollout
    m, v_lim, u_lim = ps.config_dim, float(ps.v_amp_nom), float(ps.u_amp_nom)
    A, B = nominal_AB(ps)
    log = []

    def c_arm_Xp(x, Upl, c):
        try:
            return true_rollout(x, Upl, c["dn"], c["de"], ps.dt)
        except KeyError:
            sys.exit(f"adapt c_arm_Xp: episode_ctx keys = {list(c.keys())}")

    def cb(ts, x, Xm, Um, cont, c):
        corr = cb.corr if hasattr(cb, "corr") else ExtCorrector(ps, A, B, cont.N)
        cb.corr = corr
        cs, rs = cont.cs.value, cont.rs.value
        u0 = Um[:, 0].copy()
        Xp = c_arm_Xp(x, Um, c)
        d, sl, mvec = corr.solve(Xp, cs, rs, u0)
        u_c = np.clip(u0 + d, -u_lim, u_lim)
        Uc = Um.copy(); Uc[:, 0] = u_c
        _, _, mvec_post = corr.solve(c_arm_Xp(x, Uc, c), cs, rs, u_c)  # margin vector of corrected plan
        log.append(dict(t=ts, dnorm=float(np.abs(d).max()), slack=float(sl),
                        min_margin_pre=float(np.min(mvec)), min_margin_post=float(np.min(mvec_post)),
                        plan_min_mv_minus=float((Xm[m:] + v_lim).min()), plan_min_mv_plus=float((v_lim - Xm[m:]).min())))
        return u_c

    summary = {}
    for seed in VERIFY_SEEDS:
        log.clear(); cb.__dict__.pop("corr", None)
        r, xs, us, st = run_episode(ps, seed, v2_res(), 0.0, log_cb=cb)
        for e, xn in zip(log, xs[1:]):
            e["true_next_min_vmargin"] = float((v_lim - np.abs(xn[m:])).min())
        act = [e for e in log if e["dnorm"] > 1e-9]
        neg_post = [e for e in act if e["min_margin_post"] < 0]
        neg_true = [e for e in act if e["true_next_min_vmargin"] < 0]
        summary[seed] = dict(status=r["status"], t=r["t"], fail_class=r.get("fail_class"),
                             steps_with_action=len(act), action_and_predicted_neg=len(neg_post),
                             action_and_true_neg=len(neg_true),
                             min_margin_post_over_actions=min((e["min_margin_post"] for e in act), default=None),
                             min_true_next_vmargin=min(e["true_next_min_vmargin"] for e in log))
        print(f"C seed {seed}: {json.dumps(summary[seed])}")
        json.dump(log, open(f"{OUT}/slack_{seed}.json", "w"), indent=1)
        np.savez(f"{OUT}/traj_slackC_{seed}.npz", X=xs, U=us, status=np.array(st))
    json.dump(summary, open(f"{OUT}/slack_summary.json", "w"), indent=1)


# ---------------------------------------------------------------- --gt
def stage_gt(ps):
    """Author ground-truth collision check on EXECUTED states of every traj_*.npz. sdf not used."""
    import glob
    from examples.six_dof.world import DemoWorld
    w = DemoWorld()
    per_arm = {}
    for f in sorted(glob.glob(f"{OUT}/traj_*.npz")):
        X = np.load(f)["X"]
        bad = int(sum(not w.is_collision_free_gt(x[:3]) for x in X))
        arm = os.path.basename(f).split("_")[1]
        per_arm.setdefault(arm, dict(files=0, states=0, violations=0))
        per_arm[arm]["files"] += 1; per_arm[arm]["states"] += len(X); per_arm[arm]["violations"] += bad
        if bad:
            print(f"VIOLATION {f}: {bad}/{len(X)} states")
    json.dump(per_arm, open(f"{OUT}/gt_check.json", "w"), indent=1)
    print(json.dumps(per_arm, indent=1))


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true"); ap.add_argument("--verify", action="store_true")
    ap.add_argument("--test", action="store_true"); ap.add_argument("--slack", action="store_true")
    ap.add_argument("--gt", action="store_true")
    a = ap.parse_args()
    ps = get_ps()
    H = episode_ctx(ps, 777, ZeroResidual())["cont"].N
    print(f"ps: config_dim={ps.config_dim} dt={ps.dt} v_lim={ps.v_amp_nom} u_lim={ps.u_amp_nom} H={H}")
    if a.calib: stage_calib(ps, H)
    if a.verify: stage_verify(ps)
    if a.slack: stage_slack(ps)
    if a.gt: stage_gt(ps)
    if a.test:
        b = json.load(open(f"{OUT}/b.json"))["b"]
        print(f"FROZEN b = {b:.5f}")
        stage_test(ps, b)
