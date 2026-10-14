"""
day13_reftrack.py — one task-integration fix: the joint-space MPC tracks the verified IK path as a TIME-INDEXED reference
instead of only a final joint goal. Repo root, PYTHONPATH=. ; env robmpc. Output: results/day13_reftrack/
Declared: this is a path-tracking extension built on the authors' platform, NOT the authors' task run as-is.
All later methods would share this controller.

FROZEN (no weight scan, no geometry change, acceptance unchanged from day12):
  objective  J = sum_{k=1..N} (q_k - q_ref(s(t_{t+k})))^T Qq (.)  +  sum_k u_k^T R u_k
             Qq = author position weight (10 I), R = author input weight (1e-3 I); author goal terms removed
  constraints: all author constraints unchanged (dynamics, input box, pos box, terminal v_N = 0, u_N = 0, corridor, vbox)
             + velocity tightening b (k>=1) + linearised tool constraint with trust region 0.05 rad,
               exact-rho recheck, <= 2 re-linearisations, else plan rejected (identical to day12)
  timing s(t): quintic 10tau^3 - 15tau^4 + 6tau^5, tau = t/T; T = smallest multiple of dt such that along the IK path
             max|qdot_ref| <= 0.5 (v_lim - b) and max|qddot_ref| <= 0.5 u_lim   (rule fixed before any run)
  acceptance (day12): lateral <= 5 mm, axis <= 3 deg at the same progress sigma, end <= 5 mm, exact rho >= -TOL,
             plus: episode ends only after t >= T and |x - x_end| < 1e-2, max 500 steps
  extra logging: executed rho every step, tool-constraint dual values per step (active-set evidence), time-indexed
             joint tracking error |q_t - q_ref(s(t))|.
"""
import argparse, json, os, sys
import numpy as np
import cvxpy as cp
sys.path.insert(0, ".")
import day12_tooltask as d12
from day12_tooltask import (fk, Task, tool_metrics, A, B, b, ps, sim, M, TR, MAXRELIN, TOL, LAT_TOL, ANG_TOL, END_TOL,
                            GOAL_TOL, NSTEPS, MISMATCH_SEEDS, roll)
from controllers.nom_controller import NominalController
from aux import compute_goal_state, TimeStepIntegratorContinuous, TimeStepIntegratorDiscrete
from day8_reference_arms import find_dyns
from day7_fixed_margin import make_ctx
from exec_interface import ZeroResidual

OUT = "results/day13_reftrack"; os.makedirs(OUT, exist_ok=True)
SRC = "results/day12_tooltask"
N = 20
d12.TASK = Task(*json.load(open(f"{SRC}/path_check.json"))["chosen"])
REF = np.load(f"{SRC}/reference.npz"); QP, SS = REF["Qp"], REF["ss"]
PC, PR = REF["centres"], REF["radii"]
V_LIM, U_LIM = float(ps.v_amp_nom), float(ps.u_amp_nom)


# ------------------------------------------------------------------ timing
def prof(tau):
    tau = np.clip(tau, 0, 1)
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5, (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4), (60 * tau - 180 * tau ** 2 + 120 * tau ** 3)


def q_of_s(s): return np.array([np.interp(s, SS, QP[:, j]) for j in range(M)]).T


def choose_T():
    dq = np.gradient(QP, SS, axis=0); ddq = np.gradient(dq, SS, axis=0)
    tau = np.linspace(0, 1, 2001)
    T = ps.dt
    while True:
        s, sd, sdd = prof(tau); sd, sdd = sd / T, sdd / T ** 2
        dqs = np.array([np.interp(s, SS, dq[:, j]) for j in range(M)]).T
        ddqs = np.array([np.interp(s, SS, ddq[:, j]) for j in range(M)]).T
        v = np.abs(dqs * sd[:, None]).max(); a = np.abs(ddqs * sd[:, None] ** 2 + dqs * sdd[:, None]).max()
        if v <= 0.5 * (V_LIM - b) and a <= 0.5 * U_LIM: return T, float(v), float(a)
        T = round(T + ps.dt, 10)


def q_ref_at(t_idx, T):
    return q_of_s(prof(np.asarray(t_idx) * ps.dt / T)[0])


# ------------------------------------------------------------------ controller
def build(Nh, bb):
    Qz = np.eye(2 * M) * 10; Qz[M:, M:] *= .01
    c = NominalController(A, B, Qz, np.eye(2 * M) * 1e4, np.eye(M) * 1e-3, Nh, u_lim=U_LIM, v_lim=V_LIM, p_amp=np.pi)
    c.use_solver = cp.CLARABEL
    Qq, R = 10 * np.eye(M), 1e-3 * np.eye(M)
    c.Qref = cp.Parameter((M, Nh))
    obj = sum(cp.quad_form(c.X[:M, k] - c.Qref[:, k - 1], Qq) for k in range(1, Nh + 1)) + \
          sum(cp.quad_form(c.U[:, k], R) for k in range(Nh))
    c.G, c.h, c.Qb = cp.Parameter((M, Nh)), cp.Parameter(Nh), cp.Parameter((M, Nh))
    c.tool_con = cp.sum(cp.multiply(c.G, c.X[:M, 1:]), axis=0) >= c.h
    extra = [c.tool_con, cp.abs(c.X[:M, 1:] - c.Qb) <= TR]
    if bb > 0: extra += [c.X[M:, 1:] <= V_LIM - bb, c.X[M:, 1:] >= -(V_LIM - bb)]
    c.problem = cp.Problem(cp.Minimize(obj), list(c.problem.constraints) + extra)
    return c


def episode(plant, tag, T):
    c = build(N, b); Tsteps = int(round(T / ps.dt)); p_g = PC[-1]
    x = np.r_[QP[0], np.zeros(M)]; x_end = np.r_[QP[-1], np.zeros(M)]
    ptraj = x[:M][None].repeat(N + 1, 0); Qexec, log = [x[:M].copy()], []
    status = "truncated"
    for ts in range(NSTEPS):
        cs, rs = sim.get_corridor_balls(ptraj, PC, PR, p_g)
        xg, _ = compute_goal_state(cs, rs, PC, p_g, return_index=True)
        c.set_parameter_values(x, cs.T, rs, xg)
        c.Qref.value = q_ref_at(np.arange(ts + 1, ts + N + 1), T).T
        qbar, nrel = ptraj[1:].copy(), 0
        while True:
            G, r0 = d12.set_lin(c, qbar); ok = c.solve()
            if not ok:
                status = f"infeasible:{getattr(c, 'status', 'exception')}" + (":F1" if np.abs(x[M:]).max() > V_LIM else ""); break
            X, U = c.X.value, c.U.value
            rex = np.array([d12.rho(X[:M, k]) for k in range(1, N + 1)])
            if rex.min() >= -TOL: break
            if nrel == MAXRELIN: status = "plan_rejected_linearisation"; ok = False; break
            qbar, nrel = X[:M, 1:].T.copy(), nrel + 1
        if not ok: log.append(dict(t=ts, status=status)); break
        rlin = r0 + ((X[:M, 1:].T - qbar) * G).sum(1)
        Xt = roll(plant, x, U); rtr = np.array([d12.rho(q[:M]) for q in Xt[1:]])
        dual = np.asarray(c.tool_con.dual_value, float).ravel()
        x_next = np.array(plant.solve_time_step(x, U[:, 0]), float)
        assert np.allclose(x_next, Xt[1], atol=1e-12, rtol=0), "counterfactual copy != executed step (plant not stateless)"
        rho_next_true = d12.rho(x_next[:M]); v_next_abs = float(np.abs(x_next[M:]).max())
        log.append(dict(t=ts, relin=nrel, rho_now=d12.rho(x[:M]), plan_min_rho=float(rex.min()),
                        dual_k1=float(dual[0]), dual_max=float(dual.max()),
                        geo_err=(rlin - rex).tolist(), dyn_err=(rex - rtr).tolist(),
                        track_err=float(np.abs(x[:M] - q_ref_at(ts, T)).max()),
                        dyn_prefix_pos=float(max(0.0, (rex - rtr).max())), dyn_err_k1=float(rex[0] - rho_next_true),
                        fs_next_step=bool(rex[0] >= -TOL and rho_next_true < -TOL), v_exec=v_next_abs,
                        fs_open_loop_window=bool(rex.min() >= -TOL and rtr.min() < -TOL)))
        x = x_next; Qexec.append(x[:M].copy())
        ptraj = sim.get_shifted_trajectory(X)[:, :M]
        if ts + 1 >= Tsteps and np.linalg.norm(x - x_end) < GOAL_TOL: status = "success"; break
    Qexec = np.array(Qexec); tm = tool_metrics(Qexec); _, tip_end, _ = fk(Qexec[-1])
    ok_steps = [s for s in log if "rho_now" in s]
    vdev = [float((fk(q)[1] - d12.TASK.p(s))[2]) for q, s in zip(Qexec, tm["sigma"])]
    ge = np.concatenate([s["geo_err"] for s in ok_steps]) if ok_steps else np.array([np.nan])
    de = np.concatenate([s["dyn_err"] for s in ok_steps]) if ok_steps else np.array([np.nan])
    dsig = np.diff(tm["sigma"]) * 0.20
    r = dict(status=status, steps=len(Qexec) - 1, T_ref_steps=Tsteps,
             lat_max=float(tm["lat"].max()), vertical_dev_min=float(min(vdev)), ang_max_deg=float(np.rad2deg(tm["ang"].max())),
             ang_max_step=int(np.argmax(tm["ang"])), end_err=float(np.linalg.norm(tip_end - d12.TASK.p(1.0))),
             rho_exec_min=float(tm["rho"].min()), rho_exec_median=float(np.median(tm["rho"])),
             steps_rho_below_1mm=int((tm["rho"] < 1e-3).sum()),
             steps_tool_dual_k1_active=int(sum(s["dual_k1"] > 1e-6 for s in ok_steps)),
             steps_tool_dual_any_active=int(sum(s["dual_max"] > 1e-6 for s in ok_steps)),
             track_err_max=float(max(s["track_err"] for s in ok_steps)) if ok_steps else None,
             backtrack_max_mm=float(1e3 * max(0.0, -dsig.min())) if len(dsig) else 0.0,
             wrist_exec_range_deg=[float(np.rad2deg(np.ptp(Qexec[:, j]))) for j in (3, 4, 5)],
             relin_total=int(sum(s.get("relin", 0) for s in ok_steps)),
             geo_err_absq95=float(np.quantile(np.abs(ge), .95)), dyn_err_absq95=float(np.quantile(np.abs(de), .95)),
             dyn_prefix_pos_q95=float(np.quantile([s["dyn_prefix_pos"] for s in ok_steps], .95)) if ok_steps else None,
             dyn_err_k1_absq95=float(np.quantile([abs(s["dyn_err_k1"]) for s in ok_steps], .95)) if ok_steps else None,
             fs_next_step=int(sum(s["fs_next_step"] for s in ok_steps)),
             fs_open_loop_window=int(sum(s["fs_open_loop_window"] for s in ok_steps)),
             v_exec_max=float(max(s["v_exec"] for s in ok_steps)) if ok_steps else None,
             exec_violation_steps=int((tm["rho"] < -TOL).sum()),
             exec_violation_depth=float(max(0.0, -tm["rho"].min())))
    r["task_PASS"] = (status == "success" and r["lat_max"] <= LAT_TOL and r["ang_max_deg"] <= np.rad2deg(ANG_TOL)
                      and r["end_err"] <= END_TOL and r["rho_exec_min"] >= -TOL)
    np.savez(f"{OUT}/exec_{tag}.npz", Q=Qexec, sigma=tm["sigma"], lat=tm["lat"], ang=tm["ang"], rho=tm["rho"])
    json.dump(log, open(f"{OUT}/steps_{tag}.json", "w"))
    print(tag, json.dumps(r, default=float))
    return r


def stage_zero():
    T, vmax, amax = choose_T()
    print(f"timing: T = {T:.2f} s ({int(round(T / ps.dt))} steps), ref max|qdot| {vmax:.3f} <= {0.5 * (V_LIM - b):.3f}, "
          f"max|qddot| {amax:.3f} <= {0.5 * U_LIM:.1f}")
    c0 = make_ctx(ps, MISMATCH_SEEDS[0], ZeroResidual(), 0.0)
    plant = TimeStepIntegratorDiscrete(A=A, B=B, ignore_me=True); plant.set_dyns(*find_dyns(c0["it"]))
    r = episode(plant, "zero", T)
    json.dump(dict(T=T, ref_vmax=vmax, ref_amax=amax, zero=r), open(f"{OUT}/zero.json", "w"), indent=1, default=float)
    print("ZERO-MISMATCH TOOL TASK", "PASS" if r["task_PASS"] else "FAIL — stop at engineering validation")


def stage_mismatch():
    z = json.load(open(f"{OUT}/zero.json")); assert z["zero"]["task_PASS"], "zero-mismatch stage did not pass"
    res = {}
    for sd in MISMATCH_SEEDS:
        c0 = make_ctx(ps, sd, ZeroResidual(), 0.0)
        plant = TimeStepIntegratorContinuous(dt=ps.dt); plant.set_dyns(*find_dyns(c0["it"]))
        res[sd] = episode(plant, f"mis_{sd}", z["T"])
    json.dump(res, open(f"{OUT}/mismatch.json", "w"), indent=1, default=float)
    keys = ["status", "steps", "lat_max", "ang_max_deg", "end_err", "task_PASS", "rho_exec_min", "exec_violation_steps",
            "exec_violation_depth", "steps_tool_dual_any_active", "dyn_err_k1_absq95", "dyn_err_absq95", "dyn_prefix_pos_q95",
            "fs_next_step", "fs_open_loop_window", "geo_err_absq95", "relin_total"]
    print("\nseed  " + "  ".join(keys))
    for sd, r in res.items(): print(sd, "  ".join(f"{r[k]:.4g}" if isinstance(r[k], float) else str(r[k]) for k in keys))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--zero", action="store_true"); ap.add_argument("--mismatch", action="store_true")
    a = ap.parse_args()
    if a.zero: stage_zero()
    if a.mismatch: stage_mismatch()
