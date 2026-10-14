"""
day12_tooltask.py — Stage 0 of the tool-operation task (no learning). Repo root, PYTHONPATH=. ; env robmpc.
Stages run in order, each gated by the previous PASS:  --path  ->  --zero  ->  --mismatch
Output: results/day12_tooltask/

FROZEN TASK SPEC (synthetic, interpretable; no specific industrial basis):
  tool: capsule from flange (link_6) origin along flange-local +z, L=0.10 m, r=0.01 m; no mass/inertia simulated
  plane: horizontal, safe side +z.  d_safe = 0.01 m.  rho = min(z_base, z_tip) - z_plane - r - d_safe
  task frame: e1 = horizontal travel direction (radial, azimuth PHI), e3 = world z.
  tool direction base->tip  a(s) = sin(theta) e1 - cos(theta) e3  (tip down), theta(s) = -25 deg -> +25 deg linear in s
  tip centre on a straight segment of length 0.20 m along e1, at height z_plane + 0.03 m  -> rho_ref = 10 mm
  workspace: first entry of CANDIDATES that passes the reference-path check (feasibility only, fixed order)
Engineering tolerances (diagnostic, pre-registered): lateral tip deviation <= 5 mm, axis error <= 3 deg, both at the
  SAME progress sigma (projection of executed tip onto the segment); final tip error <= 5 mm; exact rho >= -TOL.
Planner integration (shared by all arms): nominal MPC (author recipe, CLARABEL) + velocity tightening b (k>=1) +
  linearised tool constraint rho(qbar_k) + grad(qbar_k)^T (q_k - qbar_k) >= 0 for k=1..N, qbar = shifted previous plan,
  trust region |q_k - qbar_k|_inf <= TR; after each solve the EXACT rho is evaluated on the predicted plan;
  violation -> re-linearise about the new plan, at most MAXRELIN times; still violated -> plan rejected, episode ends.
Errors logged separately: geometric  rho_lin(q_hat) - rho(q_hat) ; dynamic  rho(q_hat) - rho(q_true) (open-loop true
  rollout of the planned inputs from the same state, evaluation only).
Not claimed: learning value, safety guarantee, ft/rt comparison (tube extension to the tool constraint not implemented).
"""
import argparse, json, os, sys, time
import numpy as np
import cvxpy as cp
sys.path.insert(0, ".")
from exec_interface import get_ps
from controllers.nom_controller import NominalController
from corridor_simulators.base import BaseSimulator
from aux import (get_linear_double_integrator_discrete_dynamics, interpolate_equidistant, compute_goal_state,
                 TimeStepIntegratorContinuous, TimeStepIntegratorDiscrete)
from examples.six_dof.world import DemoWorld
from day8_reference_arms import find_dyns
from day7_fixed_margin import make_ctx
from exec_interface import ZeroResidual

OUT = "results/day12_tooltask"; os.makedirs(OUT, exist_ok=True)
L_TOOL, R_TOOL, D_SAFE, STANDOFF, SEG, TH0, TH1 = 0.10, 0.01, 0.01, 0.03, 0.20, np.deg2rad(-25), np.deg2rad(25)
CANDIDATES = [(-np.pi / 2, 0.30, 0.05), (-np.pi / 2, 0.30, 0.10), (np.pi / 2, 0.30, 0.05)]   # (azimuth, start radius, tip z)
NPATH, CONT_MAX, WRIST_MIN = 81, 0.05, np.deg2rad(10)
TOL, LAT_TOL, ANG_TOL, END_TOL = 1e-6, 0.005, np.deg2rad(3), 0.005
TR, MAXRELIN, HGRAD, NSTEPS, GOAL_TOL, DELTA_PATH = 0.05, 2, 1e-6, 500, 1e-2, 0.05
MISMATCH_SEEDS = [200, 201, 202, 203]           # development seeds; test seeds untouched
M = 6

ps = get_ps(); world = DemoWorld(); man = world.man
LO, HI = man.model.lowerPositionLimit.copy(), man.model.upperPositionLimit.copy()
A, B = get_linear_double_integrator_discrete_dynamics(M, dt=ps.dt, method="zoh")
b = json.load(open("results/day7_margin/b.json"))["b"]
sim = BaseSimulator(None, None)


# ------------------------------------------------------------------ geometry
def fk(q):
    T6, = man.get_link_fk(np.asarray(q, float), links=["link_6"])
    pb = T6[:3, 3]; a = T6[:3, 2]; return pb, pb + L_TOOL * a, a


class Task:
    def __init__(self, phi, r0, zt):
        self.e1 = np.r_[np.cos(phi), np.sin(phi), 0.0]; self.e3 = np.r_[0, 0, 1.0]
        self.p0 = r0 * self.e1 + zt * self.e3; self.zp = zt - STANDOFF; self.phi, self.r0, self.zt = phi, r0, zt
    def p(self, s): return self.p0 + s * SEG * self.e1
    def th(self, s): return TH0 + s * (TH1 - TH0)
    def a(self, s): t = self.th(s); return np.sin(t) * self.e1 - np.cos(t) * self.e3
    def sigma(self, tip): return float(np.clip((tip - self.p0) @ self.e1 / SEG, 0, 1))


TASK = None
def rho(q):
    pb, pt, _ = fk(q); return min(pb[2], pt[2]) - TASK.zp - R_TOOL - D_SAFE
def grad(q):
    g = np.zeros(M)
    for j in range(M):
        e = np.zeros(M); e[j] = HGRAD; g[j] = (rho(q + e) - rho(q - e)) / (2 * HGRAD)
    return g
def angle(u, v): return float(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))


def tool_metrics(Q):
    """executed/planned joint positions (T,6) -> per-sample sigma, lateral dev, axis error at the SAME sigma, rho."""
    out = dict(sigma=[], lat=[], ang=[], rho=[])
    for q in Q:
        pb, pt, a = fk(q); s = TASK.sigma(pt)
        out["sigma"].append(s); out["lat"].append(float(np.linalg.norm(pt - TASK.p(s))))
        out["ang"].append(angle(a, TASK.a(s))); out["rho"].append(rho(q))
    return {k: np.array(v) for k, v in out.items()}


# ------------------------------------------------------------------ IK
def ik(p_des, a_des, q0, iters=300, lam=1e-3):
    q = q0.copy()
    for _ in range(iters):
        _, pt, a = fk(q); e = np.r_[pt - p_des, np.cross(a, a_des)]
        if np.linalg.norm(e) < 1e-11: break
        J = np.zeros((6, 5))
        for j in range(5):
            d = np.zeros(M); d[j] = 1e-7; _, pt2, a2 = fk(q + d)
            J[:, j] = (np.r_[pt2 - p_des, np.cross(a2, a_des)] - e) / 1e-7
        q[:5] += J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), -e)
    _, pt, a = fk(q)
    return q, float(np.linalg.norm(pt - p_des)), angle(a, a_des)


def stage_path():
    global TASK
    tried = []
    for phi, r0, zt in CANDIDATES:
        TASK = Task(phi, r0, zt); rep = dict(candidate=[phi, r0, zt], plane_z=TASK.zp)
        seeds = [np.r_[phi, np.pi / 4, np.pi / 2, 0, 0, 0], np.r_[phi, 0.3, 1.8, 0, 1.0, 0], np.r_[phi, -0.3, 2.0, 0, 1.0, 0]]
        q = None
        for sd in seeds:
            qs, ep, ea = ik(TASK.p(0), TASK.a(0), sd.astype(float))
            if ep < 1e-6 and ea < 1e-6 and np.all((qs >= LO) & (qs <= HI)): q = qs; break
        if q is None: rep["fail"] = "no IK seed converged at s=0"; tried.append(rep); continue
        ss = np.linspace(0, 1, NPATH); Qp = [q]; res = []
        for s in ss[1:]:
            q, ep, ea = ik(TASK.p(s), TASK.a(s), Qp[-1].copy()); Qp.append(q); res.append((ep, ea))
        Qp = np.array(Qp)
        centres = interpolate_equidistant(Qp, delta=DELTA_PATH)
        radii = world.sdf(centres[:, :3])
        tm_ref, tm_cen = tool_metrics(Qp), tool_metrics(centres)
        checks = dict(
            ik_pos_max=max(r[0] for r in res), ik_ang_max=max(r[1] for r in res),
            continuity_max=float(np.abs(np.diff(Qp, axis=0)).max()),
            urdf_limits=bool(np.all((Qp >= LO) & (Qp <= HI))), world_lims=bool(np.all(np.abs(Qp) <= world.lims[1] + 1e-12)),
            gt_collision_free=bool(all(world.is_collision_free_gt(qq[:3]) for qq in np.r_[Qp, centres])),
            sdf_radius_min=float(np.min(radii)),
            rho_ref_min=float(tm_ref["rho"].min()), rho_ref_max=float(tm_ref["rho"].max()),
            tip_is_lowest=bool(all(fk(qq)[1][2] <= fk(qq)[0][2] for qq in Qp)),
            wrist_range_deg=[float(np.rad2deg(np.ptp(Qp[:, j]))) for j in (3, 4, 5)],
            centres_lat_max=float(tm_cen["lat"].max()), centres_ang_max_deg=float(np.rad2deg(tm_cen["ang"].max())),
            n_centres=int(len(centres)))
        ok = (checks["ik_pos_max"] < 1e-6 and checks["ik_ang_max"] < 1e-6 and checks["continuity_max"] <= CONT_MAX
              and checks["urdf_limits"] and checks["world_lims"] and checks["gt_collision_free"] and checks["sdf_radius_min"] > 0
              and abs(checks["rho_ref_min"] - 0.01) < 1e-6 and checks["tip_is_lowest"]
              and max(checks["wrist_range_deg"]) >= np.rad2deg(WRIST_MIN)
              and checks["centres_lat_max"] <= LAT_TOL and checks["centres_ang_max_deg"] <= np.rad2deg(ANG_TOL))
        rep.update(checks=checks, PASS=ok); tried.append(rep)
        print(json.dumps(rep, indent=1, default=float))
        if ok:
            np.savez(f"{OUT}/reference.npz", Qp=Qp, centres=centres, radii=radii, ss=ss)
            json.dump(dict(chosen=[phi, r0, zt], plane_z=TASK.zp, tried=tried), open(f"{OUT}/path_check.json", "w"), indent=1, default=float)
            print("REFERENCE PATH PASS"); return
    json.dump(dict(chosen=None, tried=tried), open(f"{OUT}/path_check.json", "w"), indent=1, default=float)
    print("REFERENCE PATH FAIL — stop, controller not connected")


# ------------------------------------------------------------------ controller
def build_controller(N, bb):
    Q = np.eye(2 * M) * 10; Q[M:, M:] *= .01
    c = NominalController(A, B, Q, np.eye(2 * M) * 1e4, np.eye(M) * 1e-3, N, u_lim=ps.u_amp_nom, v_lim=ps.v_amp_nom, p_amp=np.pi)
    c.use_solver = cp.CLARABEL; v_lim = float(ps.v_amp_nom)
    c.G, c.h, c.Qb = cp.Parameter((M, N)), cp.Parameter(N), cp.Parameter((M, N))
    extra = [cp.sum(cp.multiply(c.G, c.X[:M, 1:]), axis=0) >= c.h, cp.abs(c.X[:M, 1:] - c.Qb) <= TR]
    if bb > 0: extra += [c.X[M:, 1:] <= v_lim - bb, c.X[M:, 1:] >= -(v_lim - bb)]
    c.problem = cp.Problem(c.problem.objective, list(c.problem.constraints) + extra)
    return c


def set_lin(c, qbar):
    G = np.array([grad(q) for q in qbar]); r0 = np.array([rho(q) for q in qbar])
    c.G.value = G.T; c.h.value = (G * qbar).sum(1) - r0; c.Qb.value = qbar.T
    return G, r0


def roll(plant, x, U):
    X = [x]
    for k in range(U.shape[1]): X.append(np.array(plant.solve_time_step(X[-1], U[:, k]), float))
    return np.array(X)


def episode(plant, eval_plant, tag):
    ref = np.load(f"{OUT}/reference.npz"); pc, pr = ref["centres"], ref["radii"]; p_g = pc[-1]
    N = 20; c = build_controller(N, b); v_lim = float(ps.v_amp_nom)
    x = np.r_[pc[0], np.zeros(M)]; xg_final = np.r_[p_g, np.zeros(M)]
    ptraj = x[:M][None].repeat(N + 1, 0); Qexec, steps = [x[:M].copy()], []
    status = "truncated"
    for ts in range(NSTEPS):
        cs, rs = sim.get_corridor_balls(ptraj, pc, pr, p_g)
        xg, _ = compute_goal_state(cs, rs, pc, p_g, return_index=True)
        c.set_parameter_values(x, cs.T, rs, xg)
        qbar, nrel = ptraj[1:].copy(), 0
        while True:
            G, r0 = set_lin(c, qbar); ok = c.solve()
            if not ok:
                cls = "F1_v0_over_limit" if np.abs(x[M:]).max() > v_lim else "infeasible"
                status = f"infeasible:{getattr(c, 'status', 'exception')}:{cls}"; break
            X, U = c.X.value, c.U.value
            rex = np.array([rho(X[:M, k]) for k in range(1, N + 1)])
            if rex.min() >= -TOL: break
            if nrel == MAXRELIN: status = "plan_rejected_linearisation"; ok = False; break
            qbar, nrel = X[:M, 1:].T.copy(), nrel + 1
        if not ok: steps.append(dict(t=ts, status=status)); break
        rlin = r0 + ((X[:M, 1:].T - qbar) * G).sum(1)
        Xt = roll(eval_plant, x, U); rtr = np.array([rho(q[:M]) for q in Xt[1:]])
        steps.append(dict(t=ts, relin=nrel, plan_min_rho=float(rex.min()), geo_err=(rlin - rex).tolist(),
                          dyn_err=(rex - rtr).tolist(), true_ol_min_rho=float(rtr.min()),
                          plan_safe_true_unsafe=bool(rex.min() >= -TOL and rtr.min() < -TOL)))
        x = np.array(plant.solve_time_step(x, U[:, 0]), float); Qexec.append(x[:M].copy())
        ptraj = sim.get_shifted_trajectory(X)[:, :M]
        if np.linalg.norm(x - xg_final) < GOAL_TOL: status = "success"; break
    Qexec = np.array(Qexec); tm = tool_metrics(Qexec)
    dsig = np.diff(tm["sigma"]) * SEG
    back = float(max(0.0, -dsig.min())) if len(dsig) else 0.0
    stall = int(sum(np.all(np.abs(dsig[i:i + 20]) < 1e-4) for i in range(max(0, len(dsig) - 20)) if tm["sigma"][i] < 0.99))
    _, tip_end, _ = fk(Qexec[-1])
    task = dict(status=status, steps=len(Qexec) - 1, lat_max=float(tm["lat"].max()), ang_max_deg=float(np.rad2deg(tm["ang"].max())),
                end_err=float(np.linalg.norm(tip_end - TASK.p(1.0))), rho_exec_min=float(tm["rho"].min()),
                sigma_final=float(tm["sigma"][-1]), backtrack_max_mm=1e3 * back, stall_windows=stall,
                wrist_exec_range_deg=[float(np.rad2deg(np.ptp(Qexec[:, j]))) for j in (3, 4, 5)],
                relin_total=int(sum(s.get("relin", 0) for s in steps)))
    task["task_PASS"] = (status == "success" and task["lat_max"] <= LAT_TOL and task["ang_max_deg"] <= np.rad2deg(ANG_TOL)
                         and task["end_err"] <= END_TOL and task["rho_exec_min"] >= -TOL)
    ge = np.concatenate([s["geo_err"] for s in steps if "geo_err" in s]) if any("geo_err" in s for s in steps) else np.array([np.nan])
    de = np.concatenate([s["dyn_err"] for s in steps if "dyn_err" in s]) if any("dyn_err" in s for s in steps) else np.array([np.nan])
    pm = [max(0.0, max(s["dyn_err"])) for s in steps if "dyn_err" in s]
    task.update(geo_err_absq95=float(np.quantile(np.abs(ge), .95)), geo_err_absmax=float(np.abs(ge).max()),
                dyn_err_absq95=float(np.quantile(np.abs(de), .95)), dyn_prefix_pos_q95=float(np.quantile(pm, .95)) if pm else None,
                plan_safe_true_unsafe=int(sum(s.get("plan_safe_true_unsafe", False) for s in steps)),
                near_steps_frac=float(np.mean(tm["rho"] <= 0.02)))
    np.savez(f"{OUT}/exec_{tag}.npz", Q=Qexec, sigma=tm["sigma"], lat=tm["lat"], ang=tm["ang"], rho=tm["rho"])
    json.dump(steps, open(f"{OUT}/steps_{tag}.json", "w"))
    print(tag, json.dumps(task, default=float))
    return task


def load_task():
    global TASK
    pcj = json.load(open(f"{OUT}/path_check.json")); assert pcj["chosen"], "reference path did not pass"
    TASK = Task(*pcj["chosen"])


def stage_zero():
    load_task()
    c0 = make_ctx(ps, MISMATCH_SEEDS[0], ZeroResidual(), 0.0); dn, de = find_dyns(c0["it"])
    plant = TimeStepIntegratorDiscrete(A=A, B=B, ignore_me=True); plant.set_dyns(dn, de)
    x = np.r_[np.zeros(M), np.ones(M) * .1]; u = np.ones(M)
    assert np.allclose(plant.solve_time_step(x, u), A @ x + B @ u), "zero-mismatch plant is not the nominal model"
    t = episode(plant, plant, "zero")
    json.dump(dict(zero=t), open(f"{OUT}/zero.json", "w"), indent=1, default=float)
    print("ZERO-MISMATCH TASK", "PASS" if t["task_PASS"] else "FAIL — task/planner integration incomplete, no learning discussion")


def stage_mismatch():
    load_task()
    z = json.load(open(f"{OUT}/zero.json"))["zero"]; assert z["task_PASS"], "zero-mismatch stage did not pass"
    res = {}
    for sd in MISMATCH_SEEDS:
        c0 = make_ctx(ps, sd, ZeroResidual(), 0.0)
        plant = TimeStepIntegratorContinuous(dt=ps.dt); plant.set_dyns(*find_dyns(c0["it"]))
        res[sd] = episode(plant, plant, f"mis_{sd}")
    json.dump(res, open(f"{OUT}/mismatch.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for s in ("path", "zero", "mismatch"): ap.add_argument(f"--{s}", action="store_true")
    a = ap.parse_args()
    if a.path: stage_path()
    if a.zero: stage_zero()
    if a.mismatch: stage_mismatch()
