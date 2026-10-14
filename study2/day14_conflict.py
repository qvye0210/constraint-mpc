"""
day14_conflict.py — offline diagnosis: does NOT knowing which dynamics is true create an MPC constraint conflict?
Repo root, PYTHONPATH=. ; env robmpc. Output: results/day14_conflict/. No training, no new data, task frozen (day13).

SOURCE FACTS (read from the authors' code, recorded in config.json):
 1. mismatch acts on LINK MASSES: ProblemScenarioMassAllPin.get_err_dyn_random() draws one ratio per link,
    ratio = 1 + U(-frac, frac), frac = ps.frac; the plant error enters as
    delta(q,qd,u) = -M^-1 M_err u - M^-1 (c_err + g_err)   [aux.compute_model_error], qdd = u + delta.
 2. the ratios are drawn ONCE per episode and stay fixed (DeltaDynPin holds the sampled model).
 3. the MPC predictor is the feedback-linearised double integrator (delta = 0), zoh-discretised at dt.
 4. for a GIVEN mass instance the error is AFFINE in u at fixed (q,qd):  delta = -Mtilde(q) u + w(q,qd).
    Each model therefore maps exactly to a stage-wise affine predictor about a linearisation state:
       x_{k+1} = A x_k + B[(I - Mtilde_i(qbar_k)) u_k + w_i(qbar_k, qdbar_k)]
    linearisation states = the frozen day13 reference at the checkpoint -> "APPROXIMATE PLANNING DIAGNOSIS";
    feasible candidates are re-checked against the CONTINUOUS nonlinear plant of every model.
 5. author terminal conditions kept exactly: X[m:, N] == 0 and U[:, N-1] == 0.
 6. the decision variable is one COMMON OPEN-LOOP input sequence U; each model keeps its OWN state trajectory.
    no feedback policy class is available in the repo -> POLICY_CLASS_UNRESOLVED is reported, never "learning needed".
 7. tool constraint linearised about the same reference states (G, h as in day12/13), trust region 0.05 rad;
    corridor balls frozen from the reference at the checkpoint. Velocity tightening b = 0.0862 kept (design choice).
Nothing was tightened, sped up or enlarged to produce a conflict.
"""
import argparse, csv, itertools, json, os, subprocess, sys
import numpy as np
import cvxpy as cp
sys.path.insert(0, ".")
import day12_tooltask as d12
import day13_reftrack as d13
from day12_tooltask import A, B, b, ps, sim, M, TR, TOL
from aux import compute_model_error, TimeStepIntegratorContinuous, compute_goal_state
from day7_fixed_margin import make_ctx
from day8_reference_arms import find_dyns
from exec_interface import get_ps, ZeroResidual

OUT = "results/day14_conflict"; os.makedirs(OUT, exist_ok=True)
N = 20; SEEDS = [200, 201, 202, 203]
V_LIM, U_LIM = float(ps.v_amp_nom), float(ps.u_amp_nom)
FEAS = {"optimal", "optimal_inaccurate"}


# ---------------------------------------------------------------- model set
def masses(dyn): return np.array([i.mass for i in dyn.model.inertias])


def build_models():
    ms, probe_q, probe_qd, probe_u = [], d13.QP[10].copy(), np.full(M, .3), np.full(M, 2.0)
    caps = []
    for sd in SEEDS:
        c = make_ctx(ps, sd, ZeroResidual(), 0.0); dn, de = find_dyns(c["it"])
        caps.append(dict(seed=sd, ratios=(masses(de.sampled) / masses(de.nom)).tolist(),
                         probe=compute_model_error(dn, de, probe_q, probe_qd, probe_u).tolist()))
    for cap in caps:                       # independent scenario object per model, verified against the capture
        p = get_ps(); de = p.get_err_dyn_from_ratios(np.array(cap["ratios"]))
        err = float(np.abs(compute_model_error(p.dyn_nom, de, probe_q, probe_qd, probe_u) - np.array(cap["probe"])).max())
        assert err < 1e-12, f"model rebuild mismatch seed {cap['seed']}: {err:.2e}"
        ms.append(dict(name=f"m{cap['seed']}", seed=cap["seed"], ratios=cap["ratios"], ps=p, dyn_err=de, rebuild_err=err))
    return ms, caps


def stage_affine(mdl, qbar, qdbar):
    """delta = -Mtilde u + w  ->  acc = (I - Mtilde) u + w ; built by probing compute_model_error (no internals)."""
    if mdl is None: return np.eye(M), np.zeros(M)
    dn, de = mdl["ps"].dyn_nom, mdl["dyn_err"]
    w = compute_model_error(dn, de, qbar, qdbar, np.zeros(M))
    Emat = np.zeros((M, M))
    for j in range(M):
        e = np.zeros(M); e[j] = 1.0
        Emat[:, j] = compute_model_error(dn, de, qbar, qdbar, e) - w
    return np.eye(M) + Emat, w


def affine_check(mdl, x, u):
    """one-step residual of the affine predictor vs the continuous plant of the same model."""
    it = TimeStepIntegratorContinuous(dt=ps.dt); it.set_dyns(mdl["ps"].dyn_nom, mdl["dyn_err"])
    E, w = stage_affine(mdl, x[:M], x[M:])
    return float(np.abs((A @ x + B @ (E @ u + w)) - np.array(it.solve_time_step(x, u), float)).max())


# ---------------------------------------------------------------- problem
def solve(models, ctx, drop=()):
    """models: list (None = nominal). Common U, per-model X. drop: {'terminal','tool','corridor','tighten','trust'}"""
    x0, qbar, qdbar, cs, rs, G, h, Qref = (ctx[k] for k in ("x0", "qbar", "qdbar", "cs", "rs", "G", "h", "Qref"))
    U = cp.Variable((M, N)); cons = [cp.abs(U) <= U_LIM, U[:, N - 1] == 0]
    obj = 0
    for i, mdl in enumerate(models):
        X = cp.Variable((2 * M, N + 1)); cons += [X[:, 0] == x0]
        for k in range(N):
            E, w = stage_affine(mdl, qbar[k], qdbar[k])
            cons += [X[:, k + 1] == A @ X[:, k] + B @ (E @ U[:, k] + w)]
        cons += [cp.abs(X[:M, :]) <= np.pi, cp.abs(X[M:, :]) <= V_LIM]
        if "tighten" not in drop: cons += [cp.abs(X[M:, 1:]) <= V_LIM - b]
        if "terminal" not in drop: cons += [X[M:, N] == 0]
        if "corridor" not in drop:
            cons += [cp.norm(X[:M, k] - cs[k], 2) <= rs[k] for k in range(N + 1)]
        if "tool" not in drop:
            cons += [cp.sum(cp.multiply(G, X[:M, 1:]), axis=0) >= h]
        if "trust" not in drop: cons += [cp.abs(X[:M, 1:] - np.array(qbar).T) <= TR]
        obj = obj + sum(cp.sum_squares(X[:M, k] - Qref[k - 1]) for k in range(1, N + 1))
    p = cp.Problem(cp.Minimize(10 * obj + 1e-3 * cp.sum_squares(U)), cons)
    try:
        p.solve(solver=cp.CLARABEL)
    except Exception as e:
        return dict(status=f"exception:{type(e).__name__}", U=None, value=None)
    return dict(status=p.status, value=None if p.value is None else float(p.value),
                U=None if U.value is None else U.value.copy())


def nonlinear_recheck(models, U, ctx):
    """feasible candidate -> exact continuous rollout of every model, exact geometry, no linearisation."""
    out = {}
    for mdl in models:
        it = TimeStepIntegratorContinuous(dt=ps.dt)
        it.set_dyns(ps.dyn_nom if mdl is None else mdl["ps"].dyn_nom, None if mdl is None else mdl["dyn_err"])
        x = ctx["x0"].copy(); X = [x]
        for k in range(N):
            x = (A @ x + B @ U[:, k]) if mdl is None else np.array(it.solve_time_step(x, U[:, k]), float)
            X.append(x)
        X = np.array(X)
        out["nom" if mdl is None else mdl["name"]] = dict(
            v_max=float(np.abs(X[:, M:]).max()), rho_min=float(min(d12.rho(q[:M]) for q in X[1:])),
            corridor_min=float(min(ctx["rs"][k] - np.linalg.norm(X[k, :M] - ctx["cs"][k]) for k in range(N + 1))),
            term_v=float(np.abs(X[-1, M:]).max()))
    return out


def make_ctx_window(t0, x0):
    T = json.load(open("results/day13_reftrack/zero.json"))["T"]
    qbar = d13.q_ref_at(np.arange(t0 + 1, t0 + N + 1), T)          # linearisation + reference states, k=1..N
    qdbar = np.vstack([np.zeros(M), np.diff(qbar, axis=0) / ps.dt])
    ptraj = np.vstack([x0[:M], qbar])
    cs, rs = sim.get_corridor_balls(ptraj, d13.PC, d13.PR, d13.PC[-1])
    G = np.array([d12.grad(q) for q in qbar]); r0 = np.array([d12.rho(q) for q in qbar])
    return dict(x0=x0, qbar=qbar, qdbar=qdbar, cs=np.asarray(cs), rs=np.asarray(rs),
                G=G.T, h=(G * qbar).sum(1) - r0, Qref=qbar, t0=t0)


# ---------------------------------------------------------------- main
def main():
    models, caps = build_models()
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip().splitlines()
    zero = np.load("results/day13_reftrack/exec_zero.npz")["Q"]
    checkpoints = {"start": (0, np.r_[d13.QP[0], np.zeros(M)]),
                   "mid": (68, np.r_[zero[68], (zero[68] - zero[67]) / ps.dt])}
    cfg = dict(commit=git, dirty_files=dirty, N=N, b=b, v_lim=V_LIM, u_lim=U_LIM, dt=ps.dt, frac=float(ps.frac),
               mismatch="link-mass ratios 1+U(-frac,frac), one draw per episode, fixed within the episode",
               predictor="feedback-linearised double integrator; per-model stage-wise affine about the reference",
               policy="common open-loop U, per-model states; no feedback class available -> POLICY_CLASS_UNRESOLVED",
               terminal="X[m:,N]==0 and U[:,N-1]==0 kept", tool="linearised about reference, trust region 0.05 rad",
               models=[dict(name=m["name"], seed=m["seed"], ratios=m["ratios"], rebuild_err=m["rebuild_err"]) for m in models],
               checkpoints={k: dict(t0=v[0]) for k, v in checkpoints.items()}, solver="CLARABEL")
    rows_m, rows_j, conflicts, verdicts = [], [], {}, {}
    for cp_name, (t0, x0) in checkpoints.items():
        ctx = make_ctx_window(t0, x0)
        cfg.setdefault("affine_residual", {})[cp_name] = {m["name"]: affine_check(m, x0, np.full(M, 1.0)) for m in models}
        # A: single models
        singles = {}
        for mdl in [None] + models:
            nm = "nom" if mdl is None else mdl["name"]
            r = solve([mdl], ctx); singles[nm] = r
            rows_m.append(dict(checkpoint=cp_name, model=nm, status=r["status"], value=r["value"],
                               **(nonlinear_recheck([mdl], r["U"], ctx)[nm] if r["U"] is not None else {})))
            print(f"[{cp_name}] single {nm}: {r['status']}")
        # C: duplicate regression
        dup = solve([models[0]] * 4, ctx)
        rows_j.append(dict(checkpoint=cp_name, arm="duplicate_x4_m200", status=dup["status"], value=dup["value"],
                           matches_single=bool(dup["status"] in FEAS) == bool(singles[models[0]["name"]]["status"] in FEAS)))
        # B: joint
        joint = solve(models, ctx)
        rows_j.append(dict(checkpoint=cp_name, arm="joint_all", status=joint["status"], value=joint["value"],
                           **({f"recheck_{k}": json.dumps(v) for k, v in nonlinear_recheck(models, joint["U"], ctx).items()}
                              if joint["U"] is not None else {})))
        print(f"[{cp_name}] duplicate x4: {dup['status']} | joint: {joint['status']}")
        singles_ok = all(r["status"] in FEAS for r in singles.values())
        if joint["status"] in FEAS:
            state = "COMMON_PLAN_EXISTS"
        elif not singles_ok:
            state = "MODELWISE_INFEASIBLE"
        elif joint["status"] == "infeasible":
            state = "OPEN_LOOP_CONFLICT_FOUND"
        else:
            state = "NUMERICAL_OR_MODELING_UNRESOLVED"
        if state == "OPEN_LOOP_CONFLICT_FOUND":                     # pseudo-conflict screens (diagnostic only)
            for d in ("terminal", "tool", "corridor", "trust", "tighten"):
                r = solve(models, ctx, drop=(d,))
                rows_j.append(dict(checkpoint=cp_name, arm=f"joint_drop_{d}", status=r["status"], value=r["value"],
                                   note="diagnostic relaxation, NOT the task"))
                print(f"[{cp_name}] joint without {d}: {r['status']}")
            keep = list(models)                                     # inclusion-minimal conflict subset
            trace = []
            for mdl in list(models):
                cand = [x for x in keep if x is not mdl]
                r = solve(cand, ctx) if len(cand) >= 2 else dict(status="skipped")
                trace.append(dict(removed=mdl["name"], remaining=[x["name"] for x in cand], status=r["status"]))
                if r["status"] == "infeasible": keep = cand
            verify = {m["name"]: solve([x for x in keep if x is not m], ctx)["status"] for m in keep} if len(keep) >= 2 else {}
            conflicts[cp_name] = dict(subset=[m["name"] for m in keep], removal_trace=trace, leave_one_out=verify,
                                      minimality="inclusion-minimal, NOT guaranteed cardinality-minimal")
        else:
            conflicts[cp_name] = dict(subset=None, reason=f"no reliable joint infeasibility ({joint['status']})")
        verdicts[cp_name] = dict(planning_status=state, singles={k: v["status"] for k, v in singles.items()},
                                 joint_status=joint["status"], duplicate_regression=dup["status"])
    def dump(p, rows):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, keys, restval=""); w.writeheader(); w.writerows(rows)
    dump(f"{OUT}/modelwise.csv", rows_m); dump(f"{OUT}/joint.csv", rows_j)
    json.dump(conflicts, open(f"{OUT}/conflicts.json", "w"), indent=1, default=str)
    json.dump(cfg, open(f"{OUT}/config.json", "w"), indent=1, default=str)
    v = dict(per_checkpoint=verdicts, learning_usability="SAFE_PROBE_VALUE_NOT_TESTED",
             policy_class="POLICY_CLASS_UNRESOLVED (common open-loop only; no feedback class in the repo)",
             scope="finite-horizon N=20 planning problem at fixed checkpoints under the frozen day13 task; "
                   "NOT whole-task feasibility, NOT recursive feasibility; models = 4 dev-seed mass draws + nominal, "
                   "a finite set, not the continuous uncertainty range; affine per-model predictor about the reference "
                   "= approximate planning diagnosis, feasible candidates re-checked on the continuous plants",
             not_claimed=["active identification works", "conflict removable by probing", "learning needed"])
    json.dump(v, open(f"{OUT}/verdict.json", "w"), indent=1, default=str)
    print(json.dumps(v, indent=1, default=str))


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
