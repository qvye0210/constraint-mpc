"""
day15_tiers.py — three-tier planning-form diagnosis on the frozen day13 task. Repo root, PYTHONPATH=. ; env robmpc.
No training, no task change, no new data. Output: results/day15_tiers/

Tiers (same models, same window, same constraints; only the POLICY CLASS changes):
  A  common open loop      : one U for every model                              (= day14)
  B  common fixed feedback : u_k^i = v_k + K (x_k^i - xbar_k), v shared, K fixed (discrete LQR on the nominal
                             A,B with the author weights Q=10I/0.1I, R=1e-3I; not tuned)
  C  one-step-reveal oracle: u_0 shared for every model; after the first executed step the true model is assumed
                             known, so U_{1:N-1} may branch per model. Diagnostic upper bound on the value of
                             model information; NOT a deployable method.
Terminal condition (main): the authors' own task tolerance ||v_N||_2 <= goal_tol = 0.01, identical for all models.
  The exact v_N == 0 variant is kept as an ADDITIONAL diagnostic only; no model-specific slack is ever used.
Gating (Dolly's order): A(tol) -> if A fails, C -> B only if A fails and C succeeds.
Also reported: the spread of the models' true next states after the shared u_0 (raw, no sensor model ->
  distinguishability NOT claimed).
"""
import csv, json, os, sys
import numpy as np
import cvxpy as cp
from scipy.linalg import solve_discrete_are
sys.path.insert(0, ".")
import day12_tooltask as d12
import day13_reftrack as d13
import day14_conflict as d14
from day12_tooltask import A, B, b, ps, M, TR
from aux import TimeStepIntegratorContinuous

OUT = "results/day15_tiers"; os.makedirs(OUT, exist_ok=True)
N, GOAL_TOL = d14.N, 1e-2
V_LIM, U_LIM = d14.V_LIM, d14.U_LIM
FEAS = d14.FEAS
Qw = np.eye(2 * M) * 10; Qw[M:, M:] *= .01
K = -np.linalg.solve(np.eye(M) * 1e-3 + B.T @ solve_discrete_are(A, B, Qw, np.eye(M) * 1e-3) @ B,
                     B.T @ solve_discrete_are(A, B, Qw, np.eye(M) * 1e-3) @ A)   # fixed LQR gain, not tuned


def tier(models, ctx, mode, terminal="tol"):
    qbar, cs, rs, G, h = ctx["qbar"], ctx["cs"], ctx["rs"], ctx["G"], ctx["h"]
    xbar = np.hstack([qbar, ctx["qdbar"]])
    cons, obj = [], 0
    if mode == "A": U = cp.Variable((M, N)); cons += [U[:, N - 1] == 0]
    if mode == "B": V = cp.Variable((M, N)); cons += []
    if mode == "C": u0 = cp.Variable(M); Us = [cp.Variable((M, N - 1)) for _ in models]
    for i, mdl in enumerate(models):
        X = cp.Variable((2 * M, N + 1)); cons += [X[:, 0] == ctx["x0"]]
        for k in range(N):
            E, w = d14.stage_affine(mdl, qbar[k], ctx["qdbar"][k])
            uk = (U[:, k] if mode == "A" else
                  V[:, k] + K @ (X[:, k] - xbar[k]) if mode == "B" else
                  (u0 if k == 0 else Us[i][:, k - 1]))
            cons += [X[:, k + 1] == A @ X[:, k] + B @ (E @ uk + w), cp.abs(uk) <= U_LIM]
            if k == N - 1: cons += [uk == 0]
        cons += [cp.abs(X[:M, :]) <= np.pi, cp.abs(X[M:, :]) <= V_LIM, cp.abs(X[M:, 1:]) <= V_LIM - b,
                 cp.abs(X[:M, 1:] - qbar.T) <= TR, cp.sum(cp.multiply(G, X[:M, 1:]), axis=0) >= h]
        cons += [cp.norm(X[:M, k] - cs[k], 2) <= rs[k] for k in range(N + 1)]
        cons += [X[M:, N] == 0] if terminal == "exact" else [cp.norm(X[M:, N], 2) <= GOAL_TOL]
        obj = obj + sum(cp.sum_squares(X[:M, k] - qbar[k - 1]) for k in range(1, N + 1))
    p = cp.Problem(cp.Minimize(10 * obj), cons)
    try:
        p.solve(solver=cp.CLARABEL)
    except Exception as e:
        return dict(status=f"exception:{type(e).__name__}", value=None, u0=None)
    u0v = (U.value[:, 0] if mode == "A" else V.value[:, 0] + K @ (ctx["x0"] - xbar[0]) if mode == "B" else
           (u0.value if u0.value is not None else None)) if p.status in FEAS else None
    return dict(status=p.status, value=None if p.value is None else float(p.value),
                u0=None if u0v is None else np.asarray(u0v).ravel().copy())


def next_state_spread(models, ctx, u0):
    xs = {}
    for mdl in models:
        it = TimeStepIntegratorContinuous(dt=ps.dt); it.set_dyns(mdl["ps"].dyn_nom, mdl["dyn_err"])
        xs[mdl["name"]] = np.array(it.solve_time_step(ctx["x0"], u0), float)
    names = list(xs); d = [(a, c, float(np.abs(xs[a] - xs[c]).max())) for i, a in enumerate(names) for c in names[i + 1:]]
    return dict(pairwise_max_abs_state_diff={f"{a}|{c}": v for a, c, v in d},
                min_pair=min(v for *_, v in d), max_pair=max(v for *_, v in d),
                note="raw one-step state difference, no sensor/noise model -> distinguishability NOT established")


def main():
    models, _ = d14.build_models()
    zero = np.load("results/day13_reftrack/exec_zero.npz")["Q"]
    cps = {"start": (0, np.r_[d13.QP[0], np.zeros(M)]), "mid": (68, np.r_[zero[68], (zero[68] - zero[67]) / ps.dt])}
    rows, out = [], {}
    for name, (t0, x0) in cps.items():
        ctx = d14.make_ctx_window(t0, x0)
        r = {}
        for mdl in models:                                   # single-model reference under the task tolerance
            r[f"single_{mdl['name']}"] = tier([mdl], ctx, "A")
        r["A_tol"] = tier(models, ctx, "A"); r["A_exact"] = tier(models, ctx, "A", terminal="exact")
        singles_ok = all(r[f"single_{m['name']}"]["status"] in FEAS for m in models)
        if r["A_tol"]["status"] not in FEAS:
            r["C_tol"] = tier(models, ctx, "C")
            if r["C_tol"]["status"] in FEAS:
                r["B_tol"] = tier(models, ctx, "B")
                r["spread"] = next_state_spread(models, ctx, r["C_tol"]["u0"])
        st = ("SINGLE_MODEL_INFEASIBLE" if not singles_ok else
              "COMMON_OPEN_LOOP_OK_UNDER_TASK_TOLERANCE" if r["A_tol"]["status"] in FEAS else
              "NUMERICAL_UNRESOLVED" if r["A_tol"]["status"] != "infeasible" else
              "FEEDBACK_SUFFICIENT" if r.get("B_tol", {}).get("status") in FEAS else
              "INFORMATION_VALUE_SIGNAL" if r.get("C_tol", {}).get("status") in FEAS else
              "FIRST_COMMON_ACTION_BLOCKS_ALL")
        out[name] = dict(status=st, **{k: (v if k == "spread" else v["status"]) for k, v in r.items()})
        for k, v in r.items():
            if k != "spread": rows.append(dict(checkpoint=name, arm=k, status=v["status"], value=v["value"]))
        print(f"[{name}] {st}: " + json.dumps({k: v["status"] for k, v in r.items() if k != "spread"}))
    with open(f"{OUT}/tiers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["checkpoint", "arm", "status", "value"], restval=""); w.writeheader(); w.writerows(rows)
    v = dict(per_checkpoint=out, terminal_main=f"||v_N||_2 <= {GOAL_TOL} (authors' task tolerance, same for all models)",
             terminal_extra="exact v_N == 0 reported as A_exact only",
             feedback_gain="discrete LQR on nominal A,B with author weights; fixed, not tuned",
             oracle="C reveals the true model after one executed step; diagnostic upper bound, not deployable",
             learning_usability="SAFE_PROBE_VALUE_NOT_TESTED",
             scope="finite-horizon N=20 at two checkpoints, 4 dev-seed mass draws + nominal; not whole-task, "
                   "not recursive feasibility; affine per-model predictor about the reference",
             not_claimed=["probing can remove the conflict", "identification works", "learning needed"])
    json.dump(v, open(f"{OUT}/verdict.json", "w"), indent=1, default=str)
    print(json.dumps(v, indent=1, default=str))


if __name__ == "__main__":
    main()
