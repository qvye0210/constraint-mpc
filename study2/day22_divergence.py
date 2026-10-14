"""
day22_divergence.py — two offline checks that do not depend on the oracle closed loop. Repo root, PYTHONPATH=.

ROW 3  "prediction error changes the decision": along the NOMINAL closed-loop trajectory (trusted), at every K-th
       step solve, from the same state and reference window, (a) the nominal QP and (b) the true-model QP.
       Record  |u0_nom - u0_true| / u_lim,  and roll the nominal plan through the TRUE model to measure what the
       nominal controller does not see: tracking cost gap, worst tool margin gap, velocity margin gap.
ROW 2  "the error is learnable from allowed data": fit the day2 linear residual class (x_next = f_nom + dA x + dB u)
       on the nominal closed-loop transitions of the TRAIN draws, evaluate on the TEST draws: one-step and 20-step
       prediction error vs nominal and vs the true model.  Data = only what the nominal controller produced.
Output: results/day22_div/<path>/{divergence.csv, learnability.json, summary.json}
Env: PATHS17 (one path), TRAIN20 / TEST20 seed lists, EVERY (step stride, default 5), plus the day20 controller env
     (TS17, RHO_BAND, ANG_BAND, RELIN, R17) so the QP is the same one the closed loop uses.
"""
import csv, json, os, sys
import numpy as np
sys.path.insert(0, ".")
import day12_tooltask as d12
import day13_reftrack as d13
import day14_conflict as d14
import day17_oracle as d17
import day20_run as d20
from day12_tooltask import A, B, ps, M, TOL, fk
from aux import TimeStepIntegratorContinuous

OUT = os.environ.get("OUT22", "results/day22_div")
N = d20.N
TRAIN = [int(x) for x in os.environ.get("TRAIN20", "300,301,302,303,304,305").split(",")]
TEST = [int(x) for x in os.environ.get("TEST20", "400,401,402,403").split(",")]
EVERY = int(os.environ.get("EVERY", "5"))
V_LIM, U_LIM = float(ps.v_amp_nom), float(ps.u_amp_nom)


def setup(name):
    ref = None
    for r0 in d17.RADII:
        for t0, t1 in d17.TILTS:
            if f"r{r0:g}_t{t0:+d}" == name:
                ref = d17.build_ref(d17.TiltTask(d17.PHI, r0, d17.ZT, t0, t1)); ref["name"] = name
    assert ref is not None
    d12.TASK = ref["task"]; d13.QP, d13.SS = ref["Qp"], ref["ss"]
    nseg = d20.TS_SEGS if os.environ.get("TS17") else 1
    ctrls = []
    for i in range(nseg):
        mid = int(round(0.5 * (i / nseg + (i + 1) / nseg) * (len(ref["Qp"]) - 1)))
        d17.TS_QREP = ref["Qp"][mid].copy(); ctrls.append(d17.Ctrl(d17.B0))
    return ref, ctrls


def nominal_closed_loop(ref, ctrls, plant, seed):
    """run the nominal arm, but keep every state; also return the list of (x, ts, ptraj) decision points."""
    Tsteps = int(round(ref["T"] / ps.dt))
    x = np.r_[ref["Qp"][0], np.zeros(M)]; x_end = np.r_[ref["Qp"][-1], np.zeros(M)]
    ptraj = x[:M][None].repeat(N + 1, 0)
    X, U, pts, ts = [x.copy()], [], [], 0
    while ts < d20.NSTEPS:
        W = d20.window(ref, x, ts)
        seg = min(len(ctrls) - 1, int(d12.TASK.sigma(fk(x[:M])[1]) * len(ctrls))); ctrl = ctrls[seg]
        d20.fill(ctrl, x, W, None, False, ptraj[1:].copy())
        if not ctrl.solve(): break
        Xp, Up = ctrl.X.value.copy(), ctrl.U.value.copy()
        pts.append(dict(ts=ts, x=x.copy(), ptraj=ptraj.copy(), seg=seg, U_nom=Up, X_nom=Xp))
        x = np.array(plant["it"].solve_time_step(x, Up[:, 0]), float)
        X.append(x.copy()); U.append(Up[:, 0].copy()); ts += 1
        ptraj = np.vstack([Xp.T[1:], Xp.T[-1:]])[:, :M]
        if ts >= Tsteps and np.linalg.norm(x - x_end) < d20.GOAL_TOL: break
    return np.array(X), np.array(U), pts


def true_solve(ctrl, W, plant, pt):
    d20.fill(ctrl, pt["x"], W, plant, True, pt["ptraj"][1:].copy())
    ok = ctrl.solve()
    return (ctrl.X.value.copy(), ctrl.U.value.copy()) if ok else (None, None)


def roll_true(plant, x, U):
    X = [x]
    for k in range(U.shape[1]): X.append(np.array(plant["it"].solve_time_step(X[-1], U[:, k]), float))
    return np.array(X)


def plan_metrics(Xr, qref):
    """what the plan achieves when evaluated on the given state sequence."""
    ts_err = np.array([np.linalg.norm(fk(Xr[k, :M])[1] - d12.TASK.p(d12.TASK.sigma(fk(Xr[k, :M])[1]))) for k in range(1, N + 1)])
    return dict(track_cost=float(np.sum((Xr[1:, :M] - qref) ** 2)),
                tip_dev_max=float(ts_err.max()),
                rho_min=float(min(d12.rho(Xr[k, :M]) for k in range(1, N + 1))),
                vmargin_min=float((V_LIM - np.abs(Xr[1:, M:])).min()))


# ---------------------------------------------------------------- row 3
def divergence(ref, ctrls, seeds):
    rows = []
    for seed in seeds:
        dr = d17.draw(seed); plant = d17.make_model(seed, dr["m"], dr["c"])
        X, U, pts = nominal_closed_loop(ref, ctrls, plant, seed)
        for pt in pts[::EVERY]:
            W = d20.window(ref, pt["x"], pt["ts"]); ctrl = ctrls[pt["seg"]]
            X_true, U_true = true_solve(ctrl, W, plant, pt)
            if U_true is None: continue
            U_nom = pt["U_nom"]
            # what the nominal controller BELIEVES its plan does vs what it actually does on the true plant
            believed = plan_metrics(pt["X_nom"].T, W["qref"])
            actual = plan_metrics(roll_true(plant, pt["x"], U_nom), W["qref"])
            # what the true-model plan actually does (same state, same window)
            true_plan = plan_metrics(roll_true(plant, pt["x"], U_true), W["qref"])
            rows.append(dict(seed=seed, m=dr["m"], c=dr["c"], ts=pt["ts"], progress=float(d12.TASK.sigma(fk(pt["x"][:M])[1])),
                             du0_rel=float(np.abs(U_nom[:, 0] - U_true[:, 0]).max() / U_LIM),
                             du_seq_rel=float(np.abs(U_nom - U_true).max() / U_LIM),
                             track_cost_believed=believed["track_cost"], track_cost_actual=actual["track_cost"],
                             track_cost_true_plan=true_plan["track_cost"],
                             tip_dev_believed_mm=1e3 * believed["tip_dev_max"], tip_dev_actual_mm=1e3 * actual["tip_dev_max"],
                             tip_dev_true_plan_mm=1e3 * true_plan["tip_dev_max"],
                             rho_believed_mm=1e3 * believed["rho_min"], rho_actual_mm=1e3 * actual["rho_min"],
                             vmargin_believed=believed["vmargin_min"], vmargin_actual=actual["vmargin_min"]))
        print(f"  seed {seed}: {len([r for r in rows if r['seed'] == seed])} decision points, "
              f"closed-loop steps {len(X) - 1}")
    return rows


# ---------------------------------------------------------------- row 2
def collect(ref, ctrls, seeds):
    Xs, Us, Xn, tags = [], [], [], []
    for seed in seeds:
        dr = d17.draw(seed); plant = d17.make_model(seed, dr["m"], dr["c"])
        X, U, _ = nominal_closed_loop(ref, ctrls, plant, seed)
        Xs.append(X[:-1]); Us.append(U); Xn.append(X[1:]); tags += [seed] * len(U)
    return np.vstack(Xs), np.vstack(Us), np.vstack(Xn), np.array(tags)


def fit_residual(X, U, Xn):
    """day2 class: x_next = A x + B u + dA x + dB u   (least squares on the residual)."""
    R = Xn - (X @ A.T + U @ B.T); Z = np.c_[X, U]
    W, *_ = np.linalg.lstsq(Z, R, rcond=None)
    return W[:2 * M].T, W[2 * M:].T


def learnability(ref, ctrls):
    Xtr, Utr, Xntr, _ = collect(ref, ctrls, TRAIN)
    dA, dB = fit_residual(Xtr, Utr, Xntr)
    Xte, Ute, Xnte, tag = collect(ref, ctrls, TEST)
    one = dict(nominal=float(np.abs(Xnte - (Xte @ A.T + Ute @ B.T)).max(1).mean()),
               learned=float(np.abs(Xnte - (Xte @ A.T + Ute @ B.T + Xte @ dA.T + Ute @ dB.T)).max(1).mean()))
    # 20-step open-loop on test windows: nominal / learned / true model
    err = dict(nominal=[], learned=[], true=[])
    for seed in TEST:
        idx = np.where(tag == seed)[0]; dr = d17.draw(seed); plant = d17.make_model(seed, dr["m"], dr["c"])
        for t in idx[:-N:10]:
            x = Xte[t]; xt = x.copy(); xn = x.copy(); xl = x.copy(); xo = x.copy()
            for k in range(N):
                u = Ute[t + k]
                xt = np.array(plant["it"].solve_time_step(xt, u), float)
                xn = A @ xn + B @ u
                xl = A @ xl + B @ u + dA @ xl + dB @ u
                E, w = d14.stage_affine(plant, xo[:M], xo[M:]); xo = A @ xo + B @ (E @ u + w)
            err["nominal"].append(np.abs(xn - xt).max()); err["learned"].append(np.abs(xl - xt).max()); err["true"].append(np.abs(xo - xt).max())
    return dict(n_train=int(len(Utr)), n_test=int(len(Ute)), train_seeds=TRAIN, test_seeds=TEST,
                one_step_mean_abs=one,
                twenty_step_q95={k: float(np.quantile(v, .95)) for k, v in err.items()},
                twenty_step_median={k: float(np.median(v)) for k, v in err.items()},
                note="learned = single linear residual fitted across ALL training draws (one model for a distribution of payloads); "
                     "true = per-episode affine model with the actual payload. The gap learned->true is what a per-episode "
                     "identification could still recover.")


# ---------------------------------------------------------------- main
def main():
    name = os.environ["PATHS17"]; os.makedirs(f"{OUT}/{name}", exist_ok=True)
    ref, ctrls = setup(name)
    print(f"path {name}  TS17={os.environ.get('TS17','0')} RHO_BAND={os.environ.get('RHO_BAND','0')} "
          f"ANG_BAND={os.environ.get('ANG_BAND','0')} every {EVERY} steps")
    print("ROW 3 decision divergence along the nominal closed loop (test draws):")
    rows = divergence(ref, ctrls, TEST)
    keys = list(rows[0]) if rows else []
    with open(f"{OUT}/{name}/divergence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    def q(k, p=.95): return float(np.quantile([r[k] for r in rows], p))
    summ3 = dict(n_points=len(rows),
                 du0_rel_median=q("du0_rel", .5), du0_rel_q95=q("du0_rel", .95),
                 frac_points_du0_over_10pct=float(np.mean([r["du0_rel"] > 0.10 for r in rows])),
                 tip_dev_believed_vs_actual_mm=dict(believed_q95=q("tip_dev_believed_mm"), actual_q95=q("tip_dev_actual_mm"),
                                                    true_plan_q95=q("tip_dev_true_plan_mm")),
                 rho_optimism_mm_q95=float(np.quantile([r["rho_believed_mm"] - r["rho_actual_mm"] for r in rows], .95)),
                 vmargin_optimism_q95=float(np.quantile([r["vmargin_believed"] - r["vmargin_actual"] for r in rows], .95)),
                 track_cost_ratio_actual_over_true_plan_median=float(np.median([r["track_cost_actual"] / max(r["track_cost_true_plan"], 1e-12) for r in rows])))
    print(json.dumps(summ3, indent=1))
    print("ROW 2 learnability from nominal-controller data:")
    summ2 = learnability(ref, ctrls)
    print(json.dumps(summ2, indent=1))
    json.dump(dict(row3=summ3, row2=summ2), open(f"{OUT}/{name}/summary.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
