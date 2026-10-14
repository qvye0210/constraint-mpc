"""
day19_oracle_debug.py — staged minimal repro for the oracle arm. Repo root, PYTHONPATH=. ; env robmpc.
No闭环 statistics, no new results directory beyond results/day19_debug/. Each stage prints PASS/FAIL and the
numbers behind it; stop at the first FAIL — the stage tells you which layer is broken.

A  wiring          : are E, w what they should be, and is the nominal arm really E=I, w=0?
B  QP consistency  : does the solved plan satisfy the dynamics the arm was given (manual residual)?
C  plan fidelity   : rolled through the TRUE continuous plant, is the oracle plan closer to reality than nominal?
D  first divergence: 30 closed-loop steps, both arms, per-step executed-vs-plan error and lateral error
E  sign check      : an arm given a WRONG payload (different seed). If wrong-model ≈ oracle, the compensation
                     enters with the wrong sign or at the wrong place; if wrong-model is worse than both, the
                     oracle wiring is fine and the fault is elsewhere.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, ".")
import day12_tooltask as d12
import day13_reftrack as d13
import day14_conflict as d14
import day17_oracle as d17
from day12_tooltask import A, B, ps, sim, M, TOL, fk
from aux import compute_goal_state, compute_model_error

OUT = "results/day19_debug"; os.makedirs(OUT, exist_ok=True)
N = 20
PATH = os.environ.get("PATH19", "r0.3_t-25")
SEED = int(os.environ.get("SEED19", "401"))
R17 = float(os.environ.get("R17", "1e-5"))


def setup():
    refs = []
    for r0 in d17.RADII:
        for t0, t1 in d17.TILTS:
            nm = f"r{r0:g}_t{t0:+d}"
            if nm != PATH: continue
            ref = d17.build_ref(d17.TiltTask(d17.PHI, r0, d17.ZT, t0, t1)); ref["name"] = nm; refs.append(ref)
    ref = refs[0]
    d12.TASK = ref["task"]; d13.QP, d13.SS = ref["Qp"], ref["ss"]
    d17.TS_QREP = ref["Qp"][len(ref["Qp"]) // 2].copy()
    dr = d17.draw(SEED); mdl = d17.make_model(SEED, dr["m"], dr["c"])
    print(f"path {ref['name']}  T={ref['T']:.2f}s  payload m={dr['m']:.4f} kg c={dr['c']:.3f} m  R={R17:g}")
    return ref, mdl


def window(ref, x, ts):
    """same parameter values the closed loop would set at step ts."""
    T = ref["T"]; PC, PR = ref["centres"], ref["radii"]
    ptraj = x[:M][None].repeat(N + 1, 0)
    cs, rs = sim.get_corridor_balls(ptraj, PC, PR, PC[-1])
    xg, _ = compute_goal_state(cs, rs, PC, PC[-1], return_index=True)
    qref = d13.q_ref_at(np.arange(ts + 1, ts + N + 1), T)
    xbar = np.hstack([qref, np.vstack([x[M:], np.diff(qref, axis=0) / ps.dt])])
    return dict(cs=np.asarray(cs).T, rs=np.asarray(rs), xg=xg, qref=qref, xbar=xbar)


def fill(ctrl, x, W, mdl, oracle):
    ctrl.x0.value = x; ctrl.cs.value = W["cs"]; ctrl.rs.value = W["rs"]; ctrl.Qref.value = W["qref"].T

    Es, ws = [], []
    for k in range(N):
        E, w = d14.stage_affine(mdl if oracle else None, W["xbar"][k, :M], W["xbar"][k, M:])
        ctrl.E[k].value, ctrl.w[k].value = E, w; Es.append(E); ws.append(w)
    qbar = W["xbar"][:, :M]
    G = np.array([d12.grad(q) for q in qbar]); r0 = np.array([d12.rho(q) for q in qbar])
    ctrl.G.value, ctrl.h.value, ctrl.Qb.value = G.T, (G * qbar).sum(1) - r0, qbar.T
    return np.array(Es), np.array(ws)


def roll_model(x, U, Es, ws):
    X = [x]
    for k in range(U.shape[1]): X.append(A @ X[-1] + B @ (Es[k] @ U[:, k] + ws[k]))
    return np.array(X)


def roll_true(mdl, x, U):
    X = [x]
    for k in range(U.shape[1]): X.append(np.array(mdl["it"].solve_time_step(X[-1], U[:, k]), float))
    return np.array(X)


def main():
    ref, mdl = setup()
    x0 = np.r_[ref["Qp"][0], np.zeros(M)]
    ctrl = d17.Ctrl(d17.B0)
    rep = {}

    # ---------------- A wiring
    W = window(ref, x0, 0)
    En, wn = fill(ctrl, x0, W, mdl, False)
    Eo, wo = fill(ctrl, x0, W, mdl, True)
    probe = compute_model_error(mdl["ps"].dyn_nom, mdl["dyn_err"], W["xbar"][5, :M], W["xbar"][5, M:], np.ones(M))
    pred = (Eo[5] - np.eye(M)) @ np.ones(M) + wo[5]
    a_ok = (np.abs(En - np.eye(M)).max() < 1e-12 and np.abs(wn).max() < 1e-12
            and np.abs(Eo - np.eye(M)).max() > 1e-3 and np.abs(probe - pred).max() < 1e-10)
    print(f"A wiring: nom |E-I| {np.abs(En - np.eye(M)).max():.1e} |w| {np.abs(wn).max():.1e} | "
          f"orc |E-I| {np.abs(Eo - np.eye(M)).max():.3f} |w|max {np.abs(wo).max():.3f} | "
          f"affine vs compute_model_error {np.abs(probe - pred).max():.1e} -> {'PASS' if a_ok else 'FAIL'}")
    rep["A"] = a_ok

    # ---------------- B QP consistency
    sols = {}
    for name, orc in (("nom", False), ("orc", True)):
        Es, ws = fill(ctrl, x0, W, mdl, orc)
        ok = ctrl.solve(); X, U = ctrl.X.value.copy(), ctrl.U.value.copy()
        man = roll_model(x0, U, Es, ws)
        res = np.abs(man - X.T).max()
        sols[name] = dict(X=X, U=U, Es=Es, ws=ws, status=ctrl.prob.status, res=res)
        print(f"B consistency {name}: status {ctrl.prob.status} dynamics residual |X_qp - X_manual| {res:.2e} "
              f"|u|max {np.abs(U).max():.2f} -> {'PASS' if res < 1e-6 else 'FAIL'}")
    rep["B"] = all(s["res"] < 1e-6 for s in sols.values())

    # ---------------- C plan fidelity against the true plant
    print("C plan fidelity (same U rolled through the TRUE plant):")
    for name in ("nom", "orc"):
        Xt = roll_true(mdl, x0, sols[name]["U"])
        err = np.abs(Xt - sols[name]["X"].T).max(1)
        print(f"   {name}: |plan - true| at k=1 {err[1]:.2e}  k=10 {err[10]:.2e}  k=20 {err[20]:.2e}")
    Xt_o = roll_true(mdl, x0, sols["orc"]["U"]); err_o = np.abs(Xt_o - sols["orc"]["X"].T).max()
    Xt_n = roll_true(mdl, x0, sols["nom"]["U"]); err_n = np.abs(Xt_n - sols["nom"]["X"].T).max()
    c_ok = err_o < err_n / 10
    print(f"   oracle plan error {err_o:.2e} vs nominal {err_n:.2e} -> {'PASS' if c_ok else 'FAIL'}")
    rep["C"] = c_ok

    # ---------------- D first divergence, 30 closed-loop steps at dt
    print("D closed loop, first N steps (lat mm | rho mm | |u|max | plan-vs-executed first step):")
    for name, orc in (("nom", False), ("orc", True)):
        x = x0.copy(); ptraj = x[:M][None].repeat(N + 1, 0); rows = []
        for ts in range(int(os.environ.get('NSTEP19', '30'))):
            Wt = window(ref, x, ts); Wt["xbar"][:, :M] = ptraj[1:]      # linearise about the shifted plan
            Es, ws = fill(ctrl, x, Wt, mdl, orc)
            if not ctrl.solve():
                print(f"   {name} t={ts} SOLVE FAILED status={ctrl.prob.status} "
                      f"|x| {np.abs(x).max():.3f} |v| {np.abs(x[M:]).max():.3f} "
                      f"LJ_max {max(np.abs(p.value).max() for p in ctrl.LJ) if getattr(ctrl,'LJ',None) else 0:.1f}")
                rows.append((ts, np.nan, np.nan, np.nan, np.nan)); break
            X, U = ctrl.X.value, ctrl.U.value
            xn = np.array(mdl["it"].solve_time_step(x, U[:, 0]), float)
            lat = 1e3 * np.linalg.norm(fk(xn[:M])[1] - d12.TASK.p(d12.TASK.sigma(fk(xn[:M])[1])))
            rows.append((ts, lat, 1e3 * d12.rho(xn[:M]), np.abs(U).max(), np.abs(xn - X[:, 1]).max()))
            x = xn; ptraj = np.vstack([X.T[1:], X.T[-1:]])[:, :M]
        for r in rows[::5] + rows[-1:]:
            print(f"   {name} t={r[0]:2d} lat {r[1]:7.3f} rho {r[2]:7.3f} |u| {r[3]:6.2f} |x-plan| {r[4]:.2e}")
    rep["D"] = "see table"

    # ---------------- E sign check with a WRONG payload
    dr_w = d17.draw(SEED + 1); mdl_w = d17.make_model(SEED, dr_w["m"], dr_w["c"])   # wrong payload, same base masses
    print(f"E wrong-model arm (payload m={dr_w['m']:.4f} instead of {mdl['m']:.4f}):")
    res = {}
    for name, use in (("orc(true)", mdl), ("wrong", mdl_w), ("nom", None)):
        x = x0.copy(); ptraj = x[:M][None].repeat(N + 1, 0); lat_max = 0.0; rho_min = 1.0
        for ts in range(60):
            Wt = window(ref, x, ts); Wt["xbar"][:, :M] = ptraj[1:]
            fill(ctrl, x, Wt, use, use is not None)
            if not ctrl.solve():
                print(f"   {name} stopped at t={ts} status={ctrl.prob.status}"); break
            X, U = ctrl.X.value, ctrl.U.value
            x = np.array(mdl["it"].solve_time_step(x, U[:, 0]), float)     # TRUE plant is always the real one
            lat_max = max(lat_max, 1e3 * np.linalg.norm(fk(x[:M])[1] - d12.TASK.p(d12.TASK.sigma(fk(x[:M])[1]))))
            rho_min = min(rho_min, d12.rho(x[:M]))
            ptraj = np.vstack([X.T[1:], X.T[-1:]])[:, :M]
        res[name] = (lat_max, 1e3 * rho_min)
        print(f"   {name:9s} after 60 steps: lat_max {lat_max:7.3f} mm  rho_min {1e3 * rho_min:7.3f} mm")
    rep["E"] = {k: v for k, v in res.items()}
    json.dump({k: (v if not isinstance(v, dict) else {a: list(b) for a, b in v.items()}) for k, v in rep.items()},
              open(f"{OUT}/stages.json", "w"), indent=1, default=str)
    print("\nreading: A/B/C FAIL -> the fault is in that layer. All PASS but D shows oracle diverging -> the fault is "
          "in the closed-loop interface (shift, reference index, or corridor update). E: wrong-model ~ oracle means "
          "the compensation is applied with the wrong sign or to the wrong states.")


if __name__ == "__main__":
    main()
