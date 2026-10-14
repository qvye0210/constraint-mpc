"""
day20_run.py — clean experiment driver. The closed loop is the one verified in day19 (solve, execute, shift);
the controller, models, task family and metrics are reused from day12/13/17. day17_oracle.py is NOT modified.
Repo root, PYTHONPATH=. ; env robmpc.  Output: results/day20_run/<path>/

Env:
  TS17=1        task-space tracking objective (day17.Ctrl); unset = joint-space (day13 objective)
  TERM_TOL=1    terminal ||v_N|| <= 0.01 (authors' task tolerance) instead of the exact v_N = 0
  R17=1e-5      input weight of the joint-space objective
  PATHS17=name  one path per process (required)
  ARMS20=nom,orc[,wrong]   arms to run (default nom,orc)
  SEEDS20=400,401,402,403  payload draws
  TMPC_MS=10    replanning period; the inner loop always runs at dt and holds the planned inputs in between
  RELIN=0|1     exact-rho recheck + one re-linearisation (default 0 = as in day19)
"""
import csv, json, os, sys, time
import numpy as np
sys.path.insert(0, ".")
import day12_tooltask as d12
import day13_reftrack as d13
import day14_conflict as d14
import day17_oracle as d17
from day12_tooltask import A, B, ps, sim, M, TOL, LAT_TOL, ANG_TOL, END_TOL, GOAL_TOL, NSTEPS, fk
from aux import compute_goal_state

OUT = os.environ.get("OUT20", "results/day20_run")
N = 20
SEEDS = [int(x) for x in os.environ.get("SEEDS20", "400,401,402,403").split(",")]
ARMS = os.environ.get("ARMS20", "nom,orc").split(",")
NHOLD = max(1, int(os.environ.get("TMPC_MS", "10")) // int(round(ps.dt * 1000)))
RELIN = os.environ.get("RELIN") == "1"
TS_SEGS = int(os.environ.get("TS_SEGS", "4"))   # task-space metric segments (1 = single constant metric)


def window(ref, x, ts):
    """identical to day19.window: corridor from the current configuration, reference from the timing law."""
    PC, PR = ref["centres"], ref["radii"]
    ptraj0 = x[:M][None].repeat(N + 1, 0)
    cs, rs = sim.get_corridor_balls(ptraj0, PC, PR, PC[-1])
    compute_goal_state(cs, rs, PC, PC[-1], return_index=True)
    qref = d13.q_ref_at(np.arange(ts + 1, ts + N + 1), ref["T"])
    xbar = np.hstack([qref, np.vstack([x[M:], np.diff(qref, axis=0) / ps.dt])])
    return dict(cs=np.asarray(cs).T, rs=np.asarray(rs), qref=qref, xbar=xbar)


def fill(ctrl, x, W, mdl, oracle, qbar_pos):
    ctrl.x0.value = x; ctrl.cs.value = W["cs"]; ctrl.rs.value = W["rs"]; ctrl.Qref.value = W["qref"].T
    for k in range(N):
        if isinstance(mdl, dict) and "Kb_seg" in mdl:       # piecewise input gain (day25), segment by plan progress
            _s = min(mdl["Kb_seg"].shape[0] - 1, int(d12.TASK.sigma(fk(qbar_pos[k])[1]) * mdl["Kb_seg"].shape[0]))
            E, w = np.eye(M) + mdl["Kb_seg"][_s], mdl["Ka"] @ np.r_[qbar_pos[k], W["xbar"][k, M:]]
        elif isinstance(mdl, dict) and "Ka" in mdl:        # learned linear residual, frozen at the plan like the oracle
            E, w = np.eye(M) + mdl["Kb"], mdl["Ka"] @ np.r_[qbar_pos[k], W["xbar"][k, M:]]
            if "true" in mdl:                                  # hybrid diagnostics: swap one part for the truth
                Et, wt = d14.stage_affine(mdl["true"], qbar_pos[k], W["xbar"][k, M:])
                if mdl["swap"] == "E": E = Et
                if mdl["swap"] == "w": w = wt
                if mdl["swap"] == "alpha": E, w = np.eye(M) + mdl["alpha"] * (Et - np.eye(M)), mdl["alpha"] * wt
        else:
            E, w = d14.stage_affine(mdl if oracle else None, qbar_pos[k], W["xbar"][k, M:])
        ctrl.E[k].value, ctrl.w[k].value = E, w
    G = np.array([d12.grad(q) for q in qbar_pos]); r0 = np.array([d12.rho(q) for q in qbar_pos])
    ctrl.G.value, ctrl.h.value, ctrl.Qb.value = G.T, (G * qbar_pos).sum(1) - r0, qbar_pos.T
    if os.environ.get("ANG_BAND"):
        tol = np.deg2rad(3.0); h = 1e-6
        Ga = np.zeros((len(qbar_pos), M)); a0 = np.zeros(len(qbar_pos))
        for i, q in enumerate(qbar_pos):
            s_i = d12.TASK.sigma(fk(q)[1]); aref = d12.TASK.a(s_i)
            f = lambda qq: float(np.arccos(np.clip(fk(qq)[2] @ aref, -1, 1)))
            a0[i] = f(q)
            for j in range(M):
                e = np.zeros(M); e[j] = h
                Ga[i, j] = (f(q + e) - f(q - e)) / (2 * h)
        ctrl.Ga.value = Ga.T
        ctrl.ha_lo.value = (Ga * qbar_pos).sum(1) - a0 - tol
        ctrl.ha_up.value = (Ga * qbar_pos).sum(1) - a0 + tol
    if os.environ.get("RHO_BAND"):
        band = d17.RHO_BAND_HALF
        ctrl.h.value = (G * qbar_pos).sum(1) - r0 + ((d12.STANDOFF - d12.R_TOOL - d12.D_SAFE) - band)   # rho >= design - band
        ctrl.h_up.value = (G * qbar_pos).sum(1) - r0 + ((d12.STANDOFF - d12.R_TOOL - d12.D_SAFE) + band)  # rho <= design + band


def episode(ref, mdl_plant, mdl_ctrl, ctrls, tag):
    """mdl_plant: the true plant. mdl_ctrl: what the controller believes (None = nominal)."""
    d12.TASK = ref["task"]; d13.QP, d13.SS = ref["Qp"], ref["ss"]
    Tsteps = int(round(ref["T"] / ps.dt))
    x = np.r_[ref["Qp"][0], np.zeros(M)]; x_end = np.r_[ref["Qp"][-1], np.zeros(M)]
    ptraj = x[:M][None].repeat(N + 1, 0)
    Qe, Ue, solve_ms, vmax, status, relins, ts = [x[:M].copy()], [], [], 0.0, "truncated", 0, 0
    while ts < NSTEPS:
        W = window(ref, x, ts)
        seg = min(len(ctrls) - 1, int(d12.TASK.sigma(fk(x[:M])[1]) * len(ctrls)))
        ctrl = ctrls[seg]
        qbar = ptraj[1:].copy()
        for attempt in range(2 if RELIN else 1):
            fill(ctrl, x, W, mdl_ctrl, mdl_ctrl is not None, qbar)
            t0 = time.perf_counter(); ok = ctrl.solve(); solve_ms.append(1e3 * (time.perf_counter() - t0))
            if not ok:
                status = f"infeasible:{ctrl.prob.status}"; break
            X, U = ctrl.X.value, ctrl.U.value
            if not RELIN: break
            _r = [d12.rho(X[:M, k]) for k in range(1, N + 1)]
            _lo = (d12.STANDOFF - d12.R_TOOL - d12.D_SAFE) - d17.RHO_BAND_HALF if os.environ.get('RHO_BAND') else 0.0
            _hi = (d12.STANDOFF - d12.R_TOOL - d12.D_SAFE) + d17.RHO_BAND_HALF if os.environ.get('RHO_BAND') else 1e9
            if min(_r) >= _lo - TOL and max(_r) <= _hi + TOL: break
            qbar = X[:M, 1:].T.copy(); relins += 1
        if not ok: break
        for j in range(min(NHOLD, N)):
            x = np.array(mdl_plant["it"].solve_time_step(x, U[:, j]), float)
            Qe.append(x[:M].copy()); Ue.append(U[:, j].copy()); vmax = max(vmax, float(np.abs(x[M:]).max()))
        ts += min(NHOLD, N)
        ptraj = np.vstack([X.T[min(NHOLD, N):], np.repeat(X.T[-1:], min(NHOLD, N), 0)])[:, :M]
        if ts >= Tsteps and np.linalg.norm(x - x_end) < GOAL_TOL: status = "success"; break
    Q = np.array(Qe); tm = d12.tool_metrics(Q); _, tip, _ = fk(Q[-1])
    r = dict(tag=tag, status=status, steps=len(Q) - 1, T_steps=Tsteps,
             lat_max=float(tm["lat"].max()), ang_max_deg=float(np.rad2deg(tm["ang"].max())),
             end_err=float(np.linalg.norm(tip - ref["task"].p(1.0))), rho_min=float(tm["rho"].min()),
             violations=int((tm["rho"] < -TOL).sum()),
             rho_max=float(tm["rho"].max()),
             band_high_steps=int((tm["rho"] > (d12.STANDOFF - d12.R_TOOL - d12.D_SAFE) + d17.RHO_BAND_HALF + TOL).sum()), v_exec_max=vmax, relins=relins,
             u_max=float(np.abs(Ue).max()) if Ue else None,
             u_rms=float(np.sqrt(np.mean(np.square(Ue)))) if Ue else None,
             du_max=float(np.abs(np.diff(Ue, axis=0)).max() / ps.dt) if len(Ue) > 1 else None,
             solve_ms_med=float(np.median(solve_ms)), solve_ms_p99=float(np.quantile(solve_ms, .99)),
             n_segments=len(ctrls))
    r["PASS"] = (status == "success" and r["lat_max"] <= LAT_TOL and r["ang_max_deg"] <= np.rad2deg(ANG_TOL)
                 and r["end_err"] <= END_TOL and r["rho_min"] >= -TOL)
    np.savez(f"{OUT}/{ref['name']}/exec_{tag}.npz", Q=Q, U=np.array(Ue), lat=tm["lat"], ang=tm["ang"], rho=tm["rho"])
    print(f"{ref['name']:12s} {tag:22s} {status[:12]:12s} steps {r['steps']:4d} lat {1e3*r['lat_max']:7.2f} "
          f"ang {r['ang_max_deg']:6.2f} rho {1e3*r['rho_min']:7.2f} viol {r['violations']:3d} PASS {r['PASS']}")
    return r


def main():
    name = os.environ["PATHS17"]
    ref = None
    for r0 in d17.RADII:
        for t0, t1 in d17.TILTS:
            if f"r{r0:g}_t{t0:+d}" != name: continue
            ref = d17.build_ref(d17.TiltTask(d17.PHI, r0, d17.ZT, t0, t1)); ref["name"] = name
    assert ref is not None, f"path {name} not found or failed its reference check"
    os.makedirs(f"{OUT}/{name}", exist_ok=True)
    nseg = TS_SEGS if os.environ.get("TS17") else 1
    ctrls, qreps = [], []
    for i in range(nseg):                                        # one metric (and controller) per path segment
        lo, hi = i / nseg, (i + 1) / nseg
        mid = int(round(0.5 * (lo + hi) * (len(ref["Qp"]) - 1)))
        d17.TS_QREP = ref["Qp"][mid].copy(); qreps.append(mid)
        ctrls.append(d17.Ctrl(d17.B0))
    ctrl = ctrls
    print(f"path {name} T={ref['T']:.2f}s  TS17={os.environ.get('TS17', '0')} TERM_TOL={os.environ.get('TERM_TOL', '0')} "
          f"R17={os.environ.get('R17', '1e-3')} TMPC={NHOLD * int(round(ps.dt * 1000))}ms RELIN={int(RELIN)} "
          f"arms={ARMS} metric_segments={len(ctrls)} at path idx {qreps} "
          f"rho_band={os.environ.get('RHO_BAND', '0')}")
    rows = []
    for seed in SEEDS:
        dr = d17.draw(seed)
        if os.environ.get("PAYLOAD"):                                   # fixed unknown payload protocol (day27)
            _m, _c = (float(v) for v in os.environ["PAYLOAD"].split(",")); dr = dict(seed=seed, m=_m, c=_c)
        plant = d17.make_model(seed, dr["m"], dr["c"])
        wrong = d17.make_model(seed, d17.draw(seed + 1)["m"], d17.draw(seed + 1)["c"])
        for arm in ARMS:
            mse = dict(np.load(f"results/day23_mse/{name}/mse_model.npz")) if "mse" in ARMS else None
            extra27 = {a: dict(np.load(f"results/day27/model_{a[:-2]}.npz")) for a in ARMS if a.endswith("27")}
            extra = {a: dict(np.load(f"results/day24/{name}/model_{a}.npz")) for a in ARMS if a in ("mse1", "mseN", "cerr", "mse1_cl")}
            extra.update({a: dict(np.load(f"results/day25/{name}/model_{a}_{seed}.npz")) for a in ARMS if a.startswith(("random_", "directed_"))})
            for a in [x for x in ARMS if x.startswith("shrink")]:          # shrinkβ_<tag>: identified Kb_seg scaled by β toward nominal
                beta, base = a.split("_", 1)[0][6:], a.split("_", 1)[1]
                m = dict(np.load(f"results/day25/{name}/model_{base}_{seed}.npz")); m["Kb_seg"] = float(beta) * m["Kb_seg"]; m["Ka"] = float(beta) * m["Ka"]
                extra[a] = m
            alpha = {f"alpha{a}": dict(true=plant, swap="alpha", alpha=a/100, Ka=np.zeros((M, 2*M)), Kb=np.zeros((M, M))) for a in (25, 50, 75, 90)}
            ctrl_mdl = {"nom": None, "orc": plant, "wrong": wrong, "mse": mse, **alpha,
                        "mse_trueE": dict(mse, true=plant, swap="E") if mse else None,
                        "mse_trueW": dict(mse, true=plant, swap="w") if mse else None, **extra, **extra27}[arm]
            r = episode(ref, plant, ctrl_mdl, ctrl, f"{seed}_{arm}")
            r.update(arm=arm, seed=seed, path=name, m=dr["m"], c=dr["c"]); rows.append(r)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(f"{OUT}/{name}/episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys, restval=""); w.writeheader(); w.writerows(rows)
    summ = {}
    for arm in ARMS:
        R = [r for r in rows if r["arm"] == arm]
        summ[arm] = dict(n=len(R), passed=sum(r["PASS"] for r in R),
                         lat_med_mm=float(np.median([1e3 * r["lat_max"] for r in R])),
                         ang_med_deg=float(np.median([r["ang_max_deg"] for r in R])),
                         steps_med=float(np.median([r["steps"] for r in R])),
                         rho_min_mm=float(1e3 * min(r["rho_min"] for r in R)),
                         violations=sum(r["violations"] for r in R))
    json.dump(dict(path=name, config=dict(TS17=os.environ.get("TS17"), TERM_TOL=os.environ.get("TERM_TOL"),
                                          R17=os.environ.get("R17"), tmpc_ms=NHOLD * int(round(ps.dt * 1000)),
                                          relin=RELIN, seeds=SEEDS, arms=ARMS),
                   summary=summ), open(f"{OUT}/{name}/verdict.json", "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
