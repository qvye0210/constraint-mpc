"""
day27_method.py — pre-registered fair test of method (4): correction-consistency training. Repo root, PYTHONPATH=.
Controller env as day20 (NO_TERMINAL=1 ANG_BAND=1 RELIN=1 RHO_BAND=1 TS17=1 R17=1e-5).

PROTOCOL (frozen)
  payload: FIXED unknown m*=0.05 kg, c*=0.05 m for every episode (train and test); models never see it.
  train paths  : r0.28_t-25 r0.28_t+25 r0.3_t-25 r0.3_t+25 ; draws 300-305 (authors' 2% mass ratios only)
  test  paths  : r0.32_t-25 r0.32_t+25 ; draws 500-503 (never used before)
  data (shared by every arm, one file): nominal closed loop on train paths + probes (20 steps x 3 configs per train
       episode, multisine 5-20 Hz, |u|<=1.5, uniform energy); synthetic measurement noise sigma_q=1e-5 rad on positions,
       velocity noise sqrt(2) sigma_q/dt, independent; declared as synthetic, no device basis.
  model class: delta = Ka x + Kb_seg(s) u, Kb per 3 progress segments (aligned with the metric segments), Ka global.
  arms (same data, same class):
     mse    two-stage ridge: Kb_seg from probes (lambda by validation), Ka from closed-loop data
     cerr   from mse, Adam on 20-step CONSTRAINT-MARGIN prediction error along observed trajectories
     ours   from mse, Adam on the correction-consistency loss
            L_dec = V(m_hat) - V(m) + mu(m)^T (m_hat - m)   over the shared margin-correction QP (below)
            m = observed 20-step margins along the EXECUTED inputs (noisy states), m_hat = model-predicted margins.
            No true E, no counterfactual queries: everything comes from the same noisy trajectories.
     shrink mse with Kb_seg *= beta, beta by validation (strong regularised baseline)
  correction QP (shared): given a margin vector m over the horizon and the model's sensitivity S = dm/du0 (finite
            difference through the model), V(m) = min_d ||d||^2 + rho_pen * ||max(0, -(m + S d))||^2 ; mu = dV/dm.
  validation: draws 304,305 of the train set.  offline metrics on the TEST episodes (nominal closed loop under m*):
     one-step MSE, 20-step state error, 20-step margin error, correction error |V(m_hat)-V(m)| and ||d_hat - d||.
  pre-registered continuation criteria (reviewer): ours correction error <= 0.8 x best MSE-type arm,
     one-step MSE degradation <= 10%; closed loop (day20) task error >= 10% better than the best baseline,
     safety not worse, over 3 optimiser seeds.
"""
import csv, json, os, sys
import numpy as np
sys.path.insert(0, ".")
import day12_tooltask as d12
import day13_reftrack as d13
import day14_conflict as d14
import day17_oracle as d17
import day20_run as d20
import day22_divergence as d22
import day24_train as d24
import day25_ident as d25
from day12_tooltask import A, B, ps, M, fk

OUT = os.environ.get("OUT27", "results/day27")
N = 20; V_LIM = float(ps.v_amp_nom)
M_STAR, C_STAR = 0.05, 0.05
TRAIN_PATHS = ["r0.28_t-25", "r0.28_t+25", "r0.3_t-25", "r0.3_t+25"]
TEST_PATHS = ["r0.32_t-25", "r0.32_t+25"]
TRAIN_DRAWS, VAL_DRAWS, TEST_DRAWS = [300, 301, 302, 303], [304, 305], [500, 501, 502, 503]
BUDGET, CONFIGS, NSEG = 20, (0.0, 0.5, 1.0), 3
RHO_PEN, ITERS = 1e3, int(os.environ.get("ITERS", "60"))
TAU = np.r_[np.full(2, 0.3 * float(os.environ.get('RHO_HALF', '5e-3'))), np.deg2rad(1.0), np.full(M, 0.1), np.full(M, 0.1)]   # correction back-off per margin type
Bp = np.linalg.pinv(B)
RHO_LO, RHO_HI, ANG_TOL = 10e-3 - float(os.environ.get('RHO_HALF', '5e-3')), 10e-3 + float(os.environ.get('RHO_HALF', '5e-3')), np.deg2rad(3.0)


# ------------------------------------------------------------------ setup
def refs_for(names):
    out = {}
    for r0 in d17.RADII:
        for t0, t1 in d17.TILTS:
            nm = f"r{r0:g}_t{t0:+d}"
            if nm in names:
                ref = d17.build_ref(d17.TiltTask(d17.PHI, r0, d17.ZT, t0, t1)); ref["name"] = nm; out[nm] = ref
    return out


def plant_for(draw):
    return d17.make_model(draw, M_STAR, C_STAR)


def ctrls_for(ref):
    ctrls = []
    for i in range(NSEG):
        mid = int(round(0.5 * (i / NSEG + (i + 1) / NSEG) * (len(ref["Qp"]) - 1)))
        d17.TS_QREP = ref["Qp"][mid].copy(); ctrls.append(d17.Ctrl(d17.B0))
    return ctrls


def nominal_closed_loop_hold(ref, ctrls, plant):
    Tsteps = int(round(ref["T"] / ps.dt)); x = np.r_[ref["Qp"][0], np.zeros(M)]; x_end = np.r_[ref["Qp"][-1], np.zeros(M)]
    ptraj = x[:M][None].repeat(N + 1, 0); X, U, ts = [x.copy()], [], 0
    while ts < d20.NSTEPS:
        W = d20.window(ref, x, ts); seg = min(len(ctrls) - 1, int(d12.TASK.sigma(fk(x[:M])[1]) * len(ctrls))); ctrl = ctrls[seg]
        d20.fill(ctrl, x, W, None, False, ptraj[1:].copy())
        if not ctrl.solve(): break
        Xp, Up = ctrl.X.value.copy(), ctrl.U.value.copy()
        for j in range(min(d20.NHOLD, N)):
            x = np.array(plant["it"].solve_time_step(x, Up[:, j]), float); X.append(x.copy()); U.append(Up[:, j].copy())
        ts += min(d20.NHOLD, N); ptraj = np.vstack([Xp.T[min(d20.NHOLD, N):], np.repeat(Xp.T[-1:], min(d20.NHOLD, N), 0)])[:, :M]
        if ts >= Tsteps and np.linalg.norm(x - x_end) < d20.GOAL_TOL: break
    return np.array(X), np.array(U)


def episode_data(ref, ctrls, draw, rng):
    """nominal closed loop + 3 probes on the same plant; noisy measurements; returns per-episode dict."""
    d12.TASK = ref["task"]; d13.QP, d13.SS = ref["Qp"], ref["ss"]
    plant = plant_for(draw)
    X, U = nominal_closed_loop_hold(ref, ctrls, plant)                    # holds planned inputs for TMPC_MS
    Xc, Uc, Xnc = X[:-1], U, X[1:]
    Xcn, Xncn = d25.add_noise(Xc, Xnc, rng, True)
    probes = []
    for ci, s in enumerate(CONFIGS):
        q0 = ref["Qp"][int(s * (len(ref["Qp"]) - 1))]
        Xp, Up, Xnp = d25.run_probe(ref, plant, q0, d25.probe(draw, ci, BUDGET, "random", None))
        Xpn, Xnpn = d25.add_noise(Xp, Xnp, rng, True); probes.append((Xpn, Up, Xnpn))
    return dict(path=ref["name"], draw=draw, Xc=Xcn, Uc=Uc, Xnc=Xncn, probes=probes,
                sigma=np.array([d12.TASK.sigma(fk(x[:M])[1]) for x in Xc]))


# ------------------------------------------------------------------ margins, correction QP
def margins_state(x, ref):
    q = x[:M]; _, tip, a = fk(q)
    ang = np.arccos(np.clip(a @ ref["task"].a(ref["task"].sigma(tip)), -1, 1))
    r = d12.rho(q)
    return np.r_[r - RHO_LO, RHO_HI - r, ANG_TOL - ang, V_LIM - np.abs(x[M:]), V_LIM + x[M:]]


def rollout(Ka, Kb_seg, x, U, sig):
    X = [x]
    for k in range(U.shape[0]):
        seg = min(NSEG - 1, int(sig * NSEG))
        X.append(A @ X[-1] + B @ (U[k] + Ka @ X[-1] + Kb_seg[seg] @ U[k]))
    return np.array(X)


def margins_traj(Xr, ref): return np.concatenate([margins_state(x, ref) for x in Xr[1:]])


def V_and_mu(m, S):
    m = m - np.tile(TAU, N)                      # correction is required whenever a margin falls below its back-off
    """V(m) = min_d ||d||^2 + RHO_PEN*||max(0,-(m+Sd))||^2 (small convex QP, solved by projected Newton);
    mu = dV/dm = -2 RHO_PEN * max(0,-(m+Sd*))  (envelope theorem)."""
    d = np.zeros(S.shape[1])
    for _ in range(50):
        viol = np.maximum(0, -(m + S @ d)); act = viol > 0
        g = 2 * d - 2 * RHO_PEN * S[act].T @ viol[act]
        H = 2 * np.eye(len(d)) + 2 * RHO_PEN * S[act].T @ S[act]
        step = np.linalg.solve(H, g); d = d - step
        if np.linalg.norm(step) < 1e-10: break
    viol = np.maximum(0, -(m + S @ d))
    V = float(d @ d + RHO_PEN * viol @ viol); mu = -2 * RHO_PEN * viol
    return V, mu, d


def sensitivity(Ka, Kb_seg, x, U, sig, ref, h=1e-4):
    base = margins_traj(rollout(Ka, Kb_seg, x, U, sig), ref); S = np.zeros((len(base), M))
    for j in range(M):
        Up = U.copy(); Up[0, j] += h
        S[:, j] = (margins_traj(rollout(Ka, Kb_seg, x, Up, sig), ref) - base) / h
    return base, S


# ------------------------------------------------------------------ training
def windows(ep, stride=10):
    n = len(ep["Uc"]); return [t for t in range(0, n - N, stride)]


def fit_mse(eps, lam_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
    tr = [e for e in eps if e["draw"] in TRAIN_DRAWS]; va = [e for e in eps if e["draw"] in VAL_DRAWS]
    best = None
    for lam in lam_grid:
        Kb_seg = []
        for ci in range(NSEG):
            Xp = np.vstack([e["probes"][ci][0] for e in tr]); Up = np.vstack([e["probes"][ci][1] for e in tr]); Xnp = np.vstack([e["probes"][ci][2] for e in tr])
            d = d24.residual(Xp, Up, Xnp); Z = np.c_[np.ones((len(Up), 1)), Up]
            K = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ d); Kb_seg.append(K[1:, :].T)
        Xc = np.vstack([e["Xc"] for e in tr]); Uc = np.vstack([e["Uc"] for e in tr]); Xnc = np.vstack([e["Xnc"] for e in tr])
        seg = np.concatenate([np.minimum(NSEG - 1, (e["sigma"] * NSEG).astype(int)) for e in tr])
        d = d24.residual(Xc, Uc, Xnc) - np.array([Kb_seg[s] @ u for s, u in zip(seg, Uc)])
        Ka = np.linalg.solve(Xc.T @ Xc + 1.0 * np.eye(2 * M), Xc.T @ d).T
        e20 = state_err(Ka, np.array(Kb_seg), va)
        if best is None or e20 < best[0]: best = (e20, lam, Ka, np.array(Kb_seg))
    return dict(Ka=best[2], Kb_seg=best[3], lam=best[1], val=best[0])


def state_err(Ka, Kb_seg, eps):
    out = []
    for e in eps:
        for t in windows(e):
            Xr = rollout(Ka, Kb_seg, e["Xc"][t], e["Uc"][t:t + N], e["sigma"][t]); out.append(np.abs(Xr[-1] - e["Xnc"][t + N - 1]).max())
    return float(np.median(out))


def loss_on(kind, Ka, Kb_seg, eps, refs, stride=20):
    tot, n = 0.0, 0
    for e in eps:
        ref = refs[e["path"]]; d12.TASK = ref["task"]
        for t in windows(e, stride):
            x, U, sig = e["Xc"][t], e["Uc"][t:t + N], e["sigma"][t]
            m = margins_traj(np.vstack([x[None], e["Xnc"][t:t + N]]), ref)          # observed margins (noisy states)
            if kind == "cerr":
                mh = margins_traj(rollout(Ka, Kb_seg, x, U, sig), ref); tot += np.mean((mh - m) ** 2)
            else:
                mh, S = sensitivity(Ka, Kb_seg, x, U, sig, ref)
                Vh, _, _ = V_and_mu(mh, S); Vm, mu, _ = V_and_mu(m, S)
                tot += Vh - Vm + mu @ (mh - m)
            n += 1
    return tot / max(n, 1)


def fit_adam(kind, init, eps, refs, seed, lr=2e-3):
    rng = np.random.default_rng(seed)
    th = np.r_[init["Ka"].ravel(), init["Kb_seg"].ravel()]
    def unpack(t): return t[:M * 2 * M].reshape(M, 2 * M), t[M * 2 * M:].reshape(NSEG, M, M)
    tr = [e for e in eps if e["draw"] in TRAIN_DRAWS]
    f = lambda t: loss_on(kind, *unpack(t), tr, refs)
    mom = np.zeros_like(th); v = np.zeros_like(th); best = (f(th), th.copy()); h = 1e-5
    for it in range(ITERS):
        idx = rng.choice(len(th), size=min(40, len(th)), replace=False)           # random-coordinate finite differences
        L0 = f(th); g = np.zeros_like(th)
        for i in idx:
            e = np.zeros_like(th); e[i] = h; g[i] = (f(th + e) - L0) / h
        mom = 0.9 * mom + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        th = th - lr * mom / (np.sqrt(v) + 1e-12); L = f(th)
        if L < best[0]: best = (L, th.copy())
        if it % 10 == 0: print(f"      {kind} seed{seed} it {it:3d} loss {L:.6f}")
    Ka, Kb = unpack(best[1]); return dict(Ka=Ka, Kb_seg=Kb, loss=best[0])


# ------------------------------------------------------------------ evaluation
def evaluate(model, eps, refs):
    one, s20, mrg, verr, derr = [], [], [], [], []
    for e in eps:
        ref = refs[e["path"]]; d12.TASK = ref["task"]
        pred1 = e["Xc"] @ A.T + e["Uc"] @ B.T + (model["Ka"] @ e["Xc"].T).T @ B.T + np.array([B @ (model["Kb_seg"][min(NSEG-1, int(s*NSEG))] @ u) for s, u in zip(e["sigma"], e["Uc"])])
        one.append(np.abs(pred1 - e["Xnc"]).mean())
        for t in windows(e):
            x, U, sig = e["Xc"][t], e["Uc"][t:t + N], e["sigma"][t]
            Xr = rollout(model["Ka"], model["Kb_seg"], x, U, sig); s20.append(np.abs(Xr[-1] - e["Xnc"][t + N - 1]).max())
            m = margins_traj(np.vstack([x[None], e["Xnc"][t:t + N]]), ref); mh, S = sensitivity(model["Ka"], model["Kb_seg"], x, U, sig, ref)
            mrg.append(np.abs(mh - m).max())
            Vh, _, dh = V_and_mu(mh, S); Vm, _, dm = V_and_mu(m, S); verr.append(abs(Vh - Vm)); derr.append(np.linalg.norm(dh - dm))
    active = float(np.mean([v > 1e-9 for v in verr])) if verr else 0.0
    return dict(one_step=float(np.mean(one)), state20=float(np.median(s20)), margin20=float(np.median(mrg)), frac_windows_with_correction=active,
                V_err=float(np.median(verr)), d_err=float(np.median(derr)))


def main():
    os.makedirs(OUT, exist_ok=True); rng = np.random.default_rng(2027)
    refs = refs_for(TRAIN_PATHS + TEST_PATHS)
    stage = os.environ.get("STAGE", "all")
    if stage in ("data", "all"):
        eps = []
        for nm in TRAIN_PATHS:
            ctrls = ctrls_for(refs[nm])
            for dr in TRAIN_DRAWS + VAL_DRAWS: eps.append(episode_data(refs[nm], ctrls, dr, rng)); print(f"  data {nm} draw {dr}: {len(eps[-1]['Uc'])} cl steps")
        test = []
        for nm in TEST_PATHS:
            ctrls = ctrls_for(refs[nm])
            for dr in TEST_DRAWS: test.append(episode_data(refs[nm], ctrls, dr, rng)); print(f"  test {nm} draw {dr}: {len(test[-1]['Uc'])} cl steps")
        np.save(f"{OUT}/train_eps.npy", np.array(eps, dtype=object), allow_pickle=True); np.save(f"{OUT}/test_eps.npy", np.array(test, dtype=object), allow_pickle=True)
        budget = dict(closed_loop_steps=int(sum(len(e["Uc"]) for e in eps)), probe_steps=int(sum(len(p[1]) for e in eps for p in e["probes"])))
        json.dump(dict(payload=dict(m=M_STAR, c=C_STAR), train_paths=TRAIN_PATHS, test_paths=TEST_PATHS, train_draws=TRAIN_DRAWS,
                       val_draws=VAL_DRAWS, test_draws=TEST_DRAWS, probe=dict(steps=BUDGET, configs=CONFIGS, umax=1.5, band="5-20 Hz"),
                       noise=dict(sigma_q=d25.SIG_Q, velocity="sqrt(2)*sigma_q/dt independent", basis="synthetic"),
                       budget=budget, rho_pen=RHO_PEN), open(f"{OUT}/config.json", "w"), indent=1)
    if stage in ("train", "all"):
        eps = list(np.load(f"{OUT}/train_eps.npy", allow_pickle=True)); test = list(np.load(f"{OUT}/test_eps.npy", allow_pickle=True))
        va = [e for e in eps if e["draw"] in VAL_DRAWS]
        models = {"mse": fit_mse(eps)}; print(f"mse: lambda {models['mse']['lam']} val20 {models['mse']['val']:.4f}")
        # shrink baseline: beta by validation on the correction error (the quantity the method targets) AND state error
        best = None
        for beta in (0.25, 0.5, 0.75, 1.0):
            m = dict(Ka=models["mse"]["Ka"], Kb_seg=beta * models["mse"]["Kb_seg"]); ev = evaluate(m, va, refs)
            print(f"  shrink beta {beta}: val state20 {ev['state20']:.4f} d_err {ev['d_err']:.4f}")
            if best is None or ev["d_err"] < best[0]: best = (ev["d_err"], beta, m)
        models["shrink"] = dict(best[2], beta=best[1])
        for kind in ("cerr", "ours"):
            runs = [fit_adam(kind, models["mse"], eps, refs, seed=s) for s in (0, 1, 2)]
            models[kind] = min(runs, key=lambda r: r["loss"]); models[kind]["seeds"] = [r["loss"] for r in runs]
        report = {}
        for name, m in models.items():
            np.savez(f"{OUT}/model_{name}.npz", Ka=m["Ka"], Kb_seg=m["Kb_seg"])
            report[name] = evaluate(m, test, refs); print(f"{name:7s} TEST " + " ".join(f"{k} {v:.4f}" for k, v in report[name].items()))
        base = min(report["mse"]["d_err"], report["shrink"]["d_err"], report["cerr"]["d_err"])
        report["criteria"] = dict(ours_correction_vs_best_baseline=report["ours"]["d_err"] / base,
                                  ours_onestep_vs_mse=report["ours"]["one_step"] / report["mse"]["one_step"],
                                  offline_signal=bool(report["ours"]["d_err"] <= 0.8 * base and report["ours"]["one_step"] <= 1.1 * report["mse"]["one_step"]))
        json.dump(report, open(f"{OUT}/offline.json", "w"), indent=1, default=str); print(json.dumps(report["criteria"], indent=1))


if __name__ == "__main__":
    main()
