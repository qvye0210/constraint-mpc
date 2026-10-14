"""
day25_ident.py — pre-registered identification study. Repo root, PYTHONPATH=. ; day20 controller env applies.
Axes (all cells run, main cell fixed in advance = budget 20, noise ON):
  EXCITATION : random  (equal energy per joint, multi-sine)
               directed (energy allocated per joint in proportion to the task's own reference acceleration usage,
                         computed from the NOMINAL model and the reference only; same total energy as random)
  BUDGET     : probe steps per configuration in {10, 20, 40}; configurations = 3 (path progress 0, 0.5, 1), at rest
  NOISE      : measurement noise on identification data only: sigma_q = 1e-4 rad (encoder-class), velocity by
               finite difference of noisy positions. OFF / ON. The closed loop uses the true state (declared).
Model class (fixed): delta = Ka x + Kb_seg(s) u with Kb piecewise over 3 progress segments (aligned with the metric
  segments), Ka global.  Two-stage identification: Kb_seg from that configuration's probe only (u independent of x),
  then Ka from the nominal closed-loop data with Kb fixed.  Same class, same data, same fit for every cell.
Identifiability check per cell: condition number of the probe regressor and the LS residual.
Reported: ||E_seg - E_true||_F per segment (test draws), 20-step test error, closed loop on test draws (day20).
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
from day12_tooltask import A, B, ps, M, fk

OUT = os.environ.get("OUT25", "results/day25")
N = 20; V_LIM = float(ps.v_amp_nom)
TRAIN, TEST = d22.TRAIN, d22.TEST
BUDGETS = [int(b) for b in os.environ.get("BUDGETS", "10,20,40").split(",")]
CONFIGS = (0.0, 0.5, 1.0); NSEG = 3
PROBE_UMAX = float(os.environ.get('PROBE_UMAX', '1.5')); SIG_Q = float(os.environ.get('SIG_Q', '1e-5'))
Bp = np.linalg.pinv(B)


# ------------------------------------------------------------------ excitation
def task_usage(ref):
    """per-joint share of reference acceleration energy (nominal model + reference only)."""
    T = ref["T"]; ts = np.arange(1, int(round(T / ps.dt)))
    q = d13.q_ref_at(ts, T); acc = np.diff(q, n=2, axis=0) / ps.dt ** 2
    e = np.sqrt((acc ** 2).mean(0)); return e / e.sum()


def probe(seed, cfg_i, steps, kind, usage):
    rng = np.random.default_rng(20_000 + 100 * seed + cfg_i)
    t = np.arange(steps) * ps.dt
    U = np.zeros((steps, M))
    for j in range(M):
        f = rng.uniform(5.0, 20.0, 3); ph = rng.uniform(0, 2 * np.pi, 3)   # several cycles inside a 0.1-0.4 s probe
        U[:, j] = sum(np.sin(2 * np.pi * f[i] * t + ph[i]) for i in range(3)) / 3
    if kind == "directed":
        w = np.sqrt((0.8 * usage + 0.2 / M) * M)     # 80% task-directed + 20% uniform floor; same total energy
        U = U * w[None, :]
    return np.clip(PROBE_UMAX * U, -PROBE_UMAX, PROBE_UMAX)


def run_probe(ref, plant, q0, U):
    run_probe.cuts = getattr(run_probe, 'cuts', [])
    x = np.r_[q0, np.zeros(M)]; X, Us, Xn = [], [], []
    for k in range(len(U)):
        xn = np.array(plant["it"].solve_time_step(x, U[k]), float)
        why = d24.legal(xn, ref)
        if why == 'angle':                       # probe is not the task: allow up to 10 deg during probing
            _, tip, a = fk(xn[:M]); ang = np.arccos(np.clip(a @ ref['task'].a(ref['task'].sigma(tip)), -1, 1))
            why = None if ang <= np.deg2rad(10.0) else 'angle10'
        if why:
            run_probe.cuts.append(why); break
        X.append(x.copy()); Us.append(U[k].copy()); Xn.append(xn.copy()); x = xn
    return np.array(X), np.array(Us), np.array(Xn)


def add_noise(X, Xn, rng, on):
    """encoder-class position noise; velocity noise as the finite-difference amplification sqrt(2)*sigma_q/dt,
    applied independently to x and x_next (so the acceleration residual carries ~2*sigma_v/dt of noise)."""
    if not on: return X, Xn
    sv = np.sqrt(2) * SIG_Q / ps.dt
    Xm, Xnm = X.copy(), Xn.copy()
    Xm[:, :M] += rng.normal(0, SIG_Q, (len(X), M)); Xnm[:, :M] += rng.normal(0, SIG_Q, (len(X), M))
    Xm[:, M:] += rng.normal(0, sv, (len(X), M)); Xnm[:, M:] += rng.normal(0, sv, (len(X), M))
    return Xm, Xnm


# ------------------------------------------------------------------ identification
def fit_episode(ref, seed, kind, budget, noise, usage, rng):
    """per-episode: probe at 3 configs with this episode's plant, fit its own Kb_seg; Ka from shared closed-loop data."""
    dr = d17.draw(seed); plant = d17.make_model(seed, dr["m"], dr["c"]); Kb_seg, ident = [], []
    for ci, s in enumerate(CONFIGS):
        q0 = ref["Qp"][int(s * (len(ref["Qp"]) - 1))]
        X, U, Xn = run_probe(ref, plant, q0, probe(seed, ci, budget, kind, usage)); X, Xn = add_noise(X, Xn, rng, noise)
        d = d24.residual(X, U, Xn); Z = np.c_[np.ones((len(U), 1)), U]     # [1, u]: intercept absorbs w at rest
        K, res, rank, sv = np.linalg.lstsq(Z, d, rcond=None); Kb_seg.append(K[1:, :].T)
        ident.append(dict(config=s, samples=int(len(U)), cond=float(sv[0] / max(sv[-1], 1e-300)), rank=int(rank),
                          rms_residual=float(np.sqrt(np.mean((Z @ K - d) ** 2)))))
    z = np.load("results/day24/r0.3_t-25/data.npz"); Xc, Uc, Xnc = z["Xc"], z["Uc"], z["Xnc"]
    seg = np.array([min(NSEG - 1, int(d12.TASK.sigma(fk(x[:M])[1]) * NSEG)) for x in Xc])
    d = d24.residual(Xc, Uc, Xnc) - np.array([Kb_seg[si] @ u for si, u in zip(seg, Uc)])
    Ka = np.linalg.solve(Xc.T @ Xc + 1.0 * np.eye(2 * M), Xc.T @ d).T
    return dict(Ka=Ka, Kb_seg=np.array(Kb_seg)), ident


def E_err_episode(model, ref, seed):
    dr = d17.draw(seed); plant = d17.make_model(seed, dr["m"], dr["c"]); out = []
    for si, s in enumerate(CONFIGS):
        Et, _ = d14.stage_affine(plant, ref["Qp"][int(s * (len(ref["Qp"]) - 1))], np.zeros(M))
        out.append(float(np.linalg.norm(np.eye(M) + model["Kb_seg"][si] - Et)))
    return out


def fit_cell(ref, kind, budget, noise, usage, rng):
    Kb_seg, ident = [], []
    for ci, s in enumerate(CONFIGS):
        Xs, Us, Xns = [], [], []
        for seed in TRAIN:
            dr = d17.draw(seed); plant = d17.make_model(seed, dr["m"], dr["c"])
            q0 = ref["Qp"][int(s * (len(ref["Qp"]) - 1))]
            X, U, Xn = run_probe(ref, plant, q0, probe(seed, ci, budget, kind, usage))
            X, Xn = add_noise(X, Xn, rng, noise)
            Xs.append(X); Us.append(U); Xns.append(Xn)
        X, U, Xn = np.vstack(Xs), np.vstack(Us), np.vstack(Xns)
        d = d24.residual(X, U, Xn); Z = np.c_[X[:, :M], U]
        K, res, rank, sv = np.linalg.lstsq(Z, d, rcond=None)
        Kb_seg.append(K[M:, :].T)
        ident.append(dict(config=s, samples=int(len(U)), cond=float(sv[0] / sv[-1]), rank=int(rank),
                          rms_residual=float(np.sqrt(np.mean((Z @ K - d) ** 2)))))
    # stage 2: Ka from nominal closed-loop data with Kb fixed per segment
    z = np.load("results/day24/r0.3_t-25/data.npz"); Xc, Uc, Xnc = z["Xc"], z["Uc"], z["Xnc"]
    Xc, Xnc = add_noise(Xc, Xnc, rng, noise)
    seg = np.array([min(NSEG - 1, int(d12.TASK.sigma(fk(x[:M])[1]) * NSEG)) for x in Xc])
    d = d24.residual(Xc, Uc, Xnc) - np.array([Kb_seg[si] @ u for si, u in zip(seg, Uc)])
    lam = 1.0
    Ka = np.linalg.solve(Xc.T @ Xc + lam * np.eye(2 * M), Xc.T @ d).T
    return dict(Ka=Ka, Kb_seg=np.array(Kb_seg)), ident


def E_err(model, ref):
    out = []
    for si, s in enumerate(CONFIGS):
        errs = []
        for seed in TEST:
            dr = d17.draw(seed); plant = d17.make_model(seed, dr["m"], dr["c"])
            q = ref["Qp"][int(s * (len(ref["Qp"]) - 1))]
            Et, _ = d14.stage_affine(plant, q, np.zeros(M))
            errs.append(np.linalg.norm(np.eye(M) + model["Kb_seg"][si] - Et))
        out.append(float(np.mean(errs)))
    return out


# ------------------------------------------------------------------ main
def main():
    name = os.environ["PATHS17"]; os.makedirs(f"{OUT}/{name}", exist_ok=True)
    ref, _ = d22.setup(name); usage = task_usage(ref)
    print(f"path {name}; task acceleration usage per joint: {np.round(usage, 3)}")
    rows = []
    for noise in (False, True):
        for kind in ("random", "directed"):
            for budget in BUDGETS:
                rng = np.random.default_rng(777)
                tag = f"{kind}_b{budget}_n{int(noise)}"; es, ident = [], []
                for seed in TEST:                               # PER-EPISODE identification on the test draws
                    model, idn = fit_episode(ref, seed, kind, budget, noise, usage, rng); ident += idn
                    es.append(E_err_episode(model, ref, seed))
                    np.savez(f"{OUT}/{name}/model_{tag}_{seed}.npz", Ka=model["Ka"], Kb_seg=model["Kb_seg"])
                e = np.mean(es, 0)
                from collections import Counter; cuts = Counter(run_probe.cuts); run_probe.cuts = []
                rows.append(dict(kind=kind, budget=budget, noise=int(noise), tag=tag,
                                 E_err_start=e[0], E_err_mid=e[1], E_err_end=e[2], E_err_mean=float(np.mean(e)),
                                 cond_max=max(i["cond"] for i in ident), samples_min=min(i["samples"] for i in ident),
                                 rms_res_max=max(i["rms_residual"] for i in ident)))
                print(f"{tag:18s} E_err start/mid/end {e[0]:.2f}/{e[1]:.2f}/{e[2]:.2f}  cond {rows[-1]['cond_max']:.1e}  "
                      f"samples/cfg >= {rows[-1]['samples_min']}  rms_res {rows[-1]['rms_res_max']:.3f}  cuts {dict(cuts)}")
    with open(f"{OUT}/{name}/ident.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(dict(usage=usage.tolist(), configs=CONFIGS, budgets=BUDGETS, sigma_q=SIG_Q, umax=PROBE_UMAX,
                   main_cell="budget 20, noise ON", model="Ka global + Kb per progress segment, two-stage LS"),
              open(f"{OUT}/{name}/config.json", "w"), indent=1)


if __name__ == "__main__":
    main()
