"""
day24_train.py — stage A (excitation data) + stage B (three of the four training objectives). Repo root, PYTHONPATH=.
Everything here is shared by every learning arm: same data, same 108-parameter class  delta = Ka x + Kb u,
same validation split, same closed-loop driver (day20).  "Our" (correction consistency) is a separate script.

STAGE A  excitation (FROZEN before running): per TRAIN draw, from the task start state, 40 steps of multi-sine
         virtual acceleration with random phases, |u| <= 10, executed on the TRUE plant; the segment is cut at the
         first step where |v| > 0.5 v_lim, rho leaves [5,15] mm, the tool-axis error > 3 deg, or the corridor is left.
         Test draws get no excitation.  Data budget = 6 draws x <=40 steps.
STAGE B  objectives on {nominal closed-loop data} U {excitation}:
         mse1   one-step ridge (lambda by held-out draws)              -> results/day24/<path>/model_mse1.npz
         mseN   20-step rollout MSE, Adam from the mse1 solution        -> model_mseN.npz
         cerr   20-step rollout error on the CONSTRAINT MARGINS
                (rho, tool-axis angle, velocity margins), Adam from mse1 -> model_cerr.npz
         plus the old closed-loop-only ridge (mse1_cl) for reference.
DIAGNOSTIC (not a training target): per model, ||E_learned - E_true||_F averaged over the test-draw reference
         configurations, and the 20-step open-loop error vs nominal and true.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, ".")
import day12_tooltask as d12
import day13_reftrack as d13
import day14_conflict as d14
import day17_oracle as d17
import day20_run as d20
import day22_divergence as d22
from day12_tooltask import A, B, ps, M, fk

OUT = os.environ.get("OUT24", "results/day24")
N = 20; V_LIM = float(ps.v_amp_nom)
TRAIN, TEST = d22.TRAIN, d22.TEST
PROBE_STEPS, PROBE_UMAX, PROBE_VFRAC = 40, 3.0, 0.5
RHO_LO, RHO_HI = 10e-3 - float(os.environ.get('RHO_HALF', '5e-3')), 10e-3 + float(os.environ.get('RHO_HALF', '5e-3')); ANG_TOL = np.deg2rad(3.0)
Bp = np.linalg.pinv(B)


# ------------------------------------------------------------------ stage A
def probe_inputs(seed):
    rng = np.random.default_rng(10_000 + seed)
    t = np.arange(PROBE_STEPS) * ps.dt
    freqs = rng.uniform(1.0, 4.0, size=(M, 3)); phases = rng.uniform(0, 2 * np.pi, size=(M, 3))
    U = np.zeros((PROBE_STEPS, M))
    for j in range(M):
        U[:, j] = sum(np.sin(2 * np.pi * freqs[j, i] * t + phases[j, i]) for i in range(3)) / 3
    return PROBE_UMAX * U


def legal(x, ref):
    q = x[:M]
    if np.abs(x[M:]).max() > PROBE_VFRAC * V_LIM: return "velocity"
    r = d12.rho(q)
    if not (RHO_LO - 1e-6 <= r <= RHO_HI + 1e-6): return "rho"
    _, tip, a = fk(q)
    if np.arccos(np.clip(a @ ref["task"].a(ref["task"].sigma(tip)), -1, 1)) > ANG_TOL: return "angle"
    j = int(np.argmin(np.linalg.norm(ref["centres"] - q, axis=1)))
    if np.linalg.norm(q - ref["centres"][j]) > ref["radii"][j]: return "corridor"
    return None


def excite(ref, seed):
    dr = d17.draw(seed); plant = d17.make_model(seed, dr["m"], dr["c"])
    x = np.r_[ref["Qp"][0], np.zeros(M)]; U = probe_inputs(seed)
    X, Us, Xn, cut = [], [], [], None
    for k in range(PROBE_STEPS):
        xn = np.array(plant["it"].solve_time_step(x, U[k]), float)
        why = legal(xn, ref)
        if why: cut = (k, why); break
        X.append(x.copy()); Us.append(U[k].copy()); Xn.append(xn.copy()); x = xn
    return np.array(X), np.array(Us), np.array(Xn), cut


# ------------------------------------------------------------------ stage B helpers
def residual(X, U, Xn): return (Xn - X @ A.T) @ Bp.T - U           # acceleration residual delta


def ridge(X, U, Xn, lam):
    Z = np.c_[X, U]; d = residual(X, U, Xn)
    K = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ d).T
    return K[:, :2 * M], K[:, 2 * M:]


def rollout_err(Ka, Kb, X, U, Xn, idx, H=N):
    out = []
    for t in idx:
        xm = X[t].copy()
        for k in range(H): xm = A @ xm + B @ (U[t + k] + Ka @ xm + Kb @ U[t + k])
        out.append(np.abs(xm - Xn[t + H - 1]).max())
    return np.array(out)


def margins(x, ref):
    q = x[:M]; _, tip, a = fk(q)
    ang = np.arccos(np.clip(a @ ref["task"].a(ref["task"].sigma(tip)), -1, 1))
    return np.r_[d12.rho(q) - RHO_LO, RHO_HI - d12.rho(q), ANG_TOL - ang, V_LIM - np.abs(x[M:])]


def fit_adam(Ka0, Kb0, X, U, Xn, idx, loss_kind, ref, iters=300, lr=1e-3):
    """20-step rollout objective, gradient by finite differences on the 108 parameters (small, so affordable)."""
    theta = np.r_[Ka0.ravel(), Kb0.ravel()]
    def unpack(th): return th[:M * 2 * M].reshape(M, 2 * M), th[M * 2 * M:].reshape(M, M)
    def loss(th):
        Ka, Kb = unpack(th); tot = 0.0
        for t in idx:
            xm = X[t].copy()
            for k in range(N):
                xm = A @ xm + B @ (U[t + k] + Ka @ xm + Kb @ U[t + k])
                xt = Xn[t + k]
                tot += (np.sum((xm - xt) ** 2) if loss_kind == "mseN" else np.sum((margins(xm, ref) - margins(xt, ref)) ** 2))
        return tot / len(idx)
    m = np.zeros_like(theta); v = np.zeros_like(theta); best = (loss(theta), theta.copy()); h = 1e-6
    for it in range(iters):
        g = np.zeros_like(theta); L0 = loss(theta)
        for i in range(len(theta)):
            e = np.zeros_like(theta); e[i] = h; g[i] = (loss(theta + e) - L0) / h
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        theta = theta - lr * m / (np.sqrt(v) + 1e-12)
        L = loss(theta)
        if L < best[0]: best = (L, theta.copy())
        if it % 50 == 0: print(f"      {loss_kind} it {it:3d} loss {L:.5f}")
    return unpack(best[1]), best[0]


def E_error(Ka, Kb, ref, seeds):
    errs = []
    for seed in seeds:
        dr = d17.draw(seed); plant = d17.make_model(seed, dr["m"], dr["c"])
        for q in ref["Qp"][::20]:
            Et, _ = d14.stage_affine(plant, q, np.zeros(M))
            errs.append(np.linalg.norm((np.eye(M) + Kb) - Et))
    return float(np.mean(errs))


# ------------------------------------------------------------------ main
def main():
    name = os.environ["PATHS17"]; os.makedirs(f"{OUT}/{name}", exist_ok=True)
    ref, ctrls = d22.setup(name)
    # --- closed-loop data (nominal controller) and excitation
    Xc, Uc, Xnc, tagc = d22.collect(ref, ctrls, TRAIN)
    Xe, Ue, Xne, tage, cuts = [], [], [], [], {}
    for seed in TRAIN:
        X, U, Xn, cut = excite(ref, seed); cuts[seed] = cut
        if len(X): Xe.append(X); Ue.append(U); Xne.append(Xn); tage += [seed] * len(U)
        print(f"  excitation seed {seed}: {len(U)} steps" + (f" (cut at {cut[0]}: {cut[1]})" if cut else ""))
    Xe, Ue, Xne, tage = np.vstack(Xe), np.vstack(Ue), np.vstack(Xne), np.array(tage)
    np.savez(f"{OUT}/{name}/data.npz", Xc=Xc, Uc=Uc, Xnc=Xnc, tagc=tagc, Xe=Xe, Ue=Ue, Xne=Xne, tage=tage)
    Xa, Ua, Xna, taga = np.vstack([Xc, Xe]), np.vstack([Uc, Ue]), np.vstack([Xnc, Xne]), np.r_[tagc, tage]
    print(f"data: closed-loop {len(Uc)}  excitation {len(Ue)}  total {len(Ua)}")
    # --- test data for diagnostics
    Xt, Ut, Xnt, tagt = d22.collect(ref, ctrls, TEST)
    idx_te = [t for s in TEST for t in np.where(tagt == s)[0][:-N:10]]
    val = np.isin(taga, TRAIN[-2:]); tr = ~val
    idx_val = [t for s in TRAIN[-2:] for t in np.where(taga == s)[0][:-N:10] if taga[t + N - 1] == s]
    idx_tr = [t for s in TRAIN[:-2] for t in np.where(taga == s)[0][:-N:int(os.environ.get('STRIDE', '20'))] if taga[t + N - 1] == s]
    models = {}
    # mse1 on closed-loop only (reference) and on augmented data
    for tag, (X, U, Xn, tg) in {"mse1_cl": (Xc, Uc, Xnc, tagc), "mse1": (Xa, Ua, Xna, taga)}.items():
        v = np.isin(tg, TRAIN[-2:]); best = None
        for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            Ka, Kb = ridge(X[~v], U[~v], Xn[~v], lam)
            iv = [t for s in TRAIN[-2:] for t in np.where(tg == s)[0][:-N:10] if tg[t + N - 1] == s]
            e = float(np.median(rollout_err(Ka, Kb, X, U, Xn, iv)))
            if best is None or e < best[0]: best = (e, lam, Ka, Kb)
        models[tag] = dict(Ka=best[2], Kb=best[3], lam=best[1], val20=best[0])
        print(f"{tag:8s}: lambda {best[1]:g} val 20-step {best[0]:.4f}")
    Ka0, Kb0 = models["mse1"]["Ka"], models["mse1"]["Kb"]
    for tag in ("mseN", "cerr"):
        (Ka, Kb), L = fit_adam(Ka0, Kb0, Xa, Ua, Xna, idx_tr, tag, ref, iters=int(os.environ.get("ITERS", "300")))
        models[tag] = dict(Ka=Ka, Kb=Kb, lam=models["mse1"]["lam"], val20=float(np.median(rollout_err(Ka, Kb, Xa, Ua, Xna, idx_val))))
        print(f"{tag:8s}: train loss {L:.5f} val 20-step {models[tag]['val20']:.4f}")
    # --- diagnostics on TEST draws
    en = float(np.median(rollout_err(np.zeros((M, 2 * M)), np.zeros((M, M)), Xt, Ut, Xnt, idx_te)))
    report = dict(data=dict(closed_loop=int(len(Uc)), excitation=int(len(Ue)), cuts={str(k): v for k, v in cuts.items()},
                            probe=dict(steps=PROBE_STEPS, umax=PROBE_UMAX, vfrac=PROBE_VFRAC)),
                  test_20step_nominal=en, models={})
    for tag, mdl in models.items():
        e20 = float(np.median(rollout_err(mdl["Ka"], mdl["Kb"], Xt, Ut, Xnt, idx_te)))
        rad = float(max(abs(np.linalg.eigvals(A + B @ mdl["Ka"]))))
        report["models"][tag] = dict(lam=mdl["lam"], val20=mdl["val20"], test20=e20, rho=rad,
                                     E_err_vs_true=E_error(mdl["Ka"], mdl["Kb"], ref, TEST),
                                     Kb_diag=np.round(np.diag(mdl["Kb"]), 3).tolist())
        np.savez(f"{OUT}/{name}/model_{tag}.npz", Ka=mdl["Ka"], Kb=mdl["Kb"], lam=mdl["lam"], rho=rad)
        print(f"{tag:8s}: test 20-step {e20:.4f} (nominal {en:.4f})  rho {rad:.4f}  ||E-E_true||_F {report['models'][tag]['E_err_vs_true']:.3f}")
    json.dump(report, open(f"{OUT}/{name}/report.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
