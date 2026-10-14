"""fit the MSE residual arm from NOMINAL closed-loop data (train draws), ridge on the acceleration residual,
spectral-radius check, save per path. Usage: same env as day20 + PATHS17. Output results/day23_mse/<path>/mse_model.npz"""
import os, sys, json, numpy as np; sys.path.insert(0, ".")
import day22_divergence as d22, day17_oracle as d17
from day12_tooltask import A, B, M, ps
OUT = "results/day23_mse"; name = os.environ["PATHS17"]; os.makedirs(f"{OUT}/{name}", exist_ok=True)
ref, ctrls = d22.setup(name)
X, U, Xn, tag = d22.collect(ref, ctrls, d22.TRAIN)
Bp = np.linalg.pinv(B); delta = (Xn - X @ A.T) @ Bp.T - U          # acceleration residual per transition
CLS = os.environ.get("MSE_CLASS", "full")     # full: [q, qd, u] ; noqd: [q, u] (no velocity feedback)
Z = np.c_[X, U] if CLS == "full" else np.c_[X[:, :M], U]
val = np.isin(tag, d22.TRAIN[-2:]); tr = ~val                     # last two train draws held out for lambda
best = None
for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
    K = np.linalg.solve(Z[tr].T @ Z[tr] + lam * np.eye(Z.shape[1]), Z[tr].T @ delta[tr]).T
    if CLS == "full": Ka, Kb = K[:, :2 * M], K[:, 2 * M:]
    else: Ka, Kb = np.c_[K[:, :M], np.zeros((M, M))], K[:, M:]
    rad = max(abs(np.linalg.eigvals(A + B @ Ka)))
    e1 = np.abs(delta[val] - Z[val] @ K.T).mean()
    err20 = []
    for t in np.where(val)[0][:-20:10]:
        x = X[t].copy(); xm = x.copy()
        for k in range(20):
            xm = A @ xm + B @ (U[t + k] + Ka @ xm + Kb @ U[t + k])
        err20.append(np.abs(xm - Xn[t + 19]).max())
    e20 = float(np.median(err20)); print(f"lambda {lam:6g}: rho(A+BKa) {rad:.4f}  val 1-step {e1:.4f}  20-step med {e20:.4f}")
    if rad < 1 + 1e-3 and (best is None or e20 < best[0]): best = (e20, lam, Ka, Kb, rad, e1)
# nominal 20-step error on the same validation windows, for reference
errn = []
for t in np.where(val)[0][:-20:10]:
    xm = X[t].copy()
    for k in range(20): xm = A @ xm + B @ U[t + k]
    errn.append(np.abs(xm - Xn[t + 19]).max())
print(f"nominal  : 20-step med {float(np.median(errn)):.4f}")
assert best is not None, "no candidate with rho <= 1+1e-3"
assert best[0] < float(np.median(errn)), "learned model does not beat nominal on validation -> not learnable in this class"
e20, lam, Ka, Kb, rad, e1 = best
np.savez(f"{OUT}/{name}/mse_model.npz", Ka=Ka, Kb=Kb, lam=lam, rho=rad, cls=CLS, n_train=int(tr.sum()), n_val=int(val.sum()))
json.dump(dict(lambda_=lam, spectral_radius=rad, val_one_step=e1, val_twenty_step_median=e20,
               train_seeds=d22.TRAIN[:-2], val_seeds=d22.TRAIN[-2:], note="residual fitted ONLY on nominal-controller data"),
          open(f"{OUT}/{name}/fit.json", "w"), indent=1)
print("saved", f"{OUT}/{name}/mse_model.npz", f"lambda={lam} rho={rad:.4f}")
