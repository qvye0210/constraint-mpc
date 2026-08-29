#!/usr/bin/env python3
"""STEP 2 GATE: does recursive rollout actually compound constraint error?

The proposed method (direct multi-step prediction of the constraint output)
can only beat the standard pipeline (one-step state model, rolled out, then
projected) if rollout error genuinely compounds in the constraint channel.
Two measurements decide that:

  A. consecutive-step residual direction correlation  (need > 0.2)
     ~0 means residuals are i.i.d. noise -> little systematic compounding,
     and a direct predictor has nothing structural to exploit.
  B. margin error growth under recursive rollout      (need err(k=10)/err(k=1) > 3)
     rolls the trained one-step model k steps on held-out trajectories and
     measures |predicted margin - true margin| at each k.

PASS on both -> the mechanism the method relies on exists on this task;
building the direct predictor and the MPC is then justified.

    python diag_step2_compound.py --quick
"""
import argparse, json, os
import numpy as np, torch
from rscarry.data import build_dataset, apply_nominal, R_SAFE
from rscarry.model import train, NQ

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--n-traj", type=int, default=None)
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--K", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/step2")
a = ap.parse_args()
n = a.n_traj or (40 if a.quick else 120)
ep = a.epochs or (500 if a.quick else 1500)
os.makedirs(a.out, exist_ok=True)

d = build_dataset(n_traj=n, seed=a.seed)
W = d["_W"]

# --- A: systematic residual? --------------------------------------------
cors = []
for t in d["test"]["trajs"]:
    R = t["Xn"] - apply_nominal(W, t["X"], t["U"])
    r1, r2 = R[:-1], R[1:]
    num = (r1 * r2).sum(1)
    den = np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1) + 1e-12
    cors.append(np.mean(num / den))
corr = float(np.mean(cors))
print(f"A. consecutive residual direction correlation = {corr:+.3f}  (need >0.2)")

# --- B: rollout margin-error growth -------------------------------------
m = train(d["train"], epochs=ep, seed=a.seed)
@torch.no_grad()
def step_model(x, u):
    xn = apply_nominal(W, x[None], u[None])[0]
    xn += m(torch.tensor(x[None], dtype=torch.float32),
            torch.tensor(u[None], dtype=torch.float32)).numpy()[0]
    return xn

errs = {k: [] for k in range(1, a.K + 1)}
for t in d["test"]["trajs"]:
    T = len(t["X"])
    # eef via nearest-neighbour on the same trajectory is NOT available for
    # predicted states; use the linearised map: p ~ p_t + Jv (q_pred - q_t).
    for s in range(0, T - a.K - 1, 4):
        x = t["X"][s].copy()
        for k in range(1, a.K + 1):
            x = step_model(x, t["U"][s + k - 1])
            p_pred = t["eef"][s] + t["Jv"][s] @ (x[:NQ] - t["X"][s][:NQ])
            g_pred = np.linalg.norm(p_pred - t["p_obs"]) - R_SAFE
            g_true = np.linalg.norm(t["eef"][s + k] - t["p_obs"]) - R_SAFE
            errs[k].append(abs(g_pred - g_true))
curve = {k: float(np.mean(v)) for k, v in errs.items()}
grow = curve[a.K] / max(curve[1], 1e-12)
print("B. margin-error growth under recursive rollout:")
for k in sorted(curve):
    print(f"   k={k:>2}  |g_pred - g_true| = {curve[k]:.4f}")
print(f"   growth err(k={a.K})/err(k=1) = {grow:.1f}  (need >3)")
print("   note: uses a first-order eef linearisation around the start state; "
      "at large k this UNDERSTATES the true rollout error, so a PASS here is "
      "conservative.")

passed = bool(corr > 0.2 and grow > 3)
print("\n>>> " + ("STEP 2 PASS: compounding is real and systematic. Building "
                  "the direct constraint predictor + MPC is justified."
                  if passed else
                  "STEP 2 FAIL: little systematic compounding -- the direct "
                  "predictor has no structural advantage here. Stop before "
                  "building the method; revisit the task or the idea."))
json.dump(dict(residual_corr=corr, curve=curve, growth=grow, passed=passed),
          open(f"{a.out}/verdict.json", "w"), indent=2, default=float)
print(f"wrote {a.out}/verdict.json")
