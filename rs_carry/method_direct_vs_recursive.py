#!/usr/bin/env python3
"""THE METHOD'S FIRST HEAD-TO-HEAD TEST, against the criterion registered
before any method code was written:

  Direct multi-step constraint prediction must beat the recursive pipeline
  (well-trained one-step residual model, rolled out, margins via EXACT FK) by
  >= 30% on |g_pred - g_true| at k=10, with a monotone advantage over k=5..15.
  Fail -> the method is falsified on this task; convert to the scope paper.
  No second round.

Fairness notes:
  * both use the SAME trajectories, splits and seeds;
  * the recursive baseline's margins use exact FK on predicted joint angles
    (robosuite forward kinematics), NOT the first-order linearisation used in
    the gate -- the linearisation understated the baseline's error, so this
    comparison is harder for us than the gate curve suggested... and also
    fairer to the baseline where linearisation overstated it;
  * the direct head sees fewer training samples (windows, not transitions) --
    reported, not hidden.

    python method_direct_vs_recursive.py --quick
    python method_direct_vs_recursive.py            # the one that counts
"""
import argparse, json, os
import numpy as np, torch
from rscarry.data import build_dataset, apply_nominal, R_SAFE
from rscarry.model import train as train_onestep
from rscarry.direct import train_direct, windows
from rscarry.env import make_env, Carry, NQ

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--n-traj", type=int, default=None)
ap.add_argument("--K", type=int, default=15)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/method")
a = ap.parse_args()
n = a.n_traj or (40 if a.quick else 120)
ep1 = 500 if a.quick else 1500          # one-step model budget
epD = 400 if a.quick else 800           # direct head budget
os.makedirs(a.out, exist_ok=True)

print(f"dataset: {n} trajectories, K={a.K}")
d = build_dataset(n_traj=n, seed=a.seed)
W = d["_W"]

print("training one-step residual model (recursive baseline)...")
m1 = train_onestep(d["train"], epochs=ep1, seed=a.seed)
print("training direct multi-step head...")
mD = train_direct(d["train"]["trajs"], K=a.K, epochs=epD, seed=a.seed)
Ztr, _, _ = windows(d["train"]["trajs"], a.K)
print(f"  (window samples for direct head: {len(Ztr)}; "
      f"transitions for one-step model: {len(d['train']['X'])})")

# exact FK for the baseline's predicted joint angles
env = make_env(); c = Carry(env)
_qpos0 = env.sim.data.qpos.copy(); _qvel0 = env.sim.data.qvel.copy()
def fk(q):
    env.sim.data.qpos[c.r._ref_joint_pos_indexes] = q
    env.sim.forward()
    return env.sim.data.site_xpos[c.sid].copy()

@torch.no_grad()
def rollout_margins(t, s, K):
    x = t["X"][s].copy(); out = []
    for k in range(1, K + 1):
        u = t["U"][s + k - 1]
        xn = apply_nominal(W, x[None], u[None])[0]
        xn = xn + m1(torch.tensor(x[None], dtype=torch.float32),
                     torch.tensor(u[None], dtype=torch.float32)).numpy()[0]
        x = xn
        out.append(np.linalg.norm(fk(x[:NQ]) - t["p_obs"]) - R_SAFE)
    return np.array(out)

@torch.no_grad()
def direct_margins(t, s, K):
    z = np.concatenate([t["X"][s], t["U"][s:s + K].ravel()])[None].astype(np.float32)
    P = mD(torch.tensor(z)).numpy()[0]
    return np.linalg.norm(P - t["p_obs"], axis=1) - R_SAFE

err_R = {k: [] for k in range(1, a.K + 1)}
err_D = {k: [] for k in range(1, a.K + 1)}
for t in d["test"]["trajs"]:
    T = len(t["X"])
    for s in range(0, T - a.K - 1, 4):
        g_true = np.linalg.norm(t["eef"][s + 1:s + a.K + 1] - t["p_obs"], axis=1) - R_SAFE
        gR = rollout_margins(t, s, a.K)
        gD = direct_margins(t, s, a.K)
        for k in range(1, a.K + 1):
            err_R[k].append(abs(gR[k - 1] - g_true[k - 1]))
            err_D[k].append(abs(gD[k - 1] - g_true[k - 1]))
env.sim.data.qpos[:] = _qpos0; env.sim.data.qvel[:] = _qvel0; env.sim.forward()

print(f"\n{'k':>4}{'recursive':>12}{'direct':>12}{'direct/recur':>14}")
curveR, curveD = {}, {}
for k in range(1, a.K + 1):
    r, dd = float(np.mean(err_R[k])), float(np.mean(err_D[k]))
    curveR[k], curveD[k] = r, dd
    print(f"{k:>4}{r:>12.4f}{dd:>12.4f}{dd / max(r, 1e-12):>14.2f}")

k0 = 10
adv = 1 - curveD[k0] / max(curveR[k0], 1e-12)
mono = all(curveD[k] < curveR[k] for k in range(5, a.K + 1))
passed = bool(adv >= 0.30 and mono)
print(f"\nadvantage at k=10: {adv:+.1%} (need >= +30%)   "
      f"monotone win k=5..{a.K}: {mono}")
print(">>> " + ("REGISTERED CRITERION MET. Direct constraint prediction beats "
                "the recursive pipeline. This is Fig-1 material; next step is "
                "the MPC integration."
                if passed else
                "CRITERION NOT MET. Per the registered rule the method is "
                "falsified on this task: no retuning, no second round. "
                "Convert to the scope paper with the compounding analysis."))
json.dump(dict(curve_recursive=curveR, curve_direct=curveD,
               advantage_k10=adv, monotone=mono, passed=passed,
               n_windows_train=int(len(Ztr)), config=vars(a)),
          open(f"{a.out}/verdict.json", "w"), indent=2, default=float)
print(f"wrote {a.out}/verdict.json")
