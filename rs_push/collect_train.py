#!/usr/bin/env python3
"""Collect random-push data across frictions/masses and train the FROZEN
one-step model used by every Gate-A policy. Also prints the k-step open-loop
error curve (the epsilon_k raw material for the later hold-length method).

    python collect_train.py --quick     (~10 min)
    python collect_train.py             (~30 min)
"""
import argparse, os, pickle
import numpy as np

from rspush.env import make_env, Push, DT, clearance
from rspush.model import features, train, rollout

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--episodes", type=int, default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="ckpt")
a = ap.parse_args()
n_ep = a.episodes or (60 if a.quick else 200)
os.makedirs(a.out, exist_ok=True)

env = make_env(); p = Push(env, seed=a.seed)
rng = np.random.default_rng(a.seed)
F, Y, trajs = [], [], []
ep = 0
while ep < n_ep:
    spec = dict(friction=float(rng.uniform(0.3, 1.2)),
                mass=float(rng.uniform(0.05, 0.6)),
                obj_xy=np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08)]),
                obj_yaw=float(rng.uniform(-np.pi, np.pi)),
                goal_xy=np.array([rng.uniform(-0.16, 0.16), rng.uniform(-0.16, 0.16)]))
    if np.linalg.norm(spec["goal_xy"] - spec["obj_xy"]) < 0.10:
        continue
    if not p.apply_spec(spec):
        continue
    E, S, U = [], [], []
    v = np.zeros(2)
    for t in range(70):
        if t % 4 == 0:
            direc = spec["goal_xy"] - p.obj_pose()[:2]
            direc /= (np.linalg.norm(direc) + 1e-9)
            v = 0.08 * direc + rng.normal(0, 0.04, 2)
        eef0, obj0 = p.eef()[:2].copy(), p.obj_pose().copy()
        r = p.step_eef_vel(v)
        E.append(eef0); S.append(obj0); U.append(v.copy())
        F.append(features(eef0, obj0, v))
        Y.append(r["obj"] - obj0)
    trajs.append(dict(E=np.array(E), S=np.array(S), U=np.array(U)))
    ep += 1
    if ep % 20 == 0:
        print(f"  {ep}/{n_ep} episodes", flush=True)
env.close()

F = np.array(F, dtype=np.float32); Y = np.array(Y, dtype=np.float32)
n_tr = int(0.85 * len(trajs))
tr_ids = set(range(n_tr))
print(f"transitions: {len(F)}   object |d s| rms: "
      f"{np.sqrt((Y[:, :2] ** 2).sum(1).mean()):.4f} m/step")
model = train(F[:sum(len(t['S']) for t in trajs[:n_tr])],
              Y[:sum(len(t['S']) for t in trajs[:n_tr])],
              epochs=200 if a.quick else 400, seed=a.seed)

# open-loop k-step error on held-out episodes (epsilon_k raw curve)
K = 12
errs = {k: [] for k in range(1, K + 1)}
for t in trajs[n_tr:]:
    T = len(t["S"])
    for s in range(0, T - K, 3):
        pred = rollout(model, t["E"][s], t["S"][s], t["U"][None, s:s + K])[0]
        for k in range(1, K + 1):
            errs[k].append(np.linalg.norm(pred[k - 1, :2] - t["S"][s + k, :2]))
print("\nopen-loop object position error (held-out episodes):")
for k in (1, 2, 4, 8, 12):
    print(f"  k={k:>2}: mean {np.mean(errs[k]):.4f} m   p90 {np.quantile(errs[k], .9):.4f} m")

import torch, hashlib, json
torch.save(model.state_dict(), f"{a.out}/onestep.pt")
md5 = hashlib.md5(open(f"{a.out}/onestep.pt", "rb").read()).hexdigest()
json.dump(dict(episodes=n_ep, transitions=int(len(F)),
               epochs=200 if a.quick else 400, seed=a.seed, quick=bool(a.quick),
               ckpt_md5=md5, n_train_eps=n_tr),
          open(f"{a.out}/manifest.json", "w"), indent=2)
if a.quick:
    print("NOTE: --quick checkpoint is for smoke only; the formal Gate A run "
          "must use a checkpoint trained with the full registered budget "
          "(200 episodes / 400 epochs): python collect_train.py")
with open(f"{a.out}/norm.pkl", "wb") as f:
    pickle.dump(dict(mu=model.mu.numpy(), sd=model.sd.numpy(),
                     osd=model.osd.numpy()), f)
print(f"\nfrozen checkpoint -> {a.out}/onestep.pt  (Gate A uses this, eval() only)")
