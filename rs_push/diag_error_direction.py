#!/usr/bin/env python3
"""Decompose open-loop prediction error into components parallel/perpendicular
to the push path. Decides the doorway orientation BEFORE pilot (pre-registered):

    e_perp median at k=8  >= 10 mm  ->  straight doorway, run pilot with --phi 0
    e_perp median at k=8  <  10 mm  ->  slanted doorway,  run pilot with --phi 45

A lateral doorway only couples to lateral error; if the model's error is almost
entirely longitudinal (under/over-estimated slide distance), a straight doorway
would never be hit no matter how narrow -- the geometry must be rotated so wall
normals pick up the longitudinal component.

    python diag_error_direction.py        # ~4 min, 30 fresh episodes, frozen ckpt
"""
import numpy as np, torch
from rspush.env import make_env, Push
from rspush.model import OneStep, rollout

K = 12
model = OneStep(); model.load_state_dict(torch.load("ckpt/onestep.pt")); model.eval()
env = make_env(); p = Push(env, seed=123)
rng = np.random.default_rng(123)
E, S, U, trajs = [], [], [], []
ep = 0
while ep < 30:
    spec = dict(friction=float(rng.uniform(0.3, 1.2)), mass=float(rng.uniform(0.05, 0.6)),
                obj_xy=np.array([rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06)]),
                obj_yaw=float(rng.uniform(-np.pi, np.pi)),
                goal_xy=None, np_seed=5000 + ep)
    ang = rng.uniform(-np.pi, np.pi)
    spec["goal_xy"] = spec["obj_xy"] + 0.24 * np.array([np.cos(ang), np.sin(ang)])
    if not p.apply_spec(spec):
        continue
    Ee, Ss, Uu = [], [], []
    v = np.zeros(2)
    for t in range(70):
        if t % 4 == 0:
            d = spec["goal_xy"] - p.obj_pose()[:2]; d /= (np.linalg.norm(d) + 1e-9)
            v = 0.08 * d + rng.normal(0, 0.04, 2)
        Ee.append(p.eef()[:2].copy()); Ss.append(p.obj_pose().copy()); Uu.append(v.copy())
        p.step_eef_vel(v)
    trajs.append((np.array(Ee), np.array(Ss), np.array(Uu))); ep += 1
env.close()

par = {k: [] for k in (1, 2, 4, 8, 12)}; per = {k: [] for k in (1, 2, 4, 8, 12)}
for Ee, Ss, Uu in trajs:
    for s in range(0, len(Ss) - K, 3):
        net = Ss[s + K, :2] - Ss[s, :2]
        n = np.linalg.norm(net)
        if n < 0.02:
            continue
        dpath = net / n; nperp = np.array([-dpath[1], dpath[0]])
        pred = rollout(model, Ee[s], Ss[s], Uu[None, s:s + K])[0]
        for k in par:
            e = pred[k - 1, :2] - Ss[s + k, :2]
            par[k].append(abs(e @ dpath)); per[k].append(abs(e @ nperp))
print(f"{'k':>4}{'e_par mean':>12}{'e_perp mean':>13}{'e_perp p50':>12}")
for k in par:
    print(f"{k:>4}{np.mean(par[k]):>12.4f}{np.mean(per[k]):>13.4f}"
          f"{np.median(per[k]):>12.4f}")
p50 = float(np.median(per[8]))
print(f"\ne_perp median @k=8 = {p50 * 1000:.1f} mm  ->  "
      + ("STRAIGHT doorway: run pilot with --phi 0"
         if p50 >= 0.010 else "SLANTED doorway: run pilot with --phi 45"))
