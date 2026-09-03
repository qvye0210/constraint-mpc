#!/usr/bin/env python3
"""ACCEPTANCE CHECK for the fixed planner, run BEFORE any new pilot: at the
widest door (c=0.030) both k=1 and maximal-hold must be able to do the task.

    accept iff, for BOTH policies:
        crossed  >= 80%
        success  >= 70%
        predicted-negative-clearance fraction ~ 0 (hard filter makes it exact 0)
    solve_failures are reported (occasional safe stops are acceptable;
    systematic failure means the proposal still cannot thread).

The reviewer's decision table is printed from the measured quantities:
pred_crossed vs actual crossed separates optimizer failure from model/execution
failure; k=1 vs maximal-hold separates re-anchoring problems.

    python feasibility_check.py --phi 45
"""
import argparse
import numpy as np, torch

from gate_a import LADDER, pilot_spec, run_episode
from rspush.env import make_env, Push, clearance
from rspush.model import OneStep

ap = argparse.ArgumentParser()
ap.add_argument("--phi", type=float, default=45.0)
ap.add_argument("--c", type=float, default=LADDER[0])
ap.add_argument("--H", type=int, default=12)
ap.add_argument("--T", type=int, default=90)
ap.add_argument("--r-zone", type=float, default=0.05)
ap.add_argument("--succ-tol", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

model = OneStep(); model.load_state_dict(torch.load("ckpt/onestep.pt")); model.eval()
env = make_env(); p = Push(env, seed=a.seed)
res = {}
for period, name in ((1, "k=1"), (a.H, "maximal-hold")):
    rows = []
    for slot in range(12):
        for retry in range(25):
            s = pilot_spec(90000 + a.seed, slot, retry, a.c, a.phi)
            if (clearance(s["obj_xy"], s["zone_xy"], a.r_zone) >= 0.005
                    and clearance(s["goal_xy"], s["zone_xy"], a.r_zone) >= 0.005):
                r = run_episode(p, None, model, s, period, a)
                if "setup_fail" not in r:
                    rows.append(r); break
    agg = dict(crossed=np.mean([r["crossed"] for r in rows]),
               succ=np.mean([r["success"] for r in rows]),
               viol=np.mean([r["violation"] for r in rows]),
               pred_crossed=np.mean([r["pred_crossed_frac"] for r in rows]),
               fails=int(np.sum([r["solve_failures"] for r in rows])),
               goal_err=float(np.median([r["goal_err"] for r in rows])))
    res[name] = agg
    print(f"{name:>13}: crossed {agg['crossed']:.0%}  succ {agg['succ']:.0%}  "
          f"viol {agg['viol']:.0%}  pred_crossed {agg['pred_crossed']:.0%}  "
          f"solve_failures {agg['fails']}  med goal_err {agg['goal_err']:.3f}")
env.close()
ok = all(v["crossed"] >= 0.8 and v["succ"] >= 0.7 for v in res.values())
print("\n>>> " + ("ACCEPTED: planner can do the task under both policies. "
                  "Freeze it and rerun: python gate_a.py --pilot --phi %g"
                  % a.phi if ok else
                  "NOT ACCEPTED — read the reviewer's table: pred_crossed~0 on "
                  "both => optimizer still failing; pred_crossed>0 but actual~0 "
                  "=> model/execution gap; k=1 ok but maximal-hold not => "
                  "re-anchoring bug. Send this output back; do not tune "
                  "against violations."))
