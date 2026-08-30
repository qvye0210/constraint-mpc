#!/usr/bin/env python3
"""LAYER-0 PROBE (no planner, no model): push the object straight through the
doorway with a fixed script. Separates PHYSICS from OPTIMIZATION:

  crossed high, violations 20-60%  -> task feasible, drift creates the regime;
                                      the blocker is MPPI -> fix the planner.
  crossed high, violations ~0      -> lateral drift too small for this door;
                                      revisit error-direction data, not planner.
  crossed ~0                       -> pushing physics cannot thread this gap;
                                      geometry rethink, planner is innocent.

    python probe_push.py --phi 45           # widest door c=0.030, 12 episodes
"""
import argparse
import numpy as np

from gate_a import LADDER, pilot_spec
from rspush.env import make_env, Push, clearance

ap = argparse.ArgumentParser()
ap.add_argument("--phi", type=float, default=45.0)
ap.add_argument("--c", type=float, default=LADDER[0])
ap.add_argument("--T", type=int, default=90)
ap.add_argument("--r-zone", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

env = make_env(); p = Push(env, seed=a.seed)
rows = []
for slot in range(12):
    spec = None
    for retry in range(25):
        s = pilot_spec(90000 + a.seed, slot, retry, a.c, a.phi)
        if (clearance(s["obj_xy"], s["zone_xy"], a.r_zone) >= 0.005
                and clearance(s["goal_xy"], s["zone_xy"], a.r_zone) >= 0.005
                and p.apply_spec(s)):
            spec = s; break
    if spec is None:
        continue
    path = spec["goal_xy"] - spec["obj_xy"]
    L = np.linalg.norm(path); path = path / L
    mid_s = 0.5 * L
    viol = False; min_rho = np.inf; crossed = False; drift = []
    for t in range(a.T):
        obj = p.obj_pose()[:2]
        prog = float((obj - spec["obj_xy"]) @ path)
        target = spec["obj_xy"] + path * min(prog + 0.06, L)   # carrot on the line
        # steer the PUSHER to stay behind the object on the line
        behind = obj - path * 0.035
        eef = p.eef()[:2]
        u = 2.5 * (behind - eef) + 0.06 * path
        n = np.linalg.norm(u)
        if n > 0.12:
            u *= 0.12 / n
        r = p.step_eef_vel(u)
        rho = clearance(r["obj"][:2], spec["zone_xy"], a.r_zone)
        min_rho = min(min_rho, rho)
        if rho < 0:
            viol = True
        prog = float((r["obj"][:2] - spec["obj_xy"]) @ path)
        if prog > mid_s + 0.02:
            crossed = True
        if abs(prog - mid_s) < 0.05:
            drift.append(abs(float((r["obj"][:2] - spec["obj_xy"]) @
                                   np.array([-path[1], path[0]]))))
    ge = float(np.linalg.norm(p.obj_pose()[:2] - spec["goal_xy"]))
    rows.append(dict(crossed=crossed, viol=viol, min_rho=min_rho, goal_err=ge,
                     drift=float(np.mean(drift)) if drift else np.nan))
    print(f"  ep {slot}: crossed={crossed} viol={viol} min_rho={min_rho:+.3f} "
          f"goal_err={ge:.3f} drift@gate={rows[-1]['drift']:.3f}")
env.close()
cr = np.mean([r["crossed"] for r in rows]); vi = np.mean([r["viol"] for r in rows])
print(f"\ncrossed {cr:.0%}   violations {vi:.0%}   "
      f"median drift@gate {np.nanmedian([r['drift'] for r in rows]):.3f} m")
print(">>> " + ("PHYSICS OK + REGIME EXISTS -> fix the planner (nominal proposal "
                "+ mixed sampling), then feasibility_check.py"
                if cr >= 0.8 and 0.1 <= vi
                else ("PHYSICS OK but drift too small for this door width -> "
                      "revisit error-direction data" if cr >= 0.8
                      else "PUSHING CANNOT THREAD THIS GAP -> geometry rethink; "
                           "planner is innocent")))
