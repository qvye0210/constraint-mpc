#!/usr/bin/env python3
"""LAYER-0 PROBE (no planner, no model): push the object straight through the
doorway with a fixed script. Separates PHYSICS from OPTIMIZATION:

v2: the v1 script kept the pusher behind the object ALONG THE PATH -- zero
lateral authority, so its 17% crossing rate measured the script, not physics.
v2 steers: the pusher repositions to the opposite side of the DESIRED push
direction (path + lateral correction toward the line), which is how pushing is
actually controlled. This measures the CONTROLLABILITY scale that the doorway
ladder must respect:

    regime window:  steered drift  <  c  <  k=8 open-loop error (~48mm)

Printed verdict applies the pre-registered rule:
  drift p90 + 5mm < 48mm  -> window exists; ladder goes inside it
  else                    -> switch object to a cylinder (registered design
                             intent; the cube was a shortcut) and re-measure;
                             still empty -> doorway task falsified as a Gate A
                             vehicle. Stop.

    python probe_push.py --phi 45
    python probe_push.py --phi 45 --no-steer     # reproduce v1 for comparison
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
ap.add_argument("--no-steer", action="store_true")
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
    E, S, U, t_cross = [], [], [], -1
    perp = np.array([-path[1], path[0]])
    for t in range(a.T):
        obj = p.obj_pose()[:2]
        prog = float((obj - spec["obj_xy"]) @ path)
        e_lat = float((obj - spec["obj_xy"]) @ perp)       # lateral offset
        if a.no_steer:
            d_des = path
        else:
            # desired push direction bends back toward the line (gain 6 /m)
            d_des = path - np.clip(6.0 * e_lat, -0.8, 0.8) * perp
            d_des = d_des / np.linalg.norm(d_des)
        behind = obj - d_des * 0.035        # reposition AROUND the object
        eef = p.eef()[:2]
        gap = np.linalg.norm(behind - eef)
        fwd = 0.05 * max(0.0, 1.0 - gap / 0.02)   # push only when in position
        u = 3.0 * (behind - eef) + fwd * d_des
        n = np.linalg.norm(u)
        if n > 0.12:
            u *= 0.12 / n
        E.append(eef.copy()); S.append(p.obj_pose().copy()); U.append(u.copy())
        r = p.step_eef_vel(u)
        rho = clearance(r["obj"][:2], spec["zone_xy"], a.r_zone)
        min_rho = min(min_rho, rho)
        if rho < 0:
            viol = True
        prog = float((r["obj"][:2] - spec["obj_xy"]) @ path)
        if prog > mid_s + 0.02:
            if not crossed:
                t_cross = t
            crossed = True
        if abs(prog - mid_s) < 0.05:
            drift.append(abs(float((r["obj"][:2] - spec["obj_xy"]) @
                                   np.array([-path[1], path[0]]))))
    ge = float(np.linalg.norm(p.obj_pose()[:2] - spec["goal_xy"]))
    rows.append(dict(crossed=crossed, viol=viol, min_rho=min_rho, goal_err=ge,
                     drift=float(np.mean(drift)) if drift else np.nan,
                     E=np.array(E), S=np.array(S), U=np.array(U),
                     t_cross=t_cross, spec=spec))
    print(f"  ep {slot}: crossed={crossed} viol={viol} min_rho={min_rho:+.3f} "
          f"goal_err={ge:.3f} drift@gate={rows[-1]['drift']:.3f}")
env.close()

# ---- counterfactual model check + constraint-normal error (reviewer's two
# missing diagnostics; both use the FROZEN checkpoint) --------------------
import torch
from rspush.model import OneStep, rollout as mroll
H = 12
model = OneStep(); model.load_state_dict(torch.load("ckpt/onestep.pt")); model.eval()
print("\ncounterfactual: feed each ACTUAL crossing's executed actions to the model")
cf = []
for i, r in enumerate(rows):
    if not r["crossed"] or r["t_cross"] < 1:
        continue
    s0 = max(0, r["t_cross"] - H)
    seg = r["U"][s0:s0 + H]
    if len(seg) < H:
        continue
    pred = mroll(model, r["E"][s0], r["S"][s0], seg[None])[0]
    prog = (pred[:, :2] - r["spec"]["obj_xy"]) @ (
        (r["spec"]["goal_xy"] - r["spec"]["obj_xy"])
        / np.linalg.norm(r["spec"]["goal_xy"] - r["spec"]["obj_xy"]))
    mid_s = 0.5 * np.linalg.norm(r["spec"]["goal_xy"] - r["spec"]["obj_xy"])
    m_cross = bool((prog > mid_s + 0.02).any())
    m_rho = min(clearance(pp, r["spec"]["zone_xy"], a.r_zone) for pp in pred[:, :2])
    cf.append(dict(ep=i, model_pred_crossed=m_cross, model_pred_minrho=m_rho))
    print(f"  ep {i}: model_pred_crossed={m_cross}  model_pred_minrho={m_rho:+.3f}")
if not cf:
    print("  (no crossings long enough to test)")

# constraint-normal prefix-max error E_rho on all executed windows
per_win = []
for r in rows:
    T = len(r["S"])
    for s0 in range(0, T - H, 4):
        pred = mroll(model, r["E"][s0], r["S"][s0], r["U"][None, s0:s0 + H])[0]
        e = [abs(clearance(pred[k, :2], r["spec"]["zone_xy"], a.r_zone)
                 - clearance(r["S"][s0 + k + 1, :2], r["spec"]["zone_xy"], a.r_zone))
             for k in range(H - 1)]
        per_win.append(max(e))
E_rho_p90 = float(np.quantile(per_win, 0.9)) if per_win else float("nan")
print(f"constraint-normal prefix-max error E_rho (H={H}): "
      f"p50 {np.median(per_win):.3f}  p90 {E_rho_p90:.3f}   "
      f"(this replaces the 2-D 48mm as the window's right edge)")

cr = np.mean([r["crossed"] for r in rows]); vi = np.mean([r["viol"] for r in rows])
dr = np.array([r["drift"] for r in rows], dtype=float)
stall = np.mean([r["goal_err"] > 0.15 for r in rows])
p50, p90 = np.nanmedian(dr), np.nanquantile(dr, 0.9)
print(f"\ncrossed {cr:.0%}   violations {vi:.0%}   stalled {stall:.0%}")
print(f"steered drift@gate  p50 {p50:.3f}  p90 {p90:.3f}  (k=8 open-loop error ~0.048)")
win_lo = p90 + 0.005
mc = [c["model_pred_crossed"] for c in cf]
print(">>> five-branch exit (pre-registered):")
if cr >= 0.8 and win_lo < E_rho_p90:
    print("    A: cube steered probe good AND strict window exists -> integrate "
          "the same steering primitive into ALL policies, then feasibility_check.")
elif cr >= 0.8 and cf and not any(mc):
    print("    B: actual crossings exist but the model predicts none -> MODEL "
          "problem; doorway cannot serve Gate A until the model is fixed.")
elif cr >= 0.8 and cf and any(mc):
    print("    C: actual + model both cross, MPPI pred_crossed was ~0 -> "
          "OPTIMIZER problem; physics is no longer the suspect.")
elif cr < 0.8:
    print("    D: cube window empty/low crossing -> registered branch: CYLINDER "
          "object, one re-measure.")
print("    E: cylinder also empty -> stop the doorway line entirely.")
