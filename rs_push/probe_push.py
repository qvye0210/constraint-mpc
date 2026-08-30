#!/usr/bin/env python3
"""PAIRED controllability probe: v1 (no steering) and v3 (plough steering) run
on IDENTICAL episode specs, frictions, masses and seeds. v2's repositioning
gate starved progress (83% stalled, worse than v1) so its numbers were an
instrument failure, not physics. v3 = development-stage controller repair
(gate removed = functional fix; gain 4/m, clip 0.6, low-pass = mechanism-
informed tuning, admitted as such). ONE round; frozen afterwards.

Mechanism metrics recorded per episode (to VERIFY, not speculate, why v2/v3
succeed or fail): mean forward command along path, contact-keeping fraction,
gap>2cm time fraction, max progress along path, reached-gate flag.

TWO-TIER acceptance (pre-registered):
  instrument (paired vs v1):  crossed >= 25%, stalled < 50%,
                              median max-progress >= paired v1's
  regime (only if instrument passes):
                              crossed >= 80%, stall low,
                              drift p90 + 5mm < E_rho p90 (PROVISIONAL E_rho:
                              near-gate windows only; final value must be
                              recomputed under the frozen primitive)
Branches: v3 at 25-79% crossing = instrument fixed, cube still fails regime ->
CYLINDER branch. v3 fails instrument -> no more cube repairs, cylinder v1/v3.
Cylinder fails -> terminate the doorway line (branch E).

    python probe_push.py --phi 45
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


def specs_for_slots(p):
    out = []
    for slot in range(12):
        for retry in range(25):
            s = pilot_spec(90000 + a.seed, slot, retry, a.c, a.phi)
            if (clearance(s["obj_xy"], s["zone_xy"], a.r_zone) >= 0.005
                    and clearance(s["goal_xy"], s["zone_xy"], a.r_zone) >= 0.005
                    and p.apply_spec(s)):
                out.append((slot, s)); break
    return out


def run(p, spec, mode):
    p.apply_spec(spec)
    path = spec["goal_xy"] - spec["obj_xy"]
    L = np.linalg.norm(path); path = path / L
    perp = np.array([-path[1], path[0]])
    mid_s = 0.5 * L
    viol = crossed = False; min_rho = np.inf; drift = []
    fwd_cmd = []; contact = []; gap_big = []; max_prog = -np.inf
    E, S, U, t_cross = [], [], [], -1
    d_smooth = path.copy()
    for t in range(a.T):
        obj = p.obj_pose()[:2]
        prog = float((obj - spec["obj_xy"]) @ path)
        e_lat = float((obj - spec["obj_xy"]) @ perp)
        if mode == "v1":
            d_des = path
        else:
            d_raw = path - np.clip(4.0 * e_lat, -0.6, 0.6) * perp
            d_raw = d_raw / np.linalg.norm(d_raw)
            d_smooth = 0.7 * d_smooth + 0.3 * d_raw
            d_smooth = d_smooth / np.linalg.norm(d_smooth)
            d_des = d_smooth
        behind = obj - d_des * 0.035
        eef = p.eef()[:2]
        u = 2.5 * (behind - eef) + 0.06 * d_des
        n = np.linalg.norm(u)
        if n > 0.12:
            u *= 0.12 / n
        fwd_cmd.append(float(u @ path))
        contact.append(float(np.linalg.norm(eef - obj) < 0.05))
        gap_big.append(float(np.linalg.norm(behind - eef) > 0.02))
        E.append(eef.copy()); S.append(p.obj_pose().copy()); U.append(u.copy())
        r = p.step_eef_vel(u)
        rho = clearance(r["obj"][:2], spec["zone_xy"], a.r_zone)
        min_rho = min(min_rho, rho)
        if rho < 0:
            viol = True
        prog = float((r["obj"][:2] - spec["obj_xy"]) @ path)
        max_prog = max(max_prog, prog)
        if prog > mid_s + 0.02 and not crossed:
            crossed = True; t_cross = t
        if abs(prog - mid_s) < 0.05:
            drift.append(abs(float((r["obj"][:2] - spec["obj_xy"]) @ perp)))
    ge = float(np.linalg.norm(p.obj_pose()[:2] - spec["goal_xy"]))
    return dict(crossed=crossed, viol=viol, min_rho=min_rho, goal_err=ge,
                drift=float(np.mean(drift)) if drift else np.nan,
                fwd=float(np.mean(fwd_cmd)), contact=float(np.mean(contact)),
                gap_big=float(np.mean(gap_big)), max_prog=float(max_prog),
                reached=bool(len(drift) > 0), mid_s=mid_s,
                E=np.array(E), S=np.array(S), U=np.array(U),
                t_cross=t_cross, spec=spec)


env = make_env(); p = Push(env, seed=a.seed)
pairs = specs_for_slots(p)
res = {}
for mode in ("v1", "v3"):
    rows = []
    for slot, spec in pairs:
        r = run(p, spec, mode)
        rows.append(r)
        print(f"  [{mode}] ep {slot}: crossed={r['crossed']} viol={r['viol']} "
              f"min_rho={r['min_rho']:+.3f} max_prog={r['max_prog']:.3f}/"
              f"{r['mid_s']:.3f} fwd={r['fwd']:.3f} contact={r['contact']:.0%} "
              f"gap>2cm={r['gap_big']:.0%}")
    res[mode] = rows
env.close()


def agg(rows):
    dr = np.array([r["drift"] for r in rows], dtype=float)
    return dict(crossed=np.mean([r["crossed"] for r in rows]),
                viol=np.mean([r["viol"] for r in rows]),
                stalled=np.mean([r["goal_err"] > 0.15 for r in rows]),
                reached=np.sum([r["reached"] for r in rows]),
                prog=np.median([r["max_prog"] for r in rows]),
                fwd=np.median([r["fwd"] for r in rows]),
                p50=np.nanmedian(dr), p90=np.nanquantile(dr, 0.9))


A1, A3 = agg(res["v1"]), agg(res["v3"])
print(f"\n{'':>6}{'crossed':>9}{'viol':>7}{'stalled':>9}{'reached':>9}"
      f"{'med prog':>10}{'med fwd':>9}{'drift p50/p90':>16}")
for k, A in (("v1", A1), ("v3", A3)):
    print(f"{k:>6}{A['crossed']:>9.0%}{A['viol']:>7.0%}{A['stalled']:>9.0%}"
          f"{A['reached']:>9d}{A['prog']:>10.3f}{A['fwd']:>9.3f}"
          f"{A['p50']:>8.3f}/{A['p90']:.3f}")

# PROVISIONAL E_rho on near-gate windows only (final value must be recomputed
# under the frozen primitive with adequate near-gate coverage)
import torch
from rspush.model import OneStep, rollout as mroll
H = 12
model = OneStep(); model.load_state_dict(torch.load("ckpt/onestep.pt")); model.eval()
per_win = []
for r in res["v1"] + res["v3"]:
    T = len(r["S"])
    for s0 in range(0, T - H, 4):
        true_r = [clearance(r["S"][s0 + k + 1, :2], r["spec"]["zone_xy"], a.r_zone)
                  for k in range(H - 1)]
        if min(true_r) > 0.06:
            continue                       # keep near-gate windows only
        pred = mroll(model, r["E"][s0], r["S"][s0], r["U"][None, s0:s0 + H])[0]
        per_win.append(max(abs(clearance(pred[k, :2], r["spec"]["zone_xy"],
                                         a.r_zone) - true_r[k])
                           for k in range(H - 1)))
E90 = float(np.quantile(per_win, 0.9)) if len(per_win) >= 20 else float("nan")
print(f"\nPROVISIONAL E_rho (near-gate windows, n={len(per_win)}): "
      f"p90 {E90:.3f}" + ("  [too few windows -- not usable]" if len(per_win) < 20 else ""))

print("\n>>> instrument acceptance (v3 vs PAIRED v1):")
inst = (A3["crossed"] >= 0.25 and A3["stalled"] < 0.5 and A3["prog"] >= A1["prog"])
print(f"    crossed {A3['crossed']:.0%} (>=25%), stalled {A3['stalled']:.0%} (<50%), "
      f"med progress {A3['prog']:.3f} vs v1 {A1['prog']:.3f}  -> "
      f"{'PASS' if inst else 'FAIL'}")
if not inst:
    print(">>> instrument failed twice -> NO more cube repairs. Run the CYLINDER "
          "branch (v1 and v3 there); cylinder fails -> branch E, doorway line ends.")
else:
    reg = (A3["crossed"] >= 0.8 and not np.isnan(E90)
           and A3["p90"] + 0.005 < E90)
    print(">>> regime acceptance: crossed>=80% and drift p90+5mm < E_rho p90 -> "
          + ("PASS: freeze the primitive, wire it into ALL policies, recompute "
             "E_rho under it, then feasibility_check." if reg else
             "NOT MET: instrument works but the cube doorway regime does not -> "
             "CYLINDER branch (one re-measure); cylinder fails -> branch E."))
