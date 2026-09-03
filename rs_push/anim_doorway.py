#!/usr/bin/env python3
"""Demo animation for the doorway experiments (for the weekly report).

Re-runs 3 representative paired-probe episodes (box, phi=45, c=0.030, final
steered primitive, T=180) and renders a top-down animation:
  * grey discs   = keep-out walls (dark core = zone r=0.05, light ring = the
                   effective keep-out for the object CENTRE, r+R_object=0.061)
  * green star   = goal;  dashed segment = the doorway gap
  * blue dot     = pusher (eef);  red disc = object, leaves a trail
  * banner turns RED on a violation, GREEN on crossing
Outputs:  doorway_demo.gif   (the animation)
          doorway_verdict.png (the one-slide summary: drift vs error window)

    conda activate rs_carry && python anim_doorway.py        # ~4 min
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, patches

from gate_a import pilot_spec
from rspush.env import make_env, Push, clearance, R_OBJECT

PHI, C, T, RZ = 45.0, 0.030, 180, 0.05
# slots chosen to match the reported T=180 box table exactly:
#   ep1 crossed clean, ep0 crossed with violation, ep3 stalled before the gate
SLOTS = [1, 0, 3]
TITLES = {1: "crossed, safe", 0: "crossed, VIOLATION", 3: "stalled before gate"}

env = make_env(); p = Push(env)
episodes = []
for slot in SLOTS:
    spec = None
    for retry in range(25):
        s = pilot_spec(90000, slot, retry, C, PHI)
        if (clearance(s["obj_xy"], s["zone_xy"], RZ) >= 0.005
                and clearance(s["goal_xy"], s["zone_xy"], RZ) >= 0.005
                and p.apply_spec(s)):
            spec = s; break
    path = spec["goal_xy"] - spec["obj_xy"]
    L = np.linalg.norm(path); path = path / L
    perp = np.array([-path[1], path[0]])
    obj_tr, eef_tr, rho_tr = [], [], []
    for t in range(T):
        obj = p.obj_pose()[:2]
        e_lat = float((obj - spec["obj_xy"]) @ perp)
        th = float(np.clip(-20.0 * e_lat, -np.radians(30), np.radians(30)))
        ct, st = np.cos(th), np.sin(th)
        d = np.array([ct * path[0] - st * path[1], st * path[0] + ct * path[1]])
        u = 2.5 * ((obj - d * 0.035) - p.eef()[:2]) + 0.06 * d
        n = np.linalg.norm(u)
        if n > 0.12:
            u *= 0.12 / n
        r = p.step_eef_vel(u)
        obj_tr.append(r["obj"][:2].copy()); eef_tr.append(r["eef"][:2].copy())
        rho_tr.append(clearance(r["obj"][:2], spec["zone_xy"], RZ))
    episodes.append(dict(spec=spec, obj=np.array(obj_tr), eef=np.array(eef_tr),
                         rho=np.array(rho_tr), slot=slot,
                         mid=0.5 * (spec["obj_xy"] + spec["goal_xy"]), path=path))
env.close()
print("episodes recorded; rendering...")

fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
arts = []
for ax, ep in zip(axes, episodes):
    s = ep["spec"]
    for z in s["zone_xy"]:
        ax.add_patch(patches.Circle(z, RZ + R_OBJECT, fc="#f3c9c9", ec="none"))
        ax.add_patch(patches.Circle(z, RZ, fc="#b7b7b7", ec="#777"))
    ax.plot(*s["goal_xy"], marker="*", ms=16, color="#2a9d2a")
    ax.plot(*s["obj_xy"], marker="o", ms=5, color="#666", mfc="none")
    trail, = ax.plot([], [], "-", lw=1.6, color="#c1272d")
    objp = patches.Circle((0, 0), R_OBJECT, fc="#c1272d", ec="k", zorder=5)
    ax.add_patch(objp)
    eefp, = ax.plot([], [], "o", ms=7, color="#1f77b4", zorder=6)
    ttl = ax.set_title(TITLES[ep["slot"]], fontsize=11)
    ax.set_xlim(-0.25, 0.30); ax.set_ylim(-0.28, 0.28)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    arts.append((trail, objp, eefp, ttl, ep))

def draw(f):
    out = []
    for trail, objp, eefp, ttl, ep in arts:
        k = min(f, T - 1)
        trail.set_data(ep["obj"][:k + 1, 0], ep["obj"][:k + 1, 1])
        objp.center = ep["obj"][k]
        eefp.set_data([ep["eef"][k, 0]], [ep["eef"][k, 1]])
        viol = (ep["rho"][:k + 1] < 0).any()
        prog = (ep["obj"][k] - ep["spec"]["obj_xy"]) @ ep["path"]
        crossed = prog > (0.5 * 0.24 + 0.02)
        base = TITLES[ep["slot"]]
        ttl.set_text(f"{base}   t={k}  min ρ={ep['rho'][:k+1].min():+.3f} m")
        ttl.set_color("#c1272d" if viol else ("#2a9d2a" if crossed else "k"))
        out += [trail, objp, eefp, ttl]
    return out

ani = animation.FuncAnimation(fig, draw, frames=range(0, T, 2), blit=False)
ani.save("doorway_demo.gif", writer=animation.PillowWriter(fps=18), dpi=90)
print("wrote doorway_demo.gif")

# ---- the one-slide verdict chart -----------------------------------------
fig2, ax = plt.subplots(figsize=(6.4, 4))
bars = ax.bar(["box\n(steered)", "cylinder\n(steered)"], [57, 103],
              color=["#c1272d", "#8b1a1a"], width=0.5)
ax.axhline(34, color="#2a9d2a", lw=2)
ax.axhspan(0, 34, color="#2a9d2a", alpha=0.12)
ax.text(1.28, 30, "needed: drift p90 < 34 mm\n(E_ρ p90 39 − 5 margin)",
        color="#2a6d2a", fontsize=10, va="top")
for b, v in zip(bars, (57, 103)):
    ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v} mm", ha="center")
ax.set_ylabel("lateral drift at gate, p90 (mm)")
ax.set_title("Doorway falsified: contact controllability exceeds\n"
             "the model-error window (branch E, pre-registered)")
ax.set_ylim(0, 115)
fig2.tight_layout(); fig2.savefig("doorway_verdict.png", dpi=150)
print("wrote doorway_verdict.png")
