#!/usr/bin/env python3
"""Cylinder stability self-test — run BEFORE any cylinder probe.
Correctness repair, not tuning: no doorway, no steering, no crossing rates.

Per mass level: (a) reset + settle 300 zero-action steps, every step checks
qpos/qvel/qacc finite; (b) gentle 0.03 m/s straight push 100 steps on the empty
table, checks finite + no tipping (disk axis stays vertical) + sane motion.
Also prints: DOF 15/16 joint names (measured, not guessed), object geom_size
(verifies [radius, half_height]), rest height vs table+half_height, initial
contacts and max penetration.

VERDICT: the largest mass with 100% finite + no-tip defines the cylinder-mode
mass cap. All levels failing => geometry itself must change (bigger disk +
re-parameterised R_OBJECT). 12/12 finite at probe time is required before the
paired probe may run.
    python cyl_selftest.py
"""
import numpy as np
from rspush.env import make_env, Push, NQ

env = make_env(object_shape="cylinder"); p = Push(env)
m = env.sim.model
print("DOF -> joint map (the warning named DOF 15/16):")
for dof in range(m.nv):
    j = m.dof_jntid[dof]
    if dof >= NQ:  # skip arm printout noise
        print(f"  dof {dof:2d} -> joint '{m.joint_id2name(j)}'")
gid = p.gid_cube[0]
print(f"object geom_size = {m.geom_size[gid][:2]}  (expect [radius, half_height] = [0.011 0.011])")
half_h = float(m.geom_size[gid][1])

results = {}
for mass in (0.05, 0.10, 0.20, 0.40, 0.60):
    spec = dict(friction=0.6, mass=mass, obj_xy=np.array([0.0, 0.0]),
                obj_yaw=0.0, goal_xy=np.array([0.20, 0.0]), np_seed=42)
    ok_setup = p.apply_spec(spec)
    bad = 0; max_pen = 0.0; z0 = p.obj_z()
    ncon0 = env.sim.data.ncon
    for t in range(300):
        p._step_qd(np.zeros(NQ))
        d = env.sim.data
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
                and np.isfinite(d.qacc).all()):
            bad += 1
        for c in range(d.ncon):
            max_pen = max(max_pen, -float(d.contact[c].dist))
    # gentle push, no doorway
    tip = False
    for t in range(100):
        r = p.step_eef_vel(np.array([0.03, 0.0]))
        d = env.sim.data
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
            bad += 1
        w, x, y, z = d.qpos[p.jadr + 3:p.jadr + 7]
        upz = 1 - 2 * (x * x + y * y)          # z-component of disk axis
        if upz < 0.9:
            tip = True
    rest_err = abs(z0 - (p.table_z + half_h))
    results[mass] = dict(finite=(bad == 0), tip=tip, max_pen=max_pen,
                         rest_err=rest_err, setup=ok_setup)
    print(f"mass {mass:.2f}: finite={bad == 0}  tip={tip}  "
          f"max_pen={max_pen * 1000:.2f}mm  rest_height_err={rest_err * 1000:.1f}mm  "
          f"init_contacts={ncon0}  setup_ok={ok_setup}")
env.close()
stable = [mm for mm, r in results.items() if r["finite"] and not r["tip"]]
print("\n>>> " + (f"STABLE MASS RANGE: up to {max(stable):.2f} kg. Cap the "
                  f"cylinder probe with --mass-max {max(stable):.2f}; rerun the "
                  f"paired probe only after this."
                  if stable else
                  "NO stable mass at r=0.011 -> geometry must change (larger "
                  "disk, re-parameterised R_OBJECT) or, per the timebox, drop "
                  "the cylinder and take the stable box env to the terminal "
                  "keep-out geometry."))
