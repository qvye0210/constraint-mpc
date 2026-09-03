#!/usr/bin/env python3
"""Cylinder stability self-test v2. Fixes from review of v1:
 * verdict is the FULL conjunction (finite, badqacc, no tip, setup_ok, rest
   height, per-phase penetration) -- v1 only checked finite&&!tip and crowned
   0.40 kg while the object lay 800 mm away;
 * outputs the PASSING SET, no monotonic "up to X" assumption;
 * masses follow the BOX experiments' DENSITY range (box V=7.4e-5 m^3, masses
   0.05-0.6 kg => 675-8100 kg/m^3; cylinder V=8.36e-6 => 0.006-0.068 kg), so
   the material is held fixed while the shape changes;
 * penetration counts CYLINDER-involved contacts only; rest and push phases
   reported separately; max |qacc| on the object's DOFs recorded.
    python cyl_selftest.py
"""
import numpy as np
from rspush.env import make_env, Push, NQ

env = make_env(object_shape="cylinder"); p = Push(env)
m = env.sim.model
gid = set(p.gid_cube)
jid = m.joint_name2id("cube_joint0")
obj_dofs = [d for d in range(m.nv) if m.dof_jntid[d] == jid]
print(f"object dofs {obj_dofs}; geom_size {m.geom_size[p.gid_cube[0]][:2]} "
      f"(= [radius, half_height])")
half_h = float(m.geom_size[p.gid_cube[0]][1])
V = float(np.pi * m.geom_size[p.gid_cube[0]][0] ** 2 * 2 * half_h)

def phase(n, act):
    bad = pen = qmax = 0.0; tilt = 0.0
    for _ in range(n):
        act()
        d = env.sim.data
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
                and np.isfinite(d.qacc).all()):
            bad += 1
        qmax = max(qmax, float(np.abs(d.qacc[obj_dofs]).max()))
        for c in range(d.ncon):
            if d.contact[c].geom1 in gid or d.contact[c].geom2 in gid:
                pen = max(pen, -float(d.contact[c].dist))
        w, x, y, z = d.qpos[p.jadr + 3:p.jadr + 7]
        tilt = max(tilt, float(np.degrees(np.arccos(
            np.clip(1 - 2 * (x * x + y * y), -1, 1)))))
    return bad, pen, qmax, tilt

passing = []
for mass in (0.006, 0.010, 0.020, 0.040, 0.068):
    rho = mass / V
    spec = dict(friction=0.6, mass=mass, obj_xy=np.array([0.0, 0.0]),
                obj_yaw=0.0, goal_xy=np.array([0.20, 0.0]), np_seed=42)
    ok_setup = p.apply_spec(spec)
    z0 = p.obj_z(); rest_err = abs(z0 - (p.table_z + half_h))
    b1, pen1, q1, t1 = phase(300, lambda: p._step_qd(np.zeros(NQ)))
    b2, pen2, q2, t2 = phase(100, lambda: p.step_eef_vel(np.array([0.03, 0.0])))
    z_end = p.obj_z()
    ok = (b1 + b2 == 0 and ok_setup and rest_err < 0.005
          and abs(z_end - (p.table_z + half_h)) < 0.05
          and max(t1, t2) < 20.0 and pen1 < 0.003 and pen2 < 0.005
          and max(q1, q2) < 5e3)
    if ok:
        passing.append(mass)
    print(f"mass {mass:.3f} (rho {rho:6.0f}): setup={ok_setup} "
          f"rest[pen {pen1*1000:.1f}mm qacc {q1:.0f} tilt {t1:.0f}deg "
          f"z_err {rest_err*1000:.1f}mm]  push[pen {pen2*1000:.1f}mm "
          f"qacc {q2:.0f} tilt {t2:.0f}deg z_end_err "
          f"{abs(z_end-(p.table_z+half_h))*1000:.1f}mm]  -> "
          f"{'PASS' if ok else 'fail'}")
env.close()
print("\n>>> passing mass set: " + (str(passing) if passing else "EMPTY"))
print(">>> " + (f"use --mass-max {max(passing):.3f} AND --mass-min "
               f"{min(passing):.3f} equivalents in the probe (box-matched "
               f"densities)." if passing else
               "no stable mass at box-matched densities -> ONE geometry "
               "correctness fix allowed (wider flatter puck that cannot enter "
               "the fingertip gap, push height at disk mid, R_OBJECT "
               "re-parameterised); still unstable -> timebox: drop the "
               "cylinder, take the stable box env to the terminal keep-out "
               "geometry."))
