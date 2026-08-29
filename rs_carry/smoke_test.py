#!/usr/bin/env python3
"""RUN THIS FIRST. Validates every robosuite-1.4.1 API assumption the rest of
the code relies on, then runs one full grasp+transport episode.

Developed against robosuite 1.4.1 docs/source but NOT executed end-to-end by
the author (build container is Python 3.12; robosuite 1.4.1 needs <=3.11), so
this script is the contract: if it prints all OK and one episode completes,
everything downstream uses only the calls verified here.
"""
import numpy as np
from rscarry.env import make_env, Carry, NQ

print("1. creating Lift / UR5e / JOINT_VELOCITY ...")
env = make_env()
c = Carry(env)
print(f"   action_dim={env.action_dim} (expect 7 = 6 joints + 1 gripper)")
print(f"   q0={np.round(c.q(),3)}")
print(f"   eef={np.round(c.eef(),3)}  J_v shape={c.jac_v().shape} (expect (3,6))")
print(f"   cube mass default={env.sim.model.body_mass[c.cube_bid]:.4f}")

print("2. velocity command tracking ...")
for _ in range(15): c._act(np.array([0.3,0,0,0,0,0]), grip=-1.0)
qd = c.qd()
print(f"   commanded 0.3 rad/s on joint 1 -> qd={np.round(qd,3)}")
print(f"   {'OK' if abs(qd[0]-0.3)<0.15 else 'PROBLEM: tracking far off — check controller scaling'}")

print("3. scripted grasp with mass=0.5 ...")
ok = c.reset_and_grasp(0.5)
print(f"   grasp {'OK' if ok else 'FAILED (retry or tune _servo_to gains)'}")
if ok:
    print("4. transport phase, 60 steps ...")
    tr, full = c.transport(60)
    from rscarry.data import margins
    m = margins(tr)
    print(f"   recorded {len(tr['X'])} steps, dropped={tr['dropped']}")
    print(f"   margin min={m.min():.3f} median={np.median(m):.3f} "
          f"near(<0.05)={np.mean((m>=0)&(m<0.05)):.1%} viol={np.mean(m<0):.1%}")
env.close()
print("done. If all OK -> python diag_step1_gap.py --quick")
