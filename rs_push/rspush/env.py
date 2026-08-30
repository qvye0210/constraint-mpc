"""Planar pushing on robosuite 1.4.1 Lift (UR5e): the gripper (closed) pushes
the cube along the table. Per-episode friction and mass are randomised and
UNOBSERVED. A virtual circular forbidden zone lives on the table; the object
must not enter it. Object channel is fully open-loop (no servo corrects it).

Sign convention used EVERYWHERE (code uses positive clearance only):
    clearance rho = ||obj_xy - zone_xy|| - (r_zone + r_object)
    rho > 0 safe, rho = 0 boundary, rho < 0 violation; penetration = max(0,-rho).
"""
import numpy as np

NQ = 6
DT = 0.05                      # 20 Hz
PUSH_H = 0.010                 # eef height above table surface
EEF_VMAX = 0.15                # m/s planar cap -> <=7.5 mm per step (anti-tunnel bound)
R_OBJECT = 0.011               # half of the 2.2cm robosuite cube footprint (disk proxy)


def clearance(obj_xy, zone_xy, r_zone, r_object=R_OBJECT):
    return float(np.linalg.norm(np.asarray(obj_xy) - np.asarray(zone_xy))
                 - (r_zone + r_object))


def make_env(seed=0):
    import robosuite
    from robosuite.controllers import load_controller_config
    cfg = load_controller_config(default_controller="JOINT_VELOCITY")
    env = robosuite.make("Lift", robots="UR5e", controller_configs=cfg,
                         has_renderer=False, has_offscreen_renderer=False,
                         use_camera_obs=False, use_object_obs=False,
                         control_freq=int(round(1 / DT)), horizon=20000,
                         ignore_done=True, hard_reset=False)
    env.reset()
    return env


class Push:
    def __init__(self, env, seed=0):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.r = env.robots[0]
        self.c = self.r.controller
        self.sid = env.sim.model.site_name2id(self.c.eef_name)
        self.bid = env.sim.model.body_name2id("cube_main")
        m = env.sim.model
        self.gid_cube = [i for i in range(m.ngeom)
                         if (m.geom_id2name(i) or "").startswith("cube_g")]
        self.gid_table = [i for i in range(m.ngeom)
                          if "table" in (m.geom_id2name(i) or "")
                          and "collision" in (m.geom_id2name(i) or "")]
        self.jadr = m.get_joint_qpos_addr("cube_joint0")[0]
        self.vadr = m.get_joint_qvel_addr("cube_joint0")[0]
        self._m0 = float(m.body_mass[self.bid])
        self._I0 = m.body_inertia[self.bid].copy()
        self.table_z = float(env.sim.data.body_xpos[
            m.body_name2id("table")][2]) + 0.025  # table top approx

    # ---- basic access ---------------------------------------------------
    def eef(self):
        return self.env.sim.data.site_xpos[self.sid].copy()

    def jac_v(self):
        return np.asarray(self.c.J_full)[:3, :NQ]

    def obj_pose(self):
        d = self.env.sim.data
        x, y = d.qpos[self.jadr:self.jadr + 2]
        w, qx, qy, qz = d.qpos[self.jadr + 3:self.jadr + 7]
        yaw = np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return np.array([x, y, yaw])

    def obj_z(self):
        return float(self.env.sim.data.qpos[self.jadr + 2])

    # ---- episode spec ---------------------------------------------------
    def apply_spec(self, spec):
        m = self.env.sim.model
        self.env.reset()
        m.body_mass[self.bid] = spec["mass"]
        m.body_inertia[self.bid] = self._I0 * (spec["mass"] / max(self._m0, 1e-9))
        for g in self.gid_cube + self.gid_table:
            m.geom_friction[g][0] = spec["friction"]
        d = self.env.sim.data
        d.qpos[self.jadr:self.jadr + 2] = spec["obj_xy"]
        yaw = spec.get("obj_yaw", 0.0)
        d.qpos[self.jadr + 3:self.jadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qvel[self.vadr:self.vadr + 6] = 0
        self.env.sim.forward()
        # settle, then move eef to the pre-push pose behind the object
        for _ in range(5):
            self._step_qd(np.zeros(NQ))
        direc = spec["goal_xy"] - spec["obj_xy"]
        direc = direc / (np.linalg.norm(direc) + 1e-9)
        pre = np.array([*(spec["obj_xy"] - 0.065 * direc), self.table_z + PUSH_H])
        ok = self._servo_to(pre, 120)
        return ok and abs(self.obj_z() - self.table_z) < 0.05

    # ---- low-level motion ----------------------------------------------
    def _step_qd(self, qd):
        a = np.zeros(self.env.action_dim)
        a[:NQ] = np.clip(qd, -1, 1)
        a[NQ:] = 1.0                       # keep gripper closed = rigid pusher
        self.env.step(a)

    def _servo_to(self, p, steps, gain=3.0, tol=0.008):
        for _ in range(steps):
            e = p - self.eef()
            if np.linalg.norm(e) < tol:
                return True
            u = np.linalg.pinv(self.jac_v(), rcond=1e-4) @ (gain * e)
            n = np.linalg.norm(u)
            if n > 0.8:
                u *= 0.8 / n
            self._step_qd(u)
        return np.linalg.norm(p - self.eef()) < 2 * tol

    def step_eef_vel(self, v_xy):
        """One 20 Hz control step with a planar eef velocity command (the
        planner's action space). Height held by P; returns realized states."""
        v_xy = np.asarray(v_xy, dtype=float)
        n = np.linalg.norm(v_xy)
        if n > EEF_VMAX:
            v_xy = v_xy * EEF_VMAX / n
        vz = 2.0 * (self.table_z + PUSH_H - self.eef()[2])
        u = np.linalg.pinv(self.jac_v(), rcond=1e-4) @ np.array([*v_xy, vz])
        nn = np.linalg.norm(u)
        if nn > 0.9:
            u *= 0.9 / nn
        p0 = self.obj_pose()[:2]
        self._step_qd(u)
        v_obj = self.env.sim.data.qvel[self.vadr:self.vadr + 2]
        return dict(eef=self.eef(), obj=self.obj_pose(),
                    obj_step=float(np.linalg.norm(self.obj_pose()[:2] - p0)),
                    obj_speed=float(np.linalg.norm(v_obj)))
