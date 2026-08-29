"""Carry-unknown-payload task on robosuite 1.4.1 Lift (UR5e, JOINT_VELOCITY).

Episode structure:
  1. reset, randomise cube mass (UNOBSERVED by any model)
  2. scripted grasp (not recorded, retried on failure)
  3. transport phase (recorded): follow a task-space reference toward a sampled
     target, passing near a virtual obstacle placed on the corridor

Why JOINT_VELOCITY: it is a plain P controller on joint-velocity error, the
real-UR5e equivalent is speedj over RTDE, joint-space actions make joint-limit
constraints first-class, and unlike OSC it does not use the (sim-exact) mass
matrix to cancel the very dynamics the model is supposed to learn.

Why a grasped object rather than extra wrist mass: robosuite compensates
gravity of everything in the kinematic tree (qfrc_bias), so welded/attached
mass would be cancelled by the controller. A grasped object acts through
CONTACT forces, which are not compensated -- the payload effect stays in the
residual, exactly as on the real robot.
"""

import numpy as np

NQ = 6


def make_env(control_freq=20, horizon=4000, seed=0):
    import robosuite
    from robosuite.controllers import load_controller_config
    cfg = load_controller_config(default_controller="JOINT_VELOCITY")
    env = robosuite.make(
        "Lift", robots="UR5e", controller_configs=cfg,
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, use_object_obs=False,
        control_freq=control_freq, horizon=horizon,
        ignore_done=True, hard_reset=False,       # keep model so mass edits persist
    )
    env.reset()
    return env


class Carry:
    def __init__(self, env, seed=0):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.r = env.robots[0]
        self.c = self.r.controller
        self.sid = env.sim.model.site_name2id(self.c.eef_name)
        self.cube_bid = env.sim.model.body_name2id("cube_main")
        self._m0 = float(env.sim.model.body_mass[self.cube_bid])
        self._I0 = env.sim.model.body_inertia[self.cube_bid].copy()
        self.gdim = env.action_dim - NQ

    # ---- state access ---------------------------------------------------
    def q(self):
        return self.env.sim.data.qpos[self.r._ref_joint_pos_indexes].copy()

    def qd(self):
        return self.env.sim.data.qvel[self.r._ref_joint_vel_indexes].copy()

    def eef(self):
        return self.env.sim.data.site_xpos[self.sid].copy()

    def jac_v(self):
        J = np.asarray(self.c.J_full)          # (6, dof)
        return J[:3, :NQ].copy()

    def cube_pos(self):
        return self.env.sim.data.body_xpos[self.cube_bid].copy()

    def _act(self, u, grip):
        a = np.zeros(self.env.action_dim)
        # map desired joint velocity (rad/s) into the controller's input range
        lo_i, hi_i = np.asarray(self.c.input_min), np.asarray(self.c.input_max)
        lo_o, hi_o = np.asarray(self.c.output_min), np.asarray(self.c.output_max)
        a[:NQ] = np.clip((u - lo_o[:NQ]) / (hi_o[:NQ] - lo_o[:NQ])
                         * (hi_i[:NQ] - lo_i[:NQ]) + lo_i[:NQ], lo_i[:NQ], hi_i[:NQ])
        if self.gdim:
            a[NQ:] = grip
        self.env.step(a)
        return a[:NQ] * 0 + u                 # commanded velocity in rad/s

    def _servo_to(self, p_target, steps, gain=3.0, u_cap=0.6, grip=-1.0, tol=0.01):
        for _ in range(steps):
            e = p_target - self.eef()
            if np.linalg.norm(e) < tol:
                return True
            J = self.jac_v()
            u = np.linalg.pinv(J, rcond=1e-4) @ (gain * e)
            n = np.linalg.norm(u)
            if n > u_cap:
                u = u * u_cap / n
            self._act(u, grip)
        return np.linalg.norm(p_target - self.eef()) < 2 * tol

    # ---- episode phases -------------------------------------------------
    def reset_and_grasp(self, mass, max_tries=3):
        for _ in range(max_tries):
            self.env.reset()
            self.env.sim.model.body_mass[self.cube_bid] = mass
            self.env.sim.model.body_inertia[self.cube_bid] = \
                self._I0 * (mass / max(self._m0, 1e-9))
            cube = self.cube_pos()
            ok = self._servo_to(cube + [0, 0, 0.10], 80, grip=-1.0)
            ok = ok and self._servo_to(cube + [0, 0, 0.005], 60, gain=2.0,
                                       u_cap=0.35, grip=-1.0)
            for _ in range(12):                       # close
                self._act(np.zeros(NQ), grip=1.0)
            self._servo_to(self.eef() + [0, 0, 0.12], 60, grip=1.0)
            if self.cube_pos()[2] > cube[2] + 0.06:   # actually lifted
                return True
        return False

    def transport(self, T, explore=0.25, gain=2.5, u_cap=0.8, hold_u=5):
        """Recorded phase. Returns per-step arrays; margins computed by caller."""
        start = self.eef()
        # target across the table, same height band
        tgt = start + np.array([self.rng.uniform(-0.30, 0.30),
                                self.rng.uniform(-0.35, 0.35),
                                self.rng.uniform(-0.05, 0.15)])
        # virtual obstacle on the corridor: near the midpoint, offset sideways
        mid = 0.5 * (start + tgt)
        off = self.rng.normal(0, 1, 3); off /= np.linalg.norm(off)
        p_obs = mid + off * self.rng.uniform(0.05, 0.12)

        X, U, Xn, P, Jv = [], [], [], [], []
        u_noise = np.zeros(NQ)
        for k in range(T):
            alpha = min(1.0, k / (0.8 * T))
            p_ref = start + alpha * (tgt - start)
            if k % hold_u == 0:
                u_noise = self.rng.normal(0, explore, NQ)
            J = self.jac_v()
            u = np.linalg.pinv(J, rcond=1e-4) @ (gain * (p_ref - self.eef()))
            u = u + u_noise
            n = np.linalg.norm(u)
            if n > u_cap:
                u = u * u_cap / n
            x = np.concatenate([self.q(), self.qd()])
            p = self.eef(); Jk = self.jac_v()
            self._act(u, grip=1.0)
            X.append(x); U.append(u.copy()); P.append(p); Jv.append(Jk)
            Xn.append(np.concatenate([self.q(), self.qd()]))
            if self.cube_pos()[2] < 0.85:             # dropped (table ~0.8)
                break
        return (dict(X=np.array(X), U=np.array(U), Xn=np.array(Xn),
                     eef=np.array(P), Jv=np.array(Jv), p_obs=p_obs,
                     dropped=len(X) < T),
                len(X) == T)
