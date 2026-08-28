"""UR5e plant backed by the official MuJoCo Menagerie model.

Why this replaces the hand-written plant.  `capacity_check.py --part c` showed
the planar testbed was severely over-parameterised: with a fixed architecture the
test error fell monotonically from 7.8e-5 to 1.3e-5 and never bottomed out.  With
no scarce capacity there is nothing for a decision-aware loss to re-allocate, so
every weighting experiment run there tested the wrong thing.  Random noise does
not fix this -- unbiased noise leaves the conditional mean unchanged, so weighted
and unweighted losses share the same optimum.  What creates real competition is a
large amount of learnable-but-not-fully-learnable structure, which is what full
rigid-body dynamics with friction provides.

Interface mirrors the real UR5e: the input is a joint VELOCITY command, and an
internal servo (which the learner never sees) turns it into torques.

    state  x = (q, qd) in R^12
    input  u = qd_cmd  in R^6
    dt     0.02 s outer control period; MuJoCo integrates at 0.002 s inside

Kinematics and the constraint Jacobian come from MuJoCo itself (`mj_jacSite`)
rather than from analytic DH parameters: the hand-written DH table disagreed with
the official model by ~1 mm in position and 1.2e-3 in the Jacobian, and for a
constraint of radius 0.15 m that is a 0.7% inconsistency with no upside.
"""

import os

import mujoco
import numpy as np

NQ = 6
NX = 2 * NQ
NU = NQ

_HERE = os.path.dirname(os.path.abspath(__file__))


def default_xml():
    """Locate the Menagerie UR5e scene.

    Checked in order: UR5E_XML env var, ./menagerie next to the project,
    then a pip-installed mujoco_menagerie if present.
    """
    env = os.environ.get("UR5E_XML")
    if env and os.path.exists(env):
        return env
    for rel in ("../menagerie/universal_robots_ur5e/scene.xml",
                "../../menagerie/universal_robots_ur5e/scene.xml",
                "menagerie/universal_robots_ur5e/scene.xml"):
        cand = os.path.abspath(os.path.join(_HERE, rel))
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        "UR5e scene.xml not found. Clone the model first:\n"
        "  git clone --depth 1 https://github.com/google-deepmind/"
        "mujoco_menagerie.git menagerie\n"
        "or set UR5E_XML to the scene.xml path.")


class MjParams:
    dt = 0.02                 # outer control period (50 Hz)
    u_max = 1.5               # rad/s command limit

    # The internal servo is the model's own PD position actuator (kp 2000 /
    # kd 400), driven by a position target that advances at the commanded
    # velocity -- the same construction the real arm uses for servoj.  Writing a
    # separate torque servo on top fights that actuator and diverges.

    # misspecification the learner has to absorb
    joint_damping = 1.0       # viscous
    frictionloss = 1.0        # dry friction: NON-SMOOTH at qd = 0
    payload_kg = 0.0          # UNOBSERVED, sampled per episode
    armature_scale = 1.0

    # nominal (drag-free, decoupled) model the residual is learned on top of
    tau_nom = 0.08


class UR5ePlant:
    def __init__(self, xml=None, p=MjParams, seed=0):
        self.p = p
        self.model = mujoco.MjModel.from_xml_path(xml or default_xml())
        self.model.opt.timestep = 0.002
        self.data = mujoco.MjData(self.model)
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                      "attachment_site")
        self.n_sub = int(round(p.dt / self.model.opt.timestep))
        self._apply_params()
        self._target = np.zeros(NQ)
        self.rng = np.random.default_rng(seed)

    def _apply_params(self):
        p = self.p
        self.model.dof_damping[:NQ] = p.joint_damping
        self.model.dof_frictionloss[:NQ] = p.frictionloss
        self.model.dof_armature[:NQ] = 0.1 * p.armature_scale
        # payload: add mass at the wrist
        wid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
        self._base_mass = float(self.model.body_mass[wid])
        self._wid = wid
        self.model.body_mass[wid] = self._base_mass + p.payload_kg

    def set_payload(self, kg):
        self.model.body_mass[self._wid] = self._base_mass + kg

    # -- state ------------------------------------------------------------
    def get_state(self):
        return np.concatenate([self.data.qpos[:NQ].copy(),
                               self.data.qvel[:NQ].copy()])

    def set_state(self, x):
        x = np.asarray(x, dtype=float)
        self.data.qpos[:NQ] = x[:NQ]
        self.data.qvel[:NQ] = x[NQ:]
        self._target = np.asarray(x[:NQ], dtype=float).copy()
        mujoco.mj_forward(self.model, self.data)

    # -- dynamics ---------------------------------------------------------
    def step(self, u):
        """Apply a joint-velocity command for one outer period."""
        p = self.p
        u = np.clip(np.asarray(u, dtype=float), -p.u_max, p.u_max)
        h = self.model.opt.timestep
        for _ in range(self.n_sub):
            self._target = self._target + u * h
            self.data.ctrl[:NQ] = self._target
            mujoco.mj_step(self.model, self.data)
        return self.get_state()

    # -- kinematics (from the model itself, not from a DH table) -----------
    def tcp(self, q=None):
        if q is not None:
            self.data.qpos[:NQ] = q
            mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.site].copy()

    def tcp_jacobian(self, q=None):
        if q is not None:
            self.data.qpos[:NQ] = q
            mujoco.mj_forward(self.model, self.data)
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jp, jr, self.site)
        return jp[:, :NQ].copy()


def f_nominal(x, u, p=MjParams, dt=None):
    """Decoupled first-order lag -- what the residual is learned on top of."""
    dt = p.dt if dt is None else dt
    x = np.atleast_2d(np.asarray(x, dtype=float))
    u = np.atleast_2d(np.asarray(u, dtype=float))
    q, qd = x[:, :NQ], x[:, NQ:]
    qd_n = qd + (u - qd) * dt / p.tau_nom
    return np.concatenate([q + qd * dt, qd_n], axis=1)
