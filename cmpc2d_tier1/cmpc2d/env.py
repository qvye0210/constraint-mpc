"""Tier-1 testbed: 2D point mass with a circular obstacle constraint.

Why this system (see design notes):
  * dynamics stay simple (near-double-integrator) so MPC is fast and debuggable
  * the constraint g(x) = r_safe - ||p - p_obs|| is NONLINEAR in p, so its
    gradient direction rotates with the state.  That gives a genuine
    decision-irrelevant direction (tangential) -- which is exactly the
    property the 1D double integrator lacked.

State  x = [px, py, vx, vy]
Input  u = [ax, ay]
"""

import numpy as np

NX = 4
NU = 2


class Params:
    dt = 0.15
    drag = 1.2          # quadratic drag coeff -> mild nonlinearity + plant/model mismatch
    u_max = 4.0
    r_safe = 1.0        # constraint radius
    ep_len = 60
    resolve_every = 5   # zero-order hold; makes multi-step model error observable
    viol_tol = 1e-4     # ignore numerical boundary grazing

    # reference: straight sweep whose nominal path cuts THROUGH the obstacle,
    # forcing the constraint to be genuinely active rather than decorative.
    ref_x0, ref_x1 = -5.0, 5.0
    ref_y = -0.35


# ----------------------------------------------------------------------------
# dynamics
# ----------------------------------------------------------------------------
def f_true(x, u, p=Params, dt=None):
    """Plant. Quadratic drag on velocity."""
    dt = p.dt if dt is None else dt
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    pos, vel = x[..., :2], x[..., 2:]
    speed = np.linalg.norm(vel, axis=-1, keepdims=True)
    acc = u - p.drag * speed * vel
    pos_n = pos + vel * dt + 0.5 * acc * dt * dt
    vel_n = vel + acc * dt
    return np.concatenate([pos_n, vel_n], axis=-1)


def f_nominal(x, u, p=Params, dt=None):
    """Drag-free model -> structured plant-model mismatch (what f_theta must learn)."""
    dt = p.dt if dt is None else dt
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    pos, vel = x[..., :2], x[..., 2:]
    pos_n = pos + vel * dt + 0.5 * u * dt * dt
    vel_n = vel + u * dt
    return np.concatenate([pos_n, vel_n], axis=-1)


# ----------------------------------------------------------------------------
# constraint geometry
# ----------------------------------------------------------------------------
def g_val(x, p_obs, p=Params):
    """g(x) <= 0 is satisfied.  g = r_safe - ||p - p_obs||."""
    x = np.asarray(x, dtype=float)
    d = np.linalg.norm(x[..., :2] - np.asarray(p_obs), axis=-1)
    return p.r_safe - d


def margin(x, p_obs, p=Params):
    """m = -g >= 0 : distance to constraint activation."""
    return -g_val(x, p_obs, p)


def normal_dir(x, p_obs, eps=1e-9):
    """Unit vector in POSITION space along dg/dp (radial, pointing inward).

    An error along this direction changes g at first order.
    """
    x = np.asarray(x, dtype=float)
    d = x[..., :2] - np.asarray(p_obs)
    n = d / (np.linalg.norm(d, axis=-1, keepdims=True) + eps)
    return -n  # dg/dp = -(p-p_obs)/||.||


def tangent_dir(x, p_obs):
    """Unit vector orthogonal to the constraint normal (position space).

    To first order an error along this direction does NOT change g.
    """
    n = normal_dir(x, p_obs)
    return np.stack([-n[..., 1], n[..., 0]], axis=-1)


# ----------------------------------------------------------------------------
# scenario sampling
# ----------------------------------------------------------------------------
def sample_scenario(rng, p=Params, jitter=True):
    """Randomised obstacle / reference / initial state.

    Obstacle position is randomised so the model cannot memorise one geometry.
    """
    if jitter:
        p_obs = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4)])
        ref_y = p.ref_y + rng.uniform(-0.25, 0.25)
        x0 = np.array([p.ref_x0, ref_y, rng.uniform(1.0, 2.0), rng.uniform(-0.2, 0.2)])
    else:
        p_obs = np.zeros(2)
        ref_y = p.ref_y
        x0 = np.array([p.ref_x0, ref_y, 1.5, 0.0])
    return dict(p_obs=p_obs, ref_y=ref_y, x0=x0)


def ref_traj(scn, n_steps, p=Params):
    """Position reference; sweeps left->right at height ref_y."""
    s = np.linspace(0.0, 1.0, n_steps + 1)
    px = p.ref_x0 + (p.ref_x1 - p.ref_x0) * s
    py = np.full_like(px, scn["ref_y"])
    return np.stack([px, py], axis=-1)
