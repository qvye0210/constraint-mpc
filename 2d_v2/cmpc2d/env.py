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

    # --- Stribeck friction: STRUCTURAL model misspecification -----------------
    # Unlike distractor dimensions (which consume capacity but are orthogonal to
    # the constraint geometry), friction acts ALONG THE VELOCITY DIRECTION, so
    # the residual competes with the constraint normal in the same physical
    # space: normal when approaching the obstacle, tangential when skirting it.
    # It is also non-smooth at v=0, so a smooth MLP cannot fit it away.
    stribeck = 0.0      # 0 disables; ~1.0 is a strong effect
    F_c = 0.8           # Coulomb level
    F_s = 2.4           # static / breakaway level
    v_s = 0.12          # Stribeck velocity
    delta_s = 2.0       # Stribeck exponent
    eps_v = 1e-3        # regularises v/|v| at zero (keeps the transition sharp)

    # reference: straight sweep whose nominal path cuts THROUGH the obstacle,
    # forcing the constraint to be genuinely active rather than decorative.
    ref_x0, ref_x1 = -5.0, 5.0
    ref_y = -0.35


# ----------------------------------------------------------------------------
# dynamics
# ----------------------------------------------------------------------------
def friction_accel(vel, p=Params):
    """Stribeck + Coulomb friction, opposing the velocity direction."""
    if p.stribeck == 0.0:
        return 0.0
    speed = np.linalg.norm(vel, axis=-1, keepdims=True)
    direction = vel / (speed + p.eps_v)
    mag = p.F_c + (p.F_s - p.F_c) * np.exp(-(speed / p.v_s) ** p.delta_s)
    return -p.stribeck * mag * direction


def f_true(x, u, p=Params, dt=None):
    """Plant: quadratic drag, plus Stribeck friction when enabled."""
    dt = p.dt if dt is None else dt
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    pos, vel = x[..., :2], x[..., 2:]
    speed = np.linalg.norm(vel, axis=-1, keepdims=True)
    acc = u - p.drag * speed * vel + friction_accel(vel, p)
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


# ----------------------------------------------------------------------------
# Distracting state dimensions
# ----------------------------------------------------------------------------
# Following the construction used by VaGraM (Voelcker et al., ICLR 2022): append
# superfluous state dimensions that follow an INDEPENDENT nonlinear dynamical
# system.  They are irrelevant to both the constraint and the cost, but the
# model must still predict them, so they permanently consume capacity.
#
# This is a stronger premise than "use a small network": shrinking the net can
# always be answered with "train longer / use a bigger net", whereas distractor
# dimensions impose an error floor that does not vanish with budget.

def f_distract(z, p=Params, dt=None):
    """Independent nonlinear system: pairs of Van der Pol style oscillators.

    z: (..., D) with D even.  Each pair (a, b) evolves as a limit cycle with a
    pair-specific frequency, which makes the map genuinely nonlinear without
    diverging.
    """
    dt = p.dt if dt is None else dt
    z = np.asarray(z, dtype=float)
    D = z.shape[-1]
    a, b = z[..., 0::2], z[..., 1::2]
    k = np.arange(D // 2)
    omega = 1.0 + 0.7 * k                      # different frequency per pair
    mu = 1.5 + 0.3 * k
    r2 = a ** 2 + b ** 2
    da = -omega * b + mu * (1.0 - r2) * a
    db = omega * a + mu * (1.0 - r2) * b
    out = np.empty_like(z)
    out[..., 0::2] = a + dt * da
    out[..., 1::2] = b + dt * db
    return out


def sample_distract(rng, n, D):
    """Draw distractor states covering the annulus around the limit cycle."""
    if D == 0:
        return np.zeros((n, 0))
    th = rng.uniform(0, 2 * np.pi, size=(n, D // 2))
    r = rng.uniform(0.3, 1.6, size=(n, D // 2))
    z = np.empty((n, D))
    z[:, 0::2] = r * np.cos(th)
    z[:, 1::2] = r * np.sin(th)
    return z
