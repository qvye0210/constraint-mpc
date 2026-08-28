"""Velocity-level UR5e dynamics.

Modelled at the level the hardware actually exposes: the input is a joint
velocity command (`speedJ` over RTDE), not a torque.  Torque-level modelling is
not executable on a stock UR5e, and a reviewer will ask how it was run.

    state   x = (q, qd) in R^12
    input   u = qd_cmd  in R^6

Nominal model -- first-order lag on the internal servo loop:

    q+  = q  + qd*dt
    qd+ = qd + (u - qd) * dt / tau

The residual network learns what that misses.  The `true` plant here adds three
sources that are all present on the real arm and none of which a smooth MLP can
fit away with more training:

  1. Stribeck + Coulomb friction -- non-smooth at qd = 0
  2. an UNOBSERVED payload -- enters as J_v(q)^T g, so it is configuration
     dependent and its direction is set by the same Jacobian that defines the
     constraint gradient; the model never sees the payload mass
  3. configuration-dependent servo lag -- effective inertia varies with q

Together these give an error floor that does not vanish with budget, which is
the regime the method needs and which the planar testbeds could not produce.
"""

import numpy as np

from .kinematics import NQ, jacobian_v

NX = 2 * NQ
NU = NQ
GRAV = 9.81


class URParams:
    dt = 0.02              # 50 Hz control; measure and fix on hardware
    tau = 0.06             # servo lag time constant; CALIBRATE on hardware
    u_max = 1.5            # rad/s command limit

    # --- misspecification sources (set to 0 to disable) -------------------
    friction = 1.0         # overall friction scale
    F_c = 0.05             # Coulomb level (rad/s per step, velocity domain)
    F_s = 0.14             # breakaway level
    v_s = 0.05             # Stribeck velocity
    delta_s = 2.0
    eps_v = 1e-3

    payload_kg = 0.0       # UNOBSERVED; sampled per episode, never a model input
    payload_gain = 0.02    # compliance: rad/s of velocity error per Nm

    lag_variation = 0.35   # how much tau varies with configuration


def friction_accel(qd, p=URParams):
    """Non-smooth friction opposing joint velocity."""
    if p.friction == 0.0:
        return 0.0
    s = np.abs(qd)
    direction = qd / (s + p.eps_v)
    mag = p.F_c + (p.F_s - p.F_c) * np.exp(-(s / p.v_s) ** p.delta_s)
    return -p.friction * mag * direction


def payload_effect(q, p=URParams):
    """Velocity error induced by an unobserved payload at the TCP.

    tau_g = J_v(q)^T * [0, 0, -m*g]; the joint-level compliance turns that into
    a velocity error.  Configuration dependent by construction, and it shares
    J_v(q) with the constraint gradient -- so its direction genuinely competes
    with the constraint normal instead of living in an orthogonal subspace the
    way synthetic distractor dimensions do.
    """
    if p.payload_kg == 0.0:
        return 0.0
    Jv = jacobian_v(q)                                  # (B,3,6)
    f = np.zeros((len(Jv), 3))
    f[:, 2] = -p.payload_kg * GRAV
    tau_g = np.einsum("bij,bi->bj", Jv, f)              # (B,6)
    return p.payload_gain * tau_g


def lag_factor(q, p=URParams):
    """Configuration-dependent effective servo lag (proxy for varying inertia)."""
    if p.lag_variation == 0.0:
        return 1.0
    reach = np.linalg.norm(jacobian_v(q)[:, :, 1], axis=-1, keepdims=True)
    return 1.0 + p.lag_variation * (reach - 0.5)


def f_nominal(x, u, p=URParams, dt=None):
    """Model the MPC starts from: first-order lag, no friction, no payload."""
    dt = p.dt if dt is None else dt
    x = np.atleast_2d(np.asarray(x, dtype=float))
    u = np.atleast_2d(np.asarray(u, dtype=float))
    q, qd = x[:, :NQ], x[:, NQ:]
    qd_n = qd + (u - qd) * dt / p.tau
    q_n = q + qd * dt
    return np.concatenate([q_n, qd_n], axis=1)


def f_true(x, u, p=URParams, dt=None):
    """Plant."""
    dt = p.dt if dt is None else dt
    x = np.atleast_2d(np.asarray(x, dtype=float))
    u = np.atleast_2d(np.asarray(u, dtype=float))
    q, qd = x[:, :NQ], x[:, NQ:]
    tau_eff = p.tau * lag_factor(q, p)
    qd_n = (qd + (u - qd) * dt / tau_eff
            + dt / p.dt * friction_accel(qd, p)
            + dt / p.dt * payload_effect(q, p))
    q_n = q + qd * dt
    return np.concatenate([q_n, qd_n], axis=1)


def linearised_A(p=URParams, dt=None):
    """Jacobian of the nominal map, used to propagate errors along the horizon.

    A = [[ I,  dt*I ],
         [ 0, (1 - dt/tau) I ]]
    """
    dt = p.dt if dt is None else dt
    A = np.eye(NX)
    A[:NQ, NQ:] = dt * np.eye(NQ)
    A[NQ:, NQ:] = (1.0 - dt / p.tau) * np.eye(NQ)
    return A


def sample_payload(rng, p=URParams, lo=0.0, hi=0.5):
    """Per-episode unobserved payload (kg)."""
    return float(rng.uniform(lo, hi))
