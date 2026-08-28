"""Constraints for the UR5e study, with gradients and propagated weights.

Design note -- why not input constraints.  A bound |u| <= u_max is satisfiable
exactly by clipping, independently of the model, so the causal chain this whole
project rests on ("model error -> wrong predicted state -> constraint violated")
does not exist for it.  That is what makes an input-only setup indefensible
here, not the number of constraints.  Every constraint below is a STATE
constraint whose satisfaction depends on multi-step prediction.

The set is deliberately mixed:

  SphereObstacle  nonlinear, grad_q g = -n^T J_v(q) rotates with configuration.
                  One relevant direction in R^6, five in the nullspace.
  PlaneConstraint fixed Cartesian normal; the linear-normal control for the
                  sphere, so the rotation effect can be ablated.
  JointLimit      axis-aligned in joint space -- a normal pointing somewhere
                  completely different from the Cartesian ones, which is what
                  creates genuine competition between simultaneously active
                  constraints (and is what makes dual variables meaningful).
  VelocityLimit   acts on the qd block, which the Cartesian constraints only
                  reach through propagation.
"""

import numpy as np

from .dynamics import NQ, NX, URParams, linearised_A
from .kinematics import Q_MAX, Q_MIN, QD_MAX, fk, jacobian_v


class Constraint:
    """g(x) <= 0 is satisfied.  grad returns dg/dx of shape (B, NX)."""
    name = "base"

    def value(self, x):
        raise NotImplementedError

    def grad(self, x):
        raise NotImplementedError

    def margin(self, x):
        return -self.value(x)


class SphereObstacle(Constraint):
    """g = r_safe - ||p(q) - p_obs||.  The primary constraint."""
    name = "sphere"

    def __init__(self, p_obs, r_safe=0.15):
        self.p_obs = np.asarray(p_obs, dtype=float)
        self.r = float(r_safe)

    def value(self, x):
        x = np.atleast_2d(x)
        d = fk(x[:, :NQ]) - self.p_obs
        return self.r - np.linalg.norm(d, axis=-1)

    def grad(self, x):
        x = np.atleast_2d(x)
        q = x[:, :NQ]
        d = fk(q) - self.p_obs
        n = d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12)
        g = np.zeros((len(q), NX))
        g[:, :NQ] = -np.einsum("bi,bij->bj", n, jacobian_v(q))
        return g


class PlaneConstraint(Constraint):
    """g = d_min - a^T p(q).  Fixed Cartesian normal; ablation control."""
    name = "plane"

    def __init__(self, normal=(0.0, 0.0, 1.0), offset=0.10):
        n = np.asarray(normal, dtype=float)
        self.a = n / np.linalg.norm(n)
        self.d = float(offset)

    def value(self, x):
        x = np.atleast_2d(x)
        return self.d - fk(x[:, :NQ]) @ self.a

    def grad(self, x):
        x = np.atleast_2d(x)
        q = x[:, :NQ]
        g = np.zeros((len(q), NX))
        g[:, :NQ] = -np.einsum("i,bij->bj", self.a, jacobian_v(q))
        return g


class JointLimit(Constraint):
    """g = q_i - q_max  (upper) or  q_min - q_i  (lower)."""
    name = "joint"

    def __init__(self, joint, upper=True, limit=None):
        self.j = int(joint)
        self.upper = bool(upper)
        self.lim = float(limit if limit is not None
                         else (Q_MAX[self.j] if upper else Q_MIN[self.j]))

    def value(self, x):
        x = np.atleast_2d(x)
        return (x[:, self.j] - self.lim) if self.upper else (self.lim - x[:, self.j])

    def grad(self, x):
        x = np.atleast_2d(x)
        g = np.zeros((len(x), NX))
        g[:, self.j] = 1.0 if self.upper else -1.0
        return g


class VelocityLimit(Constraint):
    """g = |qd_i| - qd_max, smoothed near zero so the gradient is defined."""
    name = "vel"

    def __init__(self, joint, limit=None, eps=1e-6):
        self.j = int(joint)
        self.lim = float(limit if limit is not None else QD_MAX[self.j])
        self.eps = eps

    def value(self, x):
        x = np.atleast_2d(x)
        return np.sqrt(x[:, NQ + self.j] ** 2 + self.eps) - self.lim

    def grad(self, x):
        x = np.atleast_2d(x)
        v = x[:, NQ + self.j]
        g = np.zeros((len(x), NX))
        g[:, NQ + self.j] = v / np.sqrt(v ** 2 + self.eps)
        return g


# ---------------------------------------------------------------------------
def default_constraint_set(p_obs=(0.45, 0.10, 0.35), r_safe=0.15,
                           joints=(1, 2), plane_offset=0.08):
    """The set recommended for the study.

    Joint limits are applied only to the joints that actually approach their
    bound on the task trajectories.  Adding all twelve would leave most of them
    permanently inactive and slow the solve for nothing.
    """
    cs = [SphereObstacle(p_obs, r_safe),
          PlaneConstraint((0.0, 0.0, 1.0), plane_offset)]
    for j in joints:
        cs.append(JointLimit(j, upper=True))
        cs.append(JointLimit(j, upper=False))
    for j in joints:
        cs.append(VelocityLimit(j))
    return cs


def propagated_metric(X_future, constraints, gamma=0.9, p=URParams,
                      eps_floor=0.05, clip_q=0.95):
    """M = sum_k gamma^{k-1} sum_j c_kj c_kj^T,  c_kj = (A^{k-1})^T grad g_j(x_{t+k}).

    Same construction validated on the planar testbed, lifted to R^12.  Joint
    velocities receive weight only through propagation, since dg/dqd = 0 for the
    Cartesian constraints.
    """
    X_future = np.asarray(X_future, dtype=float)
    N, H, _ = X_future.shape
    A = linearised_A(p)
    M = np.zeros((N, NX, NX))
    Ak = np.eye(NX)
    for k in range(1, H + 1):
        for c in constraints:
            ck = c.grad(X_future[:, k - 1]) @ Ak          # (N,NX)
            M += (gamma ** (k - 1)) * ck[:, :, None] * ck[:, None, :]
        Ak = Ak @ A
    tr = np.trace(M, axis1=1, axis2=2)
    M = M + eps_floor * tr.mean() / NX * np.eye(NX)
    cap = np.quantile(np.trace(M, axis1=1, axis2=2), clip_q)
    trm = np.trace(M, axis1=1, axis2=2)
    M = M * np.minimum(1.0, cap / np.maximum(trm, 1e-12))[:, None, None]
    return M / max(np.trace(M, axis1=1, axis2=2).mean(), 1e-12)


def activity_report(X, constraints, near=0.05):
    """Activation and active-set statistics -- report these in the paper.

    Reviewers check that constraints are neither decorative (never active) nor
    degenerate (always active). 10-40% activation is the healthy band.
    """
    out, act = {}, []
    for c in constraints:
        g = c.value(X)
        act.append(g > -near)
        out[f"{c.name}_{id(c) % 1000}"] = dict(
            frac_active=float(np.mean(g > -near)),
            frac_violated=float(np.mean(g > 0)),
            margin_median=float(np.median(-g)))
    act = np.array(act)
    switches = np.mean(np.any(act[:, 1:] != act[:, :-1], axis=0)) if act.shape[1] > 1 else 0.0
    out["_n_active_mean"] = float(act.sum(0).mean())
    out["_active_set_switch_rate"] = float(switches)
    return out
