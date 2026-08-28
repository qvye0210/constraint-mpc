"""UR5e forward kinematics and geometric Jacobian.

Standard (Denavit-Hartenberg) parameters for the UR5e, from Universal Robots'
published table.  Everything is analytic and batched over configurations, so it
can be called inside a rollout without a robotics dependency.

The position Jacobian is the object the whole study turns on: for a Cartesian
constraint g(q) = r - ||p(q) - p_obs||,

    grad_q g = -n^T J_v(q)        n = (p - p_obs) / ||p - p_obs||

is a single row in R^6.  One relevant direction, five in the nullspace -- the
5/6 codimension that makes the arm a far better testbed than the planar case,
and it comes from the real robot rather than from an artificial construction.
"""

import numpy as np

NQ = 6

# standard DH: theta_i is the joint variable
D = np.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996])
A = np.array([0.0, -0.425, -0.3922, 0.0, 0.0, 0.0])
ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])

# UR5e joint limits (rad).  The hardware allows +-2*pi; these are the tighter
# software limits used for the experiments so that limits can actually become
# active without driving the arm through awkward configurations.
Q_MIN = np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi])
Q_MAX = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])
QD_MAX = np.array([2.0, 2.0, 3.0, 3.0, 3.0, 3.0])       # rad/s, conservative


def _dh_matrix(theta, d, a, alpha):
    """(B,) joint angle -> (B,4,4) homogeneous transform."""
    theta = np.atleast_1d(theta)
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    B = len(theta)
    T = np.zeros((B, 4, 4))
    T[:, 0, 0] = ct
    T[:, 0, 1] = -st * ca
    T[:, 0, 2] = st * sa
    T[:, 0, 3] = a * ct
    T[:, 1, 0] = st
    T[:, 1, 1] = ct * ca
    T[:, 1, 2] = -ct * sa
    T[:, 1, 3] = a * st
    T[:, 2, 1] = sa
    T[:, 2, 2] = ca
    T[:, 2, 3] = d
    T[:, 3, 3] = 1.0
    return T


def fk_all(q):
    """q: (B,6) -> list of (B,4,4) cumulative transforms T_0..T_6 (7 entries)."""
    q = np.atleast_2d(np.asarray(q, dtype=float))
    B = len(q)
    T = np.tile(np.eye(4), (B, 1, 1))
    out = [T]
    for i in range(NQ):
        T = T @ _dh_matrix(q[:, i], D[i], A[i], ALPHA[i])
        out.append(T)
    return out


def fk(q):
    """TCP position, (B,3)."""
    return fk_all(q)[-1][:, :3, 3]


def fk_pose(q):
    """TCP position (B,3) and rotation (B,3,3)."""
    T = fk_all(q)[-1]
    return T[:, :3, 3], T[:, :3, :3]


def jacobian(q):
    """Geometric Jacobian, (B,6,6): rows 0-2 linear, rows 3-5 angular.

    Revolute joints only:  J_v_i = z_i x (p_e - p_i),  J_w_i = z_i.
    """
    Ts = fk_all(q)
    p_e = Ts[-1][:, :3, 3]
    B = len(p_e)
    J = np.zeros((B, 6, NQ))
    for i in range(NQ):
        z_i = Ts[i][:, :3, 2]
        p_i = Ts[i][:, :3, 3]
        J[:, :3, i] = np.cross(z_i, p_e - p_i)
        J[:, 3:, i] = z_i
    return J


def jacobian_v(q):
    """Position (linear) Jacobian only, (B,3,6)."""
    return jacobian(q)[:, :3, :]


def manipulability(q):
    """sqrt(det(J_v J_v^T)) -- small values flag near-singular configurations."""
    Jv = jacobian_v(q)
    G = Jv @ np.transpose(Jv, (0, 2, 1))
    return np.sqrt(np.maximum(np.linalg.det(G), 0.0))


def random_configs(rng, n, margin=0.25):
    """Configurations inside the software limits, away from the boundary."""
    lo, hi = Q_MIN + margin, Q_MAX - margin
    return rng.uniform(lo, hi, size=(n, NQ))
