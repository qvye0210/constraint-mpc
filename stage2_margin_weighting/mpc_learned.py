"""
MPC controller that uses a LEARNED dynamics model for prediction.

Since the learned model is a small nonlinear MLP and the existing stage-1
MPC infrastructure (src/mpc_solver.py) is a convex QP built around a FIXED
linear (A, B) pair, we adopt a standard "successive linearization" / real-
time-iteration simplification appropriate for a quick proof-of-concept (no
solver differentiation, no nonlinear programming):

  At every closed-loop control step, linearize the learned model's
  [p, v, u] -> [p_next, v_next] map (first-order Taylor expansion) around
  the CURRENT measured state and the previously applied control. This gives
  a local affine model  x_next = A_lin @ x + B_lin @ u + c_lin  which is
  held FIXED across the prediction horizon for that one MPC solve (frozen
  linearization), then the QP is solved exactly as in stage 1.

The QP itself reuses stage 1's cost weights, bounds, and horizon
(src/config.py) exactly, but with A, B, c promoted to cp.Parameter so the
compiled problem can be reused (DPP) across every closed-loop step and every
episode, method, and seed -- built once, solved thousands of times.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import cvxpy as cp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import ExperimentConfig  # noqa: E402

from models import DynamicsMLP, Normalizer  # noqa: E402


@dataclass
class MPCStepResult:
    u0: float
    status: str
    solve_success: bool
    x_pred_traj: np.ndarray = None
    objective_value: float = float("nan")


_PROBLEM_CACHE = {}


def _get_problem(horizon: int, dt: float, mpc_cfg):
    key = (horizon, round(dt, 12))
    if key in _PROBLEM_CACHE:
        return _PROBLEM_CACHE[key]

    N = horizon
    X = cp.Variable((2, N + 1))
    U = cp.Variable(N)

    x0_param = cp.Parameter(2)
    pref_param = cp.Parameter()
    vref_param = cp.Parameter()
    A_param = cp.Parameter((2, 2))
    B_param = cp.Parameter(2)
    c_param = cp.Parameter(2)

    pmin, pmax = mpc_cfg.pos_bounds
    vmin, vmax = mpc_cfg.vel_bounds
    umin, umax = mpc_cfg.input_bounds

    constraints = [X[:, 0] == x0_param]
    for k in range(N):
        constraints.append(X[:, k + 1] == A_param @ X[:, k] + B_param * U[k] + c_param)

    for k in range(1, N + 1):
        constraints += [X[0, k] <= pmax, X[0, k] >= pmin,
                         X[1, k] <= vmax, X[1, k] >= vmin]
    for k in range(N):
        constraints += [U[k] <= umax, U[k] >= umin]

    cost = 0
    for k in range(N):
        cost += mpc_cfg.q_pos * cp.square(X[0, k] - pref_param)
        cost += mpc_cfg.q_vel * cp.square(X[1, k] - vref_param)
        cost += mpc_cfg.r_u * cp.square(U[k])
    cost += mpc_cfg.q_pos_terminal * cp.square(X[0, N] - pref_param)
    cost += mpc_cfg.q_vel_terminal * cp.square(X[1, N] - vref_param)
    if mpc_cfg.r_du > 0 and N > 1:
        for k in range(1, N):
            cost += mpc_cfg.r_du * cp.square(U[k] - U[k - 1])

    problem = cp.Problem(cp.Minimize(cost), constraints)
    assert problem.is_dcp(dpp=True), "learned-dynamics MPC problem is not DPP-compliant"

    handles = dict(problem=problem, X=X, U=U, x0_param=x0_param, pref_param=pref_param,
                    vref_param=vref_param, A_param=A_param, B_param=B_param, c_param=c_param)
    _PROBLEM_CACHE[key] = handles
    return handles


def linearize_model(model: DynamicsMLP, in_norm: Normalizer, out_norm: Normalizer,
                     p0: float, v0: float, u0: float):
    """First-order Taylor expansion of the learned next-state map around
    (p0, v0, u0). Returns (A_lin (2,2), B_lin (2,), c_lin (2,)) such that
    x_next ~= A_lin @ x + B_lin * u + c_lin matches the model exactly at the
    linearization point and to first order nearby."""
    z0 = torch.tensor([p0, v0, u0], dtype=torch.float32, requires_grad=False)

    def f(z):
        p, v, u = z[0], z[1], z[2]
        x = torch.stack([p, v, u])
        xn = in_norm.normalize(x)
        dy_n = model(xn.unsqueeze(0)).squeeze(0)
        dy = out_norm.denormalize(dy_n)
        p_next = p + dy[0]
        v_next = v + dy[1]
        return torch.stack([p_next, v_next])

    with torch.no_grad():
        x_next0 = f(z0)
    J = torch.autograd.functional.jacobian(f, z0)  # (2,3)
    A_lin = J[:, :2].detach().numpy()
    B_lin = J[:, 2].detach().numpy()
    x_next0_np = x_next0.detach().numpy()
    z0_np = z0.detach().numpy()
    c_lin = x_next0_np - A_lin @ z0_np[:2] - B_lin * z0_np[2]
    return A_lin, B_lin, c_lin


def solve_mpc_step(model: DynamicsMLP, in_norm: Normalizer, out_norm: Normalizer,
                    x_current: np.ndarray, u_prev: float, p_ref: float, v_ref: float,
                    exp_cfg: ExperimentConfig) -> MPCStepResult:
    horizon = exp_cfg.mpc.horizon
    dt = exp_cfg.dynamics.dt
    handles = _get_problem(horizon, dt, exp_cfg.mpc)

    A_lin, B_lin, c_lin = linearize_model(model, in_norm, out_norm,
                                           float(x_current[0]), float(x_current[1]), float(u_prev))

    handles["x0_param"].value = np.asarray(x_current, dtype=float).reshape(2)
    handles["pref_param"].value = float(p_ref)
    handles["vref_param"].value = float(v_ref)
    handles["A_param"].value = A_lin
    handles["B_param"].value = B_lin
    handles["c_param"].value = c_lin

    problem = handles["problem"]
    solved = False
    for solver_name in exp_cfg.solver_preference:
        try:
            problem.solve(solver=getattr(cp, solver_name), warm_start=True)
            solved = True
            break
        except Exception:
            continue

    status = problem.status if solved else "SOLVER_ERROR"
    success = status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

    if not success:
        return MPCStepResult(u0=0.0, status=status, solve_success=False)

    U = handles["U"]
    X = handles["X"]
    u0 = float(U.value[0])
    u0 = float(np.clip(u0, *exp_cfg.mpc.input_bounds))
    return MPCStepResult(u0=u0, status=status, solve_success=True,
                          x_pred_traj=X.value.T.copy(), objective_value=float(problem.value))
