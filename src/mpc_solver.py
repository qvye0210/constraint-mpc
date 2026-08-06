"""
Finite-horizon tracking MPC formulated and solved as a QP with CVXPY.

Supports:
  - "oracle" mode: true dynamics used for prediction.
  - "perturbed" mode: true dynamics plus a constant additive bias vector
    used for prediction (same A, B; only the affine offset differs).

Both modes use IDENTICAL cost function structure and IDENTICAL hard
constraints -- only the prediction model differs. This is verified by a
correctness test in tests/.

Extracts, per solve:
  - full state/control trajectories
  - per-step, per-constraint slack (margin) values
  - per-step, per-constraint dual variables (raw and normalized)
  - horizon-minimum state margin / input margin and which constraint
    step attains it
  - active-set indicator vectors (state and input, separately)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cvxpy as cp

from .config import MPCConfig, ToleranceConfig
from .dynamics import get_double_integrator_matrices


@dataclass
class MPCSolution:
    status: str
    solve_success: bool

    # Trajectories
    x_traj: np.ndarray = None       # (N+1, 2) predicted state trajectory (under the model used)
    u_traj: np.ndarray = None       # (N,) control trajectory
    u0: float = np.nan              # first control action

    objective_value: float = np.nan  # optimizer-reported objective

    # State constraint bookkeeping, arrays indexed k = 1..N (length N)
    pos_upper_slack: np.ndarray = None
    pos_lower_slack: np.ndarray = None
    vel_upper_slack: np.ndarray = None
    vel_lower_slack: np.ndarray = None

    pos_upper_dual: np.ndarray = None
    pos_lower_dual: np.ndarray = None
    vel_upper_dual: np.ndarray = None
    vel_lower_dual: np.ndarray = None

    # Input constraint bookkeeping, arrays indexed k = 0..N-1 (length N)
    u_upper_slack: np.ndarray = None
    u_lower_slack: np.ndarray = None
    u_upper_dual: np.ndarray = None
    u_lower_dual: np.ndarray = None

    # Summary scalars
    min_state_margin: float = np.nan
    min_state_margin_step: int = -1
    min_state_margin_kind: str = ""

    min_input_margin: float = np.nan
    min_input_margin_step: int = -1
    min_input_margin_kind: str = ""

    # Active-set indicators (boolean arrays), state: length N x 4
    # order: [pos_upper, pos_lower, vel_upper, vel_lower]
    state_active: np.ndarray = None
    input_active: np.ndarray = None  # length N x 2: [u_upper, u_lower]

    solver_name: str = ""


def _solve_qp(x0: np.ndarray, p_ref: float, v_ref: float,
              dt: float, mpc_cfg: MPCConfig,
              bias: Optional[np.ndarray],
              solver_preference: tuple[str, ...]) -> tuple[MPCSolution, dict]:
    """Build (or reuse a cached, DPP-parametrized) QP and solve it.

    If bias is None -> oracle model (true dynamics, bias parameter set to 0).
    If bias is a 2-vector -> perturbed model (true dynamics + constant bias
    added at every predicted step).

    The problem structure (horizon, weights, bounds) only depends on
    (dt, mpc_cfg); x0, p_ref, v_ref, bias vary per call. We build the CVXPY
    problem ONCE per distinct (dt, mpc_cfg) using cp.Parameter for the
    per-call quantities (Disciplined Parametrized Programming), then reuse
    the compiled problem on every subsequent call -- this avoids CVXPY
    re-canonicalizing the problem from scratch on every solve, which
    dominates wall-clock time for many small QPs of identical structure.
    """
    handles = _get_or_build_problem(dt, mpc_cfg)

    handles["x0_param"].value = np.asarray(x0, dtype=float).reshape(2)
    handles["pref_param"].value = float(p_ref)
    handles["vref_param"].value = float(v_ref)
    bias_vec = np.zeros(2) if bias is None else np.asarray(bias, dtype=float).reshape(2)
    handles["bias_param"].value = bias_vec

    problem = handles["problem"]
    N = mpc_cfg.horizon
    X = handles["X"]
    U = handles["U"]

    solved = False
    used_solver = ""
    for solver_name in solver_preference:
        try:
            solver_attr = getattr(cp, solver_name)
            problem.solve(solver=solver_attr, warm_start=True)
            used_solver = solver_name
            solved = True
            break
        except Exception:
            continue

    status = problem.status if solved else "SOLVER_ERROR"
    success = status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

    sol = MPCSolution(status=status, solve_success=success, solver_name=used_solver)

    if not success:
        return sol, handles

    x_traj = X.value.T  # (N+1, 2)
    u_traj = U.value.flatten()  # (N,)

    sol.x_traj = x_traj
    sol.u_traj = u_traj
    sol.u0 = float(u_traj[0])
    sol.objective_value = float(problem.value)

    pmin, pmax = mpc_cfg.pos_bounds
    vmin, vmax = mpc_cfg.vel_bounds
    umin, umax = mpc_cfg.input_bounds

    # Compute slacks directly and robustly from bound values (not from cvxpy internals)
    pos_upper_slack = np.array([pmax - x_traj[k, 0] for k in range(1, N + 1)])
    pos_lower_slack = np.array([x_traj[k, 0] - pmin for k in range(1, N + 1)])
    vel_upper_slack = np.array([vmax - x_traj[k, 1] for k in range(1, N + 1)])
    vel_lower_slack = np.array([x_traj[k, 1] - vmin for k in range(1, N + 1)])

    u_upper_slack = np.array([umax - u_traj[k] for k in range(N)])
    u_lower_slack = np.array([u_traj[k] - umin for k in range(N)])

    def dual_arr(cons_list):
        vals = []
        for c in cons_list:
            d = c.dual_value
            if d is None:
                vals.append(0.0)
            else:
                vals.append(float(np.atleast_1d(d)[0]))
        return np.array(vals)

    pos_upper_dual = dual_arr(handles["pos_upper_cons"])
    pos_lower_dual = dual_arr(handles["pos_lower_cons"])
    vel_upper_dual = dual_arr(handles["vel_upper_cons"])
    vel_lower_dual = dual_arr(handles["vel_lower_cons"])
    u_upper_dual = dual_arr(handles["u_upper_cons"])
    u_lower_dual = dual_arr(handles["u_lower_cons"])

    sol.pos_upper_slack = pos_upper_slack
    sol.pos_lower_slack = pos_lower_slack
    sol.vel_upper_slack = vel_upper_slack
    sol.vel_lower_slack = vel_lower_slack
    sol.pos_upper_dual = pos_upper_dual
    sol.pos_lower_dual = pos_lower_dual
    sol.vel_upper_dual = vel_upper_dual
    sol.vel_lower_dual = vel_lower_dual

    sol.u_upper_slack = u_upper_slack
    sol.u_lower_slack = u_lower_slack
    sol.u_upper_dual = u_upper_dual
    sol.u_lower_dual = u_lower_dual

    # Horizon-minimum STATE margin (across pos+vel, all steps) -- primary quantity
    state_slacks = np.stack([pos_upper_slack, pos_lower_slack,
                              vel_upper_slack, vel_lower_slack], axis=1)  # (N,4)
    kinds = ["pos_upper", "pos_lower", "vel_upper", "vel_lower"]
    flat_idx = np.argmin(state_slacks)
    step_idx, kind_idx = np.unravel_index(flat_idx, state_slacks.shape)
    sol.min_state_margin = float(state_slacks[step_idx, kind_idx])
    sol.min_state_margin_step = int(step_idx + 1)  # horizon step (1-indexed)
    sol.min_state_margin_kind = kinds[kind_idx]

    input_slacks = np.stack([u_upper_slack, u_lower_slack], axis=1)  # (N,2)
    ikinds = ["u_upper", "u_lower"]
    iflat = np.argmin(input_slacks)
    istep, ikind = np.unravel_index(iflat, input_slacks.shape)
    sol.min_input_margin = float(input_slacks[istep, ikind])
    sol.min_input_margin_step = int(istep)  # 0-indexed control step
    sol.min_input_margin_kind = ikinds[ikind]

    return sol, handles


_PROBLEM_CACHE: dict = {}


def _mpc_cfg_key(dt: float, mpc_cfg: MPCConfig) -> tuple:
    return (
        round(dt, 12), mpc_cfg.horizon,
        round(mpc_cfg.q_pos, 12), round(mpc_cfg.q_vel, 12),
        round(mpc_cfg.r_u, 12), round(mpc_cfg.r_du, 12),
        round(mpc_cfg.q_pos_terminal, 12), round(mpc_cfg.q_vel_terminal, 12),
        tuple(round(b, 12) for b in mpc_cfg.pos_bounds),
        tuple(round(b, 12) for b in mpc_cfg.vel_bounds),
        tuple(round(b, 12) for b in mpc_cfg.input_bounds),
    )


def _get_or_build_problem(dt: float, mpc_cfg: MPCConfig) -> dict:
    """Return cached CVXPY problem handles for this (dt, mpc_cfg), building
    and compiling them the first time this exact configuration is seen."""
    key = _mpc_cfg_key(dt, mpc_cfg)
    if key in _PROBLEM_CACHE:
        return _PROBLEM_CACHE[key]

    N = mpc_cfg.horizon
    A, B = get_double_integrator_matrices(dt)

    X = cp.Variable((2, N + 1))
    U = cp.Variable(N)

    x0_param = cp.Parameter(2)
    pref_param = cp.Parameter()
    vref_param = cp.Parameter()
    bias_param = cp.Parameter(2)

    pmin, pmax = mpc_cfg.pos_bounds
    vmin, vmax = mpc_cfg.vel_bounds
    umin, umax = mpc_cfg.input_bounds

    constraints = [X[:, 0] == x0_param]
    for k in range(N):
        constraints.append(X[:, k + 1] == A @ X[:, k] + B.flatten() * U[k] + bias_param)

    pos_upper_cons, pos_lower_cons = [], []
    vel_upper_cons, vel_lower_cons = [], []
    for k in range(1, N + 1):
        cu = X[0, k] <= pmax
        cl = X[0, k] >= pmin
        vu = X[1, k] <= vmax
        vl = X[1, k] >= vmin
        constraints += [cu, cl, vu, vl]
        pos_upper_cons.append(cu)
        pos_lower_cons.append(cl)
        vel_upper_cons.append(vu)
        vel_lower_cons.append(vl)

    u_upper_cons, u_lower_cons = [], []
    for k in range(N):
        cu = U[k] <= umax
        cl = U[k] >= umin
        constraints += [cu, cl]
        u_upper_cons.append(cu)
        u_lower_cons.append(cl)

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
    assert problem.is_dcp(dpp=True), "MPC problem is not DPP-compliant; check parameter usage"

    handles = dict(
        problem=problem, X=X, U=U,
        x0_param=x0_param, pref_param=pref_param, vref_param=vref_param, bias_param=bias_param,
        pos_upper_cons=pos_upper_cons, pos_lower_cons=pos_lower_cons,
        vel_upper_cons=vel_upper_cons, vel_lower_cons=vel_lower_cons,
        u_upper_cons=u_upper_cons, u_lower_cons=u_lower_cons,
    )
    _PROBLEM_CACHE[key] = handles
    return handles


def solve_oracle_mpc(x0, p_ref, v_ref, dt, mpc_cfg, solver_preference=("OSQP", "CLARABEL", "SCS")):
    sol, _ = _solve_qp(x0, p_ref, v_ref, dt, mpc_cfg, bias=None, solver_preference=solver_preference)
    return sol


def solve_perturbed_mpc(x0, p_ref, v_ref, dt, mpc_cfg, bias,
                         solver_preference=("OSQP", "CLARABEL", "SCS")):
    sol, _ = _solve_qp(x0, p_ref, v_ref, dt, mpc_cfg, bias=bias, solver_preference=solver_preference)
    return sol


def compute_active_set(sol: MPCSolution, tol: ToleranceConfig) -> tuple[np.ndarray, np.ndarray]:
    """Boolean active-set indicators requiring BOTH small slack AND a
    meaningful nonnegative dual value. Returns (state_active (N,4), input_active (N,2))."""
    if not sol.solve_success:
        return None, None

    def active(slack, dual):
        return (slack < tol.active_slack_tol) & (dual > tol.active_dual_tol)

    state_active = np.stack([
        active(sol.pos_upper_slack, sol.pos_upper_dual),
        active(sol.pos_lower_slack, sol.pos_lower_dual),
        active(sol.vel_upper_slack, sol.vel_upper_dual),
        active(sol.vel_lower_slack, sol.vel_lower_dual),
    ], axis=1)

    input_active = np.stack([
        active(sol.u_upper_slack, sol.u_upper_dual),
        active(sol.u_lower_slack, sol.u_lower_dual),
    ], axis=1)

    sol.state_active = state_active
    sol.input_active = input_active
    return state_active, input_active


def normalized_duals(sol: MPCSolution, mpc_cfg: MPCConfig) -> dict:
    """Scale raw duals by the physical range of the corresponding bound so
    that position-, velocity- and input-constraint duals become comparable.
    Raw duals are preserved separately on the MPCSolution object."""
    pos_range = mpc_cfg.pos_bounds[1] - mpc_cfg.pos_bounds[0]
    vel_range = mpc_cfg.vel_bounds[1] - mpc_cfg.vel_bounds[0]
    u_range = mpc_cfg.input_bounds[1] - mpc_cfg.input_bounds[0]
    return dict(
        pos_upper_dual_norm=sol.pos_upper_dual * pos_range,
        pos_lower_dual_norm=sol.pos_lower_dual * pos_range,
        vel_upper_dual_norm=sol.vel_upper_dual * vel_range,
        vel_lower_dual_norm=sol.vel_lower_dual * vel_range,
        u_upper_dual_norm=sol.u_upper_dual * u_range,
        u_lower_dual_norm=sol.u_lower_dual * u_range,
    )
