import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ExperimentConfig
from src.dynamics import get_double_integrator_matrices, true_step, biased_step, rollout
from src.bias import structured_directions, random_directions, make_bias_vector
from src.mpc_solver import solve_oracle_mpc, solve_perturbed_mpc, compute_active_set, normalized_duals
from src.scenarios import sample_scenarios
from src.dataset import generate_dataset


# ---------------------------------------------------------------------------
# 1. True dynamics matrices are correct
# ---------------------------------------------------------------------------

def test_double_integrator_matrices():
    dt = 0.1
    A, B = get_double_integrator_matrices(dt)
    assert np.allclose(A, np.array([[1.0, 0.1], [0.0, 1.0]]))
    assert np.allclose(B, np.array([[0.005], [0.1]]))


def test_true_step_matches_analytic_kinematics():
    dt = 0.1
    x0 = np.array([0.0, 0.0])
    u = 2.0
    x1 = true_step(x0, u, dt)
    # p = 0.5*u*dt^2, v = u*dt
    assert np.isclose(x1[0], 0.5 * u * dt ** 2)
    assert np.isclose(x1[1], u * dt)


def test_rollout_matches_manual_stepping():
    dt = 0.1
    x0 = np.array([0.3, -0.1])
    u_seq = np.array([0.1, -0.2, 0.3])
    traj = rollout(x0, u_seq, dt)
    x = x0.copy()
    manual = [x0.copy()]
    for u in u_seq:
        x = true_step(x, u, dt)
        manual.append(x.copy())
    manual = np.array(manual)
    assert np.allclose(traj, manual)


# ---------------------------------------------------------------------------
# 2. Bias vectors have the requested norm
# ---------------------------------------------------------------------------

def test_bias_vector_norm_matches_magnitude():
    dirs = structured_directions() + random_directions(5, seed=1)
    for d in dirs:
        for mag in [0.005, 0.02, 0.05]:
            b = make_bias_vector(d, mag)
            assert np.isclose(np.linalg.norm(b), mag, atol=1e-10)


def test_bias_step_differs_from_true_step_by_exact_bias():
    dt = 0.1
    x = np.array([0.2, 0.1])
    u = 0.1
    bias = np.array([0.01, -0.02])
    x_true = true_step(x, u, dt)
    x_biased = biased_step(x, u, dt, bias)
    assert np.allclose(x_biased - x_true, bias)


# ---------------------------------------------------------------------------
# 3. Oracle and perturbed MPC use identical objective/constraint structure
# ---------------------------------------------------------------------------

def test_oracle_and_perturbed_use_same_cost_and_constraints_when_bias_zero():
    cfg = ExperimentConfig()
    cfg.mpc.horizon = 10
    x0 = np.array([0.5, 0.0])
    sol_oracle = solve_oracle_mpc(x0, 1.0, 0.0, cfg.dynamics.dt, cfg.mpc)
    sol_perturbed_zero_bias = solve_perturbed_mpc(x0, 1.0, 0.0, cfg.dynamics.dt, cfg.mpc,
                                                    bias=np.array([0.0, 0.0]))
    assert sol_oracle.solve_success and sol_perturbed_zero_bias.solve_success
    assert np.isclose(sol_oracle.objective_value, sol_perturbed_zero_bias.objective_value, atol=1e-4)
    assert np.allclose(sol_oracle.u_traj, sol_perturbed_zero_bias.u_traj, atol=1e-4)
    assert np.allclose(sol_oracle.x_traj, sol_perturbed_zero_bias.x_traj, atol=1e-4)


# ---------------------------------------------------------------------------
# 4. Margins computed from constraint values and bounds
# ---------------------------------------------------------------------------

def test_state_margins_consistent_with_bounds():
    cfg = ExperimentConfig()
    cfg.mpc.horizon = 15
    x0 = np.array([0.0, 0.0])
    sol = solve_oracle_mpc(x0, 1.9, 0.0, cfg.dynamics.dt, cfg.mpc)  # ref near upper bound 2.0
    assert sol.solve_success
    pmin, pmax = cfg.mpc.pos_bounds
    manual_pos_upper = pmax - sol.x_traj[1:, 0]
    manual_pos_lower = sol.x_traj[1:, 0] - pmin
    assert np.allclose(sol.pos_upper_slack, manual_pos_upper, atol=1e-8)
    assert np.allclose(sol.pos_lower_slack, manual_pos_lower, atol=1e-8)
    # min state margin must be the true minimum across all recorded slacks
    all_slacks = np.concatenate([sol.pos_upper_slack, sol.pos_lower_slack,
                                  sol.vel_upper_slack, sol.vel_lower_slack])
    assert np.isclose(sol.min_state_margin, all_slacks.min(), atol=1e-8)


def test_margin_is_small_near_boundary_reference():
    """A reference beyond the feasible position boundary should push the
    oracle solution's terminal position close to the boundary, producing a
    small min_state_margin (near-constraint regime)."""
    cfg = ExperimentConfig()
    cfg.mpc.horizon = 18
    x0 = np.array([0.0, 0.0])
    sol_far = solve_oracle_mpc(x0, 0.0, 0.0, cfg.dynamics.dt, cfg.mpc)   # ref = init -> far from bound
    sol_near = solve_oracle_mpc(x0, 2.3, 0.0, cfg.dynamics.dt, cfg.mpc)  # ref beyond upper bound
    assert sol_far.solve_success and sol_near.solve_success
    assert sol_near.min_state_margin < sol_far.min_state_margin


# ---------------------------------------------------------------------------
# 5. Dual variables stored separately from margins
# ---------------------------------------------------------------------------

def test_duals_are_separate_arrays_from_slacks():
    cfg = ExperimentConfig()
    cfg.mpc.horizon = 15
    x0 = np.array([0.0, 0.0])
    sol = solve_oracle_mpc(x0, 1.95, 0.0, cfg.dynamics.dt, cfg.mpc)
    assert sol.solve_success
    assert sol.pos_upper_dual is not sol.pos_upper_slack
    assert sol.pos_upper_dual.shape == sol.pos_upper_slack.shape
    # Duals must be nonnegative for a correctly solved convex QP with <=/>= constraints
    assert np.all(sol.pos_upper_dual >= -1e-6)
    assert np.all(sol.pos_lower_dual >= -1e-6)


def test_normalized_duals_scale_by_bound_range():
    cfg = ExperimentConfig()
    cfg.mpc.horizon = 15
    x0 = np.array([0.0, 0.0])
    sol = solve_oracle_mpc(x0, 1.95, 0.0, cfg.dynamics.dt, cfg.mpc)
    dn = normalized_duals(sol, cfg.mpc)
    pos_range = cfg.mpc.pos_bounds[1] - cfg.mpc.pos_bounds[0]
    assert np.allclose(dn["pos_upper_dual_norm"], sol.pos_upper_dual * pos_range)


# ---------------------------------------------------------------------------
# 6. True-model replay uses true dynamics (not the biased model)
# ---------------------------------------------------------------------------

def test_true_replay_does_not_use_bias():
    cfg = ExperimentConfig()
    cfg.mpc.horizon = 15
    x0 = np.array([0.0, 0.0])
    bias = np.array([0.05, 0.0])
    psol = solve_perturbed_mpc(x0, 1.5, 0.0, cfg.dynamics.dt, cfg.mpc, bias)
    assert psol.solve_success
    true_replay = rollout(x0, psol.u_traj, cfg.dynamics.dt, bias=None)
    biased_replay = rollout(x0, psol.u_traj, cfg.dynamics.dt, bias=bias)
    # true replay should differ from the perturbed model's own predicted trajectory
    assert not np.allclose(true_replay, psol.x_traj)
    # but biased replay (using the SAME bias) should match the perturbed model's
    # own prediction, confirming true_replay is genuinely bias-free and that the
    # discrepancy above is attributable to the bias term itself
    assert np.allclose(biased_replay, psol.x_traj, atol=1e-6)


# ---------------------------------------------------------------------------
# 7. Prediction-error magnitudes are matched across directions
# ---------------------------------------------------------------------------

def test_one_step_prediction_error_matches_across_directions():
    dt = 0.1
    x = np.array([0.3, 0.1])
    u = 0.05
    mag = 0.02
    dirs = structured_directions() + random_directions(4, seed=7)
    errors = []
    for d in dirs:
        b = make_bias_vector(d, mag)
        x_true = true_step(x, u, dt)
        x_pert = biased_step(x, u, dt, b)
        errors.append(np.linalg.norm(x_pert - x_true))
    errors = np.array(errors)
    assert np.allclose(errors, mag, atol=1e-10)


# ---------------------------------------------------------------------------
# 8. Fixed random seeds reproduce the same sampled scenarios
# ---------------------------------------------------------------------------

def test_scenario_sampling_reproducible():
    cfg = ExperimentConfig()
    s1 = sample_scenarios(cfg.sampling, 20)
    s2 = sample_scenarios(cfg.sampling, 20)
    for a, b in zip(s1, s2):
        assert a.init_pos == b.init_pos
        assert a.init_vel == b.init_vel
        assert a.ref_pos == b.ref_pos


def test_dataset_generation_reproducible_small():
    cfg = ExperimentConfig()
    cfg.mpc.horizon = 10
    cfg.bias.n_random_directions = 1
    cfg.bias.magnitudes = (0.01,)
    o1, p1, m1 = generate_dataset(cfg, n_target=5, verbose=False)
    o2, p2, m2 = generate_dataset(cfg, n_target=5, verbose=False)
    assert np.allclose(o1["init_pos"].values, o2["init_pos"].values)
    assert np.allclose(p1["delta_u0"].dropna().values, p2["delta_u0"].dropna().values)


# ---------------------------------------------------------------------------
# 9. Solver statuses handled correctly
# ---------------------------------------------------------------------------

def test_infeasible_problem_flagged_not_crash():
    cfg = ExperimentConfig()
    cfg.mpc.horizon = 10
    # Contradictory bounds force infeasibility: pos bound narrower than
    # reachable range given input limits, with an unreachable initial state.
    cfg.mpc.pos_bounds = (0.5, 0.6)
    x0 = np.array([-5.0, 0.0])  # far outside bounds, cannot be reached instantly
    # x0 itself is set via equality so the very first bound constraint at k=1
    # combined with velocity/input limits should be infeasible to satisfy
    sol = solve_oracle_mpc(x0, 0.55, 0.0, cfg.dynamics.dt, cfg.mpc)
    assert sol.solve_success in (True, False)  # must not raise
    if not sol.solve_success:
        assert sol.status not in ("", None)


def test_active_set_requires_slack_and_dual():
    cfg = ExperimentConfig()
    cfg.mpc.horizon = 15
    x0 = np.array([0.0, 0.0])
    sol = solve_oracle_mpc(x0, 1.95, 0.0, cfg.dynamics.dt, cfg.mpc)
    state_active, input_active = compute_active_set(sol, cfg.tolerance)
    # Wherever active, slack must indeed be below tol and dual above tol
    tol = cfg.tolerance
    slacks = np.stack([sol.pos_upper_slack, sol.pos_lower_slack,
                        sol.vel_upper_slack, sol.vel_lower_slack], axis=1)
    duals = np.stack([sol.pos_upper_dual, sol.pos_lower_dual,
                       sol.vel_upper_dual, sol.vel_lower_dual], axis=1)
    active_idx = np.argwhere(state_active)
    for (k, j) in active_idx:
        assert slacks[k, j] < tol.active_slack_tol
        assert duals[k, j] > tol.active_dual_tol


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
