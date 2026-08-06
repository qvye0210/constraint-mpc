"""
Dataset generation pipeline.

For each feasible oracle scenario, solves the perturbed MPC under every
(bias direction, bias magnitude) combination and records downstream
decision-impact labels, always evaluating the perturbed control sequence
under the TRUE dynamics (never the biased model) for the true-model
outcomes.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import List

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .scenarios import Scenario, sample_scenarios
from .mpc_solver import (solve_oracle_mpc, solve_perturbed_mpc,
                          compute_active_set, normalized_duals, MPCSolution)
from .dynamics import rollout
from .bias import all_directions, make_bias_vector, BiasDirection


def _oracle_row(sid: int, sc: Scenario, sol: MPCSolution, state_active, input_active,
                 dual_norm: dict) -> dict:
    # Whether the oracle's FIRST control action u0 is at an active input bound
    # (corner solution). Such solutions can be locally invariant to small
    # model perturbations regardless of state-constraint proximity, which is
    # a potential confound for the near/far state-margin analysis and is
    # tracked explicitly so it can be checked for and reported on.
    u0_saturated = bool(input_active[0, 0] or input_active[0, 1]) if input_active is not None else False

    row = dict(
        scenario_id=sid,
        init_pos=sc.init_pos, init_vel=sc.init_vel,
        ref_pos=sc.ref_pos, ref_vel=sc.ref_vel,
        oracle_status=sol.status,
        oracle_u0=sol.u0,
        oracle_objective=sol.objective_value,
        oracle_min_state_margin=sol.min_state_margin,
        oracle_min_state_margin_step=sol.min_state_margin_step,
        oracle_min_state_margin_kind=sol.min_state_margin_kind,
        oracle_min_input_margin=sol.min_input_margin,
        oracle_min_input_margin_step=sol.min_input_margin_step,
        oracle_min_input_margin_kind=sol.min_input_margin_kind,
        oracle_n_active_state=int(state_active.sum()) if state_active is not None else np.nan,
        oracle_n_active_input=int(input_active.sum()) if input_active is not None else np.nan,
        oracle_x_traj=sol.x_traj.tolist(),
        oracle_u_traj=sol.u_traj.tolist(),
        oracle_state_active=state_active.tolist(),
        oracle_input_active=input_active.tolist(),
        oracle_pos_upper_dual_norm_max=float(np.max(dual_norm["pos_upper_dual_norm"])),
        oracle_pos_lower_dual_norm_max=float(np.max(dual_norm["pos_lower_dual_norm"])),
        oracle_vel_upper_dual_norm_max=float(np.max(dual_norm["vel_upper_dual_norm"])),
        oracle_vel_lower_dual_norm_max=float(np.max(dual_norm["vel_lower_dual_norm"])),
        oracle_u0_saturated=u0_saturated,
    )
    # aggregate normalized state dual (max across pos/vel upper/lower) as a single
    # scalar feature for the "dual information" analysis
    row["oracle_max_state_dual_norm"] = float(max(
        row["oracle_pos_upper_dual_norm_max"], row["oracle_pos_lower_dual_norm_max"],
        row["oracle_vel_upper_dual_norm_max"], row["oracle_vel_lower_dual_norm_max"]))
    return row


def generate_dataset(cfg: ExperimentConfig, n_target: int, verbose: bool = True):
    """Generate the full oracle + perturbed-pair dataset.

    Returns (oracle_df, pairs_df, meta) where meta records sampling stats.
    """
    t_start = time.time()
    sampling = cfg.sampling
    n_candidates = n_target * sampling.max_attempts_factor
    candidates = sample_scenarios(sampling, n_candidates)

    directions = all_directions(cfg.bias.n_random_directions, cfg.bias.random_direction_seed)

    oracle_rows = []
    pair_rows = []

    n_feasible = 0
    n_tried = 0
    infeasible_count = 0

    for sc in candidates:
        if n_feasible >= n_target:
            break
        n_tried += 1
        sol = solve_oracle_mpc(np.array([sc.init_pos, sc.init_vel]), sc.ref_pos, sc.ref_vel,
                                cfg.dynamics.dt, cfg.mpc, solver_preference=cfg.solver_preference)
        if not sol.solve_success:
            infeasible_count += 1
            continue

        state_active, input_active = compute_active_set(sol, cfg.tolerance)
        dual_norm = normalized_duals(sol, cfg.mpc)

        sid = n_feasible  # re-index feasible scenarios 0..n_target-1
        oracle_rows.append(_oracle_row(sid, sc, sol, state_active, input_active, dual_norm))

        # For every bias direction and magnitude, solve perturbed MPC
        for direction in directions:
            for mag in cfg.bias.magnitudes:
                bias_vec = make_bias_vector(direction, mag)
                psol = solve_perturbed_mpc(np.array([sc.init_pos, sc.init_vel]), sc.ref_pos,
                                            sc.ref_vel, cfg.dynamics.dt, cfg.mpc, bias_vec,
                                            solver_preference=cfg.solver_preference)

                row = dict(
                    scenario_id=sid,
                    bias_direction=direction.name,
                    bias_dir_p=float(direction.unit_vector[0] / np.linalg.norm(direction.unit_vector)),
                    bias_dir_v=float(direction.unit_vector[1] / np.linalg.norm(direction.unit_vector)),
                    bias_magnitude=mag,
                    perturbed_status=psol.status,
                    perturbed_feasible=bool(psol.solve_success),
                )

                if psol.solve_success:
                    p_state_active, p_input_active = compute_active_set(psol, cfg.tolerance)
                    p_dual_norm = normalized_duals(psol, cfg.mpc)

                    delta_u0 = float(abs(psol.u0 - sol.u0))

                    active_set_changed = bool(not np.array_equal(state_active, p_state_active))
                    n_active_changed = int(np.sum(state_active != p_state_active))

                    objective_diff = float(psol.objective_value - sol.objective_value)

                    # True-model open-loop evaluation of the PERTURBED control sequence
                    true_replay = rollout(np.array([sc.init_pos, sc.init_vel]), psol.u_traj,
                                           cfg.dynamics.dt, bias=None)

                    pmin, pmax = cfg.mpc.pos_bounds
                    vmin, vmax = cfg.mpc.vel_bounds
                    # margins of the true replay trajectory (k=1..N)
                    replay_pos = true_replay[1:, 0]
                    replay_vel = true_replay[1:, 1]
                    pos_margin = np.minimum(pmax - replay_pos, replay_pos - pmin)
                    vel_margin = np.minimum(vmax - replay_vel, replay_vel - vmin)
                    all_margin = np.minimum(pos_margin, vel_margin)
                    min_true_margin = float(np.min(all_margin))
                    max_violation = float(max(0.0, -min_true_margin))
                    true_violation = bool(max_violation > cfg.tolerance.active_slack_tol)

                    # true-model objective of perturbed control sequence: same cost
                    # structure evaluated on the true replay trajectory
                    true_cost = 0.0
                    for k in range(cfg.mpc.horizon):
                        true_cost += cfg.mpc.q_pos * (true_replay[k, 0] - sc.ref_pos) ** 2
                        true_cost += cfg.mpc.q_vel * (true_replay[k, 1] - sc.ref_vel) ** 2
                        true_cost += cfg.mpc.r_u * (psol.u_traj[k]) ** 2
                    true_cost += cfg.mpc.q_pos_terminal * (true_replay[-1, 0] - sc.ref_pos) ** 2
                    true_cost += cfg.mpc.q_vel_terminal * (true_replay[-1, 1] - sc.ref_vel) ** 2
                    if cfg.mpc.r_du > 0:
                        for k in range(1, cfg.mpc.horizon):
                            true_cost += cfg.mpc.r_du * (psol.u_traj[k] - psol.u_traj[k - 1]) ** 2

                    row.update(dict(
                        perturbed_u0=psol.u0,
                        delta_u0=delta_u0,
                        perturbed_objective=psol.objective_value,
                        objective_diff=objective_diff,
                        active_set_changed=active_set_changed,
                        n_active_state_changed=n_active_changed,
                        perturbed_min_state_margin=psol.min_state_margin,
                        true_replay_min_margin=min_true_margin,
                        true_replay_max_violation=max_violation,
                        true_replay_violated=true_violation,
                        true_replay_objective=float(true_cost),
                        true_vs_perturbed_objective_gap=float(true_cost - psol.objective_value),
                        perturbed_max_state_dual_norm=float(max(
                            np.max(p_dual_norm["pos_upper_dual_norm"]),
                            np.max(p_dual_norm["pos_lower_dual_norm"]),
                            np.max(p_dual_norm["vel_upper_dual_norm"]),
                            np.max(p_dual_norm["vel_lower_dual_norm"]))),
                    ))
                else:
                    # perturbed infeasible: fill downstream fields with NaN /flags
                    row.update(dict(
                        perturbed_u0=np.nan,
                        delta_u0=np.nan,
                        perturbed_objective=np.nan,
                        objective_diff=np.nan,
                        active_set_changed=True,  # infeasibility counts as a decision regime change
                        n_active_state_changed=np.nan,
                        perturbed_min_state_margin=np.nan,
                        true_replay_min_margin=np.nan,
                        true_replay_max_violation=np.nan,
                        true_replay_violated=np.nan,
                        true_replay_objective=np.nan,
                        true_vs_perturbed_objective_gap=np.nan,
                        perturbed_max_state_dual_norm=np.nan,
                    ))

                pair_rows.append(row)

        n_feasible += 1

    oracle_df = pd.DataFrame(oracle_rows)
    pairs_df = pd.DataFrame(pair_rows)

    meta = dict(
        n_target=n_target,
        n_feasible=n_feasible,
        n_tried=n_tried,
        n_candidates_available=n_candidates,
        oracle_infeasible_count=infeasible_count,
        oracle_feasibility_rate=(n_feasible / n_tried) if n_tried > 0 else float("nan"),
        n_directions=len(directions),
        n_magnitudes=len(cfg.bias.magnitudes),
        n_pairs=len(pair_rows),
        runtime_sec=time.time() - t_start,
    )
    if verbose:
        print(f"[dataset] feasible oracle scenarios: {n_feasible}/{n_tried} tried "
              f"(target {n_target}), infeasible={infeasible_count}, "
              f"pairs={len(pair_rows)}, runtime={meta['runtime_sec']:.1f}s")

    return oracle_df, pairs_df, meta
