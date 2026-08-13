"""
Stage 3 data generation.

Collects closed-loop trajectories by running the EXISTING stage-1 MPC
controller (src/mpc_solver.py: solve_oracle_mpc, linear internal model,
same cost/bounds/horizon as stage 1 -- reused unchanged) on the EXISTING
stage-2 TRUE nonlinear plant (stage2_margin_weighting/generate_data.py:
true_step, reused unchanged). The linear-model MPC vs. nonlinear true plant
mismatch is itself a natural, realistic source of the near/far/active
constraint coverage we need -- no extra clipping or synthetic discontinuity
is introduced (lesson carried over from stage 2's post-mortem).

Two data-collection regimes, as specified:
  - "nominal": apply the MPC's own u0 directly.
  - "noisy":   apply u0 + Gaussian exploration noise (clipped to input
    bounds), to broaden state-space coverage beyond the MPC's own optimal
    manifold.

For every timestep we additionally record the MPC's OWN horizon-predicted
margin and (signed, binding-side) dual variable for the next H steps
(k=1..H) -- privileged information available at data-collection time only,
used later as the m_{t,k} / lambda_{t,k} terms in the dual-weighted
constraint-rollout loss. This is NOT the same as the true future margin
(which is computed from the true rollout at training time); it is the
MPC's own belief at time t about upcoming constraint proximity/activity.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGE3_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(STAGE3_DIR)
sys.path.insert(0, ROOT_DIR)

from src.config import ExperimentConfig  # noqa: E402
from src.mpc_solver import solve_oracle_mpc  # noqa: E402
from _stage2_bridge import true_step as true_step_stage2  # noqa: E402

from config import Stage3Config, DataConfig  # noqa: E402

MAX_H = 18  # stage-1 MPC horizon; H must be <= this


def true_step(p: float, v: float, u: float, dt: float) -> tuple[float, float]:
    return true_step_stage2(p, v, u, dt=dt)


def collect_trajectory(rng: np.random.Generator, exp_cfg: ExperimentConfig,
                        data_cfg: DataConfig, H: int, noisy: bool, traj_id: int,
                        source: str):
    p0 = rng.uniform(*data_cfg.init_pos_range)
    v0 = rng.uniform(*data_cfg.init_vel_range)
    p_ref = rng.uniform(*data_cfg.ref_pos_range)
    v_ref = 0.0

    p, v = p0, v0
    rows = []
    n_infeasible = 0
    for t in range(data_cfg.traj_len):
        sol = solve_oracle_mpc(np.array([p, v]), p_ref, v_ref, exp_cfg.dynamics.dt,
                                exp_cfg.mpc, solver_preference=exp_cfg.solver_preference)
        if sol.solve_success:
            u_mpc = sol.u0
            pos_upper_slack = sol.pos_upper_slack[:H]
            pos_lower_slack = sol.pos_lower_slack[:H]
            pos_upper_dual = sol.pos_upper_dual[:H]
            pos_lower_dual = sol.pos_lower_dual[:H]
            margin_k = np.minimum(pos_upper_slack, pos_lower_slack)
            upper_tighter = pos_upper_slack <= pos_lower_slack
            dual_k = np.where(upper_tighter, pos_upper_dual, pos_lower_dual)
        else:
            # BUG FIX (caught during smoke testing): filling margin/dual with
            # NaN on solve failure propagated NaN into exp(-margin/tau) in
            # the dual-weighted loss and produced NaN loss on the very first
            # batch that touched an infeasible-solve row. Use a neutral,
            # finite fallback instead: repeat the CURRENT (known-finite)
            # margin across the horizon (a "assume no better information"
            # default) and zero dual (no privileged urgency signal).
            n_infeasible += 1
            u_mpc = 0.0
            current_margin = exp_cfg.mpc.pos_bounds[1] - abs(p)
            margin_k = np.full(H, current_margin)
            dual_k = np.zeros(H)

        if noisy:
            u_applied = float(np.clip(u_mpc + rng.normal(0.0, data_cfg.action_noise_std),
                                       *exp_cfg.mpc.input_bounds))
        else:
            u_applied = float(np.clip(u_mpc, *exp_cfg.mpc.input_bounds))

        p_next, v_next = true_step(p, v, u_applied, exp_cfg.dynamics.dt)

        row = dict(trajectory_id=traj_id, t_local=t, source=source,
                   p_t=p, v_t=v, u_t=u_applied, p_next=p_next, v_next=v_next,
                   margin_t=exp_cfg.mpc.pos_bounds[1] - abs(p),
                   mpc_solve_success=bool(sol.solve_success))
        for k in range(H):
            row[f"mpc_margin_{k+1}"] = float(margin_k[k])
            row[f"mpc_dual_{k+1}"] = float(dual_k[k])
        rows.append(row)

        p, v = p_next, v_next

    return rows, n_infeasible


def generate_all(outdir: str, cfg: Stage3Config, quick: bool = False):
    os.makedirs(outdir, exist_ok=True)
    exp_cfg = ExperimentConfig()
    data_cfg = cfg.data
    H = cfg.train.H
    assert H <= MAX_H, f"H={H} must be <= stage-1 MPC horizon ({MAX_H})"

    rng = np.random.default_rng(data_cfg.seed)

    n_nom = data_cfg.n_traj_nominal
    n_noisy = data_cfg.n_traj_noisy
    traj_len = data_cfg.traj_len
    if quick:
        n_nom, n_noisy, traj_len = 8, 8, 15

    all_rows = []
    total_infeasible = 0
    traj_id = 0
    for i in range(n_nom):
        rows, ninf = collect_trajectory(rng, exp_cfg, data_cfg, H, noisy=False,
                                         traj_id=traj_id, source="nominal")
        all_rows.extend(rows)
        total_infeasible += ninf
        traj_id += 1
    for i in range(n_noisy):
        rows, ninf = collect_trajectory(rng, exp_cfg, data_cfg, H, noisy=True,
                                         traj_id=traj_id, source="noisy")
        all_rows.extend(rows)
        total_infeasible += ninf
        traj_id += 1

    df = pd.DataFrame(all_rows)
    n_traj_total = traj_id

    # Trajectory-level split
    traj_ids = np.arange(n_traj_total)
    rng.shuffle(traj_ids)
    n_train = int(data_cfg.train_frac * n_traj_total)
    n_val = int(data_cfg.val_frac * n_traj_total)
    train_ids = set(traj_ids[:n_train].tolist())
    val_ids = set(traj_ids[n_train:n_train + n_val].tolist())
    test_ids = set(traj_ids[n_train + n_val:].tolist())

    train_df = df[df["trajectory_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["trajectory_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["trajectory_id"].isin(test_ids)].reset_index(drop=True)

    train_df.to_csv(os.path.join(outdir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(outdir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(outdir, "test.csv"), index=False)

    def frac_near(d, thresh=0.4):
        return float((d["margin_t"] < thresh).mean())

    def frac_active(d, thresh=0.0):
        return float((d["margin_t"] < thresh).mean())

    stats = dict(
        seed=data_cfg.seed, n_traj_nominal=n_nom, n_traj_noisy=n_noisy,
        n_traj_total=n_traj_total, traj_len=traj_len, H=H,
        n_transitions=len(df), n_train=len(train_df), n_val=len(val_df), n_test=len(test_df),
        n_train_traj=len(train_ids), n_val_traj=len(val_ids), n_test_traj=len(test_ids),
        frac_near_overall=frac_near(df), frac_near_train=frac_near(train_df),
        frac_near_val=frac_near(val_df), frac_near_test=frac_near(test_df),
        frac_active_overall=frac_active(df),
        margin_min=float(df["margin_t"].min()), margin_max=float(df["margin_t"].max()),
        margin_mean=float(df["margin_t"].mean()),
        total_mpc_infeasible_steps=int(total_infeasible),
        mpc_infeasible_rate=float(total_infeasible / len(df)) if len(df) else float("nan"),
    )
    with open(os.path.join(outdir, "data_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df["margin_t"], bins=40, color="steelblue", edgecolor="white")
    axes[0].axvline(0.0, color="black", ls="-", label="boundary (active)")
    axes[0].axvline(0.4, color="red", ls="--", label="near threshold")
    axes[0].set_xlabel("margin_t = 2.0 - |p_t|")
    axes[0].set_title(f"Margin distribution (n={len(df)})")
    axes[0].legend(fontsize=8)
    for src_name, g in df.groupby("source"):
        axes[1].hist(g["margin_t"], bins=40, alpha=0.5, label=src_name, density=True)
    axes[1].set_xlabel("margin_t")
    axes[1].set_title("By data source")
    axes[1].legend(fontsize=8)
    fig.savefig(os.path.join(outdir, "margin_histogram.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[generate_data] n_transitions={len(df)} (train/val/test={len(train_df)}/"
          f"{len(val_df)}/{len(test_df)}), near-frac={stats['frac_near_overall']:.1%}, "
          f"active-frac(margin<0)={stats['frac_active_overall']:.1%}, "
          f"mpc_infeasible_rate={stats['mpc_infeasible_rate']:.2%}")
    return stats


if __name__ == "__main__":
    cfg = Stage3Config()
    generate_all(os.path.join(STAGE3_DIR, "data"), cfg)
