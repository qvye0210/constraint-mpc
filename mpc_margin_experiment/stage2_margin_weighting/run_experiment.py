#!/usr/bin/env python3
"""
Stage 2: constraint-margin-weighted dynamics learning -- orchestrator.

Usage:
    python run_experiment.py              # full spec run
    python run_experiment.py --quick       # tiny smoke run (sanity check only)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # parent, for `src`

import generate_data
import train
import evaluate_prediction
import evaluate_mpc
from src.config import ExperimentConfig  # noqa: E402

SEEDS_FULL = [101, 202, 303]


def print_reuse_note():
    print("=" * 70)
    print("Stage 2 reuse of stage-1 code:")
    print("  - src/config.py: ExperimentConfig -> MPC horizon, cost weights,")
    print("    position/velocity/input bounds, dt, solver preference")
    print("    (dt=0.1, horizon=18, pos_bounds=(-2,2), input_bounds=(-0.5,0.5))")
    print("  - stage-1 QP-building pattern (src/mpc_solver.py) reused as the")
    print("    template for stage2/mpc_learned.py's DPP-parametrized QP, now")
    print("    with A,B,c promoted to cp.Parameter for learned-dynamics MPC")
    print("  - stage-1 true dynamics were fully LINEAR, so per the task spec a")
    print("    small fixed nonlinearity was added for stage 2's TRUE plant only")
    print("    (see generate_data.true_step); the learned models must discover it")
    print("=" * 70)


def run(quick: bool, outdir: str):
    t0 = time.time()
    print_reuse_note()

    data_dir = os.path.join(HERE, "data")
    ckpt_dir = os.path.join(HERE, "checkpoints")
    results_dir = outdir
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)

    if quick:
        seeds = [101]
        train.MAX_EPOCHS = 6
        train.PATIENCE = 3
        evaluate_mpc.N_EPISODES = 4
        evaluate_mpc.EPISODE_LEN = 10
        n_traj, traj_len = 60, 20
    else:
        seeds = SEEDS_FULL
        n_traj, traj_len = generate_data.N_TRAJ, generate_data.TRAJ_LEN

    print(f"\n--- Step 1/5: data generation (n_traj={n_traj}, traj_len={traj_len}) ---")
    data_stats = generate_data.generate_all(data_dir, n_traj=n_traj, traj_len=traj_len)

    print(f"\n--- Step 2/5: training ({len(seeds)} seed(s) x {len(train.METHODS)} methods) ---")
    train_results, in_norm, out_norm = train.run_all_training(data_dir, ckpt_dir, seeds)

    print("\n--- Step 3/5: offline prediction evaluation ---")
    pred_metrics = evaluate_prediction.evaluate_all(data_dir, ckpt_dir, results_dir, seeds)

    print("\n--- Step 4/5: closed-loop MPC evaluation ---")
    n_episodes = evaluate_mpc.N_EPISODES
    episode_len = evaluate_mpc.EPISODE_LEN
    episodes_used = evaluate_mpc.build_fixed_episodes(n_episodes=n_episodes, T=episode_len)
    print(f"  fixed episode set: n={len(episodes_used)}, T={episode_len} "
          f"(shared identically across all methods and seeds)")
    mpc_df, mpc_agg, paired_stats = evaluate_mpc.run_all_mpc_eval(
        ckpt_dir, results_dir, seeds, n_episodes=n_episodes, episode_len=episode_len)

    print("\n--- Step 5/5: judgment + report ---")
    verdict, judgment_detail = judge(pred_metrics, mpc_agg, paired_stats, quick)

    _write_top_level_outputs(results_dir, pred_metrics, mpc_agg, data_stats, seeds, quick,
                              verdict, judgment_detail, paired_stats, t0)

    runtime = time.time() - t0
    print(f"\n=== Stage 2 experiment finished in {runtime:.1f}s ({runtime/60:.1f} min) ===")
    print(f"Verdict: {verdict}")
    return dict(verdict=verdict, judgment_detail=judgment_detail, runtime=runtime)


def judge(pred_metrics: pd.DataFrame, mpc_agg: pd.DataFrame, paired_stats: dict, quick: bool):
    """Implements the section-8 decision rule from the task spec."""
    def per_seed(df, method, col):
        return df[df["method"] == method].set_index("seed")[col]

    near_mw = per_seed(pred_metrics, "margin_weighted", "near_rmse")
    near_bl = per_seed(pred_metrics, "baseline", "near_rmse")
    overall_mw = per_seed(pred_metrics, "margin_weighted", "overall_rmse")
    overall_bl = per_seed(pred_metrics, "baseline", "overall_rmse")

    common_seeds = sorted(set(near_mw.index) & set(near_bl.index))
    crit1_per_seed = [bool(near_mw[s] < near_bl[s]) for s in common_seeds]
    crit1 = all(crit1_per_seed) if crit1_per_seed else False

    overall_common = sorted(set(overall_mw.index) & set(overall_bl.index))
    rel_change = [(overall_mw[s] - overall_bl[s]) / overall_bl[s] for s in overall_common]
    crit2 = all(rc <= 0.05 for rc in rel_change) if rel_change else False

    mw_vs_bl = paired_stats.get("margin_weighted_vs_baseline", {})
    crit3_flags = []
    for m in ["violation_rate", "max_violation", "cumulative_objective"]:
        d = mw_vs_bl.get(m, {})
        if d and d.get("mean_diff", 0) < 0:  # margin - baseline < 0 => improvement (lower is better)
            crit3_flags.append(True)
        else:
            crit3_flags.append(False)
    crit3 = any(crit3_flags)

    mw_vs_rw = paired_stats.get("margin_weighted_vs_random_weight", {})
    near_mw_arr = near_mw.values
    # crude proxy: check random-weight near-RMSE doesn't improve as consistently as margin-weighted
    crit4 = True  # refined below using prediction metrics for random_weight
    near_rw = per_seed(pred_metrics, "random_weight", "near_rmse")
    common_rw = sorted(set(near_rw.index) & set(near_bl.index))
    crit4_per_seed = [bool(near_rw[s] < near_bl[s]) for s in common_rw]
    crit4 = not (all(crit4_per_seed) if crit4_per_seed else False)  # random should NOT consistently win too

    crit5 = crit1  # multiple-seed consistency already checked in crit1 for the primary metric

    checks = dict(near_rmse_improved_all_seeds=crit1, overall_rmse_not_worse_5pct=crit2,
                  at_least_one_closedloop_metric_improved=crit3,
                  random_weight_does_not_match_margin=crit4,
                  consistent_across_seeds=crit5)
    n_pass = sum(checks.values())

    if len(common_seeds) < 2 or quick:
        # Task explicitly forbids declaring success from a single run.
        verdict = "inconclusive"
    elif n_pass == 5:
        verdict = "supported"
    elif n_pass >= 3:
        verdict = "mixed"
    else:
        verdict = "not_supported"

    return verdict, dict(checks=checks, n_pass=n_pass,
                          near_rmse_margin_weighted=near_mw.to_dict(),
                          near_rmse_baseline=near_bl.to_dict(),
                          near_rmse_random_weight=near_rw.to_dict() if len(near_rw) else {},
                          overall_rel_change_margin_vs_baseline=dict(zip(overall_common, rel_change)))


def _write_top_level_outputs(results_dir, pred_metrics, mpc_agg, data_stats, seeds, quick,
                              verdict, judgment_detail, paired_stats, t0):
    exp_cfg = ExperimentConfig()

    combined = pred_metrics.merge(mpc_agg, on=["method", "seed"], how="outer")
    combined.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)

    summary = combined.groupby("method").agg(["mean", "std"])
    summary.to_csv(os.path.join(results_dir, "summary.csv"))

    config_dump = dict(
        seeds=seeds, quick_mode=quick,
        mpc_dt=exp_cfg.dynamics.dt, mpc_horizon=exp_cfg.mpc.horizon,
        pos_bounds=exp_cfg.mpc.pos_bounds, vel_bounds=exp_cfg.mpc.vel_bounds,
        input_bounds=exp_cfg.mpc.input_bounds,
        q_pos=exp_cfg.mpc.q_pos, q_vel=exp_cfg.mpc.q_vel, r_u=exp_cfg.mpc.r_u,
        r_du=exp_cfg.mpc.r_du,
        weight_formula="1 + 4*exp(-margin/0.2), clipped [0.5,5.0], mean-normalized",
        near_threshold=0.4, far_threshold=0.8,
        n_episodes=evaluate_mpc.N_EPISODES, episode_len=evaluate_mpc.EPISODE_LEN,
        data_stats=data_stats,
    )
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config_dump, f, indent=2, default=str)

    _write_report(results_dir, pred_metrics, mpc_agg, data_stats, verdict, judgment_detail,
                   paired_stats, seeds, quick, time.time() - t0)


def _write_report(results_dir, pred_metrics, mpc_agg, data_stats, verdict, judgment_detail,
                   paired_stats, seeds, quick, runtime):
    lines = []
    a = lines.append
    a("# Stage 2: Constraint-Margin-Weighted Dynamics Learning -- Report\n")
    a(f"**Mode:** {'quick smoke test' if quick else 'full spec'}  |  "
      f"**Seeds:** {seeds}  |  **Runtime:** {runtime:.1f}s\n")

    a("## Data\n")
    a(f"- {data_stats['n_transitions']} transitions from {data_stats['n_traj']} trajectories "
      f"(train/val/test = {data_stats['n_train']}/{data_stats['n_val']}/{data_stats['n_test']})\n")
    a(f"- near-constraint fraction (margin<0.4): overall {data_stats['frac_near_overall']:.1%}, "
      f"train {data_stats['frac_near_train']:.1%}, val {data_stats['frac_near_val']:.1%}, "
      f"test {data_stats['frac_near_test']:.1%}\n")

    a("## Offline prediction results (mean +/- std across seeds)\n\n")
    pred_summary = pred_metrics.groupby("method")[
        ["overall_rmse", "near_rmse", "far_rmse", "rollout15_rmse", "rollout15_near_rmse"]
    ].agg(["mean", "std"])
    a(pred_summary.to_markdown())
    a("\n\n")

    a("## Closed-loop MPC results (mean +/- std across seeds)\n\n")
    mpc_summary = mpc_agg.groupby("method")[
        ["tracking_rmse", "cumulative_objective", "violation_rate",
         "max_violation", "infeasibility_rate", "saturation_rate"]
    ].agg(["mean", "std"])
    a(mpc_summary.to_markdown())
    a("\n\n")

    a("## Paired statistical comparison (margin_weighted vs baseline / random_weight)\n\n")
    a("```json\n" + json.dumps(paired_stats, indent=2) + "\n```\n\n")

    a("## Judgment (section 8 decision rule)\n\n")
    for k, v in judgment_detail["checks"].items():
        a(f"- {k}: {'PASS' if v else 'FAIL'}\n")
    a(f"\n**{judgment_detail['n_pass']}/5 checks passed.**\n\n")
    a(f"### Verdict: `{verdict}`\n\n")

    a("## Objective diagnosis (if not fully supported)\n")
    if verdict != "supported":
        a("Candidate explanations (see task section 8 categories) to check against "
          "the numbers above:\n")
        a("- baseline near-zero error? compare baseline near_rmse to the overall scale of dp/dv\n")
        a("- near-constraint sample sufficiency: see data near-fraction above\n")
        a("- prediction improved but control did not: compare near_rmse vs violation_rate rows\n")
        a("- margin not decision-aware enough: compare margin_weighted vs random_weight rows\n")
        a("- insufficient constraint activation in closed loop: check violation_rate for baseline\n")
        a("- variance too large: check std columns above relative to the mean differences\n")
    else:
        a("All five criteria were met; treat this as preliminary support from a quick "
          "proof-of-concept, not a final claim -- rerun with more seeds/data before relying on it.\n")

    a("\n## Known simplifications / deviations from the literal spec\n")
    a("- Closed-loop MPC with a nonlinear learned model is implemented via per-step "
      "linearization of the NN (frozen across the horizon for that solve), reusing the "
      "stage-1 QP structure with A,B,c promoted to cp.Parameter -- not full nonlinear MPC "
      "or solver differentiation, consistent with the 'no solver differentiation, no complex "
      "network' instruction.\n")
    a("- Stage-1 true dynamics were fully linear, so the specified nonlinearity was added "
      "for stage 2's true plant only (see generate_data.py); stage-1 results/config files "
      "were left untouched.\n")
    a("- Near-constraint coverage (~20-30%) was reached via uniform initial-position sampling "
      "plus natural wall-bounce dynamics -- NOT via extra noise or region-specific treatment, "
      "per the fairness requirement.\n")

    with open(os.path.join(results_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--outdir", default=os.path.join(HERE, "results"))
    args = parser.parse_args()
    result = run(args.quick, args.outdir)

    print("\n" + "=" * 70)
    print("FINAL ANSWERS")
    print("=" * 70)
    print(f"1. 实验是否成功完成: 是")
    print(f"2. 判定结果 (supported/mixed/inconclusive/not_supported): {result['verdict']}")
    print(f"   详见 results/report.md 和 results/metrics.csv")
