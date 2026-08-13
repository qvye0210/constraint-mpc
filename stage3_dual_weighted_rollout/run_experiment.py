#!/usr/bin/env python3
"""
Stage 3: Dual-Weighted Constraint Rollout Learning -- orchestrator.

Usage:
    python run_experiment.py              # full spec run
    python run_experiment.py --quick       # tiny smoke run (sanity check only)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import generate_data
import tune_beta
import train
import evaluate_prediction
import evaluate_mpc
from config import Stage3Config
from src.config import ExperimentConfig  # noqa: E402


def print_reuse_note():
    print("=" * 70)
    print("Stage 3 reuse of existing code:")
    print("  - src/config.py + src/mpc_solver.py: the NOMINAL/NOISY data-")
    print("    collection MPC controller is stage 1's linear-model MPC,")
    print("    completely unmodified (same cost/bounds/horizon).")
    print("  - stage2_margin_weighting/generate_data.py: true_step (the true")
    print("    nonlinear plant) and POS_BOUND/INPUT_BOUND, reused unchanged.")
    print("  - stage2_margin_weighting/mpc_learned.py: the successive-")
    print("    linearization + DPP-QP pattern for closed-loop learned-")
    print("    dynamics MPC is reused as a self-contained copy in")
    print("    stage3/mpc_learned.py (adapted for the ResidualMLP class)")
    print("    since stage2 and stage3 each have their own models.py.")
    print("  - margin_weighted_mse reuses stage2's exact weight formula")
    print("    (1 + 4*exp(-margin/0.2), clipped [0.5,5], mean-normalized).")
    print("=" * 70)


def _archive_previous_run_if_present(outdir: str):
    report_path = os.path.join(outdir, "report.md")
    if not os.path.exists(report_path):
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(HERE, f"archive_{stamp}")
    os.makedirs(archive_dir, exist_ok=True)
    for name in ("data", "checkpoints", "results"):
        src = os.path.join(HERE, name)
        if os.path.isdir(src):
            shutil.move(src, os.path.join(archive_dir, name))
    print(f"[archive] previous run preserved at: {archive_dir}")
    return archive_dir


def run(quick: bool, outdir: str, skip_tuning: bool = False):
    t0 = time.time()
    print_reuse_note()

    if not quick:
        _archive_previous_run_if_present(outdir)

    cfg = Stage3Config()
    data_dir = os.path.join(HERE, "data")
    ckpt_dir = os.path.join(HERE, "checkpoints")
    results_dir = outdir
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)

    if quick:
        cfg.data.n_traj_nominal = 6
        cfg.data.n_traj_noisy = 6
        cfg.data.traj_len = 15
        cfg.train.max_epochs = 6
        cfg.train.patience = 3
        cfg.mpc_eval.n_episodes = 4
        cfg.mpc_eval.episode_len = 10
        skip_tuning = True

    cfg.to_json(os.path.join(results_dir, "config.json"))

    print(f"\n--- Step 1/6: data generation ---")
    data_stats = generate_data.generate_all(data_dir, cfg, quick=quick)

    beta_results = None
    if not skip_tuning:
        print(f"\n--- Step 2/6: beta tuning (validation-only, seed[0], "
              f"grid={cfg.train.beta_grid}) ---")
        beta_results = tune_beta.tune(data_dir, results_dir, cfg)
        # apply chosen beta for constraint_rollout / dual_weighted_constraint_rollout
        # (both share cfg.train.beta as currently structured; use the
        # dual-weighted method's chosen beta as the primary hypothesis test,
        # since that is the method under test)
        chosen = beta_results["dual_weighted_constraint_rollout"]["chosen_beta"]
        cfg.train.beta = chosen
        print(f"  chosen beta = {chosen} (applied to full training run)")
    else:
        print("\n--- Step 2/6: beta tuning SKIPPED "
              f"(quick mode or --skip-tuning; using beta={cfg.train.beta}) ---")

    print(f"\n--- Step 3/6: training ({len(cfg.seeds) if not quick else 1} seed(s) "
          f"x {len(train.METHODS)} methods) ---")
    train_results, in_norm, out_norm, lambda_bar = train.run_all_training(
        data_dir, ckpt_dir, cfg, quick=quick)

    seeds = list(cfg.seeds) if not quick else list(cfg.seeds[:1])

    print("\n--- Step 4/6: offline prediction evaluation ---")
    pred_metrics = evaluate_prediction.evaluate_all(data_dir, ckpt_dir, results_dir, cfg, seeds)

    print("\n--- Step 5/6: closed-loop MPC evaluation ---")
    mpc_df, mpc_agg, paired_stats = evaluate_mpc.run_all_mpc_eval(
        data_dir, ckpt_dir, results_dir, cfg, seeds,
        n_episodes=cfg.mpc_eval.n_episodes, episode_len=cfg.mpc_eval.episode_len)

    print("\n--- Step 6/6: judgment + report ---")
    verdict, judgment_detail = judge(pred_metrics, mpc_agg, paired_stats, quick, seeds)

    _write_top_level_outputs(results_dir, pred_metrics, mpc_agg, data_stats, seeds, quick,
                              verdict, judgment_detail, paired_stats, beta_results, cfg, t0)

    runtime = time.time() - t0
    print(f"\n=== Stage 3 experiment finished in {runtime:.1f}s ({runtime/60:.1f} min) ===")
    print(f"Verdict: {verdict}")
    return dict(verdict=verdict, judgment_detail=judgment_detail, runtime=runtime)


def judge(pred_metrics: pd.DataFrame, mpc_agg: pd.DataFrame, paired_stats: dict,
          quick: bool, seeds: list):
    FULL = "dual_weighted_constraint_rollout"

    def per_seed(df, method, col):
        return df[df["method"] == method].set_index("seed")[col]

    con_full = per_seed(pred_metrics, FULL, "constraint_rmse")
    con_base = per_seed(pred_metrics, "baseline_mse", "constraint_rmse")
    common = sorted(set(con_full.index) & set(con_base.index))
    crit_constraint_improved = ([bool(con_full[s] < con_base[s]) for s in common]
                                 if common else [])
    crit1 = all(crit_constraint_improved) if crit_constraint_improved else False

    track_full = per_seed(mpc_agg, FULL, "tracking_rmse")
    track_base = per_seed(mpc_agg, "baseline_mse", "tracking_rmse")
    common_t = sorted(set(track_full.index) & set(track_base.index))
    rel_track = [(track_full[s] - track_base[s]) / track_base[s] for s in common_t]
    crit2_tracking_ok = all(rc <= 0.1 for rc in rel_track) if rel_track else False

    obj_full = per_seed(mpc_agg, FULL, "cumulative_objective")
    obj_base = per_seed(mpc_agg, "baseline_mse", "cumulative_objective")
    common_o = sorted(set(obj_full.index) & set(obj_base.index))
    rel_obj = [(obj_full[s] - obj_base[s]) / obj_base[s] for s in common_o]
    crit2_objective_ok = all(rc <= 0.1 for rc in rel_obj) if rel_obj else False
    crit2 = crit2_tracking_ok and crit2_objective_ok

    key = f"{FULL}_vs_baseline_mse"
    d = paired_stats.get(key, {}).get("violation_rate", {})
    crit3_stat_sig_improved = bool(d and d.get("mean_diff", 0) < 0 and d.get("wilcoxon_p", 1) < 0.05)

    viol_full = per_seed(mpc_agg, FULL, "violation_rate")
    viol_base = per_seed(mpc_agg, "baseline_mse", "violation_rate")
    common_v = sorted(set(viol_full.index) & set(viol_base.index))
    crit_violation_per_seed = [bool(viol_full[s] <= viol_base[s]) for s in common_v]
    crit4_consistent = all(crit_violation_per_seed) if crit_violation_per_seed else False

    checks = dict(constraint_error_improved_all_seeds=crit1,
                  tracking_and_objective_not_worse_10pct=crit2,
                  violation_rate_significantly_improved=crit3_stat_sig_improved,
                  violation_improvement_consistent_across_seeds=crit4_consistent)
    n_pass = sum(checks.values())

    if len(seeds) < 2 or quick:
        verdict = "inconclusive"
    elif crit1 and not (crit3_stat_sig_improved and crit4_consistent):
        # the exact failure mode the task asked to flag explicitly
        verdict = "proxy_improved_no_control_benefit" if crit1 else "not_supported"
    elif n_pass == 4:
        verdict = "supported"
    elif n_pass >= 2:
        verdict = "mixed"
    else:
        verdict = "not_supported"

    return verdict, dict(checks=checks, n_pass=n_pass,
                          constraint_rmse_full=con_full.to_dict(),
                          constraint_rmse_baseline=con_base.to_dict(),
                          violation_rate_full=viol_full.to_dict(),
                          violation_rate_baseline=viol_base.to_dict())


def _write_top_level_outputs(results_dir, pred_metrics, mpc_agg, data_stats, seeds, quick,
                              verdict, judgment_detail, paired_stats, beta_results, cfg, t0):
    combined = pred_metrics.merge(mpc_agg, on=["method", "seed"], how="outer")
    combined.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    combined.groupby("method").agg(["mean", "std"]).to_csv(os.path.join(results_dir, "summary.csv"))

    _write_report(results_dir, pred_metrics, mpc_agg, data_stats, verdict, judgment_detail,
                   paired_stats, beta_results, seeds, quick, cfg, time.time() - t0)


def _write_report(results_dir, pred_metrics, mpc_agg, data_stats, verdict, judgment_detail,
                   paired_stats, beta_results, seeds, quick, cfg, runtime):
    lines = []
    a = lines.append
    a("# Stage 3: Dual-Weighted Constraint Rollout Learning -- Report\n")
    a(f"**Mode:** {'quick smoke test' if quick else 'full spec'}  |  **Seeds:** {seeds}  |  "
      f"**Runtime:** {runtime:.1f}s  |  **H:** {cfg.train.H}  |  **gamma:** {cfg.train.gamma}  |  "
      f"**beta:** {cfg.train.beta}\n")

    a("## Data\n")
    a(f"- {data_stats['n_transitions']} transitions from {data_stats['n_traj_total']} "
      f"trajectories ({data_stats['n_traj_nominal']} nominal-MPC + "
      f"{data_stats['n_traj_noisy']} noisy-MPC)\n")
    a(f"- near-constraint fraction (margin<0.4): {data_stats['frac_near_overall']:.1%}, "
      f"active fraction (margin<0, i.e. already past the bound): "
      f"{data_stats['frac_active_overall']:.1%}\n")
    a(f"- data-collection MPC infeasibility rate: {data_stats['mpc_infeasible_rate']:.2%}\n")

    if beta_results:
        a("\n## Beta tuning (validation-only, seed[0], small grid)\n\n")
        for method, res in beta_results.items():
            a(f"**{method}**: chosen beta = {res['chosen_beta']} "
              f"(baseline val one-step RMSE = {res['baseline_one_step_rmse']:.5f}, "
              f"guard = {res['guard']:.5f})\n\n")
            df = pd.DataFrame(res["rows"])
            a(df.to_markdown(index=False))
            a("\n\n")

    a("## Offline prediction results (mean +/- std across seeds)\n\n")
    cols = ["overall_one_step_rmse", "near_rmse", "rollout_rmse",
            "constraint_rmse", "active_constraint_rmse"]
    a(pred_metrics.groupby("method")[cols].agg(["mean", "std"]).to_markdown())
    a("\n\n")

    a("## Closed-loop MPC results (mean +/- std across seeds)\n\n")
    cols2 = ["tracking_rmse", "cumulative_objective", "violation_rate",
             "violation_frequency", "max_violation", "infeasibility_rate"]
    a(mpc_agg.groupby("method")[cols2].agg(["mean", "std"]).to_markdown())
    a("\n\n")

    a("## Paired statistical comparison (dual_weighted_constraint_rollout vs others)\n\n")
    a("```json\n" + json.dumps(paired_stats, indent=2) + "\n```\n\n")

    a("## Judgment\n\n")
    for k, v in judgment_detail["checks"].items():
        a(f"- {k}: {'PASS' if v else 'FAIL'}\n")
    a(f"\n**{judgment_detail['n_pass']}/4 checks passed.**\n\n")
    a(f"### Verdict: `{verdict}`\n\n")
    if verdict == "proxy_improved_no_control_benefit":
        a("**Explicit flag per task instructions**: the constraint-value prediction "
          "error (the proxy objective) improved consistently, but this did NOT "
          "translate into a statistically significant, seed-consistent reduction "
          "in closed-loop constraint violations. Report this as 'proxy objective "
          "improved, but not yet translated into control benefit' rather than as "
          "either full support or full rejection of the hypothesis.\n\n")

    a("## Known simplifications / deviations from the literal spec\n")
    a("- Closed-loop learned-dynamics MPC uses per-step linearization of the NN "
      "(frozen across the horizon for that solve), same successive-linearization "
      "simplification as stage 2 -- not full nonlinear MPC.\n")
    a("- `margin_weighted_mse` reuses stage 2's exact weighting formula and "
      "constants unchanged, applied to the current-step margin only (not "
      "horizon-indexed), consistent with 'use the existing margin-weighting "
      "setting'.\n")
    a("- All four methods are trained on the SAME set of H-step windows (same "
      "starting indices t, same batch order per seed) -- baseline_mse and "
      "margin_weighted_mse simply do not use the extra future-step columns in "
      "their loss. This means the last H timesteps of each trajectory are not "
      "usable as window starts for ANY method, including the two that don't "
      "strictly need the future data -- a deliberate trade-off to guarantee all "
      "four methods see identical training samples.\n")
    a("- Beta tuning followed the task's small-scale/validation-only/no-test-set "
      "rule; the SAME chosen beta was applied to both constraint_rollout and "
      "dual_weighted_constraint_rollout for simplicity (only the dual-weighted "
      "method's validation curve was used to choose it, since it is the primary "
      "method under test).\n")

    with open(os.path.join(results_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-tuning", action="store_true",
                         help="skip beta tuning and use the default beta from config.py")
    parser.add_argument("--outdir", default=os.path.join(HERE, "results"))
    args = parser.parse_args()
    result = run(args.quick, args.outdir, skip_tuning=args.skip_tuning)

    print("\n" + "=" * 70)
    print("FINAL ANSWERS")
    print("=" * 70)
    print(f"判定结果: {result['verdict']}")
    print(f"详见 results/report.md 和 results/metrics.csv")
