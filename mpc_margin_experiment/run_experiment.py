#!/usr/bin/env python3
"""
Run the near-vs-far constraint MPC decision-impact diagnostic experiment.

Usage:
    python run_experiment.py --mode smoke     # ~50 scenarios, fast sanity run
    python run_experiment.py --mode full       # full configured run
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.dataset import generate_dataset
from src.grouping import add_fixed_threshold_group, add_quantile_group, group_summary
from src.analysis import (near_far_comparison, rate_comparison,
                           spearman_margin_vs_impact, spearman_dual_vs_impact,
                           diagnostic_regression_comparison)
from src.plots import plot_all
from src.report import write_summary_report


def flatten_oracle_for_csv(oracle_df: pd.DataFrame) -> pd.DataFrame:
    """Drop/serialize array-valued columns for a lightweight scalar CSV;
    the full oracle_df (with array columns) is saved separately as Parquet/pickle."""
    df = oracle_df.copy()
    array_cols = ["oracle_x_traj", "oracle_u_traj", "oracle_state_active", "oracle_input_active"]
    for c in array_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda v: json.dumps(v))
    return df


def run(mode: str, outdir: str):
    t0 = time.time()
    cfg = ExperimentConfig()

    n_target = cfg.sampling.n_scenarios_smoke if mode == "smoke" else cfg.sampling.n_scenarios_full

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)

    print(f"=== Running experiment in '{mode}' mode, target={n_target} scenarios ===")
    oracle_df, pairs_df, meta = generate_dataset(cfg, n_target=n_target, verbose=True)

    if len(oracle_df) < 10:
        raise RuntimeError(f"Too few feasible oracle scenarios ({len(oracle_df)}); "
                            "check sampling ranges / MPC bounds configuration.")

    # Merge oracle scalar columns needed downstream onto pairs_df
    oracle_scalar_cols = [c for c in oracle_df.columns if c not in
                           ("oracle_x_traj", "oracle_u_traj", "oracle_state_active", "oracle_input_active")]
    pairs_df = pairs_df.merge(oracle_df[oracle_scalar_cols], on="scenario_id", how="left",
                               suffixes=("", "_dup"))

    # Grouping
    oracle_df = add_fixed_threshold_group(oracle_df, cfg.grouping)
    oracle_df = add_quantile_group(oracle_df, cfg.grouping)
    quantile_thresholds = dict(
        near_threshold=oracle_df.attrs.get("quantile_near_threshold"),
        far_threshold=oracle_df.attrs.get("quantile_far_threshold"),
    )
    group_map_fixed = oracle_df.set_index("scenario_id")["group_fixed"]
    group_map_quantile = oracle_df.set_index("scenario_id")["group_quantile"]
    pairs_df["group_fixed"] = pairs_df["scenario_id"].map(group_map_fixed)
    pairs_df["group_quantile"] = pairs_df["scenario_id"].map(group_map_quantile)

    # Save datasets
    oracle_csv_path = os.path.join(outdir, "oracle_scenarios.csv")
    pairs_csv_path = os.path.join(outdir, "sample_level_dataset.csv")
    flatten_oracle_for_csv(oracle_df).to_csv(oracle_csv_path, index=False)
    pairs_df.to_csv(pairs_csv_path, index=False)

    # Config archive
    cfg.to_json(os.path.join(outdir, "experiment_config.json"))
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # ---- Statistics ----
    magnitudes = sorted(cfg.bias.magnitudes)
    stat_rows = []
    rate_rows = []
    for group_col in ["group_fixed", "group_quantile"]:
        for mag in magnitudes:
            stat_rows.append(near_far_comparison(pairs_df, group_col, mag, "delta_u0"))
            rate_rows.append(rate_comparison(pairs_df, group_col, mag, "active_set_changed"))
            rate_rows.append(rate_comparison(pairs_df, group_col, mag, "perturbed_feasible"))
            rate_rows.append(rate_comparison(
                pairs_df.assign(true_replay_violated=pairs_df["true_replay_violated"].fillna(False)),
                group_col, mag, "true_replay_violated"))

    stats_df = pd.DataFrame(stat_rows)
    rates_df = pd.DataFrame(rate_rows)
    stats_df.to_csv(os.path.join(outdir, "stat_near_far_comparison.csv"), index=False)
    rates_df.to_csv(os.path.join(outdir, "stat_rate_comparisons.csv"), index=False)

    # Confound check: oracle_u0 saturation at an input bound (corner solution)
    # can suppress delta_u0 independent of state-constraint proximity. Report
    # saturation rates by group and re-run the near/far comparison restricted
    # to non-saturated scenarios.
    sat_rate_rows = []
    unsat_stat_rows = []
    pairs_unsat = pairs_df[~pairs_df["oracle_u0_saturated"].astype(bool)]
    for group_col in ["group_fixed", "group_quantile"]:
        sat_rate_rows.append(rate_comparison(pairs_df, group_col, magnitudes[0], "oracle_u0_saturated"))
        for mag in magnitudes:
            unsat_stat_rows.append(near_far_comparison(pairs_unsat, group_col, mag, "delta_u0"))
    sat_rate_df = pd.DataFrame(sat_rate_rows)
    unsat_stats_df = pd.DataFrame(unsat_stat_rows)
    sat_rate_df.to_csv(os.path.join(outdir, "stat_u0_saturation_confound.csv"), index=False)
    unsat_stats_df.to_csv(os.path.join(outdir, "stat_near_far_comparison_unsaturated.csv"), index=False)

    # Supplementary continuous outcomes that are NOT gated by u0 saturation
    # the way delta_u0 can be: objective_diff (whole-horizon cost impact) and
    # true-model max constraint violation on replay.
    pairs_df["objective_diff_abs"] = pairs_df["objective_diff"].abs()
    pairs_df["true_replay_max_violation_filled"] = pairs_df["true_replay_max_violation"].fillna(0.0)
    supp_rows = []
    for group_col in ["group_fixed", "group_quantile"]:
        for mag in magnitudes:
            supp_rows.append(near_far_comparison(pairs_df, group_col, mag, "objective_diff_abs"))
            supp_rows.append(near_far_comparison(pairs_df, group_col, mag, "true_replay_max_violation_filled"))
    supp_stats_df = pd.DataFrame(supp_rows)
    supp_stats_df.to_csv(os.path.join(outdir, "stat_near_far_comparison_supplementary.csv"), index=False)

    spearman_margin = spearman_margin_vs_impact(pairs_df)
    spearman_dual = spearman_dual_vs_impact(pairs_df)
    diag_reg = diagnostic_regression_comparison(pairs_df)

    with open(os.path.join(outdir, "stat_correlations_and_regressions.json"), "w") as f:
        json.dump(dict(spearman_margin_vs_delta_u0=spearman_margin,
                        spearman_dual_vs_delta_u0=spearman_dual,
                        diagnostic_regressions=diag_reg,
                        quantile_thresholds=quantile_thresholds), f, indent=2, default=str)

    group_summary_fixed = group_summary(oracle_df, "group_fixed")
    group_summary_quantile = group_summary(oracle_df, "group_quantile")
    group_summary_fixed.to_csv(os.path.join(outdir, "group_summary_fixed.csv"), index=False)
    group_summary_quantile.to_csv(os.path.join(outdir, "group_summary_quantile.csv"), index=False)

    # ---- Plots ----
    plot_all(oracle_df, pairs_df, cfg.mpc, cfg.dynamics.dt, os.path.join(outdir, "plots"))

    runtime = time.time() - t0

    # ---- Report ----
    write_summary_report(
        outdir=outdir, cfg=cfg, meta=meta, oracle_df=oracle_df, pairs_df=pairs_df,
        stats_df=stats_df, rates_df=rates_df,
        spearman_margin=spearman_margin, spearman_dual=spearman_dual,
        diag_reg=diag_reg, group_summary_fixed=group_summary_fixed,
        group_summary_quantile=group_summary_quantile,
        quantile_thresholds=quantile_thresholds, runtime=runtime, mode=mode,
        sat_rate_df=sat_rate_df, unsat_stats_df=unsat_stats_df, supp_stats_df=supp_stats_df,
    )

    print(f"=== Done in {runtime:.1f}s. Outputs in {outdir} ===")
    return dict(oracle_df=oracle_df, pairs_df=pairs_df, meta=meta, runtime=runtime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()
    run(args.mode, args.outdir)
