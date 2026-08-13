"""
Stage 3 closed-loop MPC evaluation. Reuses stage-2's true nonlinear plant
(POS_BOUND, INPUT_BOUND, true_step) unchanged.
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy import stats as sstats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

STAGE3_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(STAGE3_DIR)
sys.path.insert(0, ROOT_DIR)

from src.config import ExperimentConfig  # noqa: E402
from _stage2_bridge import true_step, POS_BOUND, INPUT_BOUND  # noqa: E402

from evaluate_prediction import load_checkpoint  # noqa: E402
from mpc_learned import solve_mpc_step  # noqa: E402
from train import METHODS  # noqa: E402
from config import Stage3Config  # noqa: E402

sns.set_theme(style="whitegrid")

VIOLATION_TOL = 1e-6


def build_fixed_episodes(cfg: Stage3Config, n_episodes: int = None, T: int = None):
    mcfg = cfg.mpc_eval
    if n_episodes is None:
        n_episodes = mcfg.n_episodes
    if T is None:
        T = mcfg.episode_len
    rng = np.random.default_rng(mcfg.episode_seed)
    episodes = []
    for i in range(n_episodes):
        if rng.uniform() < 0.6:
            side = rng.choice([-1.0, 1.0])
            p_ref = side * rng.uniform(1.6, 1.95)
        else:
            p_ref = rng.uniform(-1.2, 1.2)
        p0 = rng.uniform(-1.0, 1.0)
        v0 = rng.uniform(-0.3, 0.3)
        disturbance = rng.normal(0.0, mcfg.disturbance_std, size=T)
        episodes.append(dict(episode_id=i, p0=p0, v0=v0, p_ref=p_ref, v_ref=0.0,
                              disturbance=disturbance))
    return episodes


def run_episode(model, in_norm, out_norm, exp_cfg: ExperimentConfig, episode: dict, dt: float):
    p, v = episode["p0"], episode["v0"]
    p_ref, v_ref = episode["p_ref"], episode["v_ref"]
    u_prev = 0.0
    T = len(episode["disturbance"])

    ps, vs, us, objs = [], [], [], []
    infeasible_count = 0

    mpc = exp_cfg.mpc
    for t in range(T):
        res = solve_mpc_step(model, in_norm, out_norm, np.array([p, v]), u_prev,
                              p_ref, v_ref, exp_cfg)
        if res.solve_success:
            u = res.u0
        else:
            infeasible_count += 1
            u = 0.0

        stage_cost = (mpc.q_pos * (p - p_ref) ** 2 + mpc.q_vel * (v - v_ref) ** 2
                      + mpc.r_u * u ** 2)
        objs.append(stage_cost)
        ps.append(p); vs.append(v); us.append(u)

        p_next, v_next = true_step(p, v, u, dt=dt)
        v_next = v_next + episode["disturbance"][t]
        p_next = p + dt * v_next

        p, v = p_next, v_next
        u_prev = u

    ps = np.array(ps); us = np.array(us); objs = np.array(objs)
    violations = np.maximum(0.0, np.abs(ps) - POS_BOUND)
    violated_steps = violations > VIOLATION_TOL

    return dict(
        tracking_rmse=float(np.sqrt(np.mean((ps - p_ref) ** 2))),
        cumulative_objective=float(np.sum(objs)),
        violation_rate=float(violated_steps.mean()),
        violation_frequency=int(violated_steps.sum()),
        episode_violated=bool(violated_steps.any()),
        max_violation=float(violations.max()),
        mean_violation=float(violations[violated_steps].mean()) if violated_steps.any() else 0.0,
        infeasibility_rate=float(infeasible_count / T),
        p_traj=ps, u_traj=us, p_ref=p_ref,
    )


def run_all_mpc_eval(data_dir: str, ckpt_dir: str, results_dir: str, cfg: Stage3Config,
                      seeds: list[int], n_episodes: int = None, episode_len: int = None):
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
    exp_cfg = ExperimentConfig()
    episodes = build_fixed_episodes(cfg, n_episodes=n_episodes, T=episode_len)
    print(f"[eval_mpc] fixed episode set: n={len(episodes)}, T={len(episodes[0]['disturbance'])} "
          f"(shared identically across all methods and seeds)")

    rows = []
    example_trajs = {}

    for method in METHODS:
        for si, seed in enumerate(seeds):
            ckpt_path = os.path.join(ckpt_dir, f"{method}_seed{seed}.pt")
            if not os.path.exists(ckpt_path):
                print(f"[eval_mpc] MISSING checkpoint: {ckpt_path} -- skipping")
                continue
            model, in_norm, out_norm, ckpt = load_checkpoint(ckpt_path, cfg)

            for ep in episodes:
                res = run_episode(model, in_norm, out_norm, exp_cfg, ep, exp_cfg.dynamics.dt)
                rows.append(dict(method=method, seed=seed, episode_id=ep["episode_id"],
                                  **{k: v for k, v in res.items() if k not in ("p_traj", "u_traj")}))
                if si == 0 and abs(ep["p_ref"]) > 1.5 and method not in example_trajs:
                    example_trajs[method] = dict(p_traj=res["p_traj"], u_traj=res["u_traj"],
                                                  p_ref=res["p_ref"])
            seed_viol = np.mean([r["violation_rate"] for r in rows
                                  if r["method"] == method and r["seed"] == seed])
            print(f"[eval_mpc] {method:32s} seed={seed} mean_violation_rate={seed_viol:.4f}")

    mpc_df = pd.DataFrame(rows)
    mpc_df.to_csv(os.path.join(results_dir, "mpc_metrics.csv"), index=False)

    agg = (mpc_df.groupby(["method", "seed"])
           .agg(tracking_rmse=("tracking_rmse", "mean"),
                cumulative_objective=("cumulative_objective", "mean"),
                violation_rate=("violation_rate", "mean"),
                episode_violation_rate=("episode_violated", "mean"),
                violation_frequency=("violation_frequency", "mean"),
                max_violation=("max_violation", "max"),
                mean_violation=("mean_violation", "mean"),
                infeasibility_rate=("infeasibility_rate", "mean"))
           .reset_index())
    agg.to_csv(os.path.join(results_dir, "mpc_summary_per_seed.csv"), index=False)
    agg.groupby("method").agg(["mean", "std"]).to_csv(os.path.join(results_dir, "mpc_summary.csv"))

    paired_stats = _paired_comparison(mpc_df)
    with open(os.path.join(results_dir, "mpc_paired_stats.json"), "w") as f:
        json.dump(paired_stats, f, indent=2, default=str)

    _make_plot(agg, results_dir)

    return mpc_df, agg, paired_stats


def _bootstrap_ci(diffs, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs)
    boots = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean()
                       for _ in range(n_boot)])
    return float(diffs.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _paired_comparison(mpc_df: pd.DataFrame):
    metrics = ["violation_rate", "cumulative_objective", "tracking_rmse", "max_violation"]
    piv = {m: mpc_df.pivot_table(index=["seed", "episode_id"], columns="method", values=m)
           for m in metrics}

    results = {}
    for other in [m for m in METHODS if m != "dual_weighted_constraint_rollout"]:
        key = f"dual_weighted_constraint_rollout_vs_{other}"
        results[key] = {}
        for m in metrics:
            if "dual_weighted_constraint_rollout" not in piv[m].columns or other not in piv[m].columns:
                continue
            a = piv[m]["dual_weighted_constraint_rollout"].values
            b = piv[m][other].values
            diffs = a - b
            mean_diff, lo, hi = _bootstrap_ci(diffs)
            rel_improve = (-mean_diff / b.mean() * 100.0) if b.mean() != 0 else float("nan")
            try:
                if np.allclose(diffs, 0.0):
                    wstat, wp = 0.0, 1.0  # no difference at all -> trivially non-significant
                else:
                    wstat, wp = sstats.wilcoxon(a, b)
            except Exception:
                wstat, wp = float("nan"), float("nan")
            n = len(diffs)
            pooled_std = np.std(np.concatenate([a, b]), ddof=1)
            cohens_d = float(mean_diff / pooled_std) if pooled_std > 0 else float("nan")
            results[key][m] = dict(mean_diff=float(mean_diff), ci_low=lo, ci_high=hi,
                                    relative_improvement_pct=float(rel_improve),
                                    wilcoxon_stat=float(wstat), wilcoxon_p=float(wp),
                                    cohens_d=cohens_d, n_pairs=n)
    return results


def _make_plot(agg, results_dir):
    figdir = os.path.join(results_dir, "figures")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(data=agg, x="method", y="violation_rate", ax=axes[0], errorbar="sd")
    axes[0].set_title("Closed-loop violation rate")
    axes[0].tick_params(axis="x", rotation=30)
    sns.barplot(data=agg, x="method", y="cumulative_objective", ax=axes[1], errorbar="sd")
    axes[1].set_title("Cumulative MPC objective")
    axes[1].tick_params(axis="x", rotation=30)
    fig.savefig(os.path.join(figdir, "04_closedloop_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = Stage3Config()
    run_all_mpc_eval(os.path.join(here, "data"), os.path.join(here, "checkpoints"),
                      os.path.join(here, "results"), cfg, list(cfg.seeds))
