"""
Closed-loop MPC evaluation for all methods/seeds, using a FIXED set of
episodes (initial states, references, disturbances) shared identically
across every method and seed, so comparisons are apples-to-apples.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import ExperimentConfig  # noqa: E402

from generate_data import true_step, POS_BOUND, INPUT_BOUND, DT  # noqa: E402
from evaluate_prediction import load_checkpoint  # noqa: E402
from mpc_learned import solve_mpc_step  # noqa: E402
from train import METHODS  # noqa: E402

sns.set_theme(style="whitegrid")

N_EPISODES = 24
EPISODE_LEN = 40
EPISODE_SEED = 555
DISTURBANCE_STD = 0.01
VIOLATION_TOL = 1e-6


def build_fixed_episodes(n_episodes: int = None, T: int = None, seed: int = EPISODE_SEED):
    """Fixed episode configurations (init state, reference, disturbance
    sequence), generated ONCE and reused identically for every method and
    seed. Mix of boundary-seeking and moderate references so that
    constraints are sometimes but not always activated."""
    if n_episodes is None:
        n_episodes = N_EPISODES
    if T is None:
        T = EPISODE_LEN
    rng = np.random.default_rng(seed)
    episodes = []
    for i in range(n_episodes):
        if rng.uniform() < 0.6:
            side = rng.choice([-1.0, 1.0])
            p_ref = side * rng.uniform(1.6, 1.95)
        else:
            p_ref = rng.uniform(-1.2, 1.2)
        p0 = rng.uniform(-1.0, 1.0)
        v0 = rng.uniform(-0.3, 0.3)
        disturbance = rng.normal(0.0, DISTURBANCE_STD, size=T)
        episodes.append(dict(episode_id=i, p0=p0, v0=v0, p_ref=p_ref, v_ref=0.0,
                              disturbance=disturbance))
    return episodes


def run_episode(model, in_norm, out_norm, exp_cfg: ExperimentConfig, episode: dict):
    p, v = episode["p0"], episode["v0"]
    p_ref, v_ref = episode["p_ref"], episode["v_ref"]
    u_prev = 0.0
    T = len(episode["disturbance"])

    ps, vs, us, objs = [], [], [], []
    infeasible_count = 0
    sat_count = 0

    mpc = exp_cfg.mpc
    for t in range(T):
        res = solve_mpc_step(model, in_norm, out_norm, np.array([p, v]), u_prev,
                              p_ref, v_ref, exp_cfg)
        if res.solve_success:
            u = res.u0
        else:
            infeasible_count += 1
            u = 0.0  # fallback control on infeasibility

        stage_cost = (mpc.q_pos * (p - p_ref) ** 2 + mpc.q_vel * (v - v_ref) ** 2
                      + mpc.r_u * u ** 2)
        objs.append(stage_cost)
        ps.append(p); vs.append(v); us.append(u)
        if abs(u) >= INPUT_BOUND - 1e-4:
            sat_count += 1

        # apply TRUE nonlinear dynamics with this episode's fixed disturbance
        # (no wall-clipping here -- we want to genuinely measure violations)
        p_next, v_next = true_step(p, v, u, dt=DT)
        v_next = v_next + episode["disturbance"][t]
        p_next = p + DT * v_next

        p, v = p_next, v_next
        u_prev = u

    ps = np.array(ps); vs = np.array(vs); us = np.array(us); objs = np.array(objs)
    violations = np.maximum(0.0, np.abs(ps) - POS_BOUND)
    violated_steps = violations > VIOLATION_TOL

    tracking_rmse = float(np.sqrt(np.mean((ps - p_ref) ** 2)))
    cumulative_objective = float(np.sum(objs))
    violation_rate = float(violated_steps.mean())
    episode_violated = bool(violated_steps.any())
    max_violation = float(violations.max())
    mean_violation = float(violations[violated_steps].mean()) if violated_steps.any() else 0.0
    infeasibility_rate = float(infeasible_count / T)
    saturation_rate = float(sat_count / T)

    return dict(tracking_rmse=tracking_rmse, cumulative_objective=cumulative_objective,
                violation_rate=violation_rate, episode_violated=episode_violated,
                max_violation=max_violation, mean_violation=mean_violation,
                infeasibility_rate=infeasibility_rate, saturation_rate=saturation_rate,
                p_traj=ps, v_traj=vs, u_traj=us, p_ref=p_ref)


def run_all_mpc_eval(ckpt_dir: str, results_dir: str, seeds: list[int],
                      n_episodes: int = None, episode_len: int = None):
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
    exp_cfg = ExperimentConfig()
    episodes = build_fixed_episodes(n_episodes=n_episodes, T=episode_len)

    rows = []
    example_trajs = {}  # method -> one near-constraint episode trajectory (seed[0])

    for method in METHODS:
        for si, seed in enumerate(seeds):
            ckpt_path = os.path.join(ckpt_dir, f"{method}_seed{seed}.pt")
            model, in_norm, out_norm, ckpt = load_checkpoint(ckpt_path)

            for ep in episodes:
                res = run_episode(model, in_norm, out_norm, exp_cfg, ep)
                rows.append(dict(method=method, seed=seed, episode_id=ep["episode_id"],
                                  **{k: v for k, v in res.items()
                                     if k not in ("p_traj", "v_traj", "u_traj")}))
                if si == 0 and abs(ep["p_ref"]) > 1.5 and method not in example_trajs:
                    example_trajs[method] = dict(p_traj=res["p_traj"], u_traj=res["u_traj"],
                                                  p_ref=res["p_ref"])
            print(f"[eval_mpc] {method:16s} seed={seed} "
                  f"mean_viol_rate={np.mean([r['violation_rate'] for r in rows if r['method']==method and r['seed']==seed]):.4f}")

    mpc_df = pd.DataFrame(rows)
    mpc_df.to_csv(os.path.join(results_dir, "mpc_metrics.csv"), index=False)

    agg = (mpc_df.groupby(["method", "seed"])
           .agg(tracking_rmse=("tracking_rmse", "mean"),
                cumulative_objective=("cumulative_objective", "mean"),
                violation_rate=("violation_rate", "mean"),
                episode_violation_rate=("episode_violated", "mean"),
                max_violation=("max_violation", "max"),
                mean_violation=("mean_violation", "mean"),
                infeasibility_rate=("infeasibility_rate", "mean"),
                saturation_rate=("saturation_rate", "mean"))
           .reset_index())
    agg.to_csv(os.path.join(results_dir, "mpc_summary_per_seed.csv"), index=False)

    seed_summary = agg.groupby("method").agg(["mean", "std"])
    seed_summary.to_csv(os.path.join(results_dir, "mpc_summary.csv"))

    paired_stats = _paired_comparison(mpc_df)
    with open(os.path.join(results_dir, "mpc_paired_stats.json"), "w") as f:
        json.dump(paired_stats, f, indent=2, default=str)

    _make_mpc_plots(agg, mpc_df, example_trajs, exp_cfg, results_dir)

    return mpc_df, agg, paired_stats


def _bootstrap_ci(diffs, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs)
    boots = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean()
                       for _ in range(n_boot)])
    return float(diffs.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _paired_comparison(mpc_df: pd.DataFrame):
    """Paired comparison (matched by seed x episode_id) for
    margin_weighted vs baseline and margin_weighted vs random_weight."""
    metrics = ["violation_rate", "cumulative_objective", "tracking_rmse", "max_violation"]
    results = {}
    piv = {}
    for m in metrics:
        piv[m] = mpc_df.pivot_table(index=["seed", "episode_id"], columns="method", values=m)

    for other in ["baseline", "random_weight"]:
        results[f"margin_weighted_vs_{other}"] = {}
        for m in metrics:
            a = piv[m]["margin_weighted"].values
            b = piv[m][other].values
            diffs = a - b
            mean_diff, lo, hi = _bootstrap_ci(diffs)
            rel_improve = (-mean_diff / b.mean() * 100.0) if b.mean() != 0 else float("nan")
            try:
                wstat, wp = sstats.wilcoxon(a, b)
            except Exception:
                wstat, wp = float("nan"), float("nan")
            results[f"margin_weighted_vs_{other}"][m] = dict(
                mean_diff=float(mean_diff), ci_low=lo, ci_high=hi,
                relative_improvement_pct=float(rel_improve),
                wilcoxon_stat=float(wstat), wilcoxon_p=float(wp),
                n_pairs=len(diffs))
    return results


def _make_mpc_plots(agg, mpc_df, example_trajs, exp_cfg, results_dir):
    figdir = os.path.join(results_dir, "figures")

    # 1. violation rate comparison
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=agg, x="method", y="violation_rate", ax=ax, errorbar="sd")
    ax.set_title("Closed-loop constraint violation rate")
    fig.savefig(os.path.join(figdir, "06_violation_rate_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. objective comparison
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=agg, x="method", y="cumulative_objective", ax=ax, errorbar="sd")
    ax.set_title("Cumulative MPC objective (closed loop, mean over episodes)")
    fig.savefig(os.path.join(figdir, "07_objective_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3 & 4. typical near-constraint closed-loop trajectory + control input
    if example_trajs:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for method, tr in example_trajs.items():
            axes[0].plot(tr["p_traj"], label=method)
            axes[1].step(range(len(tr["u_traj"])), tr["u_traj"], where="post", label=method)
        p_ref = list(example_trajs.values())[0]["p_ref"]
        axes[0].axhline(p_ref, color="gray", ls=":", label="reference")
        axes[0].axhline(POS_BOUND, color="red", ls="--", label="position bound")
        axes[0].axhline(-POS_BOUND, color="red", ls="--")
        axes[0].set_ylabel("position")
        axes[0].set_title("Typical near-constraint closed-loop episode")
        axes[0].legend(fontsize=8)
        axes[1].axhline(INPUT_BOUND, color="red", ls="--")
        axes[1].axhline(-INPUT_BOUND, color="red", ls="--")
        axes[1].set_ylabel("control input u")
        axes[1].set_xlabel("time step")
        axes[1].legend(fontsize=8)
        fig.savefig(os.path.join(figdir, "08_example_closed_loop_trajectory.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    run_all_mpc_eval(os.path.join(here, "checkpoints"), os.path.join(here, "results"),
                      seeds=[101, 202, 303])
