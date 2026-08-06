"""
Publication-readable figures for the near/far constraint decision-impact
experiment.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


def _save(fig, path_no_ext):
    fig.savefig(path_no_ext + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_margin_vs_impact_scatter(pairs_df: pd.DataFrame, outdir: str):
    magnitudes = sorted(pairs_df["bias_magnitude"].unique())
    fig, axes = plt.subplots(1, len(magnitudes), figsize=(6 * len(magnitudes), 5), sharey=True)
    if len(magnitudes) == 1:
        axes = [axes]
    for ax, mag in zip(axes, magnitudes):
        sub = pairs_df[pairs_df["bias_magnitude"] == mag]
        sns.scatterplot(data=sub, x="oracle_min_state_margin", y="delta_u0",
                         hue="bias_direction", alpha=0.5, s=25, ax=ax, legend=(ax is axes[-1]))
        ax.set_title(f"|bias| = {mag}")
        ax.set_xlabel("Oracle horizon-min state-constraint margin")
        ax.set_ylabel(r"$\Delta u_0$" if ax is axes[0] else "")
    if len(magnitudes) > 0:
        axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, title="bias dir")
    fig.suptitle("First-action discrepancy vs. oracle constraint margin")
    _save(fig, os.path.join(outdir, "01_margin_vs_delta_u0_scatter"))


def plot_near_far_box(pairs_df: pd.DataFrame, group_col: str, outdir: str, suffix: str = ""):
    sub = pairs_df[pairs_df[group_col].isin(["near", "far"])]
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=sub, x="bias_magnitude", y="delta_u0", hue=group_col,
                hue_order=["near", "far"], ax=ax, showfliers=False)
    sns.stripplot(data=sub, x="bias_magnitude", y="delta_u0", hue=group_col,
                   hue_order=["near", "far"], dodge=True, ax=ax, alpha=0.15, size=2,
                   legend=False)
    ax.set_xlabel("Bias magnitude")
    ax.set_ylabel(r"First-action discrepancy $\Delta u_0$")
    ax.set_title(f"Near vs far ({group_col}) decision impact{suffix}")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title="group")
    _save(fig, os.path.join(outdir, f"02_near_far_box_{group_col}"))


def plot_active_set_change_rate(pairs_df: pd.DataFrame, group_col: str, outdir: str):
    sub = pairs_df[pairs_df[group_col].isin(["near", "far"])]
    rate = (sub.groupby(["bias_magnitude", group_col])["active_set_changed"]
               .mean().reset_index())
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=rate, x="bias_magnitude", y="active_set_changed", hue=group_col,
                hue_order=["near", "far"], ax=ax)
    ax.set_ylabel("Active-set-change rate")
    ax.set_xlabel("Bias magnitude")
    ax.set_title(f"Active-set-change rate: near vs far ({group_col})")
    _save(fig, os.path.join(outdir, f"03_active_set_change_rate_{group_col}"))


def plot_true_violation_rate(pairs_df: pd.DataFrame, group_col: str, outdir: str):
    sub = pairs_df[pairs_df[group_col].isin(["near", "far"])].copy()
    sub["true_replay_violated"] = sub["true_replay_violated"].fillna(False).astype(bool)
    rate = (sub.groupby(["bias_magnitude", group_col])["true_replay_violated"]
               .mean().reset_index())
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=rate, x="bias_magnitude", y="true_replay_violated", hue=group_col,
                hue_order=["near", "far"], ax=ax)
    ax.set_ylabel("True-model constraint violation rate")
    ax.set_xlabel("Bias magnitude")
    ax.set_title(f"True-model violation rate: near vs far ({group_col})")
    _save(fig, os.path.join(outdir, f"04_true_violation_rate_{group_col}"))


def plot_margin_direction_heatmap(pairs_df: pd.DataFrame, outdir: str):
    df = pairs_df.copy()
    df["margin_bin"] = pd.qcut(df["oracle_min_state_margin"], q=6, duplicates="drop")
    pivot = df.pivot_table(index="margin_bin", columns="bias_direction",
                            values="delta_u0", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax,
                cbar_kws={"label": r"mean $\Delta u_0$"})
    ax.set_xlabel("Bias direction")
    ax.set_ylabel("Oracle margin bin (narrow -> wide)")
    ax.set_title("Mean decision impact by margin bin x bias direction")
    _save(fig, os.path.join(outdir, "05_margin_direction_heatmap"))


def plot_representative_trajectories(oracle_df: pd.DataFrame, pairs_df: pd.DataFrame,
                                      mpc_cfg, dt: float, outdir: str, seed: int = 0):
    """Select representative examples via documented rule: median-impact and
    high-impact (95th percentile) scenario at the largest bias magnitude,
    plus one near-constraint and one far-from-constraint example."""
    from .dynamics import rollout

    max_mag = pairs_df["bias_magnitude"].max()
    sub = pairs_df[(pairs_df["bias_magnitude"] == max_mag) & (pairs_df["perturbed_feasible"])]
    sub = sub.dropna(subset=["delta_u0"])

    med_val = sub["delta_u0"].median()
    hi_val = sub["delta_u0"].quantile(0.95)

    med_row = sub.iloc[(sub["delta_u0"] - med_val).abs().argsort().iloc[0]]
    hi_row = sub.iloc[(sub["delta_u0"] - hi_val).abs().argsort().iloc[0]]

    selections = [("median_impact", med_row), ("high_impact_p95", hi_row)]

    pmin, pmax = mpc_cfg.pos_bounds

    for label, row in selections:
        sid = int(row["scenario_id"])
        orow = oracle_df[oracle_df["scenario_id"] == sid].iloc[0]
        x0 = np.array([orow["init_pos"], orow["init_vel"]])

        oracle_traj = np.array(orow["oracle_x_traj"])

        # Reconstruct perturbed predicted trajectory via bias-augmented rollout
        direction_p = row["bias_dir_p"]
        direction_v = row["bias_dir_v"]
        mag = row["bias_magnitude"]
        bias_vec = np.array([direction_p, direction_v]) * mag

        # Re-solve perturbed to get its own predicted trajectory & control seq
        from .mpc_solver import solve_perturbed_mpc
        psol = solve_perturbed_mpc(x0, orow["ref_pos"], orow["ref_vel"], dt, mpc_cfg, bias_vec)
        perturbed_pred = psol.x_traj
        true_replay = rollout(x0, psol.u_traj, dt, bias=None)

        t = np.arange(oracle_traj.shape[0]) * dt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        ax.plot(t, oracle_traj[:, 0], label="oracle predicted", lw=2)
        ax.plot(t, perturbed_pred[:, 0], "--", label="perturbed predicted (biased model)", lw=2)
        ax.plot(t, true_replay[:, 0], ":", label="perturbed control, true dynamics replay", lw=2.5)
        ax.axhline(pmax, color="k", lw=1, ls="-")
        ax.axhline(pmin, color="k", lw=1, ls="-")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("position")
        ax.set_title(f"Position -- {label} (scenario {sid}, dir={row['bias_direction']}, "
                      f"|bias|={mag})")
        ax.legend(fontsize=9)

        ax2 = axes[1]
        ax2.step(t[:-1], np.array(orow["oracle_u_traj"]), where="post", label="oracle u")
        ax2.step(t[:-1], psol.u_traj, where="post", label="perturbed u", ls="--")
        ax2.set_xlabel("time [s]")
        ax2.set_ylabel("control (acceleration)")
        ax2.set_title(r"Control sequence, $\Delta u_0$" + f" = {row['delta_u0']:.4f}")
        ax2.legend(fontsize=9)

        _save(fig, os.path.join(outdir, f"06_trajectory_{label}_scenario{sid}"))


def plot_all(oracle_df, pairs_df, mpc_cfg, dt, outdir):
    os.makedirs(outdir, exist_ok=True)
    plot_margin_vs_impact_scatter(pairs_df, outdir)
    plot_near_far_box(pairs_df, "group_fixed", outdir)
    plot_near_far_box(pairs_df, "group_quantile", outdir)
    plot_active_set_change_rate(pairs_df, "group_fixed", outdir)
    plot_active_set_change_rate(pairs_df, "group_quantile", outdir)
    plot_true_violation_rate(pairs_df, "group_fixed", outdir)
    plot_true_violation_rate(pairs_df, "group_quantile", outdir)
    plot_margin_direction_heatmap(pairs_df, outdir)
    plot_representative_trajectories(oracle_df, pairs_df, mpc_cfg, dt, outdir)
