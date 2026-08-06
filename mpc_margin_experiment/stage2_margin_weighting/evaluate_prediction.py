"""
Offline prediction evaluation for all methods/seeds. Uses batched torch
tensor operations (no per-sample Python loops) for both one-step and
rollout evaluation to keep this fast.
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from models import DynamicsMLP, Normalizer, build_model
from train import METHODS, to_tensors

sns.set_theme(style="whitegrid")

ROLLOUT_LEN = 15
ROLLOUT_STRIDE = 2


def load_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = DynamicsMLP()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    in_norm = Normalizer.from_state_dict(ckpt["in_norm"])
    out_norm = Normalizer.from_state_dict(ckpt["out_norm"])
    return model, in_norm, out_norm, ckpt


@torch.no_grad()
def one_step_predictions(model, in_norm, out_norm, df: pd.DataFrame):
    X, Y = to_tensors(df)  # X=[p,v,u], Y=[dp,dv] (true increments)
    Xn = in_norm.normalize(X)
    dy_n = model(Xn)
    dy = out_norm.denormalize(dy_n)
    pred_p_next = X[:, 0] + dy[:, 0]
    pred_v_next = X[:, 1] + dy[:, 1]
    true_p_next = X[:, 0] + Y[:, 0]
    true_v_next = X[:, 1] + Y[:, 1]
    return pred_p_next.numpy(), pred_v_next.numpy(), true_p_next.numpy(), true_v_next.numpy()


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


@torch.no_grad()
def batched_rollout(model, in_norm, out_norm, p0: torch.Tensor, v0: torch.Tensor,
                     u_seq: torch.Tensor):
    """Vectorized multi-step rollout across a BATCH of windows simultaneously.
    p0, v0: (B,)   u_seq: (B, T)  -> returns p_traj, v_traj each (B, T+1)."""
    B, T = u_seq.shape
    p_traj = [p0]
    v_traj = [v0]
    p, v = p0, v0
    for t in range(T):
        u = u_seq[:, t]
        x = torch.stack([p, v, u], dim=1)
        xn = in_norm.normalize(x)
        dy = out_norm.denormalize(model(xn))
        p = p + dy[:, 0]
        v = v + dy[:, 1]
        p_traj.append(p)
        v_traj.append(v)
    return torch.stack(p_traj, dim=1), torch.stack(v_traj, dim=1)


def build_rollout_windows(df: pd.DataFrame, T: int = ROLLOUT_LEN, stride: int = ROLLOUT_STRIDE):
    """Build (p0, v0, u_seq, true_p_traj, true_v_traj, start_margin) windows,
    each fully contained within a single trajectory (no cross-trajectory
    leakage), using the true recorded transition chain as ground truth."""
    windows = []
    for tid, g in df.groupby("trajectory_id"):
        g = g.reset_index(drop=True)
        n = len(g)
        for start in range(0, n - T + 1, stride):
            seg = g.iloc[start:start + T]
            p0 = seg.iloc[0]["p_t"]
            v0 = seg.iloc[0]["v_t"]
            u_seq = seg["u_t"].values
            true_p = np.concatenate([[p0], seg["p_next"].values])
            true_v = np.concatenate([[v0], seg["v_next"].values])
            windows.append(dict(p0=p0, v0=v0, u_seq=u_seq, true_p=true_p, true_v=true_v,
                                 start_margin=seg.iloc[0]["margin"]))
    return windows


def rollout_rmse_for_model(model, in_norm, out_norm, windows, near_only: bool = False):
    sel = [w for w in windows if (w["start_margin"] < 0.4)] if near_only else windows
    if len(sel) == 0:
        return float("nan"), None
    p0 = torch.tensor([w["p0"] for w in sel], dtype=torch.float32)
    v0 = torch.tensor([w["v0"] for w in sel], dtype=torch.float32)
    u_seq = torch.tensor(np.stack([w["u_seq"] for w in sel]), dtype=torch.float32)
    true_p = np.stack([w["true_p"] for w in sel])
    true_v = np.stack([w["true_v"] for w in sel])

    pred_p, pred_v = batched_rollout(model, in_norm, out_norm, p0, v0, u_seq)
    pred_p, pred_v = pred_p.numpy(), pred_v.numpy()

    overall_rmse = rmse(np.concatenate([pred_p, pred_v], axis=1),
                         np.concatenate([true_p, true_v], axis=1))
    per_step_err = np.sqrt(np.mean((pred_p - true_p) ** 2 + (pred_v - true_v) ** 2, axis=0))
    return overall_rmse, per_step_err


def evaluate_all(data_dir: str, ckpt_dir: str, results_dir: str, seeds: list[int]):
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    near_df = test_df[test_df["margin"] < 0.4]
    far_df = test_df[test_df["margin"] > 0.8]
    windows = build_rollout_windows(test_df)

    rows = []
    rollout_curves = {}  # (method) -> list of per-step-error arrays across seeds
    loss_curves = {}     # (method, seed) -> history

    for method in METHODS:
        rollout_curves[method] = []
        for seed in seeds:
            ckpt_path = os.path.join(ckpt_dir, f"{method}_seed{seed}.pt")
            model, in_norm, out_norm, ckpt = load_checkpoint(ckpt_path)

            pred_p, pred_v, true_p, true_v = one_step_predictions(model, in_norm, out_norm, test_df)
            overall_rmse = rmse(np.stack([pred_p, pred_v], 1), np.stack([true_p, true_v], 1))
            pos_rmse = rmse(pred_p, true_p)
            vel_rmse = rmse(pred_v, true_v)

            near_pred_p, near_pred_v, near_true_p, near_true_v = one_step_predictions(
                model, in_norm, out_norm, near_df)
            near_rmse = rmse(np.stack([near_pred_p, near_pred_v], 1),
                              np.stack([near_true_p, near_true_v], 1)) if len(near_df) else float("nan")

            far_pred_p, far_pred_v, far_true_p, far_true_v = one_step_predictions(
                model, in_norm, out_norm, far_df)
            far_rmse = rmse(np.stack([far_pred_p, far_pred_v], 1),
                             np.stack([far_true_p, far_true_v], 1)) if len(far_df) else float("nan")

            rollout_overall, per_step = rollout_rmse_for_model(model, in_norm, out_norm, windows, near_only=False)
            rollout_near, _ = rollout_rmse_for_model(model, in_norm, out_norm, windows, near_only=True)
            if per_step is not None:
                rollout_curves[method].append(per_step)

            log_path = os.path.join(ckpt_dir, f"{method}_seed{seed}_history.json")
            with open(log_path) as f:
                loss_curves[(method, seed)] = json.load(f)

            rows.append(dict(method=method, seed=seed,
                              overall_rmse=overall_rmse, near_rmse=near_rmse, far_rmse=far_rmse,
                              position_rmse=pos_rmse, velocity_rmse=vel_rmse,
                              rollout15_rmse=rollout_overall, rollout15_near_rmse=rollout_near,
                              best_val_loss=ckpt["best_val_loss"], n_epochs=ckpt["n_epochs_trained"]))
            print(f"[eval_pred] {method:16s} seed={seed} overall={overall_rmse:.5f} "
                  f"near={near_rmse:.5f} far={far_rmse:.5f} rollout15={rollout_overall:.5f} "
                  f"rollout15_near={rollout_near:.5f}")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(results_dir, "prediction_metrics.csv"), index=False)

    summary = (metrics_df.groupby("method").agg(["mean", "std"]))
    summary.to_csv(os.path.join(results_dir, "prediction_summary.csv"))

    _make_plots(metrics_df, test_df, rollout_curves, loss_curves, results_dir)
    return metrics_df


def _make_plots(metrics_df, test_df, rollout_curves, loss_curves, results_dir):
    figdir = os.path.join(results_dir, "figures")

    # 1. overall/near/far RMSE grouped bar chart
    melt = metrics_df.melt(id_vars=["method", "seed"],
                            value_vars=["overall_rmse", "near_rmse", "far_rmse"],
                            var_name="region", value_name="rmse")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=melt, x="region", y="rmse", hue="method", ax=ax, errorbar="sd")
    ax.set_title("One-step prediction RMSE by region")
    fig.savefig(os.path.join(figdir, "01_rmse_by_region.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. RMSE vs margin bin
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
    test_df = test_df.copy()
    test_df["margin_bin"] = pd.cut(test_df["margin"], bins=bins)
    rows = []
    for method in metrics_df["method"].unique():
        for seed in metrics_df["seed"].unique():
            ckpt_dir = os.path.join(results_dir, "..", "checkpoints")
            ckpt_path = os.path.join(ckpt_dir, f"{method}_seed{seed}.pt")
            if not os.path.exists(ckpt_path):
                continue
            model, in_norm, out_norm, ckpt = load_checkpoint(ckpt_path)
            for mbin, g in test_df.groupby("margin_bin", observed=True):
                if len(g) == 0:
                    continue
                pred_p, pred_v, true_p, true_v = one_step_predictions(model, in_norm, out_norm, g)
                r = rmse(np.stack([pred_p, pred_v], 1), np.stack([true_p, true_v], 1))
                rows.append(dict(method=method, seed=seed, margin_bin=str(mbin), rmse=r,
                                  bin_mid=mbin.mid if hasattr(mbin, "mid") else 0))
    bin_df = pd.DataFrame(rows)
    if len(bin_df):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=bin_df, x="bin_mid", y="rmse", hue="method", marker="o",
                     errorbar="sd", ax=ax)
        ax.set_xlabel("constraint margin bin (midpoint)")
        ax.set_ylabel("one-step RMSE")
        ax.set_title("RMSE vs. constraint margin")
        fig.savefig(os.path.join(figdir, "02_rmse_vs_margin.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 3. rollout error curve
    fig, ax = plt.subplots(figsize=(9, 6))
    for method, curves in rollout_curves.items():
        if len(curves) == 0:
            continue
        arr = np.stack(curves)  # (n_seeds, T+1)
        mean_curve = arr.mean(axis=0)
        std_curve = arr.std(axis=0)
        steps = np.arange(len(mean_curve))
        ax.plot(steps, mean_curve, marker="o", label=method)
        ax.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)
    ax.set_xlabel("rollout step")
    ax.set_ylabel("RMSE (position+velocity)")
    ax.set_title("15-step open-loop rollout error growth")
    ax.legend()
    fig.savefig(os.path.join(figdir, "03_rollout_error_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. example predicted trajectory near the position constraint
    near_traj_candidates = test_df[test_df["margin"] < 0.3]
    if len(near_traj_candidates) > 0:
        tid = near_traj_candidates.iloc[0]["trajectory_id"]
        g = test_df[test_df["trajectory_id"] == tid].reset_index(drop=True)
        if len(g) >= ROLLOUT_LEN:
            p0, v0 = g.iloc[0]["p_t"], g.iloc[0]["v_t"]
            u_seq = torch.tensor(g["u_t"].values[:ROLLOUT_LEN], dtype=torch.float32).unsqueeze(0)
            true_p = np.concatenate([[p0], g["p_next"].values[:ROLLOUT_LEN]])

            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(true_p, "k-", lw=2.5, label="true")
            for method in metrics_df["method"].unique():
                seed0 = sorted(metrics_df["seed"].unique())[0]
                ckpt_path = os.path.join(results_dir, "..", "checkpoints", f"{method}_seed{seed0}.pt")
                if not os.path.exists(ckpt_path):
                    continue
                model, in_norm, out_norm, ckpt = load_checkpoint(ckpt_path)
                p0t = torch.tensor([p0], dtype=torch.float32)
                v0t = torch.tensor([v0], dtype=torch.float32)
                pred_p, pred_v = batched_rollout(model, in_norm, out_norm, p0t, v0t, u_seq)
                ax.plot(pred_p.numpy()[0], "--", label=method)
            ax.axhline(2.0, color="red", ls=":", label="position bound")
            ax.axhline(-2.0, color="red", ls=":")
            ax.set_xlabel("rollout step")
            ax.set_ylabel("position")
            ax.set_title("Example predicted trajectory near the constraint")
            ax.legend()
            fig.savefig(os.path.join(figdir, "04_example_trajectory_near_constraint.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    # 5. training/validation loss curves
    fig, axes = plt.subplots(1, len(METHODS), figsize=(6 * len(METHODS), 5), sharey=True)
    first_seed_per_method = {}
    for (m, seed) in loss_curves.keys():
        first_seed_per_method.setdefault(m, seed)
    for ax, method in zip(axes, METHODS):
        for (m, seed), hist in loss_curves.items():
            if m != method:
                continue
            epochs = [h["epoch"] for h in hist]
            train_loss = [h["train_loss"] for h in hist]
            val_loss = [h["val_loss"] for h in hist]
            show_label = (seed == first_seed_per_method[m])
            ax.plot(epochs, train_loss, alpha=0.5, color="C0",
                     label="train" if show_label else None)
            ax.plot(epochs, val_loss, alpha=0.9, ls="--", color="C1",
                     label="val" if show_label else None)
        ax.set_title(method)
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("loss (log scale)")
    fig.suptitle("Training / validation loss curves (all seeds overlaid)")
    fig.savefig(os.path.join(figdir, "05_loss_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    evaluate_all(os.path.join(here, "data"), os.path.join(here, "checkpoints"),
                 os.path.join(here, "results"), seeds=[101, 202, 303])
