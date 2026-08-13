"""
Stage 3 offline prediction evaluation.
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

from models import ResidualMLP, Normalizer, predict_next_state, constraint_value
from train import METHODS, build_windows, windows_to_tensors, rollout_predict
from config import Stage3Config

sns.set_theme(style="whitegrid")


def load_checkpoint(ckpt_path: str, cfg: Stage3Config):
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = ResidualMLP(hidden_dims=cfg.model.hidden_dims, activation=cfg.model.activation)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    in_norm = Normalizer.from_state_dict(ckpt["in_norm"])
    out_norm = Normalizer.from_state_dict(ckpt["out_norm"])
    return model, in_norm, out_norm, ckpt


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


@torch.no_grad()
def evaluate_one(model, in_norm, out_norm, Wt: dict):
    p1, v1 = predict_next_state(model, in_norm, out_norm, Wt["p0"], Wt["v0"], Wt["u0"])
    one_step_rmse = rmse(torch.stack([p1, v1], 1).numpy(),
                          torch.stack([Wt["p_next_true"], Wt["v_next_true"]], 1).numpy())

    near_mask = (Wt["margin_t"] < 0.4).numpy()
    if near_mask.sum() > 0:
        near_rmse = rmse(torch.stack([p1, v1], 1).numpy()[near_mask],
                          torch.stack([Wt["p_next_true"], Wt["v_next_true"]], 1).numpy()[near_mask])
    else:
        near_rmse = float("nan")

    p_pred_seq, v_pred_seq = rollout_predict(model, in_norm, out_norm, Wt["p0"], Wt["v0"], Wt["u_seq"])
    rollout_rmse = rmse(torch.stack([p_pred_seq, v_pred_seq], -1).numpy(),
                         torch.stack([Wt["p_true_seq"], Wt["v_true_seq"]], -1).numpy())

    g_pred = constraint_value(p_pred_seq).numpy()
    g_true = constraint_value(Wt["p_true_seq"]).numpy()
    constraint_rmse = rmse(g_pred, g_true)

    active_mask = g_true < 0.0
    if active_mask.sum() > 0:
        active_constraint_rmse = rmse(g_pred[active_mask], g_true[active_mask])
    else:
        active_constraint_rmse = float("nan")

    per_step_constraint_err = np.sqrt(np.mean((g_pred - g_true) ** 2, axis=0))  # (H,)

    return dict(overall_one_step_rmse=one_step_rmse, near_rmse=near_rmse,
                rollout_rmse=rollout_rmse, constraint_rmse=constraint_rmse,
                active_constraint_rmse=active_constraint_rmse,
                n_active=int(active_mask.sum()), n_total=int(g_true.size)), per_step_constraint_err


def evaluate_all(data_dir: str, ckpt_dir: str, results_dir: str, cfg: Stage3Config,
                  seeds: list[int]):
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    Wt = windows_to_tensors(build_windows(test_df, cfg.train.H))

    rows = []
    constraint_err_curves = {m: [] for m in METHODS}
    loss_curves = {}

    for method in METHODS:
        for seed in seeds:
            ckpt_path = os.path.join(ckpt_dir, f"{method}_seed{seed}.pt")
            if not os.path.exists(ckpt_path):
                print(f"[eval_pred] MISSING checkpoint: {ckpt_path} -- skipping")
                continue
            model, in_norm, out_norm, ckpt = load_checkpoint(ckpt_path, cfg)
            metrics, per_step_err = evaluate_one(model, in_norm, out_norm, Wt)
            constraint_err_curves[method].append(per_step_err)

            log_path = os.path.join(ckpt_dir, f"{method}_seed{seed}_history.json")
            if os.path.exists(log_path):
                with open(log_path) as f:
                    loss_curves[(method, seed)] = json.load(f)

            rows.append(dict(method=method, seed=seed, best_val_loss=ckpt["best_val_loss"],
                              n_epochs=ckpt["n_epochs_trained"], aborted=ckpt.get("aborted", False),
                              **metrics))
            print(f"[eval_pred] {method:32s} seed={seed} one_step={metrics['overall_one_step_rmse']:.5f} "
                  f"near={metrics['near_rmse']:.5f} rollout={metrics['rollout_rmse']:.5f} "
                  f"constraint={metrics['constraint_rmse']:.5f} "
                  f"active_constraint={metrics['active_constraint_rmse']:.5f} (n_active={metrics['n_active']})")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(results_dir, "prediction_metrics.csv"), index=False)
    metrics_df.groupby("method").agg(["mean", "std"]).to_csv(
        os.path.join(results_dir, "prediction_summary.csv"))

    _make_plots(metrics_df, constraint_err_curves, loss_curves, results_dir)
    return metrics_df


def _make_plots(metrics_df, constraint_err_curves, loss_curves, results_dir):
    figdir = os.path.join(results_dir, "figures")

    # 1. training / validation loss curves (one_step val loss, all methods overlaid)
    fig, ax = plt.subplots(figsize=(9, 6))
    for (m, seed), hist in loss_curves.items():
        epochs = [h["epoch"] for h in hist]
        val_loss = [h["val_loss"] for h in hist]
        ax.plot(epochs, val_loss, alpha=0.7, label=f"{m} (seed {seed})" if seed == min(
            s for (mm, s) in loss_curves if mm == m) else None)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("unweighted val one-step MSE (log scale)")
    ax.set_title("Validation loss curves (all methods, all seeds)")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(os.path.join(figdir, "01_training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. constraint-value error by method (bar, overall + active-region)
    melt = metrics_df.melt(id_vars=["method", "seed"],
                            value_vars=["constraint_rmse", "active_constraint_rmse"],
                            var_name="region", value_name="rmse")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=melt, x="region", y="rmse", hue="method", ax=ax, errorbar="sd")
    ax.set_title("Future constraint-value prediction error by method")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(os.path.join(figdir, "02_constraint_error_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. per-horizon-step constraint error curve
    fig, ax = plt.subplots(figsize=(9, 6))
    for method, curves in constraint_err_curves.items():
        if not curves:
            continue
        arr = np.stack(curves)
        mean_curve = arr.mean(axis=0)
        ax.plot(np.arange(1, len(mean_curve) + 1), mean_curve, marker="o", label=method)
    ax.set_xlabel("horizon step k")
    ax.set_ylabel("constraint-value RMSE")
    ax.set_title("Constraint-value error growth over the rollout horizon")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(figdir, "03_constraint_error_vs_horizon.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = Stage3Config()
    evaluate_all(os.path.join(here, "data"), os.path.join(here, "checkpoints"),
                 os.path.join(here, "results"), cfg, list(cfg.seeds))
