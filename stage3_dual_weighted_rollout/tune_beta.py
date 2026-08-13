"""
Small-scale beta tuning for constraint_rollout / dual_weighted_constraint_rollout.

Per task spec: only tune on beta in {0.1, 0.5, 1, 2}, validation set only,
NOT the test set, reduced-epoch runs on seed[0] only (this is explicitly a
"small-scale validation tuning" step, not a full sweep).

Selection rule (documented, not test-set-informed): among beta candidates
whose validation one-step RMSE does not exceed baseline_mse's validation
one-step RMSE by more than 20% (guard against beta so large it wrecks basic
dynamics accuracy), pick the one with the lowest validation H-step
constraint-rollout RMSE. If no candidate satisfies the guard, pick the
smallest beta (most conservative).
"""
from __future__ import annotations

import copy
import json
import os
import numpy as np
import torch

from models import build_model, predict_next_state, constraint_value, Normalizer
from train import (build_windows, windows_to_tensors, compute_loss, rollout_predict,
                    compute_lambda_bar)
from config import Stage3Config

TUNE_EPOCHS = 15


def _val_one_step_rmse(model, in_norm, out_norm, Wv):
    with torch.no_grad():
        p1, v1 = predict_next_state(model, in_norm, out_norm, Wv["p0"], Wv["v0"], Wv["u0"])
        err = torch.stack([p1 - Wv["p_next_true"], v1 - Wv["v_next_true"]], dim=1)
        return float(torch.sqrt((err ** 2).mean()).item())


def _val_constraint_rollout_rmse(model, in_norm, out_norm, Wv):
    with torch.no_grad():
        p_pred, v_pred = rollout_predict(model, in_norm, out_norm, Wv["p0"], Wv["v0"], Wv["u_seq"])
        g_pred = constraint_value(p_pred)
        g_true = constraint_value(Wv["p_true_seq"])
        return float(torch.sqrt(((g_pred - g_true) ** 2).mean()).item())


def _short_train(method, seed, windows_train, windows_val, in_norm, out_norm, lambda_bar,
                  cfg, beta, epochs):
    cfg_local = copy.deepcopy(cfg)
    cfg_local.train.beta = beta
    cfg_local.train.max_epochs = epochs
    cfg_local.train.patience = epochs  # no early stopping during tuning, fixed short budget

    Wt = windows_to_tensors(windows_train)
    Wv = windows_to_tensors(windows_val)
    model = build_model(seed, cfg.model.hidden_dims, cfg.model.activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    n = len(Wt["p0"])
    n_batches = int(np.ceil(n / cfg.train.batch_size))
    batch_rng = np.random.default_rng(seed * 1000 + 7)

    for epoch in range(epochs):
        perm = batch_rng.permutation(n)
        model.train()
        for b in range(n_batches):
            idx = perm[b * cfg.train.batch_size:(b + 1) * cfg.train.batch_size]
            batch = {k: v[idx] for k, v in Wt.items()}
            sanity = {}
            loss, _ = compute_loss(method, model, in_norm, out_norm, batch, cfg_local,
                                    lambda_bar, sanity)
            if not torch.isfinite(loss):
                break
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
            optimizer.step()

    model.eval()
    one_step_rmse = _val_one_step_rmse(model, in_norm, out_norm, Wv)
    con_rmse = _val_constraint_rollout_rmse(model, in_norm, out_norm, Wv)
    return one_step_rmse, con_rmse


def tune(data_dir: str, results_dir: str, cfg: Stage3Config, methods=None):
    import pandas as pd
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    H = cfg.train.H
    windows_train = build_windows(train_df, H)
    windows_val = build_windows(val_df, H)

    X_train_raw = np.stack([windows_train["p0"], windows_train["v0"], windows_train["u0"]], axis=1)
    Y_train_raw = np.stack([windows_train["p_next_true"] - windows_train["p0"],
                             windows_train["v_next_true"] - windows_train["v0"]], axis=1)
    in_norm = Normalizer.fit(X_train_raw)
    out_norm = Normalizer.fit(Y_train_raw)
    lambda_bar = compute_lambda_bar(windows_train)

    seed0 = cfg.seeds[0]

    baseline_one_step, _ = _short_train("baseline_mse", seed0, windows_train, windows_val,
                                         in_norm, out_norm, lambda_bar, cfg, beta=0.0,
                                         epochs=TUNE_EPOCHS)

    if methods is None:
        methods = ["constraint_rollout", "dual_weighted_constraint_rollout"]

    results = {}
    for method in methods:
        rows = []
        for beta in cfg.train.beta_grid:
            one_step, con = _short_train(method, seed0, windows_train, windows_val,
                                          in_norm, out_norm, lambda_bar, cfg, beta, TUNE_EPOCHS)
            rows.append(dict(beta=beta, val_one_step_rmse=one_step, val_constraint_rollout_rmse=con))
            print(f"[tune_beta] {method:32s} beta={beta:<5} "
                  f"val_one_step_rmse={one_step:.5f} val_constraint_rmse={con:.5f}")

        guard = baseline_one_step * 1.2
        candidates = [r for r in rows if r["val_one_step_rmse"] <= guard]
        if candidates:
            best = min(candidates, key=lambda r: r["val_constraint_rollout_rmse"])
        else:
            best = min(rows, key=lambda r: r["beta"])
        results[method] = dict(rows=rows, chosen_beta=best["beta"],
                                baseline_one_step_rmse=baseline_one_step, guard=guard)
        print(f"[tune_beta] {method}: chosen beta = {best['beta']}")

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "beta_tuning.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = Stage3Config()
    tune(os.path.join(here, "data"), os.path.join(here, "results"), cfg)
