"""
Trains the three dynamics-learning methods (Baseline / Random-weight /
Margin-weighted) under IDENTICAL conditions (data, architecture, init,
optimizer, batch order, epochs, early stopping) -- the only thing that
differs between methods is the per-sample loss weight.
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models import build_model, Normalizer

METHODS = ["baseline", "random_weight", "margin_weighted"]

BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-3
WEIGHT_MIN, WEIGHT_MAX = 0.5, 5.0
WEIGHT_MARGIN_SCALE = 0.2


def compute_margin_weights(margin: np.ndarray) -> np.ndarray:
    raw = 1.0 + 4.0 * np.exp(-margin / WEIGHT_MARGIN_SCALE)
    clipped = np.clip(raw, WEIGHT_MIN, WEIGHT_MAX)
    normalized = clipped / clipped.mean()
    return normalized


def load_split(data_dir: str):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return train, val, test


def to_tensors(df: pd.DataFrame):
    p = torch.tensor(df["p_t"].values, dtype=torch.float32)
    v = torch.tensor(df["v_t"].values, dtype=torch.float32)
    u = torch.tensor(df["u_t"].values, dtype=torch.float32)
    dp = torch.tensor((df["p_next"] - df["p_t"]).values, dtype=torch.float32)
    dv = torch.tensor((df["v_next"] - df["v_t"]).values, dtype=torch.float32)
    X = torch.stack([p, v, u], dim=1)
    Y = torch.stack([dp, dv], dim=1)
    return X, Y


def train_one(method: str, seed: int, train_df: pd.DataFrame, val_df: pd.DataFrame,
              in_norm: Normalizer, out_norm: Normalizer,
              margin_weights_train: np.ndarray, random_weights_train: np.ndarray,
              ckpt_path: str, log_path: str):
    X_train, Y_train = to_tensors(train_df)
    X_val, Y_val = to_tensors(val_df)

    X_train_n = in_norm.normalize(X_train)
    Y_train_n = (Y_train - out_norm.mean) / out_norm.std
    X_val_n = in_norm.normalize(X_val)
    Y_val_n = (Y_val - out_norm.mean) / out_norm.std

    if method == "baseline":
        weights = np.ones(len(train_df), dtype=np.float32)
    elif method == "random_weight":
        weights = random_weights_train.astype(np.float32)
    elif method == "margin_weighted":
        weights = margin_weights_train.astype(np.float32)
    else:
        raise ValueError(method)
    weights_t = torch.tensor(weights, dtype=torch.float32)

    model = build_model(seed)  # identical init across methods for a given seed
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    n = len(train_df)
    n_batches = int(np.ceil(n / BATCH_SIZE))
    # Same batch-order RNG seed across methods (for a given training seed) so
    # that all three methods see EXACTLY the same sequence of mini-batches;
    # only the per-sample weight inside the loss differs.
    batch_rng = np.random.default_rng(seed * 1000 + 7)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(MAX_EPOCHS):
        perm = batch_rng.permutation(n)
        model.train()
        train_loss_sum = 0.0
        for b in range(n_batches):
            idx = perm[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            xb = X_train_n[idx]
            yb = Y_train_n[idx]
            wb = weights_t[idx]

            pred = model(xb)
            sq_err = ((pred - yb) ** 2).mean(dim=1)  # per-sample MSE (both dims)
            loss = (sq_err * wb).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(idx)
        train_loss = train_loss_sum / n

        model.eval()
        with torch.no_grad():
            pred_val = model(X_val_n)
            val_loss = float(((pred_val - Y_val_n) ** 2).mean().item())  # UNWEIGHTED, same metric all methods

        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss))

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    torch.save(dict(model_state=model.state_dict(),
                     in_norm=in_norm.state_dict(), out_norm=out_norm.state_dict(),
                     method=method, seed=seed, best_val_loss=best_val,
                     n_epochs_trained=len(history)), ckpt_path)
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    n_params = sum(p.numel() for p in model.parameters())
    return dict(method=method, seed=seed, best_val_loss=best_val,
                n_epochs_trained=len(history), n_params=n_params)


def run_all_training(data_dir: str, ckpt_dir: str, seeds: list[int]):
    os.makedirs(ckpt_dir, exist_ok=True)
    train_df, val_df, test_df = load_split(data_dir)

    # Normalization statistics from TRAIN SET ONLY
    X_train_raw = train_df[["p_t", "v_t", "u_t"]].values
    Y_train_raw = np.stack([(train_df["p_next"] - train_df["p_t"]).values,
                             (train_df["v_next"] - train_df["v_t"]).values], axis=1)
    in_norm = Normalizer.fit(X_train_raw)
    out_norm = Normalizer.fit(Y_train_raw)

    margin_weights_train = compute_margin_weights(train_df["margin"].values)

    fairness_report = []
    all_results = []

    for seed in seeds:
        # Random-weight control: SAME weight VALUES as margin-weighted, shuffled
        # within the training set so they no longer correspond to margin. Fixed
        # per seed for reproducibility.
        shuffle_rng = np.random.default_rng(seed * 999 + 3)
        random_weights_train = margin_weights_train.copy()
        shuffle_rng.shuffle(random_weights_train)

        # fairness check: weight distributions identical (as multisets)
        assert np.allclose(np.sort(random_weights_train), np.sort(margin_weights_train)), \
            "random-weight and margin-weight distributions must be identical (only order differs)"

        results_this_seed = {}
        for method in METHODS:
            ckpt_path = os.path.join(ckpt_dir, f"{method}_seed{seed}.pt")
            log_path = os.path.join(ckpt_dir, f"{method}_seed{seed}_history.json")
            res = train_one(method, seed, train_df, val_df, in_norm, out_norm,
                             margin_weights_train, random_weights_train,
                             ckpt_path, log_path)
            results_this_seed[method] = res
            all_results.append(res)
            print(f"[train] seed={seed} method={method:16s} "
                  f"epochs={res['n_epochs_trained']:3d} best_val_loss={res['best_val_loss']:.6f} "
                  f"n_params={res['n_params']}")

        # per-seed fairness check: identical param counts across methods
        n_params_set = set(r["n_params"] for r in results_this_seed.values())
        assert len(n_params_set) == 1, "param counts differ across methods!"
        fairness_report.append(dict(seed=seed, n_params=list(n_params_set)[0],
                                     epochs=[results_this_seed[m]["n_epochs_trained"] for m in METHODS]))

    with open(os.path.join(ckpt_dir, "fairness_report.json"), "w") as f:
        json.dump(dict(per_seed=fairness_report,
                        weight_check="random_weight and margin_weighted weight multisets identical: PASSED",
                        normalization="fit on train set only",
                        note="baseline uses uniform weight=1; early stopping uses UNWEIGHTED val MSE for all methods"),
                   f, indent=2)

    return all_results, in_norm, out_norm


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    run_all_training(os.path.join(here, "data"), os.path.join(here, "checkpoints"), seeds=[101, 202, 303])
