"""
Stage 3 training: baseline_mse / margin_weighted_mse / constraint_rollout /
dual_weighted_constraint_rollout.

All four methods share: identical MLP init per seed, identical optimizer,
identical mini-batch order (separate fixed-seed RNG per training seed,
independent of method), identical max epochs / patience, identical
early-stopping metric (unweighted one-step validation MSE), and are all
trained on the SAME underlying set of H-step windows (baseline/margin
methods simply ignore the extra future-step columns in their loss).
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
import torch

from models import build_model, Normalizer, predict_next_state, constraint_value
from config import Stage3Config

METHODS = ["baseline_mse", "margin_weighted_mse", "constraint_rollout",
           "dual_weighted_constraint_rollout"]

POS_BOUND = 2.0


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def build_windows(df: pd.DataFrame, H: int):
    """Build H-step windows, one per valid starting index t within each
    trajectory (t+H must stay inside the same trajectory -- no cross-
    trajectory leakage). Returns a dict of numpy arrays, one row per window:
      p0, v0, u0                      (N,)     -- the t-th transition
      p_next_true                     (N,)     -- true p_{t+1} (one-step target)
      v_next_true                     (N,)
      u_seq                           (N,H)    -- true u_t..u_{t+H-1}
      p_true_seq, v_true_seq          (N,H)    -- true p_{t+1..t+H}, v_{t+1..t+H}
      mpc_margin, mpc_dual            (N,H)    -- MPC's own privileged
                                                    horizon prediction at t
      margin_t                        (N,)     -- current-step margin (for
                                                    margin_weighted_mse)
    """
    margin_cols = [f"mpc_margin_{k}" for k in range(1, H + 1)]
    dual_cols = [f"mpc_dual_{k}" for k in range(1, H + 1)]

    p0s, v0s, u0s = [], [], []
    p_next_trues, v_next_trues = [], []
    u_seqs, p_true_seqs, v_true_seqs = [], [], []
    mpc_margins, mpc_duals = [], []
    margin_ts = []

    for tid, g in df.groupby("trajectory_id"):
        g = g.sort_values("t_local").reset_index(drop=True)
        n = len(g)
        for t in range(0, n - H):
            seg = g.iloc[t:t + H]
            p0s.append(g.iloc[t]["p_t"])
            v0s.append(g.iloc[t]["v_t"])
            u0s.append(g.iloc[t]["u_t"])
            p_next_trues.append(g.iloc[t]["p_next"])
            v_next_trues.append(g.iloc[t]["v_next"])
            margin_ts.append(g.iloc[t]["margin_t"])

            u_seqs.append(seg["u_t"].values.astype(np.float32))
            p_true_seqs.append(seg["p_next"].values.astype(np.float32))
            v_true_seqs.append(seg["v_next"].values.astype(np.float32))
            mpc_margins.append(g.iloc[t][margin_cols].values.astype(np.float32))
            mpc_duals.append(g.iloc[t][dual_cols].values.astype(np.float32))

    return dict(
        p0=np.array(p0s, dtype=np.float32), v0=np.array(v0s, dtype=np.float32),
        u0=np.array(u0s, dtype=np.float32),
        p_next_true=np.array(p_next_trues, dtype=np.float32),
        v_next_true=np.array(v_next_trues, dtype=np.float32),
        u_seq=np.stack(u_seqs), p_true_seq=np.stack(p_true_seqs), v_true_seq=np.stack(v_true_seqs),
        mpc_margin=np.stack(mpc_margins), mpc_dual=np.stack(mpc_duals),
        margin_t=np.array(margin_ts, dtype=np.float32),
    )


def windows_to_tensors(w: dict):
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in w.items()}


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

def margin_weight_raw(margin: torch.Tensor, alpha_m: float, tau: float,
                       wmin: float, wmax: float) -> torch.Tensor:
    raw = 1.0 + alpha_m * torch.exp(-margin / tau)
    return torch.clamp(raw, wmin, wmax)


def dual_weight_raw(margin: torch.Tensor, dual: torch.Tensor, lambda_bar: float,
                     alpha_m: float, alpha_lambda: float, tau: float, eps: float,
                     wmin: float, wmax: float) -> torch.Tensor:
    # Defensive guard (belt-and-suspenders on top of the generate_data.py
    # fix): any residual NaN/Inf in the privileged margin/dual columns must
    # not be allowed to poison the whole batch's loss. Replace with neutral
    # values (a large margin -> negligible weight contribution; zero dual).
    margin = torch.nan_to_num(margin, nan=10.0, posinf=10.0, neginf=-10.0)
    dual = torch.nan_to_num(dual, nan=0.0, posinf=0.0, neginf=0.0)
    raw = (1.0 + alpha_m * torch.exp(-margin / tau)
           + alpha_lambda * (dual / (lambda_bar + eps)))
    return torch.clamp(raw, wmin, wmax)


# ---------------------------------------------------------------------------
# Rollout (differentiable through the whole horizon)
# ---------------------------------------------------------------------------

def rollout_predict(model, in_norm, out_norm, p0, v0, u_seq):
    """p0,v0: (B,)  u_seq: (B,H) true recorded controls (teacher forcing on
    ACTIONS, not states). Returns p_pred_seq, v_pred_seq each (B,H), fully
    differentiable through all H steps."""
    B, H = u_seq.shape
    p, v = p0, v0
    p_preds, v_preds = [], []
    for k in range(H):
        p, v = predict_next_state(model, in_norm, out_norm, p, v, u_seq[:, k])
        p_preds.append(p)
        v_preds.append(v)
    return torch.stack(p_preds, dim=1), torch.stack(v_preds, dim=1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_loss(method: str, model, in_norm, out_norm, batch: dict, cfg: Stage3Config,
                  lambda_bar: float, sanity: dict):
    tc = cfg.train

    # --- one-step dynamics loss (used by ALL methods) ---
    p1_pred, v1_pred = predict_next_state(model, in_norm, out_norm,
                                           batch["p0"], batch["v0"], batch["u0"])
    dp_true = (batch["p_next_true"] - batch["p0"])
    dv_true = (batch["v_next_true"] - batch["v0"])
    dp_pred = (p1_pred - batch["p0"])
    dv_pred = (v1_pred - batch["v0"])
    # normalized-space squared error (consistent scale across p/v), matching
    # stage 2's convention of computing loss in normalized increment space
    dy_true_n = out_norm.normalize(torch.stack([dp_true, dv_true], dim=1))
    dy_pred_n = out_norm.normalize(torch.stack([dp_pred, dv_pred], dim=1))
    l_dyn_per_sample = ((dy_pred_n - dy_true_n) ** 2).mean(dim=1)  # (B,)

    if method == "baseline_mse":
        loss = l_dyn_per_sample.mean()
        return loss, dict(l_dyn=float(loss.item()), l_con=0.0)

    if method == "margin_weighted_mse":
        w_raw = margin_weight_raw(batch["margin_t"], tc.margin_alpha, tc.margin_scale_tau,
                                   tc.weight_min, tc.weight_max)
        w = w_raw / w_raw.mean().clamp_min(1e-8)
        loss = (w * l_dyn_per_sample).mean()
        sanity["weight_min"] = float(w.min().item())
        sanity["weight_max"] = float(w.max().item())
        return loss, dict(l_dyn=float(loss.item()), l_con=0.0)

    # --- methods that need a differentiable H-step rollout ---
    p_pred_seq, v_pred_seq = rollout_predict(model, in_norm, out_norm,
                                              batch["p0"], batch["v0"], batch["u_seq"])
    g_pred = constraint_value(p_pred_seq, POS_BOUND)       # (B,H)
    g_true = constraint_value(batch["p_true_seq"], POS_BOUND)  # (B,H)
    sq_err = (g_pred - g_true) ** 2  # (B,H)

    H = sq_err.shape[1]
    gamma_pows = (tc.gamma ** torch.arange(H, dtype=torch.float32)).to(sq_err.device)  # (H,)

    if method == "constraint_rollout":
        l_con_per_sample = (sq_err * gamma_pows.unsqueeze(0)).sum(dim=1)  # (B,)
        loss = l_dyn_per_sample.mean() + tc.beta * l_con_per_sample.mean()
        return loss, dict(l_dyn=float(l_dyn_per_sample.mean().item()),
                           l_con=float(l_con_per_sample.mean().item()))

    if method == "dual_weighted_constraint_rollout":
        w_raw = dual_weight_raw(batch["mpc_margin"], batch["mpc_dual"], lambda_bar,
                                 tc.margin_alpha, tc.dual_alpha, tc.margin_scale_tau,
                                 tc.dual_eps, tc.weight_min, tc.weight_max)  # (B,H)
        w = w_raw / w_raw.mean().clamp_min(1e-8)  # batch-mean normalized over B*H entries
        weighted_err = w * sq_err * gamma_pows.unsqueeze(0)
        l_con_per_sample = weighted_err.sum(dim=1)  # (B,)
        loss = l_dyn_per_sample.mean() + tc.beta * l_con_per_sample.mean()
        sanity["weight_min"] = float(w.min().item())
        sanity["weight_max"] = float(w.max().item())
        sanity["dual_max_raw"] = float(batch["mpc_dual"].max().item())
        return loss, dict(l_dyn=float(l_dyn_per_sample.mean().item()),
                           l_con=float(l_con_per_sample.mean().item()))

    raise ValueError(method)


def train_one(method: str, seed: int, windows_train: dict, windows_val: dict,
              in_norm: Normalizer, out_norm: Normalizer, lambda_bar: float,
              cfg: Stage3Config, ckpt_path: str, log_path: str):
    tc = cfg.train
    Wt = windows_to_tensors(windows_train)
    Wv = windows_to_tensors(windows_val)

    model = build_model(seed, cfg.model.hidden_dims, cfg.model.activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=tc.lr)

    n = len(Wt["p0"])
    n_batches = int(np.ceil(n / tc.batch_size))
    batch_rng = np.random.default_rng(seed * 1000 + 7)  # same across methods for this seed

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []
    aborted = False
    max_grad_norm_seen = 0.0
    weight_min_seen, weight_max_seen = float("inf"), float("-inf")

    for epoch in range(tc.max_epochs):
        perm = batch_rng.permutation(n)
        model.train()
        l_dyn_sum, l_con_sum = 0.0, 0.0
        for b in range(n_batches):
            idx = perm[b * tc.batch_size:(b + 1) * tc.batch_size]
            batch = {k: v[idx] for k, v in Wt.items()}
            sanity = {}
            loss, parts = compute_loss(method, model, in_norm, out_norm, batch, cfg,
                                        lambda_bar, sanity)

            if not torch.isfinite(loss):
                print(f"[train][{method}][seed {seed}] NaN/Inf loss at epoch {epoch}, batch {b} "
                      f"-- aborting this method/seed.")
                aborted = True
                break

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip_norm)
            max_grad_norm_seen = max(max_grad_norm_seen, float(grad_norm))
            optimizer.step()

            l_dyn_sum += parts["l_dyn"] * len(idx)
            l_con_sum += parts["l_con"] * len(idx)
            if "weight_min" in sanity:
                weight_min_seen = min(weight_min_seen, sanity["weight_min"])
                weight_max_seen = max(weight_max_seen, sanity["weight_max"])

        if aborted:
            break

        model.eval()
        with torch.no_grad():
            p1_pred, v1_pred = predict_next_state(model, in_norm, out_norm,
                                                    Wv["p0"], Wv["v0"], Wv["u0"])
            dy_true_n = out_norm.normalize(torch.stack(
                [Wv["p_next_true"] - Wv["p0"], Wv["v_next_true"] - Wv["v0"]], dim=1))
            dy_pred_n = out_norm.normalize(torch.stack(
                [p1_pred - Wv["p0"], v1_pred - Wv["v0"]], dim=1))
            val_loss = float(((dy_pred_n - dy_true_n) ** 2).mean().item())  # UNWEIGHTED, same for all methods

        history.append(dict(epoch=epoch, l_dyn=l_dyn_sum / n, l_con=l_con_sum / n,
                             val_loss=val_loss, max_grad_norm=max_grad_norm_seen))

        if not np.isfinite(val_loss):
            print(f"[train][{method}][seed {seed}] NaN/Inf val_loss at epoch {epoch} -- aborting.")
            aborted = True
            break

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= tc.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(dict(model_state=model.state_dict(), in_norm=in_norm.state_dict(),
                     out_norm=out_norm.state_dict(), method=method, seed=seed,
                     best_val_loss=best_val, n_epochs_trained=len(history),
                     aborted=aborted, lambda_bar=lambda_bar,
                     H=cfg.train.H, gamma=cfg.train.gamma, beta=cfg.train.beta),
               ckpt_path)
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    n_params = sum(p.numel() for p in model.parameters())
    return dict(method=method, seed=seed, best_val_loss=best_val,
                n_epochs_trained=len(history), n_params=n_params, aborted=aborted,
                max_grad_norm=max_grad_norm_seen,
                weight_min=(weight_min_seen if np.isfinite(weight_min_seen) else None),
                weight_max=(weight_max_seen if np.isfinite(weight_max_seen) else None))


def compute_lambda_bar(windows_train: dict) -> float:
    """Fixed normalizing constant for the dual term, analogous to a
    normalizer statistic: mean of the MPC's own recorded dual values over
    the full training set (fit on train only)."""
    return float(np.mean(windows_train["mpc_dual"]))


def run_all_training(data_dir: str, ckpt_dir: str, cfg: Stage3Config, quick: bool = False):
    os.makedirs(ckpt_dir, exist_ok=True)
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))

    H = cfg.train.H
    windows_train = build_windows(train_df, H)
    windows_val = build_windows(val_df, H)
    print(f"[train] windows: train={len(windows_train['p0'])}, val={len(windows_val['p0'])}")

    X_train_raw = np.stack([windows_train["p0"], windows_train["v0"], windows_train["u0"]], axis=1)
    Y_train_raw = np.stack([windows_train["p_next_true"] - windows_train["p0"],
                             windows_train["v_next_true"] - windows_train["v0"]], axis=1)
    in_norm = Normalizer.fit(X_train_raw)
    out_norm = Normalizer.fit(Y_train_raw)
    lambda_bar = compute_lambda_bar(windows_train)
    print(f"[train] lambda_bar (mean MPC dual, train set) = {lambda_bar:.6g}")

    seeds = cfg.seeds if not quick else cfg.seeds[:1]

    all_results = []
    fairness_report = []
    for seed in seeds:
        results_this_seed = {}
        for method in METHODS:
            ckpt_path = os.path.join(ckpt_dir, f"{method}_seed{seed}.pt")
            log_path = os.path.join(ckpt_dir, f"{method}_seed{seed}_history.json")
            res = train_one(method, seed, windows_train, windows_val, in_norm, out_norm,
                             lambda_bar, cfg, ckpt_path, log_path)
            results_this_seed[method] = res
            all_results.append(res)
            print(f"[train] seed={seed} method={method:32s} epochs={res['n_epochs_trained']:3d} "
                  f"best_val={res['best_val_loss']:.6f} max_grad_norm={res['max_grad_norm']:.3f} "
                  f"weights=[{res['weight_min']},{res['weight_max']}] aborted={res['aborted']}")

        n_params_set = set(r["n_params"] for r in results_this_seed.values())
        assert len(n_params_set) == 1, "param counts differ across methods!"
        fairness_report.append(dict(seed=seed, n_params=list(n_params_set)[0],
                                     epochs={m: results_this_seed[m]["n_epochs_trained"] for m in METHODS},
                                     any_aborted=any(results_this_seed[m]["aborted"] for m in METHODS)))

    with open(os.path.join(ckpt_dir, "fairness_report.json"), "w") as f:
        json.dump(dict(per_seed=fairness_report, lambda_bar=lambda_bar,
                        H=H, gamma=cfg.train.gamma, beta=cfg.train.beta,
                        normalization="fit on train set only",
                        note="all methods trained on the SAME window set and batch order; "
                             "early stopping uses UNWEIGHTED one-step val MSE for all methods"),
                  f, indent=2)

    return all_results, in_norm, out_norm, lambda_bar


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = Stage3Config()
    run_all_training(os.path.join(here, "data"), os.path.join(here, "checkpoints"), cfg)
