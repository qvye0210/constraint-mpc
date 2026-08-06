"""
Statistical analysis for the near/far constraint decision-impact hypothesis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score


def bootstrap_ci(x: np.ndarray, stat_fn=np.mean, n_boot: int = 5000, alpha: float = 0.05,
                  seed: int = 0) -> tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high) via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    point = stat_fn(x)
    boots = np.empty(n_boot)
    n = len(x)
    for i in range(n_boot):
        sample = rng.choice(x, size=n, replace=True)
        boots[i] = stat_fn(sample)
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return float(point), float(lo), float(hi)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size for two independent samples (non-parametric)."""
    a = np.asarray(a); b = np.asarray(b)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    # Efficient O(n log n) computation via ranks
    all_vals = np.concatenate([a, b])
    order = np.argsort(all_vals)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(all_vals) + 1)
    # Handle ties with average ranks
    sorted_vals = all_vals[order]
    rank_avg = ranks.copy()
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = ranks[order[i:j + 1]].mean()
            rank_avg[order[i:j + 1]] = avg
        i = j + 1
    r_a = rank_avg[:len(a)].sum()
    n1, n2 = len(a), len(b)
    u = r_a - n1 * (n1 + 1) / 2.0
    delta = (2 * u) / (n1 * n2) - 1.0
    return float(delta)


def near_far_comparison(pairs_df: pd.DataFrame, group_col: str, magnitude: float,
                         outcome_col: str = "delta_u0") -> dict:
    """Compare near vs far groups at a fixed bias magnitude for one outcome."""
    sub = pairs_df[pairs_df["bias_magnitude"] == magnitude]
    near = sub[sub[group_col] == "near"][outcome_col].dropna().values
    far = sub[sub[group_col] == "far"][outcome_col].dropna().values

    result = dict(magnitude=magnitude, group_col=group_col, outcome=outcome_col,
                   n_near=len(near), n_far=len(far))

    if len(near) < 3 or len(far) < 3:
        result.update(dict(mean_near=np.nan, mean_far=np.nan, median_near=np.nan,
                            median_far=np.nan, mwu_stat=np.nan, mwu_p=np.nan,
                            cliffs_delta=np.nan,
                            mean_near_ci_low=np.nan, mean_near_ci_high=np.nan,
                            mean_far_ci_low=np.nan, mean_far_ci_high=np.nan))
        return result

    mean_near, near_lo, near_hi = bootstrap_ci(near, np.mean, seed=1)
    mean_far, far_lo, far_hi = bootstrap_ci(far, np.mean, seed=2)

    mwu_stat, mwu_p = stats.mannwhitneyu(near, far, alternative="greater")
    delta = cliffs_delta(near, far)

    result.update(dict(
        mean_near=mean_near, mean_near_ci_low=near_lo, mean_near_ci_high=near_hi,
        mean_far=mean_far, mean_far_ci_low=far_lo, mean_far_ci_high=far_hi,
        median_near=float(np.median(near)), median_far=float(np.median(far)),
        mwu_stat=float(mwu_stat), mwu_p=float(mwu_p),
        cliffs_delta=delta,
    ))
    return result


def rate_comparison(pairs_df: pd.DataFrame, group_col: str, magnitude: float,
                     flag_col: str) -> dict:
    sub = pairs_df[pairs_df["bias_magnitude"] == magnitude]
    near = sub[sub[group_col] == "near"][flag_col]
    far = sub[sub[group_col] == "far"][flag_col]
    near_rate = float(near.mean()) if len(near) else np.nan
    far_rate = float(far.mean()) if len(far) else np.nan

    # Two-proportion z-test (approx) if possible
    p_val = np.nan
    if len(near) > 0 and len(far) > 0:
        try:
            count = np.array([near.sum(), far.sum()])
            nobs = np.array([len(near), len(far)])
            p_pool = count.sum() / nobs.sum()
            if 0 < p_pool < 1:
                se = np.sqrt(p_pool * (1 - p_pool) * (1 / nobs[0] + 1 / nobs[1]))
                z = (near_rate - far_rate) / se if se > 0 else 0.0
                p_val = float(2 * (1 - stats.norm.cdf(abs(z))))
        except Exception:
            pass

    return dict(magnitude=magnitude, group_col=group_col, flag=flag_col,
                n_near=len(near), n_far=len(far),
                rate_near=near_rate, rate_far=far_rate, p_value=p_val)


def spearman_margin_vs_impact(pairs_df: pd.DataFrame, margin_col: str = "oracle_min_state_margin",
                               outcome_col: str = "delta_u0") -> dict:
    sub = pairs_df[[margin_col, outcome_col]].dropna()
    if len(sub) < 3:
        return dict(n=len(sub), rho=np.nan, p=np.nan)
    rho, p = stats.spearmanr(sub[margin_col], sub[outcome_col])
    return dict(n=len(sub), rho=float(rho), p=float(p))


def spearman_dual_vs_impact(pairs_df: pd.DataFrame, dual_col: str = "oracle_max_state_dual_norm",
                             outcome_col: str = "delta_u0") -> dict:
    sub = pairs_df[[dual_col, outcome_col]].dropna()
    if len(sub) < 3:
        return dict(n=len(sub), rho=np.nan, p=np.nan)
    rho, p = stats.spearmanr(sub[dual_col], sub[outcome_col])
    return dict(n=len(sub), rho=float(rho), p=float(p))


def diagnostic_regression_comparison(pairs_df: pd.DataFrame) -> dict:
    """Compare a model using only bias magnitude against a model using
    magnitude + margin + active-set + dual features, for explaining
    first-action discrepancy (linear) and high-impact / active-set-change
    (logistic). This is a diagnostic tool only, not a proposed method."""
    df = pairs_df.dropna(subset=["delta_u0", "oracle_min_state_margin",
                                  "oracle_max_state_dual_norm", "bias_magnitude"]).copy()
    df["oracle_active_flag"] = (df["oracle_n_active_state"] > 0).astype(float)

    results = {}

    # --- Linear regression: delta_u0 ---
    X_mag = df[["bias_magnitude"]].values
    X_full = df[["bias_magnitude", "oracle_min_state_margin",
                 "oracle_max_state_dual_norm", "oracle_active_flag"]].values
    y = df["delta_u0"].values

    lr_mag = LinearRegression().fit(X_mag, y)
    r2_mag = r2_score(y, lr_mag.predict(X_mag))

    lr_full = LinearRegression().fit(X_full, y)
    r2_full = r2_score(y, lr_full.predict(X_full))

    results["linear_delta_u0"] = dict(r2_magnitude_only=float(r2_mag),
                                       r2_full_features=float(r2_full),
                                       n=len(df),
                                       full_coefs=dict(zip(
                                           ["bias_magnitude", "oracle_min_state_margin",
                                            "oracle_max_state_dual_norm", "oracle_active_flag"],
                                           [float(c) for c in lr_full.coef_])))

    # --- Logistic regression: high-impact (top quartile of delta_u0) ---
    thresh = df["delta_u0"].quantile(0.75)
    y_bin = (df["delta_u0"] >= thresh).astype(int).values

    if y_bin.sum() >= 5 and (len(y_bin) - y_bin.sum()) >= 5:
        log_mag = LogisticRegression().fit(X_mag, y_bin)
        auc_mag = roc_auc_score(y_bin, log_mag.predict_proba(X_mag)[:, 1])

        log_full = LogisticRegression(max_iter=1000).fit(X_full, y_bin)
        auc_full = roc_auc_score(y_bin, log_full.predict_proba(X_full)[:, 1])

        results["logistic_high_impact"] = dict(auc_magnitude_only=float(auc_mag),
                                                 auc_full_features=float(auc_full),
                                                 threshold=float(thresh), n=len(df))
    else:
        results["logistic_high_impact"] = dict(note="insufficient class balance for logistic diagnostic")

    # --- Logistic regression: active-set change ---
    df2 = pairs_df.dropna(subset=["active_set_changed", "oracle_min_state_margin",
                                   "oracle_max_state_dual_norm", "bias_magnitude"]).copy()
    df2["oracle_active_flag"] = (df2["oracle_n_active_state"] > 0).astype(float)
    y2 = df2["active_set_changed"].astype(int).values
    X2_mag = df2[["bias_magnitude"]].values
    X2_full = df2[["bias_magnitude", "oracle_min_state_margin",
                    "oracle_max_state_dual_norm", "oracle_active_flag"]].values

    if y2.sum() >= 5 and (len(y2) - y2.sum()) >= 5:
        log2_mag = LogisticRegression().fit(X2_mag, y2)
        auc2_mag = roc_auc_score(y2, log2_mag.predict_proba(X2_mag)[:, 1])
        log2_full = LogisticRegression(max_iter=1000).fit(X2_full, y2)
        auc2_full = roc_auc_score(y2, log2_full.predict_proba(X2_full)[:, 1])
        results["logistic_active_set_change"] = dict(auc_magnitude_only=float(auc2_mag),
                                                       auc_full_features=float(auc2_full),
                                                       n=len(df2))
    else:
        results["logistic_active_set_change"] = dict(note="insufficient class balance for logistic diagnostic")

    return results
