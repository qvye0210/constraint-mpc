"""
Near-constraint / far-from-constraint grouping of oracle scenarios.

Two complementary, pre-registered definitions:
  1. Fixed thresholds on oracle_min_state_margin, chosen before examining
     any decision-impact results (see config.GroupingConfig).
  2. Quantile-based groups: nearest 30% / farthest 30% of the oracle
     scenario margin distribution, excluding the middle 40%.

Grouping is defined ONLY from the oracle solution's minimum state-constraint
margin -- never from the perturbed solution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GroupingConfig


def add_fixed_threshold_group(oracle_df: pd.DataFrame, cfg: GroupingConfig) -> pd.DataFrame:
    df = oracle_df.copy()
    margin = df["oracle_min_state_margin"]
    conditions = [margin <= cfg.near_margin_threshold, margin >= cfg.far_margin_threshold]
    choices = ["near", "far"]
    df["group_fixed"] = np.select(conditions, choices, default="mid")
    return df


def add_quantile_group(oracle_df: pd.DataFrame, cfg: GroupingConfig) -> pd.DataFrame:
    df = oracle_df.copy()
    margin = df["oracle_min_state_margin"]
    q_near = margin.quantile(cfg.near_quantile)
    q_far = margin.quantile(cfg.far_quantile)
    conditions = [margin <= q_near, margin >= q_far]
    choices = ["near", "far"]
    df["group_quantile"] = np.select(conditions, choices, default="mid")
    df.attrs["quantile_near_threshold"] = float(q_near)
    df.attrs["quantile_far_threshold"] = float(q_far)
    return df


def group_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (df.groupby(group_col)["oracle_min_state_margin"]
              .agg(["count", "mean", "median", "std", "min", "max"])
              .reset_index())
