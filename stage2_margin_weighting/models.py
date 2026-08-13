"""
Dynamics model: MLP predicting state increment [dp, dv] from [p, v, u].

Identical architecture, initialization scheme, and normalization procedure
are used for ALL THREE methods (Baseline / Random-weight / Margin-weighted)
-- only the training loss weighting differs. This module is shared so there
is no possibility of accidental architectural drift between methods.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class DynamicsMLP(nn.Module):
    def __init__(self, hidden: int = 32, activation: str = "silu"):
        super().__init__()
        act = nn.SiLU if activation == "silu" else nn.ReLU
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            act(),
            nn.Linear(hidden, hidden),
            act(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


class Normalizer:
    """Standardization using TRAIN-SET statistics only."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.std = torch.clamp(self.std, min=1e-6)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    @staticmethod
    def fit(x: np.ndarray) -> "Normalizer":
        return Normalizer(mean=x.mean(axis=0), std=x.std(axis=0))

    def state_dict(self):
        return dict(mean=self.mean.numpy().tolist(), std=self.std.numpy().tolist())

    @staticmethod
    def from_state_dict(d):
        return Normalizer(mean=np.array(d["mean"]), std=np.array(d["std"]))


def build_model(seed: int, hidden: int = 32, activation: str = "silu") -> DynamicsMLP:
    """Deterministic initialization given a seed, so all three methods start
    from IDENTICAL initial weights for a given seed."""
    torch.manual_seed(seed)
    return DynamicsMLP(hidden=hidden, activation=activation)


def predict_next_state(model: DynamicsMLP, in_normalizer: Normalizer, out_normalizer: Normalizer,
                        p: torch.Tensor, v: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Given raw (unnormalized) p, v, u tensors (any leading batch shape),
    return predicted raw (p_next, v_next). The model predicts the
    normalized state increment; we denormalize and add to the raw state."""
    x = torch.stack([p, v, u], dim=-1)
    x_n = in_normalizer.normalize(x)
    dy_n = model(x_n)
    dy = out_normalizer.denormalize(dy_n)
    p_next = p + dy[..., 0]
    v_next = v + dy[..., 1]
    return p_next, v_next
