"""
Residual MLP dynamics model: x_next = x_t + f_theta(x_t, u_t).

Shared identically (architecture, init scheme, normalization procedure)
across all four training methods -- only the loss differs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class ResidualMLP(nn.Module):
    def __init__(self, hidden_dims=(256, 256, 128), activation: str = "silu"):
        super().__init__()
        act = nn.SiLU if activation == "silu" else nn.ReLU
        dims = [3] + list(hidden_dims) + [2]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x: (..., 3) = [p, v, u] (RAW, unnormalized). Internally normalizes
        with a Normalizer passed via `set_normalizers`; call `predict` for
        the full raw-in/raw-out convenience path."""
        return self.net(x)


class Normalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.clamp(torch.tensor(std, dtype=torch.float32), min=1e-6)

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


def build_model(seed: int, hidden_dims=(256, 256, 128), activation: str = "silu") -> ResidualMLP:
    """Deterministic init given a seed -- identical initial weights across
    all four methods for a given seed."""
    torch.manual_seed(seed)
    return ResidualMLP(hidden_dims=hidden_dims, activation=activation)


def predict_next_state(model: ResidualMLP, in_norm: Normalizer, out_norm: Normalizer,
                        p: torch.Tensor, v: torch.Tensor, u: torch.Tensor):
    """One residual step: raw (p, v, u) -> raw (p_next, v_next). Fully
    differentiable end-to-end (used for multi-step rollout)."""
    x = torch.stack([p, v, u], dim=-1)
    xn = in_norm.normalize(x)
    dy_n = model(xn)
    dy = out_norm.denormalize(dy_n)
    p_next = p + dy[..., 0]
    v_next = v + dy[..., 1]
    return p_next, v_next


def constraint_value(p: torch.Tensor, pos_bound: float = 2.0) -> torch.Tensor:
    """g(x) = margin = pos_bound - |p|. Positive when feasible, negative
    when violating. Differentiable a.e. (subgradient at p=0 is fine)."""
    return pos_bound - torch.abs(p)
