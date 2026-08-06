"""
Additive model-bias directions and magnitudes.

The perturbed model uses the same discrete double-integrator dynamics as
the oracle model, but with a constant additive bias vector added to the
one-step state prediction:

    x_{k+1}^perturbed = A x_k + B u_k + bias

The bias vector's Euclidean norm equals the requested magnitude exactly
(bias = magnitude * unit_direction), so that all compared perturbations
share the exact same one-step prediction-error magnitude, isolating
*direction* as the only varying factor at a given magnitude.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class BiasDirection:
    name: str
    unit_vector: np.ndarray  # shape (2,), norm == 1


def structured_directions() -> list[BiasDirection]:
    """Axis-aligned bias directions in [position, velocity] bias space."""
    return [
        BiasDirection("pos_plus", np.array([1.0, 0.0])),
        BiasDirection("pos_minus", np.array([-1.0, 0.0])),
        BiasDirection("vel_plus", np.array([0.0, 1.0])),
        BiasDirection("vel_minus", np.array([0.0, -1.0])),
    ]


def random_directions(n: int, seed: int) -> list[BiasDirection]:
    """n random unit directions in 2D bias space, deterministic given seed."""
    rng = np.random.default_rng(seed)
    dirs = []
    for i in range(n):
        theta = rng.uniform(0, 2 * np.pi)
        v = np.array([np.cos(theta), np.sin(theta)])
        dirs.append(BiasDirection(f"random_{i}", v))
    return dirs


def all_directions(n_random: int, seed: int) -> list[BiasDirection]:
    return structured_directions() + random_directions(n_random, seed)


def make_bias_vector(direction: BiasDirection, magnitude: float) -> np.ndarray:
    """Return bias vector = magnitude * unit_direction. Norm is exactly magnitude."""
    v = direction.unit_vector
    norm = np.linalg.norm(v)
    assert norm > 1e-12, "direction must be a nonzero vector"
    unit = v / norm
    return magnitude * unit
