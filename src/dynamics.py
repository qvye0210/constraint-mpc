"""
Discrete-time 1D double-integrator dynamics.

State x = [p, v] (position, velocity). Control u = a (acceleration).

Standard discrete double integrator (exact zero-order-hold discretization
of the continuous double integrator):

    p_{k+1} = p_k + dt * v_k + 0.5 * dt^2 * u_k
    v_{k+1} = v_k + dt * u_k

    x_{k+1} = A x_k + B u_k

    A = [[1, dt], [0, 1]]
    B = [[0.5*dt^2], [dt]]
"""
from __future__ import annotations

import numpy as np


def get_double_integrator_matrices(dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (A, B) for the discrete double integrator with sample time dt."""
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.5 * dt ** 2], [dt]])
    return A, B


def true_step(x: np.ndarray, u: float, dt: float) -> np.ndarray:
    """Propagate true (unbiased) dynamics one step. x = [p, v], scalar u."""
    A, B = get_double_integrator_matrices(dt)
    x = np.asarray(x, dtype=float).reshape(2)
    return A @ x + B.flatten() * u


def biased_step(x: np.ndarray, u: float, dt: float, bias: np.ndarray) -> np.ndarray:
    """Propagate dynamics with an additive constant prediction bias.

    bias is a 2-vector [bias_p, bias_v] added after the nominal linear
    prediction step. This represents a small constant model mismatch
    (e.g. unmodeled friction/offset) of controlled norm.
    """
    x_next = true_step(x, u, dt)
    return x_next + np.asarray(bias, dtype=float).reshape(2)


def rollout(x0: np.ndarray, u_seq: np.ndarray, dt: float,
            bias: np.ndarray | None = None) -> np.ndarray:
    """Roll out a control sequence under true (bias=None) or biased dynamics.

    Returns array of shape (len(u_seq)+1, 2) including x0.
    """
    x0 = np.asarray(x0, dtype=float).reshape(2)
    u_seq = np.asarray(u_seq, dtype=float).flatten()
    N = len(u_seq)
    xs = np.zeros((N + 1, 2))
    xs[0] = x0
    for k in range(N):
        if bias is None:
            xs[k + 1] = true_step(xs[k], u_seq[k], dt)
        else:
            xs[k + 1] = biased_step(xs[k], u_seq[k], dt, bias)
    return xs
