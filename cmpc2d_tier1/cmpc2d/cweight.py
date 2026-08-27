"""Propagated constraint-gradient weighting.

Implements the corrected method from EXPERIMENT_SPEC.md.

The weight metric for a one-step error e_t is

    M_t = sum_k gamma^{k-1} c_k c_k^T ,     c_k = (A^{k-1})^T grad g(x_{t+k})

For the point mass, A is block [[I, dt I], [0, I]] on the core state and block
diagonal w.r.t. the distractors, so

    c_k = [ n_k ,  (k-1) dt n_k ,  0...0 ]          n_k = grad_p g(x_{t+k})

i.e. distractor dimensions receive EXACTLY zero weight at every k -- the
mechanism the earlier runs never tested -- and velocity receives weight only
through propagation, which is why the static k=1 form was degenerate.

--- a note on rank protection -------------------------------------------------
VaGraM protects against the rank deficiency of c c^T with a Cauchy-Schwarz bound
that yields a diagonal, positive-definite matrix, and the authors note this
makes the scaling axis-aligned rather than rotated to follow the value function.

That fix cannot be inherited here. Our hypothesis is precisely about direction
WITHIN the position subspace, and n_k rotates along the horizon; an axis-aligned
diagonal keeps only the componentwise squares and destroys exactly the structure
under test. We therefore use M + eps*I as the primary form and keep the diagonal
version as a separate arm, since it doubles as a test of that claim.
"""

import numpy as np
import torch

from .env import NU, NX, Params, normal_dir, tangent_dir


def build_metric(win_X, p_obs, n_dist=0, H=10, gamma=0.9, mode="prop",
                 eps_floor=0.05, clip_q=0.95, params=Params, seed=0,
                 strength=1.0):
    """Per-sample weight metric M of shape (N, nx_aug, nx_aug).

    mode:
      'uniform'   identity
      'static'    k=1 only (no propagation) -- the degenerate form
      'prop'      full propagated metric  (primary)
      'diag'      VaGraM-style axis-aligned diagonal bound
      'mask'      zero on irrelevant dims, UNIFORM on core dims.  Isolates
                  "stop wasting capacity" from "allocate by direction": it has
                  the masking but none of the directional structure.
      'random'    same eigenvalue spectrum as 'prop', random orientation

    strength: 1.0 = the metric as derived; <1 interpolates toward isotropic,
    keeping the trace fixed.  Used to compare arms at MATCHED anisotropy so that
    "weighting strength" can be ruled out as the active variable.
    """
    N = len(win_X)
    nx = NX + n_dist
    dt = params.dt

    if mode == "uniform":
        M = np.tile(np.eye(nx), (N, 1, 1))
        M = _finalise(M, clip_q)
        return M, M.copy()

    if mode == "mask":
        core = np.zeros(nx); core[:NX] = 1.0
        M = np.tile(np.diag(core), (N, 1, 1))
        Ms = _finalise(M.copy(), clip_q)
        floor = eps_floor / nx
        return _finalise(M + floor * np.eye(nx), clip_q), Ms

    K = 1 if mode == "static" else H
    M = np.zeros((N, nx, nx))
    for k in range(1, K + 1):
        xk = win_X[:, k - 1]                      # true state at t+k
        n_k = normal_dir(xk, p_obs)               # (N,2), grad_p g
        c = np.zeros((N, nx))
        c[:, :2] = n_k
        c[:, 2:4] = (k - 1) * dt * n_k            # propagation through A
        # distractor block stays exactly zero
        M += (gamma ** (k - 1)) * c[:, :, None] * c[:, None, :]

    if mode == "diag":
        d = np.einsum("bii->bi", M).copy()        # keep only the diagonal
        M = np.zeros_like(M)
        idx = np.arange(nx)
        M[:, idx, idx] = d
    elif mode == "random":
        w, V = np.linalg.eigh(M)                  # same spectrum, new basis
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(N, nx, nx))
        Q, _ = np.linalg.qr(A)
        M = Q @ (w[:, :, None] * np.transpose(Q, (0, 2, 1)))

    if strength != 1.0:
        # Interpolate toward isotropic at fixed trace, but ONLY WITHIN THE
        # RELEVANT SUBSPACE.  Interpolating over the full space would put weight
        # back on the irrelevant dimensions and silently undo the masking, which
        # would confound the very comparison this knob exists for.
        tr = np.trace(M, axis1=1, axis2=2)[:, None, None]
        core = np.zeros(nx); core[:NX] = 1.0
        iso = tr / NX * np.diag(core)
        M = strength * M + (1.0 - strength) * iso

    # rank protection: keep the metric positive definite.  NOTE this floor puts
    # a small uniform weight on EVERY dimension, distractors included, so the
    # structural (floor-free) weight is what must be checked for zero.
    floor = eps_floor * np.trace(M, axis1=1, axis2=2).mean() / nx
    M_struct = _finalise(M.copy(), clip_q)
    M = _finalise(M + floor * np.eye(nx), clip_q)
    return M, M_struct


def _finalise(M, clip_q):
    """Clip heavy-tailed samples, then normalise to mean trace 1.

    Clipping follows VaGraM, who clip value-gradient norms at the 95th
    percentile because empirical gradients can spike and destabilise training.
    grad g spikes the same way near the obstacle.  Normalisation to mean trace 1
    matches the loss SCALE across arms, so that Adam sees comparable step sizes
    and 'better allocation' cannot be confused with 'different effective
    learning rate' -- the largest uncontrolled variable in the earlier runs.
    """
    tr = np.trace(M, axis1=1, axis2=2)
    cap = np.quantile(tr, clip_q)
    scale = np.minimum(1.0, cap / np.maximum(tr, 1e-12))
    M = M * scale[:, None, None]
    tr = np.trace(M, axis1=1, axis2=2)
    return M / max(tr.mean(), 1e-12)


# ---------------------------------------------------------------------------
def weight_report(M, X, p_obs, n_dist=0):
    """Diagnostics required by the spec (and by the supervisor's original ask).

    Distractor weight must be ~0 for the propagated metric; a non-zero value
    means the implementation is still wrong.
    """
    nx = NX + n_dist
    d = np.einsum("bii->bi", M)
    out = dict(
        w_position=float(d[:, :2].sum(-1).mean()),
        w_velocity=float(d[:, 2:4].sum(-1).mean()),
        w_distract=float(d[:, 4:].sum(-1).mean()) if n_dist else 0.0,
        trace_mean=float(np.trace(M, axis1=1, axis2=2).mean()),
        trace_p95_over_median=float(
            np.quantile(np.trace(M, axis1=1, axis2=2), 0.95)
            / max(np.median(np.trace(M, axis1=1, axis2=2)), 1e-12)),
    )
    n = normal_dir(X, p_obs)
    t = tangent_dir(X, p_obs)
    en = np.einsum("bi,bij,bj->b", _pad(n, nx), M, _pad(n, nx))
    et = np.einsum("bi,bij,bj->b", _pad(t, nx), M, _pad(t, nx))
    out["normal_over_tangent"] = float((en / np.maximum(et, 1e-12)).mean())
    out["frac_weight_on_irrelevant"] = (
        float(d[:, 4:].sum() / max(d.sum(), 1e-12)) if n_dist else 0.0)
    return out


def _pad(v2, nx):
    out = np.zeros((len(v2), nx))
    out[:, :2] = v2
    return out


def metric_ratio_by_k(win_X, p_obs, H=10, gamma=0.9, params=Params):
    """Normal/tangential weight ratio contributed by each horizon step.

    Should track the measured sin(theta_k) decay; this is the quantitative link
    between the rotation law and the weighting.
    """
    rows = []
    dt = params.dt
    for k in range(1, H + 1):
        xk = win_X[:, k - 1]
        n_k = normal_dir(xk, p_obs)
        n0 = normal_dir(win_X[:, 0], p_obs)
        t0 = tangent_dir(win_X[:, 0], p_obs)
        cn = (n_k * n0).sum(-1) ** 2
        ct = (n_k * t0).sum(-1) ** 2
        rows.append(dict(k=k, weight=gamma ** (k - 1),
                         normal_align=float(cn.mean()),
                         tangent_leak=float(ct.mean()),
                         ratio=float((cn / np.maximum(ct, 1e-12)).mean())))
    return rows


# ---------------------------------------------------------------------------
def metric_loss(err, M_batch):
    """e^T M e, batched."""
    return torch.einsum("bi,bij,bj->b", err, M_batch, err).mean()
