"""MPPI over planar eef velocities with the recursive one-step model.

Determinism: sampling RNG is seeded by (episode_id, solve_index) -- never the
global RNG -- so every replan policy sees identical solver randomness given the
same solve sequence position.
"""
import numpy as np

from .env import DT, EEF_VMAX, R_OBJECT
from .model import rollout


class MPPI:
    def __init__(self, model, H=12, N=256, iters=2, sigma=0.08, lam=0.02,
                 w_goal=1.0, w_zone=400.0, soft_margin=0.015, w_u=0.05):
        self.m = model
        self.H, self.N, self.iters = H, N, iters
        self.sigma, self.lam = sigma, lam
        self.w_goal, self.w_zone, self.w_u = w_goal, w_zone, w_u
        self.soft = soft_margin
        self.n_solves = 0
        self.solve_time = 0.0

    def solve(self, eef_xy, obj, goal_xy, zone_xy, r_zone, episode_id, solve_idx,
              u_init=None):
        import time
        t0 = time.perf_counter()
        rng = np.random.default_rng((int(episode_id) * 1000003 + int(solve_idx))
                                    & 0x7fffffff)
        mean = np.zeros((self.H, 2)) if u_init is None else u_init.copy()
        for _ in range(self.iters):
            eps = rng.normal(0, self.sigma, (self.N, self.H, 2))
            U = np.clip(mean[None] + eps, -EEF_VMAX, EEF_VMAX)
            S = rollout(self.m, eef_xy, obj, U)
            d_goal = np.linalg.norm(S[:, :, :2] - goal_xy[None, None], axis=-1)
            rho = (np.linalg.norm(S[:, :, :2] - zone_xy[None, None], axis=-1)
                   - (r_zone + R_OBJECT))
            pen = np.maximum(0.0, self.soft - rho)
            cost = (self.w_goal * (d_goal.mean(1) + 2 * d_goal[:, -1])
                    + self.w_zone * (pen ** 2).sum(1)
                    + self.w_u * (U ** 2).sum((1, 2)))
            w = np.exp(-(cost - cost.min()) / self.lam)
            w /= w.sum()
            mean = (w[:, None, None] * U).sum(0)
        self.n_solves += 1
        self.solve_time += time.perf_counter() - t0
        return mean
