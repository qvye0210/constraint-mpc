"""MPPI over planar eef velocities with the recursive one-step model.

v2 (after the zero-task-progress pilot): plain Gaussian MPPI cannot thread a
3-6cm doorway -- almost every sample clips a pillar, weights collapse onto
"stop short", which is exactly what the pilot measured (crossed 0% at every
width). Changes, all applied identically to every replan policy:
  * nominal proposal: a straight push toward the gate centre (then the goal)
    anchors HALF the samples with small noise;
  * hard feasibility filter: candidates with predicted min clearance < 0 get
    zero weight -- the planner may not knowingly plan a violation;
  * safe stop: if no candidate is predicted feasible, return zero action and
    record solve_failure instead of silently executing garbage.
Tuning boundary (pre-registered): planner knobs may be tuned ONLY against k=1
task success on separate tuning seeds, never against violation rates; frozen
before pilot/formal runs. Determinism: SeedSequence(1234, episode, solve).
"""
import time

import numpy as np

from .env import DT, EEF_VMAX, R_OBJECT
from .model import rollout


class MPPI:
    def __init__(self, model, H=12, N=256, iters=2, sigma=0.08, lam=0.02,
                 w_goal=1.0, w_zone=400.0, soft_margin=0.015, w_u=0.05,
                 push_speed=0.08):
        self.m = model
        self.H, self.N, self.iters = H, N, iters
        self.sigma, self.lam = sigma, lam
        self.w_goal, self.w_zone, self.w_u = w_goal, w_zone, w_u
        self.soft = soft_margin
        self.push_speed = push_speed
        self.n_solves = 0
        self.solve_time = 0.0
        self.plan_rhos = []          # predicted min clearance per chosen plan
        self.pred_crossed = []       # chosen plan predicted to pass the gate
        self.pred_term_prog = []     # predicted terminal progress past the gate
        self.solve_failures = 0

    def solve(self, eef_xy, obj, goal_xy, zone_xy, r_zone, episode_id, solve_idx,
              u_init=None, aux=None):
        t0 = time.perf_counter()
        ss = np.random.SeedSequence([1234, int(episode_id), int(solve_idx)])
        rng = np.random.default_rng(ss)
        Z = np.atleast_2d(zone_xy)

        # nominal proposal: straight push toward gate centre, then the goal
        if aux is not None:
            prog = float((obj[:2] - aux["obj0"]) @ aux["path"])
            tgt = aux["gate_mid"] if prog < aux["mid_s"] + 0.02 else goal_xy
        else:
            tgt = goal_xy
        d = tgt - obj[:2]
        d = d / (np.linalg.norm(d) + 1e-12)
        nominal = np.tile(self.push_speed * d, (self.H, 1))

        mean = nominal.copy() if u_init is None else u_init.copy()
        chosen = None
        for _ in range(self.iters):
            n_half = self.N // 2
            U = np.concatenate([
                np.clip(mean[None] + rng.normal(0, self.sigma,
                                                (n_half, self.H, 2)),
                        -EEF_VMAX, EEF_VMAX),
                np.clip(nominal[None] + rng.normal(0, self.sigma * 0.5,
                                                   (self.N - n_half, self.H, 2)),
                        -EEF_VMAX, EEF_VMAX)])
            S = rollout(self.m, eef_xy, obj, U)
            d_goal = np.linalg.norm(S[:, :, :2] - goal_xy[None, None], axis=-1)
            rho = np.min(np.linalg.norm(S[:, :, None, :2] - Z[None, None],
                                        axis=-1) - (r_zone + R_OBJECT), axis=2)
            pen = np.maximum(0.0, self.soft - rho)
            cost = (self.w_goal * (d_goal.mean(1) + 2 * d_goal[:, -1])
                    + self.w_zone * (pen ** 2).sum(1)
                    + self.w_u * (U ** 2).sum((1, 2)))
            feasible = rho.min(1) > 0.0            # hard filter
            if not feasible.any():
                continue                           # try next iteration
            cost = np.where(feasible, cost, np.inf)
            w = np.exp(-(cost - cost[feasible].min()) / self.lam)
            w[~feasible] = 0.0
            w /= w.sum()
            mean = (w[:, None, None] * U).sum(0)
            chosen = mean

        self.n_solves += 1
        if chosen is None:                         # no feasible candidate found
            self.solve_failures += 1
            self.plan_rhos.append(np.nan)
            self.pred_crossed.append(False)
            self.pred_term_prog.append(np.nan)
            self.solve_time += time.perf_counter() - t0
            return np.zeros((self.H, 2))           # safe stop
        Sm = rollout(self.m, eef_xy, obj, mean[None])[0]
        rho_m = float((np.linalg.norm(Sm[:, None, :2] - Z[None], axis=-1)
                       - (r_zone + R_OBJECT)).min())
        self.plan_rhos.append(rho_m)
        if aux is not None:
            prog_traj = (Sm[:, :2] - aux["obj0"]) @ aux["path"]
            self.pred_crossed.append(bool((prog_traj > aux["mid_s"] + 0.02).any()))
            self.pred_term_prog.append(float(prog_traj[-1] - aux["mid_s"]))
        self.solve_time += time.perf_counter() - t0
        return mean
