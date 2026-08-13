"""
Stage 3: Dual-Weighted Constraint Rollout Learning -- central config.

Reuses stage-1's MPC cost/bounds/horizon (src/config.py: dt=0.1, horizon=18,
pos_bounds=(-2,2), input_bounds=(-0.5,0.5)) as the internal model used by
the NOMINAL/NOISY data-collection MPC and by the learned-dynamics
closed-loop MPC. Reuses stage-2's TRUE nonlinear plant
(stage2_margin_weighting/generate_data.py: true_step) unchanged.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DataConfig:
    seed: int = 20260813
    n_traj_nominal: int = 160     # closed-loop trajectories under nominal MPC
    n_traj_noisy: int = 160       # closed-loop trajectories under noisy MPC
    traj_len: int = 30            # steps per trajectory
    action_noise_std: float = 0.15  # additive noise on applied control for "noisy MPC"
    # Reference sampling mixture (mirrors stage 1): some references pushed
    # beyond the feasible boundary so trajectories naturally cover far/near/
    # active constraint regions.
    ref_pos_range: Tuple[float, float] = (-2.3, 2.3)
    init_pos_range: Tuple[float, float] = (-1.6, 1.6)
    init_vel_range: Tuple[float, float] = (-0.6, 0.6)
    train_frac: float = 0.8
    val_frac: float = 0.1
    # test_frac is the remainder


@dataclass
class ModelConfig:
    hidden_dims: Tuple[int, ...] = (256, 256, 128)
    activation: str = "silu"


@dataclass
class TrainConfig:
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 10
    lr: float = 1e-3
    grad_clip_norm: float = 10.0       # clip gradients (rollout methods can explode)
    weight_min: float = 0.5
    weight_max: float = 5.0
    margin_scale_tau: float = 0.2       # tau in exp(-margin/tau)
    margin_alpha: float = 4.0           # alpha_m
    dual_alpha: float = 4.0             # alpha_lambda
    dual_eps: float = 1e-3              # epsilon in lambda / (lambda_bar + eps)
    H: int = 10                          # rollout horizon for constraint_rollout methods
    gamma: float = 0.9                   # discount for rollout constraint loss
    beta: float = 1.0                    # weight of constraint-rollout loss term
    beta_grid: Tuple[float, ...] = (0.1, 0.5, 1.0, 2.0)  # for validation-only beta tuning
    nan_abort: bool = True


@dataclass
class MPCEvalConfig:
    n_episodes: int = 50          # >= 50 per method per seed, per task spec
    episode_len: int = 40
    episode_seed: int = 777
    disturbance_std: float = 0.01


@dataclass
class Stage3Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    mpc_eval: MPCEvalConfig = field(default_factory=MPCEvalConfig)
    seeds: Tuple[int, ...] = (101, 202, 303)

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)
