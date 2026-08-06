"""
Central experiment configuration.

All numerical settings that define the experiment live here so that the
full configuration can be serialized to JSON and archived alongside
results for reproducibility.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, asdict
from typing import List, Tuple


@dataclass
class DynamicsConfig:
    dt: float = 0.1  # sampling interval [s]


@dataclass
class MPCConfig:
    horizon: int = 18  # N, number of control intervals

    # Cost weights
    q_pos: float = 10.0      # position tracking weight
    q_vel: float = 1.0       # velocity tracking weight
    r_u: float = 0.5         # control effort weight
    r_du: float = 0.05       # control smoothness (delta-u) weight
    q_pos_terminal: float = 20.0  # terminal position weight
    q_vel_terminal: float = 2.0   # terminal velocity weight

    # Hard constraints
    pos_bounds: Tuple[float, float] = (-2.0, 2.0)
    vel_bounds: Tuple[float, float] = (-1.0, 1.0)
    input_bounds: Tuple[float, float] = (-0.5, 0.5)

    # Reference velocity is held at zero unless sampled otherwise
    default_ref_vel: float = 0.0


@dataclass
class BiasConfig:
    # Bias magnitudes (Euclidean norm of the 2D additive bias vector
    # [bias_p, bias_v] applied to the one-step prediction).
    # Chosen small relative to state bounds (pos range 4.0, vel range 2.0)
    # and relative to one dt-step of nominal dynamics motion, so that the
    # perturbation studies *local* sensitivity, not catastrophic failure.
    magnitudes: Tuple[float, ...] = (0.005, 0.015, 0.03)

    # Named structured directions (unit vectors in [pos, vel] bias space)
    # plus a set of random unit directions generated at runtime.
    n_random_directions: int = 4
    random_direction_seed: int = 20260730


@dataclass
class SamplingConfig:
    n_scenarios_full: int = 600       # target scenarios for the full run
    n_scenarios_smoke: int = 50       # smoke-test scenario count
    seed: int = 42

    # Sampling ranges for initial state and reference
    init_pos_range: Tuple[float, float] = (-1.6, 1.6)
    init_vel_range: Tuple[float, float] = (-0.6, 0.6)
    # Reference position range intentionally extends slightly beyond the
    # feasible position bound (+/-2.0) so the oracle solution naturally
    # produces both near-constraint and far-from-constraint trajectories.
    ref_pos_range: Tuple[float, float] = (-2.3, 2.3)
    ref_vel_range: Tuple[float, float] = (0.0, 0.0)  # kept at 0 by default
    sample_ref_vel: bool = False

    max_attempts_factor: int = 6  # try up to factor*N draws to hit target feasible count


@dataclass
class ToleranceConfig:
    # Numerical tolerances for identifying "active" constraints.
    active_slack_tol: float = 1e-3     # slack below this => "tight"
    active_dual_tol: float = 1e-6      # dual value above this => "binding"
    solver_feas_tol: float = 1e-4      # primal residual tolerance for feasibility acceptance


@dataclass
class GroupingConfig:
    # Fixed-threshold definition (chosen BEFORE examining decision-impact
    # results): margins measured in position units (state constraint slack).
    near_margin_threshold: float = 0.08
    far_margin_threshold: float = 0.30

    # Quantile-based grouping
    near_quantile: float = 0.30
    far_quantile: float = 0.70  # scenarios ABOVE this quantile of margin are "far"


@dataclass
class ExperimentConfig:
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)
    bias: BiasConfig = field(default_factory=BiasConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    tolerance: ToleranceConfig = field(default_factory=ToleranceConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    solver_preference: Tuple[str, ...] = ("OSQP", "CLARABEL", "SCS")

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)

    @staticmethod
    def from_json(path: str) -> "ExperimentConfig":
        with open(path) as f:
            d = json.load(f)
        return ExperimentConfig(
            dynamics=DynamicsConfig(**d["dynamics"]),
            mpc=MPCConfig(**d["mpc"]),
            bias=BiasConfig(**d["bias"]),
            sampling=SamplingConfig(**d["sampling"]),
            tolerance=ToleranceConfig(**d["tolerance"]),
            grouping=GroupingConfig(**d["grouping"]),
            solver_preference=tuple(d["solver_preference"]),
        )
