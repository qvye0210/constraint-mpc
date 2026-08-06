"""
Synthetic scenario sampling: initial states and references.

Scenarios are sampled with a fixed seed for reproducibility. Feasibility
filtering (oracle MPC must solve to OPTIMAL) happens in the dataset
generation step, not here -- this module only proposes candidate scenarios.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import SamplingConfig


@dataclass
class Scenario:
    scenario_id: int
    init_pos: float
    init_vel: float
    ref_pos: float
    ref_vel: float


def sample_scenarios(cfg: SamplingConfig, n_candidates: int) -> list[Scenario]:
    """Deterministically sample n_candidates scenarios using cfg.seed."""
    rng = np.random.default_rng(cfg.seed)
    scenarios = []
    for i in range(n_candidates):
        p0 = rng.uniform(*cfg.init_pos_range)
        v0 = rng.uniform(*cfg.init_vel_range)
        pref = rng.uniform(*cfg.ref_pos_range)
        vref = rng.uniform(*cfg.ref_vel_range) if cfg.sample_ref_vel else cfg.ref_vel_range[0]
        scenarios.append(Scenario(scenario_id=i, init_pos=p0, init_vel=v0,
                                   ref_pos=pref, ref_vel=vref))
    return scenarios
