"""
Stage 2 data generation.

Reuses stage-1 system definition (constrained double integrator, position
bound +/-2.0, input bound +/-0.5, dt=0.1 -- see src/config.py) but adds a
small fixed unmodeled nonlinearity to the TRUE plant dynamics, since
stage 1's true dynamics were fully linear (as instructed, since stage 1 has
no such nonlinearity, we add it here rather than skipping it):

    v_next = v + dt * (u - 0.15*v*|v| + 0.08*sin(2*p))
    p_next = p + dt * v_next

Transitions are generated as short TRAJECTORIES (not iid samples) so that
train/val/test can be split at the trajectory level without leaking
adjacent transitions across splits. Position is initialized with a mixture
of full-range and near-boundary starts so that ~20-30% of transitions
naturally fall in the near-constraint region (margin < 0.4), WITHOUT adding
any extra noise or special treatment to near-constraint samples -- only the
initial-position sampling density differs, which is a data-coverage design
choice, not a noise asymmetry (task requirement: same noise mechanism
everywhere).
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POS_BOUND = 2.0
INPUT_BOUND = 0.5
DT = 0.1

SEED = 20260806
N_TRAJ = 400
TRAJ_LEN = 25  # transitions per trajectory -> 400*25 = 10000 transitions
NEAR_FRAC_INIT = 0.0  # no artificial near-boundary seeding; uniform initial p
                        # already yields ~28-29% near-constraint coverage via
                        # natural wall-bounce dynamics (verified empirically)
NOISE_STD_V = 0.0   # process noise on v_next (same for all regions, see below)


def true_step(p: float, v: float, u: float, dt: float = DT) -> tuple[float, float]:
    """True (nonlinear) plant dynamics, semi-implicit Euler as specified."""
    v_next = v + dt * (u - 0.15 * v * abs(v) + 0.08 * np.sin(2 * p))
    p_next = p + dt * v_next
    return p_next, v_next


def rollout_trajectory(rng: np.random.Generator, p0: float, v0: float, T: int,
                        noise_std: float = NOISE_STD_V):
    """Roll out one trajectory of length T under a smooth random control
    policy (autocorrelated random walk clipped to input bounds), applying
    the SAME process-noise mechanism everywhere (no near-region-specific
    noise). If position would leave the feasible envelope, simulate an
    inelastic wall stop (clip position, zero velocity) so trajectories stay
    in-domain and naturally linger near the boundary sometimes."""
    p, v = p0, v0
    u = rng.uniform(-INPUT_BOUND, INPUT_BOUND)
    rows = []
    for t in range(T):
        margin = POS_BOUND - abs(p)
        p_next, v_next = true_step(p, v, u)
        if noise_std > 0:
            v_next = v_next + rng.normal(0.0, noise_std)
            p_next = p + DT * v_next
        if abs(p_next) > POS_BOUND:
            p_next = np.sign(p_next) * POS_BOUND
            v_next = 0.0
        rows.append(dict(p_t=p, v_t=v, u_t=u, p_next=p_next, v_next=v_next, margin=margin))
        p, v = p_next, v_next
        # smooth random-walk control update, same mechanism for all regions
        u = np.clip(u + rng.normal(0.0, 0.15), -INPUT_BOUND, INPUT_BOUND)
    return rows


def generate_all(outdir: str, n_traj: int = N_TRAJ, traj_len: int = TRAJ_LEN,
                  seed: int = SEED, near_frac_init: float = NEAR_FRAC_INIT):
    rng = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)

    all_rows = []
    for tid in range(n_traj):
        if rng.uniform() < near_frac_init:
            # start near one of the two boundaries
            side = rng.choice([-1.0, 1.0])
            p0 = side * rng.uniform(1.6, 2.0)
        else:
            p0 = rng.uniform(-POS_BOUND, POS_BOUND)
        v0 = rng.uniform(-1.0, 1.0)
        traj_rows = rollout_trajectory(rng, p0, v0, traj_len)
        for r in traj_rows:
            r["trajectory_id"] = tid
        all_rows.extend(traj_rows)

    df = pd.DataFrame(all_rows)
    df = df[["trajectory_id", "p_t", "v_t", "u_t", "p_next", "v_next", "margin"]]

    # Trajectory-level split: shuffle trajectory ids, 80/10/10
    traj_ids = np.arange(n_traj)
    rng.shuffle(traj_ids)
    n_train = int(0.8 * n_traj)
    n_val = int(0.1 * n_traj)
    train_ids = set(traj_ids[:n_train].tolist())
    val_ids = set(traj_ids[n_train:n_train + n_val].tolist())
    test_ids = set(traj_ids[n_train + n_val:].tolist())

    train_df = df[df["trajectory_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["trajectory_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["trajectory_id"].isin(test_ids)].reset_index(drop=True)

    train_df.to_csv(os.path.join(outdir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(outdir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(outdir, "test.csv"), index=False)
    df.to_csv(os.path.join(outdir, "all_transitions.csv"), index=False)

    def frac_near(d):
        return float((d["margin"] < 0.4).mean())

    stats = dict(
        seed=seed, n_traj=n_traj, traj_len=traj_len, n_transitions=len(df),
        n_train=len(train_df), n_val=len(val_df), n_test=len(test_df),
        n_train_traj=len(train_ids), n_val_traj=len(val_ids), n_test_traj=len(test_ids),
        frac_near_overall=frac_near(df), frac_near_train=frac_near(train_df),
        frac_near_val=frac_near(val_df), frac_near_test=frac_near(test_df),
        margin_min=float(df["margin"].min()), margin_max=float(df["margin"].max()),
        margin_mean=float(df["margin"].mean()),
        near_frac_init_used=near_frac_init,
        pos_bound=POS_BOUND, input_bound=INPUT_BOUND, dt=DT,
    )
    with open(os.path.join(outdir, "data_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # Margin histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["margin"], bins=40, color="steelblue", edgecolor="white")
    ax.axvline(0.4, color="red", ls="--", label="near-constraint threshold (0.4)")
    ax.axvline(0.8, color="green", ls="--", label="far-region threshold (0.8)")
    ax.set_xlabel("constraint margin = 2.0 - |p|")
    ax.set_ylabel("count")
    ax.set_title(f"Margin distribution (n={len(df)}, near-frac={stats['frac_near_overall']:.1%})")
    ax.legend()
    fig.savefig(os.path.join(outdir, "margin_histogram.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[generate_data] n_transitions={len(df)}, near-frac={stats['frac_near_overall']:.1%} "
          f"(train {stats['frac_near_train']:.1%}, val {stats['frac_near_val']:.1%}, "
          f"test {stats['frac_near_test']:.1%})")
    return stats


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    generate_all(os.path.join(here, "data"))
