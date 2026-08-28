#!/usr/bin/env python3
"""Where does the error floor actually come from?

`capacity_check.py` reported an error floor, but follow-up numbers were unstable
in a way a genuine data-limited regime would not be: test MSE ranged from 2.2e-3
to 3.6e-2 across configurations, the train/test gap ran from 8.5x to 235x, and
quadrupling the data left the gap unchanged (75.6 -> 78.4) while test error got
slightly worse. A fixed payload gave the same test error as a random one
(3.59e-2 vs 3.56e-2), so the unobserved payload is not the cause either.

The remaining suspect is COVERAGE. With a 12-dimensional state and a few thousand
samples drawn from widely-spread start configurations, held-out trajectories may
simply visit regions the training set never saw. That produces an error floor
whose source is extrapolation, not scarce capacity -- and extrapolation error
gives a decision-aware loss nothing to re-allocate, exactly like unbiased noise.
Getting this wrong is what made the whole planar testbed campaign meaningless, so
it is worth an hour to settle.

Three probes:

  A  trajectory split vs transition split
     A transition split leaks correlation between neighbours and so is NOT a
     valid way to report results -- but the comparison is diagnostic. If the
     transition split has a far lower test error, the trajectory-split error is
     dominated by distribution shift, not by model capacity.

  B  nearest-neighbour distance from each test state to the training set
     Directly measures coverage. If test states sit far outside the training
     cloud, the floor is extrapolation.

  C  error vs neighbour distance
     If error grows sharply with distance to the nearest training point, the
     floor is coverage. If it is flat, the model is genuinely capacity-limited.

    python diagnose_floor.py                      # ~15 min
    python diagnose_floor.py --n-traj 40 --epochs 500     # faster
"""

import argparse
import json
import os

import numpy as np

from urmj.data import DEFAULT_OBS, collect
from urmj.model import evaluate, train
from urmj.plant import NQ, NX, UR5ePlant, f_nominal


def pack(trajs):
    d = {k: np.concatenate([t[k] for t in trajs]) for k in ("X", "U", "Xn", "g")}
    d["R"] = d["Xn"] - f_nominal(d["X"], d["U"])
    d["margin"] = -d["g"]
    return d


def subset(d, idx):
    return {k: v[idx] for k, v in d.items()}


def nn_distance(A, B, scale=None):
    """Distance from every row of A to its nearest row in B, in a scaled space."""
    if scale is None:
        scale = B.std(0) + 1e-8
    a, b = A / scale, B / scale
    out = np.empty(len(a))
    step = 512
    for i in range(0, len(a), step):
        chunk = a[i:i + step]
        d2 = ((chunk[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        out[i:i + step] = np.sqrt(d2.min(1))
    return out, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-traj", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", default="64,64")
    ap.add_argument("--out", default="results/floor")
    a = ap.parse_args()

    hidden = tuple(int(h) for h in a.hidden.split(","))
    os.makedirs(a.out, exist_ok=True)
    plant = UR5ePlant()
    M0 = lambda n: np.tile(np.eye(NX), (n, 1, 1))

    print(f"collecting {a.n_traj} trajectories ...")
    trajs = collect(a.n_traj, seed=a.seed)
    n_tr = int(0.8 * len(trajs))

    # ---- probe A -------------------------------------------------------
    print("\n=== A. trajectory split vs transition split ===")
    A_tr, A_te = pack(trajs[:n_tr]), pack(trajs[n_tr:])
    allp = pack(trajs)
    n = len(allp["X"])
    idx = np.random.default_rng(a.seed).permutation(n)
    B_tr, B_te = subset(allp, idx[:int(0.8 * n)]), subset(allp, idx[int(0.8 * n):])

    print(f"  {'split':>12}{'train MSE':>12}{'test MSE':>12}{'gap':>8}")
    out = {}
    models = {}
    for lbl, tr, te in (("trajectory", A_tr, A_te), ("transition", B_tr, B_te)):
        m = train(tr, M0(len(tr["X"])), hidden=hidden, epochs=a.epochs, seed=a.seed)
        models[lbl] = (m, tr, te)
        e_tr = evaluate(m, tr, plant, DEFAULT_OBS)["mse"]
        e_te = evaluate(m, te, plant, DEFAULT_OBS)["mse"]
        out[lbl] = dict(train=e_tr, test=e_te, gap=e_te / e_tr)
        print(f"  {lbl:>12}{e_tr:>12.3e}{e_te:>12.3e}{e_te / e_tr:>8.1f}")

    ratio = out["trajectory"]["test"] / out["transition"]["test"]
    print(f"\n  trajectory-split test error is {ratio:.1f}x the transition-split one")
    print("  (>5x means the floor is distribution shift, not capacity)")

    # ---- probe B -------------------------------------------------------
    print("\n=== B. coverage: distance from test states to the training set ===")
    d_tr_self, scale = nn_distance(A_tr["X"][::5], A_tr["X"][::5])
    d_te, _ = nn_distance(A_te["X"][::3], A_tr["X"][::5], scale)
    print(f"  train-to-train NN distance : median {np.median(d_tr_self):.3f}")
    print(f"  test-to-train  NN distance : median {np.median(d_te):.3f}  "
          f"p90 {np.quantile(d_te, 0.9):.3f}")
    cov_ratio = float(np.median(d_te) / max(np.median(d_tr_self), 1e-9))
    print(f"  ratio {cov_ratio:.1f}x  (>3x means test states sit outside the "
          f"training cloud)")

    # ---- probe C -------------------------------------------------------
    print("\n=== C. does error grow with distance to the training set? ===")
    m, tr, te = models["trajectory"]
    import torch
    with torch.no_grad():
        pred = m(torch.tensor(te["X"].astype(np.float32)),
                 torch.tensor(te["U"].astype(np.float32))).numpy()
    err = ((pred - te["R"]) ** 2).sum(1)
    d_all, _ = nn_distance(te["X"], A_tr["X"][::5], scale)
    q = np.quantile(d_all, [0, .25, .5, .75, 1.0])
    print(f"  {'NN-distance bin':>20}{'n':>7}{'mean MSE':>12}")
    bins = []
    for i in range(4):
        sel = (d_all >= q[i]) & (d_all <= q[i + 1])
        if sel.sum():
            bins.append(float(err[sel].mean()))
            print(f"  {f'{q[i]:.2f}-{q[i+1]:.2f}':>20}{int(sel.sum()):>7}"
                  f"{err[sel].mean():>12.3e}")
    slope = bins[-1] / max(bins[0], 1e-20) if len(bins) >= 2 else float("nan")
    print(f"\n  farthest/nearest quartile error ratio = {slope:.1f}")
    print("  (>5x means the floor is coverage; ~1x means it is genuinely capacity)")

    verdict = dict(
        traj_over_transition=float(ratio),
        coverage_ratio=cov_ratio,
        error_distance_slope=float(slope),
        splits=out,
        coverage_dominates=bool(ratio > 5 or slope > 5))
    print("\n" + "=" * 70)
    print(">>> " + (
        "COVERAGE DOMINATES. The floor is extrapolation, not scarce capacity, so "
        "it gives a decision-aware loss nothing to re-allocate. Narrow the start "
        "distribution / workspace so trajectories densely cover a compact region, "
        "then re-run capacity_check.py. Do NOT run compare_arms.py yet."
        if verdict["coverage_dominates"] else
        "CAPACITY-LIMITED. The floor survives with good coverage, so the regime "
        "supports the weighting experiment. Run compare_arms.py."))
    with open(f"{a.out}/verdict.json", "w") as f:
        json.dump(dict(verdict=verdict, config=vars(a)), f, indent=2, default=float)
    print(f"wrote {a.out}/verdict.json")


if __name__ == "__main__":
    main()
