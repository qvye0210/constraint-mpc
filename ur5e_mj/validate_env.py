#!/usr/bin/env python3
"""Validate the environment before any method is tested on it.

Two conditions must hold SIMULTANEOUSLY, and they pull against each other:

  COVERAGE   held-out trajectories must visit states the training set has seen.
             Otherwise the test error measures extrapolation, and extrapolation
             error gives a decision-aware loss nothing to re-allocate -- the same
             trap as unbiased noise. Measured by comparing a trajectory split
             against a transition split: a transition split leaks correlation and
             is not a valid way to report results, but if it is far easier, the
             trajectory-split error is dominated by distribution shift.

  FLOOR      test error must stop falling with more training budget while train
             error keeps falling. Otherwise the model fits everything and there
             is nothing scarce to allocate.

             CAVEAT worth reading before trusting a PASS: a floor with a SMALL
             train/test gap can mean the model has converged and the remainder is
             ALEATORIC -- driven by the unobserved payload, which no amount of
             capacity can predict. Unobservable variation gives a weighted loss
             the same optimum as an unweighted one, so it is the same dead end as
             noise. If `gap` is below ~3, run the payload probe below before
             running compare_arms.py:

                 python validate_env.py --sigmas 0.15 --payload-fixed
                 # compare test_mse against the same run without the flag;
                 # if they match, the floor is aleatoric and unusable.

The tension is real: narrowing the start distribution fixes coverage but makes
the task easier, which can remove the floor. This sweeps the start-spread knob
and looks for a setting where both hold.

STOPPING RULE (fixed before running):
    If no setting in the sweep satisfies both, do NOT keep adding difficulty
    knobs one at a time. Two attempts at raising difficulty (friction, network
    size) are allowed; past that, record that this plant cannot produce the
    regime and report the scope result instead.

    python validate_env.py                    # ~20 min
    python validate_env.py --quick            # ~5 min, coarse
"""

import argparse
import json
import os

import numpy as np

from urmj.data import DEFAULT_OBS, collect
from urmj.model import evaluate, train
from urmj.plant import NX, MjParams, UR5ePlant, f_nominal


def pack(trajs):
    d = {k: np.concatenate([t[k] for t in trajs]) for k in ("X", "U", "Xn", "g")}
    d["R"] = d["Xn"] - f_nominal(d["X"], d["U"])
    d["margin"] = -d["g"]
    return d


def nn_distance(A, B, scale=None, exclude_self=False):
    """Nearest-neighbour distance from rows of A to rows of B, scaled per-dim.

    `exclude_self` takes the SECOND nearest when A and B are the same set;
    without it every point matches itself and the distance is identically zero,
    which made the earlier coverage number meaningless.
    """
    if scale is None:
        scale = B.std(0) + 1e-8
    a, b = A / scale, B / scale
    out = np.empty(len(a))
    for i in range(0, len(a), 512):
        d2 = ((a[i:i + 512, None, :] - b[None, :, :]) ** 2).sum(-1)
        if exclude_self:
            d2 = np.sort(d2, axis=1)[:, 1]
        else:
            d2 = d2.min(1)
        out[i:i + 512] = np.sqrt(np.maximum(d2, 0.0))
    return out, scale


def assess(sigma, n_traj, epochs_lo, epochs_hi, hidden, seed, plant, friction=None,
           payload_fixed=False):
    params = MjParams
    if friction is not None:
        class P(MjParams):
            pass
        P.frictionloss = friction
        params = P
    pr = (0.75, 0.75) if payload_fixed else (0.0, 1.5)
    trajs = collect(n_traj, seed=seed, sigma=sigma, params=params, payload_range=pr)
    n_tr = int(0.8 * len(trajs))
    A_tr, A_te = pack(trajs[:n_tr]), pack(trajs[n_tr:])
    allp = pack(trajs)
    n = len(allp["X"])
    idx = np.random.default_rng(seed).permutation(n)
    B_tr = {k: v[idx[:int(0.8 * n)]] for k, v in allp.items()}
    B_te = {k: v[idx[int(0.8 * n):]] for k, v in allp.items()}

    M0 = lambda m: np.tile(np.eye(NX), (m, 1, 1))
    res = {}
    for lbl, tr, te in (("traj", A_tr, A_te), ("trans", B_tr, B_te)):
        m = train(tr, M0(len(tr["X"])), hidden=hidden, epochs=epochs_hi, seed=seed)
        res[lbl] = (evaluate(m, tr, plant, DEFAULT_OBS)["mse"],
                    evaluate(m, te, plant, DEFAULT_OBS)["mse"])
    # floor: same split, two budgets
    m_lo = train(A_tr, M0(len(A_tr["X"])), hidden=hidden, epochs=epochs_lo, seed=seed)
    lo = evaluate(m_lo, A_te, plant, DEFAULT_OBS)["mse"]
    hi = res["traj"][1]

    d_self, scale = nn_distance(A_tr["X"][::5], A_tr["X"][::5], exclude_self=True)
    d_te, _ = nn_distance(A_te["X"][::3], A_tr["X"][::5], scale)
    marg = A_tr["margin"]
    return dict(
        sigma=sigma, n_traj=n_traj,
        coverage_ratio=float(np.median(d_te) / max(np.median(d_self), 1e-9)),
        traj_over_trans=float(res["traj"][1] / max(res["trans"][1], 1e-20)),
        floor_ratio=float(hi / max(lo, 1e-20)),
        gap=float(res["traj"][1] / max(res["traj"][0], 1e-20)),
        train_mse=float(res["traj"][0]), test_mse=float(hi),
        frac_near=float(np.mean((marg >= 0) & (marg < 0.10))),
        frac_viol=float(np.mean(marg < 0)))


def ok(r):
    return (r["traj_over_trans"] < 5.0 and r["coverage_ratio"] < 3.0
            and r["floor_ratio"] > 0.7 and r["frac_near"] + r["frac_viol"] > 0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sigmas", default=None)
    ap.add_argument("--n-traj", type=int, default=None)
    ap.add_argument("--hidden", default="64,64")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--friction", type=float, default=None)
    ap.add_argument("--payload-fixed", action="store_true",
                    help="hold the payload constant; if test error is unchanged, "
                         "the error floor is aleatoric and cannot be re-allocated")
    ap.add_argument("--out", default="results/validate")
    a = ap.parse_args()

    sigmas = [float(x) for x in (a.sigmas or ("0.15,0.35" if a.quick
                                              else "0.10,0.20,0.35,0.60")).split(",")]
    n_traj = a.n_traj or (60 if a.quick else 150)
    lo, hi = (200, 800) if a.quick else (200, 2000)
    hidden = tuple(int(h) for h in a.hidden.split(","))
    os.makedirs(a.out, exist_ok=True)
    plant = UR5ePlant()

    print(f"n_traj={n_traj} hidden={hidden} budgets={lo}/{hi} "
          f"friction={a.friction if a.friction is not None else MjParams.frictionloss}")
    print("\nPASS needs ALL of: traj/trans < 5, coverage < 3, floor > 0.7, "
          "near+viol > 5%\n")
    print(f"  {'sigma':>7}{'traj/trans':>12}{'coverage':>10}{'floor':>8}"
          f"{'gap':>8}{'near+viol':>11}{'verdict':>9}")
    rows = []
    for sg in sigmas:
        r = assess(sg, n_traj, lo, hi, hidden, a.seed, plant, a.friction,
                   a.payload_fixed)
        rows.append(r)
        print(f"  {sg:>7.2f}{r['traj_over_trans']:>12.2f}{r['coverage_ratio']:>10.2f}"
              f"{r['floor_ratio']:>8.2f}{r['gap']:>8.1f}"
              f"{r['frac_near'] + r['frac_viol']:>10.1%}"
              f"{'PASS' if ok(r) else 'fail':>9}")
        if ok(r) and r["gap"] < 3.0:
            print(f"           note: gap {r['gap']:.1f} is small -- the floor may "
                  f"be aleatoric. Re-run with --payload-fixed and compare "
                  f"test_mse ({r['test_mse']:.3e}) before trusting this PASS.")

    good = [r for r in rows if ok(r)]
    print("\n" + "=" * 74)
    if good:
        b = min(good, key=lambda r: r["coverage_ratio"])
        print(f">>> USABLE at sigma={b['sigma']}. Set START_SIGMA={b['sigma']} in "
              f"urmj/data.py, then run compare_arms.py.")
    else:
        worst = {k: [r[k] for r in rows] for k in
                 ("traj_over_trans", "coverage_ratio", "floor_ratio")}
        print(">>> NO SETTING PASSES. Which condition fails tells you what to do:")
        if min(worst["floor_ratio"]) > 0.7 and max(worst["traj_over_trans"]) >= 5:
            print("    coverage still bad -> narrow sigma further")
        elif max(worst["floor_ratio"]) <= 0.7:
            print("    coverage fine but the floor is gone -> the task is too easy "
                  "at this spread. ONE difficulty increase is allowed: rerun with "
                  "--friction 3.0, or --hidden 32,32. If that also fails, stop: "
                  "record that this plant cannot produce the regime and report the "
                  "scope result instead of adding more knobs.")
        else:
            print("    mixed failure -> report the table and stop; do not tune "
                  "further without deciding which condition matters more.")
    with open(f"{a.out}/verdict.json", "w") as f:
        json.dump(dict(rows=rows, config=vars(a)), f, indent=2, default=float)
    print(f"wrote {a.out}/verdict.json")


if __name__ == "__main__":
    main()
