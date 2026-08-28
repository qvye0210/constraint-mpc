#!/usr/bin/env python3
"""Step 0 -- the regime check that must pass before anything else is run.

Decision-aware model learning can only help when the model cannot fit everything.
On the planar testbed this was never true: with a fixed architecture the test
error fell monotonically from 7.8e-5 to 1.3e-5 and never bottomed out, so every
weighting experiment run there was measuring nothing. That is the single mistake
that cost the most time, so it is now the first thing checked on any new plant.

PASS means the test error stops falling (or rises) with more budget while the
train error keeps falling. That is a genuine error floor, and here it comes from
limited DATA rather than from an artificially small network -- which is both the
real robot's actual condition and much harder for a reviewer to dismiss with
"just train longer" or "use a bigger net".

    python capacity_check.py                     # default 150 trajectories
    python capacity_check.py --n-traj 400 --seeds 3
"""

import argparse
import csv
import json
import os

import numpy as np
import torch

from urmj.data import DEFAULT_OBS, build_dataset, coverage
from urmj.model import ResidualMLP, evaluate, train
from urmj.plant import NQ, NX, UR5ePlant


def plain_metric(n):
    return np.tile(np.eye(NX), (n, 1, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-traj", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--hidden", default="64,64")
    ap.add_argument("--epoch-list", default="200,500,1000,2000,4000")
    ap.add_argument("--out", default="results/capacity")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    hidden = tuple(int(h) for h in a.hidden.split(","))
    eps = [int(x) for x in a.epoch_list.split(",")]
    seeds = list(range(a.seed_offset, a.seed_offset + a.seeds))
    os.makedirs(a.out, exist_ok=True)
    plant = UR5ePlant()

    rows = []
    for s in seeds:
        d = build_dataset(n_traj=a.n_traj, seed=s, verbose=True)
        cov = coverage(d["train"])
        print(f"\nseed {s}: {cov['n']} train transitions")
        print(f"  constraint coverage  violating {cov['frac_violating']:.1%}  "
              f"near {cov['frac_near']:.1%}  far {cov['frac_far']:.1%}")
        print(f"  residual rms  q {cov['residual_rms_q']:.5f}  "
              f"qd {cov['residual_rms_qd']:.5f}")
        print(f"\n  {'epochs':>8}{'train MSE':>13}{'test MSE':>13}{'vs prev':>10}")
        prev = None
        for ep in eps:
            m = train(d["train"], plain_metric(len(d["train"]["X"])),
                      hidden=hidden, epochs=ep, seed=s, device=a.device)
            tr = evaluate(m, d["train"], plant, DEFAULT_OBS, a.device)
            te = evaluate(m, d["test"], plant, DEFAULT_OBS, a.device)
            rows.append(dict(seed=s, epochs=ep, train_mse=tr["mse"],
                             test_mse=te["mse"], **{f"cov_{k}": v
                                                    for k, v in cov.items()}))
            print(f"  {ep:>8}{tr['mse']:>13.3e}{te['mse']:>13.3e}"
                  f"{(te['mse'] / prev if prev else float('nan')):>10.3f}")
            prev = te["mse"]

    with open(f"{a.out}/raw.csv", "w", newline="") as f:
        k = sorted({x for r in rows for x in r})
        w = csv.DictWriter(f, fieldnames=k); w.writeheader(); w.writerows(rows)

    te_by_ep = {ep: np.mean([r["test_mse"] for r in rows if r["epochs"] == ep])
                for ep in eps}
    tr_by_ep = {ep: np.mean([r["train_mse"] for r in rows if r["epochs"] == ep])
                for ep in eps}
    floor = te_by_ep[eps[-1]] / te_by_ep[eps[0]]
    gap = te_by_ep[eps[-1]] / tr_by_ep[eps[-1]]
    passed = bool(floor > 0.7)
    print("\n" + "=" * 66)
    print(f"  test error, last/first budget = {floor:.3f}   "
          f"(>0.7 means it bottomed out)")
    print(f"  train/test gap at the largest budget = {gap:.1f}x")
    print(">>> " + ("PASS: an error floor exists. The regime supports "
                    "decision-aware weighting; run compare_arms.py."
                    if passed else
                    "FAIL: error still falling. The model is not limited here -- "
                    "reduce data or raise task difficulty before testing any "
                    "weighting method."))
    with open(f"{a.out}/verdict.json", "w") as f:
        json.dump(dict(floor_ratio=float(floor), train_test_gap=float(gap),
                       passed=passed, test_by_epoch=te_by_ep,
                       train_by_epoch=tr_by_ep, config=vars(a)), f, indent=2)
    print(f"wrote {a.out}/")


if __name__ == "__main__":
    main()
