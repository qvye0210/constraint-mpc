#!/usr/bin/env python3
"""Step 1 -- the four-arm comparison, run only after capacity_check.py passes.

    uniform   plain MSE
    mask      irrelevant dimensions zeroed, the rest UNIFORM
    prop      propagated constraint-gradient metric
    random    same spectrum as prop, random orientation

The comparison that decides the contribution is prop vs MASK, not prop vs
uniform. Both mask and prop stop spending capacity where the constraint cannot
reach; only prop additionally allocates BY DIRECTION. If prop is not clearly
better than mask, the gain is masking alone -- which is the mechanism VaGraM
already established, with grad g swapped in for grad V.

PRE-REGISTERED DECISION RULE (do not revise after seeing results):

  * prop must beat uniform by >5% on near-constraint directional error, AND
  * prop must beat mask by >5% on the same metric, AND
  * both must hold at the largest training budget, not only the smallest.

If any fails: the directional part is judged ineffective on this system. Record
it, do not introduce a new auxiliary explanation, and fall back to reporting the
scope result (masking works, direction does not) with the mechanism analysis.

    python compare_arms.py --seeds 3
    python compare_arms.py --seeds 1 --seed-offset 2   # parallel across seeds
"""

import argparse
import csv
import json
import os

import numpy as np

from urmj.data import DEFAULT_OBS, build_dataset, coverage
from urmj.model import build_metric, evaluate, train, weight_report
from urmj.plant import NQ, NX, UR5ePlant

ARMS = ["uniform", "mask", "prop", "random"]
KEY = "err_constraint_dir_near"          # primary metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-traj", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--hidden", default="64,64")
    ap.add_argument("--epoch-list", default="200,1000")
    ap.add_argument("--H", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--eps-floor", type=float, default=0.05)
    ap.add_argument("--p-obs", default=",".join(str(v) for v in DEFAULT_OBS))
    ap.add_argument("--out", default="results/arms")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    arms = a.arms.split(",")
    hidden = tuple(int(h) for h in a.hidden.split(","))
    eps = [int(x) for x in a.epoch_list.split(",")]
    p_obs = tuple(float(x) for x in a.p_obs.split(","))
    seeds = list(range(a.seed_offset, a.seed_offset + a.seeds))
    os.makedirs(a.out, exist_ok=True)
    plant = UR5ePlant()

    print(f"arms={arms} hidden={hidden} epochs={eps} seeds={seeds}")
    rows, wrows = [], []

    data = {}
    for s in seeds:
        data[s] = build_dataset(n_traj=a.n_traj, seed=s)
    cov = coverage(data[seeds[0]]["train"])
    print(f"\ncoverage (seed {seeds[0]}): violating {cov['frac_violating']:.1%}  "
          f"near {cov['frac_near']:.1%}  far {cov['frac_far']:.1%}")
    if cov["frac_near"] + cov["frac_violating"] < 0.05:
        print("  WARNING: the constraint is almost never approached. Nothing "
              "constraint-related can be measured on this data -- retune "
              "p_obs / r_safe / attract before trusting any result below.")

    # weight diagnostics
    print(f"\n  {'arm':10}{'w(q)':>9}{'w(qd)':>9}{'cond(qd)':>12}")
    for arm in arms:
        M = build_metric(plant, data[seeds[0]]["train"]["X"], p_obs, arm,
                         a.H, a.gamma, a.eps_floor, seed=seeds[0])
        wr = weight_report(M); wr.update(arm=arm)
        wrows.append(wr)
        print(f"  {arm:10}{wr['w_q']:>9.4f}{wr['w_qd']:>9.4f}{wr['cond_qd']:>12.1f}")

    for ep in eps:
        print(f"\n=== {ep} epochs ===")
        print(f"  {'arm':10}{'mse':>12}{'mse_near':>12}"
              f"{'err_dir':>12}{'err_dir_near':>14}{'vs unif':>10}")
        ref = None
        for arm in arms:
            res = []
            for s in seeds:
                M = build_metric(plant, data[s]["train"]["X"], p_obs, arm,
                                 a.H, a.gamma, a.eps_floor, seed=s)
                m = train(data[s]["train"], M, hidden, ep, seed=s, device=a.device)
                res.append(evaluate(m, data[s]["test"], plant, p_obs, a.device))
            e = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
            if ref is None:
                ref = e[KEY]
            rel = (e[KEY] - ref) / ref
            rows.append(dict(arm=arm, epochs=ep, rel=rel, **e))
            print(f"  {arm:10}{e['mse']:>12.3e}{e.get('mse_near', float('nan')):>12.3e}"
                  f"{e['err_constraint_dir']:>12.3e}{e[KEY]:>14.3e}{rel:>+10.1%}")

    for name, data_ in (("raw", rows), ("weights", wrows)):
        with open(f"{a.out}/{name}.csv", "w", newline="") as f:
            k = sorted({x for r in data_ for x in r})
            w = csv.DictWriter(f, fieldnames=k); w.writeheader(); w.writerows(data_)

    print("\n" + "=" * 74)
    verdict = {}
    for ep in eps:
        sel = {r["arm"]: r for r in rows if r["epochs"] == ep}
        if not {"prop", "mask", "uniform"} <= set(sel):
            continue
        vs_u = (sel["prop"][KEY] - sel["uniform"][KEY]) / sel["uniform"][KEY]
        vs_m = (sel["prop"][KEY] - sel["mask"][KEY]) / sel["mask"][KEY]
        vs_r = ((sel["prop"][KEY] - sel["random"][KEY]) / sel["random"][KEY]
                if "random" in sel else float("nan"))
        verdict[str(ep)] = dict(prop_vs_uniform=vs_u, prop_vs_mask=vs_m,
                                prop_vs_random=vs_r,
                                direction_adds_value=bool(vs_u < -0.05 and vs_m < -0.05))
        print(f"  {ep:>5} epochs: prop vs uniform {vs_u:+.1%},  vs MASK {vs_m:+.1%},"
              f"  vs random {vs_r:+.1%}")
    ok = all(v["direction_adds_value"] for v in verdict.values()) and verdict
    print("\n>>> " + (
        "DIRECTION ADDS VALUE at every budget. The directional part of the "
        "hypothesis holds on this system; proceed to closed-loop MPC."
        if ok else
        "DIRECTION DOES NOT ADD VALUE. Per the pre-registered rule: report the "
        "scope result (masking works, direction does not) with the mechanism "
        "analysis. Do not add a new auxiliary explanation."))
    with open(f"{a.out}/verdict.json", "w") as f:
        json.dump(dict(verdict=verdict, coverage=cov, config=vars(a)),
                  f, indent=2, default=float)
    print(f"wrote {a.out}/")


if __name__ == "__main__":
    main()
