#!/usr/bin/env python3
"""Mechanism discriminator 1: directional ill-conditioning vs inter-sample
gradient conflict. Run AFTER revival_gate.py (reuses its rotating-arm rows).

    PYTHONPATH=. python fixed_axis_probe.py

Trick: same build_metric code path, but p_obs replaced by a fake obstacle at
1e6 distance -> normal_dir is (nearly) constant across samples -> same 74:1
(or 10:1) directional weighting WITHOUT rotation. Two orthogonal fake axes
(+x, +y) guard against picking a lucky axis. Evaluation stays in the TRUE
local (n,t) frame of the real obstacle.

PRE-REGISTERED READOUT (harm = q95_near/q95_near(mask) - 1, per seed/width):
  fixed harm >= 70% of rotating harm in the majority of cells
      -> CONDITIONING dominant (rotation unnecessary for the damage)
  fixed harm <= 30% of rotating harm in the majority of cells
      -> INTER-SAMPLE GRADIENT CONFLICT dominant (rotation is the poison)
  otherwise -> mixed; decoupled-head experiment required to go further.
Shared-representation (mechanism 3) is NOT discriminated here.
"""

import csv, os, sys
import numpy as np
import torch

sys.path.insert(0, ".")
from revival_gate import get_apis, metric_for, decompose, stats

KAPPA_S = {10.0: 0.85642, 74.0: 1.0}     # strengths from revival_gate inversion
AXES = {"fx": np.array([1e6, 0.0]), "fy": np.array([0.0, 1e6])}
SEEDS = [0, 1, 2]
WIDTHS = [(64, 64), (32, 32)]
EPOCHS = 2000
D = 8
OUT = "results/fixed_axis"


def load_reference():
    rows = list(csv.DictReader(open("results/revival_gate/final.csv")))
    ref = {}
    for r in rows:
        if r["split"] != "test" or int(r["epoch"]) != EPOCHS:
            continue
        key = (r["width"], int(r["seed"]), r["arm"])
        ref[key] = dict(q95=float(r["q95_near"]), rmse_n=float(r["rmse_n"]))
    return ref


def main():
    api = get_apis(); cwc = api["cwc"]
    os.makedirs(OUT, exist_ok=True)
    ref = load_reference()
    rows = []
    for hidden in WIDTHS:
        for seed in SEEDS:
            d = api["build_dataset"](n_traj=60, seed=seed)
            ptr = cwc.prepare(d["train"], D, seed)
            pte = cwc.prepare(d["test"], D, seed + 500)
            r = np.linalg.norm(ptr["X"][:, :2] - ptr["p_obs"], axis=1) - ptr["margin"]
            radius = float(np.median(r))
            for ax_name, ax in AXES.items():
                fake = np.zeros_like(ptr["p_obs"]) + ax
                for kappa, s in KAPPA_S.items():
                    p_fake = dict(ptr); p_fake["p_obs"] = fake
                    M = metric_for(api, p_fake, D, "prop", s, seed)
                    # validation: ratio in the fake frame must hit kappa;
                    # ratio in the REAL frame must be much lower (mixture)
                    rep_f = api["weight_report"](M, ptr["X"], fake, D)
                    rep_r = api["weight_report"](M, ptr["X"], ptr["p_obs"], D)
                    model, _, _ = cwc.train(ptr, M, hidden, EPOCHS, seed, D)
                    dec = decompose(api, model, pte, None, D)
                    st = stats(dec, radius)
                    key_m = (str(hidden), seed, "mask")
                    key_r = (str(hidden), seed, f"k{kappa:g}")
                    harm_f = st["q95_near"] / ref[key_m]["q95"] - 1
                    harm_r = ref[key_r]["q95"] / ref[key_m]["q95"] - 1
                    rows.append(dict(width=str(hidden), seed=seed, axis=ax_name,
                                     kappa=kappa,
                                     ratio_fake_frame=float(rep_f["normal_over_tangent"]),
                                     ratio_real_frame=float(rep_r["normal_over_tangent"]),
                                     q95_near=st["q95_near"], rmse_n=st["rmse_n"],
                                     harm_fixed=harm_f, harm_rot=harm_r,
                                     frac=harm_f / harm_r if harm_r > 0 else float("nan")))
                    print(f"w={hidden} s={seed} {ax_name} k={kappa:g}: "
                          f"ratio fake/real {rep_f['normal_over_tangent']:.1f}/"
                          f"{rep_r['normal_over_tangent']:.1f}  "
                          f"harm fixed {harm_f:+.0%} vs rot {harm_r:+.0%} "
                          f"-> frac {rows[-1]['frac']:.2f}")
    with open(f"{OUT}/probe.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    fr = np.array([r["frac"] for r in rows if r["frac"] == r["frac"]])
    hi = float(np.mean(fr >= 0.7)); lo = float(np.mean(fr <= 0.3))
    print("\n" + "=" * 80)
    print(f"cells: {len(fr)}   median frac (fixed harm / rotating harm): "
          f"{np.median(fr):.2f}   >=0.7: {hi:.0%}   <=0.3: {lo:.0%}")
    if hi > 0.5:
        v = "CONDITIONING dominant: fixed-axis reproduces most of the harm."
    elif lo > 0.5:
        v = "GRADIENT-CONFLICT dominant: harm largely requires rotation."
    else:
        v = "MIXED: neither threshold met -> decoupled-head experiment needed."
    print(">>> " + v)
    print(f"wrote {OUT}/probe.csv")


if __name__ == "__main__":
    main()
