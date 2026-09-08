#!/usr/bin/env python3
"""Mechanism discriminator 0 (zero training): read revival_gate curves.csv
and classify each kappa arm per the pre-registered 4-way taxonomy:

  A train e_n ~mask, test e_n worse        -> generalization / overfitting
  B train e_n ALSO worse than mask         -> optimization or representation
                                              conflict (not overfitting)
  C harm still shrinking at 2000 epochs    -> partly slow convergence
  D e_t degrades before e_n along training -> shared-representation coupling
                                              signature (suggestive only)

    PYTHONPATH=. python curve_report.py
"""
import csv
from collections import defaultdict
import numpy as np

ARMS = ["k1", "k5", "k10", "k74"]
ROWS = list(csv.DictReader(open("results/revival_gate/curves.csv")))


def get(width, arm, split, field):
    """epoch -> mean over seeds"""
    acc = defaultdict(list)
    for r in ROWS:
        if r["width"] == width and r["arm"] == arm and r["split"] == split:
            acc[int(r["epoch"])].append(float(r[field]))
    return {e: float(np.mean(v)) for e, v in sorted(acc.items())}


widths = sorted({r["width"] for r in ROWS})
for w in widths:
    print("=" * 78)
    print(f"width {w}   (values = rmse_n x1e4, mean over seeds)")
    mask_tr = get(w, "mask", "train", "rmse_n")
    mask_te = get(w, "mask", "test", "rmse_n")
    eps = sorted(mask_tr)
    hdr = "  ".join(f"{e:>6}" for e in eps)
    print(f"  {'arm':>5} {'side':>5}  {hdr}")
    for arm in ["mask"] + ARMS:
        for split, src in (("train", get(w, arm, "train", "rmse_n")),
                           ("test", get(w, arm, "test", "rmse_n"))):
            if not src:
                continue
            print(f"  {arm:>5} {split:>5}  " +
                  "  ".join(f"{src[e]*1e4:6.2f}" for e in eps))
    print("-" * 78)
    for arm in ARMS:
        tr, te = get(w, arm, "train", "rmse_n"), get(w, arm, "test", "rmse_n")
        tt = get(w, arm, "test", "rmse_t")
        mtt = get(w, "mask", "test", "rmse_t")
        if not tr:
            continue
        e_end = eps[-1]; e_mid = eps[len(eps) // 2]
        r_tr = tr[e_end] / mask_tr[e_end]
        r_te = te[e_end] / mask_te[e_end]
        conv = te[e_end] / te[e_mid]          # <1 and falling = still converging
        conv_m = mask_te[e_end] / mask_te[e_mid]
        # e_t-before-e_n timing: epoch where each first exceeds 1.5x mask
        def first_bad(curve, ref, f=1.5):
            for e in eps:
                if curve.get(e, 0) > f * ref.get(e, np.inf):
                    return e
            return None
        tb_t = first_bad(tt, mtt); tb_n = first_bad(te, mask_te)
        tags = []
        if r_tr <= 1.10 and r_te > 1.10:
            tags.append("A: overfitting-like (train ok, test worse)")
        if r_tr > 1.10:
            tags.append(f"B: train-side also worse x{r_tr:.2f} "
                        "(optimization/representation, NOT overfitting)")
        if conv < 0.85 and conv < conv_m - 0.05:
            tags.append("C: still converging at 2000ep (harm partly = slower)")
        if tb_t is not None and (tb_n is None or tb_t < tb_n):
            tags.append(f"D: e_t degrades first (ep{tb_t} vs "
                        f"{'never' if tb_n is None else 'ep%d' % tb_n}) "
                        "-- coupling signature, suggestive")
        print(f"  {arm:>5}: train x{r_tr:.2f}  test x{r_te:.2f}  "
              f"late-conv {conv:.2f} (mask {conv_m:.2f})")
        for t in tags:
            print(f"         {t}")
print("=" * 78)
print("Read B vs A first: it decides overfitting vs optimization. "
      "C only caveats magnitude. D is suggestive, not proof (decoupled-head "
      "experiment remains the positive test for mechanism 3).")
