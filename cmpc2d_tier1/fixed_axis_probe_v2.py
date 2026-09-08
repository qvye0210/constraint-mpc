#!/usr/bin/env python3
"""Fixed-axis probe v2 -- the single permitted correction run. No further
mechanism experiments after this regardless of outcome.

Changes vs v1 (per review):
  1. strength re-inverted PER FIXED AXIS against the fake geometry;
     HARD DOSE GATES: kappa=10 -> achieved in [9.5, 10.5];
                      kappa=74 -> achieved in [70.3, 77.7].
     Any gate miss => abort, no mechanism output.
  2. same-code-path construction (fake obstacle at 1e6 along the axis)
     keeps velocity/distractor/propagation/normalisation IDENTICAL to the
     rotating family by construction; the position block then equals
     M_k(d) = (2/(k+1)) (k dd^T + (I - dd^T)) up to the family's own
     normalisation (used here as the reference for achieved-kappa).
  3. HARD IDENTITY GATE: at s -> 0, fixed-family matrix must equal the
     rotating-family matrix elementwise (this is what licenses sharing the
     kappa=1 baseline). Abort on violation.
  4. runs ONLY width 64x64, seeds 0/1/2, kappa in {10, 74}, axes fx/fy
     (12 trainings). kappa=1 baseline is NOT retrained: by gate 3 it is
     the rotating k1 rows from results/revival_gate/final.csv.
  5. readout: per-family log(q95_near(kappa)/q95_near(kappa=1)) and the
     same for rmse_n, per seed + median. No single scalar verdict.
"""

import csv, os, sys
import numpy as np

sys.path.insert(0, ".")
from revival_gate import get_apis, metric_for, decompose, stats

WIDTH = (64, 64)
SEEDS = [0, 1, 2]
EPOCHS = 2000
D = 8
GATES = {10.0: (9.5, 10.5), 74.0: (70.3, 77.7)}
AXES = {"fx": np.array([1e6, 0.0]), "fy": np.array([0.0, 1e6])}
OUT = "results/fixed_axis_v2"
S_LO = 1e-4


def ratio_fixed(api, ptr, fake, s):
    p = dict(ptr); p["p_obs"] = np.zeros_like(ptr["p_obs"]) + fake
    M = metric_for(api, p, D, "prop", s, SEEDS[0])
    return api["weight_report"](M, ptr["X"], p["p_obs"], D)["normal_over_tangent"], M


def invert(api, ptr, fake, target):
    a, b = S_LO, 1.0
    ka, _ = ratio_fixed(api, ptr, fake, a)
    kb, _ = ratio_fixed(api, ptr, fake, b)
    if not (ka < target < kb):
        return None, min(max(target, ka), kb)
    for _ in range(48):
        m = 0.5 * (a + b)
        km, _ = ratio_fixed(api, ptr, fake, m)
        if km < target: a = m
        else: b = m
    s = 0.5 * (a + b)
    return s, ratio_fixed(api, ptr, fake, s)[0]


def main():
    api = get_apis(); cwc = api["cwc"]
    os.makedirs(OUT, exist_ok=True)
    d0 = api["build_dataset"](n_traj=60, seed=SEEDS[0])
    ptr0 = cwc.prepare(d0["train"], D, SEEDS[0])

    # ---- gate 3: kappa=1 identity between fixed and rotating families
    for ax_name, ax in AXES.items():
        _, M_fix = ratio_fixed(api, ptr0, ax, S_LO)
        M_rot = metric_for(api, ptr0, D, "prop", S_LO, SEEDS[0])
        dmax = float(np.abs(M_fix - M_rot).max())
        print(f"gate3 kappa=1 identity ({ax_name}): max|diff| = {dmax:.3e}")
        if dmax > 1e-6:
            sys.exit("ABORT gate3: fixed and rotating families differ at s->0; "
                     "the shared kappa=1 baseline is not licensed. No output.")

    # ---- gate 1: per-axis dose inversion
    doses = {}
    for ax_name, ax in AXES.items():
        for kt, (lo, hi) in GATES.items():
            s, ach = invert(api, ptr0, ax, kt)
            ok = s is not None and lo <= ach <= hi
            print(f"gate1 dose ({ax_name}, target {kt:g}): s={s} achieved={ach:.3f} "
                  f"required [{lo},{hi}] -> {'OK' if ok else 'FAIL'}")
            if not ok:
                sys.exit("ABORT gate1: dose gate missed. No mechanism output.")
            doses[(ax_name, kt)] = s

    # ---- rotating references (kappa=1 shared baseline + rotating 10/74)
    ref = {}
    for r in csv.DictReader(open("results/revival_gate/final.csv")):
        if (r["split"] == "test" and int(r["epoch"]) == EPOCHS
                and r["width"] == str(WIDTH)):
            ref[(r["arm"], int(r["seed"]))] = dict(q95=float(r["q95_near"]),
                                                   rmse_n=float(r["rmse_n"]))

    rows = []
    for seed in SEEDS:
        d = api["build_dataset"](n_traj=60, seed=seed)
        ptr = cwc.prepare(d["train"], D, seed)
        pte = cwc.prepare(d["test"], D, seed + 500)
        rr = np.linalg.norm(ptr["X"][:, :2] - ptr["p_obs"], axis=1) - ptr["margin"]
        radius = float(np.median(rr))
        for (ax_name, kt), s in doses.items():
            p = dict(ptr); p["p_obs"] = np.zeros_like(ptr["p_obs"]) + AXES[ax_name]
            M = metric_for(api, p, D, "prop", s, seed)
            model, _, _ = cwc.train(ptr, M, WIDTH, EPOCHS, seed, D)
            st = stats(decompose(api, model, pte, None, D), radius)
            rows.append(dict(family=ax_name, kappa=kt, seed=seed,
                             q95_near=st["q95_near"], rmse_n=st["rmse_n"]))
            print(f"  trained {ax_name} k={kt:g} seed{seed}: "
                  f"q95 {st['q95_near']:.2e} rmse_n {st['rmse_n']:.2e}")
    for kt in GATES:
        for seed in SEEDS:
            r = ref[(f"k{kt:g}", seed)]
            rows.append(dict(family="rot", kappa=kt, seed=seed,
                             q95_near=r["q95"], rmse_n=r["rmse_n"]))
    with open(f"{OUT}/probe.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ---- readout: log-ratios vs the SHARED kappa=1 baseline (rotating k1)
    print("\n" + "=" * 86)
    print("log10( metric(kappa) / metric(kappa=1) ), shared k1 baseline; "
          "per seed [s0 s1 s2] and median")
    for metric in ("q95_near", "rmse_n"):
        print(f"-- {metric}")
        for kt in GATES:
            line = f"  k={kt:>4g}: "
            for fam in ("rot", "fx", "fy"):
                vals = []
                for seed in SEEDS:
                    base = ref[("k1", seed)]["q95" if metric == "q95_near" else "rmse_n"]
                    cur = next(r[metric] for r in rows
                               if r["family"] == fam and r["kappa"] == kt
                               and r["seed"] == seed)
                    vals.append(np.log10(cur / base))
                line += (f"{fam}: [" + " ".join(f"{v:+.2f}" for v in vals)
                         + f"] med {np.median(vals):+.2f}   ")
            print(line)
    print("=" * 86)
    print("Reading (pre-registered): rot vs fx/fy medians within ~0.1 dex of "
          "each other => rotation unnecessary for the harm; fixed medians "
          "clearly below rot => rotation contributes; fx vs fy gap at equal "
          "dose = alignment effect, report as is. This is the last mechanism "
          "run; anything further goes to the paper plan only.")
    print(f"wrote {OUT}/probe.csv")


if __name__ == "__main__":
    main()
