#!/usr/bin/env python3
"""Cross-sign probe. Arm 'B-flip' = full rotH metric with pos-vel cross
blocks NEGATED (congruence by diag(I,-I) => spectrum identical to B-full,
asserted). Trains only the flip arm; A-mse and B-full are read from
rescue_gate's rescue.csv (same seeds/protocol). Run AFTER rescue_gate:
    EPS=0.0001 PYTHONPATH=. python cross_sign_probe.py
Pre-registered readout: flip ~ B-full (both worse than A) => coupling
STRUCTURE harmful regardless of sign (F-arm story confirmed by symmetry);
flip ~ A or better => harm is sign-specific interaction with the true
e_p/e_v error correlation -- new information, report as such."""
import csv, json, os, sys
import numpy as np
import torch

sys.path.insert(0, ".")
from revival_gate import get_apis
from autopsy_weighting import rollout_windows
from gate12_matched import build_matched, solve_s, EPS

W = (64, 64); SEEDS = [10, 11, 12]; D = 0; EP = 2000; KS = [1, 3, 5, 10]
RES = f"results/rescue_gate_eps{EPS:g}/rescue.csv"
OUT = f"results/cross_sign_eps{EPS:g}"; os.makedirs(OUT, exist_ok=True)
api = get_apis(); cwc = api["cwc"]
assert os.path.exists(RES), "run rescue_gate.py first (provides A/B arms)"
ref = list(csv.DictReader(open(RES)))


def metrics(model, d, seed_z, radius):
    res = rollout_windows(model, d, 0, seed_z, 10, api["Params"])
    near = res["m0"] <= np.quantile(res["m0"], .25)
    d_opt = np.maximum(res["r_rho"], 0.0)
    pref = np.maximum.accumulate(d_opt, axis=1)
    rho_t = res["dist_true"] - radius
    out = {}
    for k in KS:
        j = k - 1
        out[k] = dict(
            rmse_n=float(np.sqrt((res["en"][:, j] ** 2).mean())),
            rmse_all=float(np.sqrt((res["en"][:, j] ** 2
                                    + res["et"][:, j] ** 2).mean())),
            q95_near=float(np.quantile(d_opt[near, j], .95)),
            pref_p90=float(np.quantile(pref[:, j], .90)),
            false_safe=float(np.mean((rho_t[:, j] < 0)
                                     & (res["r_rho"][:, j] > 0))))
    return out


rows = []
for seed in SEEDS:
    d = api["build_dataset"](n_traj=60, seed=seed)
    ptr = cwc.prepare(d["train"], D, seed)
    radius = float(np.median(np.linalg.norm(
        ptr["X"][:, :2] - ptr["p_obs"], axis=1) - ptr["margin"]))
    s10, _ = solve_s(api, ptr, "rot", 10.0)
    E = build_matched(api, ptr, "rot", s10)
    F = E.copy()
    F[:, :2, 2:4] *= -1.0
    F[:, 2:4, :2] *= -1.0
    lE = np.sort(np.linalg.eigvalsh(E[0])); lF = np.sort(np.linalg.eigvalsh(F[0]))
    assert np.abs(lE - lF).max() < 1e-10, "congruence spectrum check failed"
    m_, _, _ = cwc.train(ptr, F, W, EP, seed, D)
    mt = metrics(m_, d["test"], seed + 500, radius)
    for k, v in mt.items():
        rows.append(dict(arm="B-flip", seed=seed, k=k, **v))
    print(f"  seed{seed} B-flip: k5 q95n {mt[5]['q95_near']:.2e} "
          f"k10 {mt[10]['q95_near']:.2e}", flush=True)
with open(f"{OUT}/flip.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
    w.writerows(rows)

g = lambda a, s, k, f: (next(float(r[f]) for r in ref if r["arm"] == a
                             and int(r["seed"]) == s and int(r["k"]) == k)
                        if a != "B-flip" else
                        next(r[f] for r in rows if r["seed"] == s
                             and r["k"] == k))
print(f"\n{'arm':>13} {'k':>3} {'q95_near(mean)':>15} {'d_vs_A':>8}")
for nm in ("A-mse", "B-full", "B-flip"):
    for k in (5, 10):
        q = np.mean([g(nm, s, k, "q95_near") for s in SEEDS])
        qa = np.mean([g("A-mse", s, k, "q95_near") for s in SEEDS])
        print(f"{nm:>13} {k:>3} {q:>15.3e} {q/qa-1:>+8.1%}")
print("\nreadout: flip~full(both worse) => coupling structure per se; "
      "flip~A/better => sign-specific error-correlation interaction")
print(f"wrote {OUT}/")
