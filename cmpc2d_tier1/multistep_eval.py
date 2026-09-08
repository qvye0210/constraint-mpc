#!/usr/bin/env python3
"""Multi-step evaluation, clean floor. Usage:
   EPS=0.0001 PYTHONPATH=. python multistep_eval.py --width 64,64
   EPS=0.0001 PYTHONPATH=. python multistep_eval.py --width 32,32
Pre-registered readout: multi-step (k=5,10) rmse_n / dangerous q95 still
worse for rot arms in >=2/3 seeds => horizon-mismatch excluded, optimization/
shared-representation verdict stands, idea sealed; one-step worse but k=5-10
better => one-step judgement was unfair, horizon-aware weighting works."""
import argparse, csv, os, sys
import numpy as np

sys.path.insert(0, ".")
from revival_gate import get_apis, rollout_windows
from gate12_matched import build_matched, solve_s, EPS

ap = argparse.ArgumentParser(); ap.add_argument("--width", default="64,64")
W = tuple(int(x) for x in ap.parse_args().width.split(","))
SEEDS = [0, 1, 2]; D = 8; EPOCHS = 2000; KS = [1, 3, 5, 10]; H = 10
OUT = f"results/multistep_{W[0]}_eps{EPS:g}"; os.makedirs(OUT, exist_ok=True)
api = get_apis(); cwc = api["cwc"]

d0 = api["build_dataset"](n_traj=60, seed=0)
p00 = cwc.prepare(d0["train"], D, 0)
S5, a5 = solve_s(api, p00, "rot", 5.0)
S10, a10 = solve_s(api, p00, "rot", 10.0)
print(f"EPS={EPS:g} width={W} doses k5 {a5:.3f} k10 {a10:.3f}")
ARMS = [("mask", 0.0), ("rot5", S5), ("rot10", S10)]

rows = []
for seed in SEEDS:
    d = api["build_dataset"](n_traj=60, seed=seed)
    ptr = cwc.prepare(d["train"], D, seed)
    r0 = float(np.median(np.linalg.norm(
        ptr["X"][:, :2] - ptr["p_obs"], axis=1) - ptr["margin"]))
    for nm, s in ARMS:
        M = build_matched(api, ptr, "rot", s)
        model, _, _ = cwc.train(ptr, M, W, EPOCHS, seed, D)
        for split, dd, sz in (("train", d["train"], seed),
                              ("test", d["test"], seed + 500)):
            res = rollout_windows(model, dd, D, sz, H, api["Params"])
            near = res["m0"] <= np.quantile(res["m0"], .25)
            rho = res["dist_true"] - r0
            pref = np.maximum.accumulate(np.abs(res["r_rho"]), axis=1)
            for k in KS:
                j = k - 1; rr = res["r_rho"][:, j]
                rows.append(dict(
                    arm=nm, seed=seed, split=split, k=k,
                    rmse_n=float(np.sqrt((res["en"][:, j] ** 2).mean())),
                    rmse_t=float(np.sqrt((res["et"][:, j] ** 2).mean())),
                    q95_pos_near=float(np.quantile(rr[near], .95)),
                    false_safe=float(np.mean((rho[:, j] < 0)
                                             & (rho[:, j] + rr[:, j] > 0))),
                    prefmax_p90=float(np.quantile(pref[:, j], .90))))
        print(f"  {nm} seed{seed} done")
with open(f"{OUT}/multistep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

print("\nTEST, mean over seeds (delta vs mask in parentheses):")
for k in KS:
    base = {r["seed"]: r for r in rows
            if r["arm"] == "mask" and r["split"] == "test" and r["k"] == k}
    for nm, _ in ARMS:
        sel = [r for r in rows if r["arm"] == nm
               and r["split"] == "test" and r["k"] == k]
        dr = np.mean([r["rmse_n"] / base[r["seed"]]["rmse_n"] - 1 for r in sel])
        dq = np.mean([r["q95_pos_near"]
                      / max(abs(base[r["seed"]]["q95_pos_near"]), 1e-12) - 1
                      for r in sel])
        print(f"  k={k:2d} {nm:>6}: rmse_n {np.mean([r['rmse_n'] for r in sel]):.2e}"
              f" ({dr:+.0%})  q95+near {np.mean([r['q95_pos_near'] for r in sel]):.2e}"
              f" ({dq:+.0%})  fs {np.mean([r['false_safe'] for r in sel]):.4f}"
              f"  prefmax90 {np.mean([r['prefmax_p90'] for r in sel]):.2e}")
print(f"wrote {OUT}/multistep.csv")
