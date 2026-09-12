#!/usr/bin/env python3
"""RESCUE GATE (the idea's final shot, pre-registered before any result).
Arms on FRESH seeds {10,11,12}, D=0, 64x64, 2000ep, same batch protocol:
  A mse | B our-full (matched rotH k10, WITH pos-vel coupling)
        | C our-simplified: L = e_p^T [(2/(2+l))(I + l nn^T)] e_p + ||e_v||^2
          (trace-normalised position tilt, achieved n/t ratio = 10, beta=1,
           velocity plain MSE, ZERO coupling)
Multi-step k in {1,3,5,10} on held-out episodes: rmse_n, one-sided
dangerous q95 (near stratum), prefix-max p90, false-safe, overall rmse.
PASS (all required): mean q95_near improvement of C vs A at k=5 AND k=10
>= 10%; improvement direction in >= 2/3 seeds; overall state rmse
worsening <= 5%; false-safe not increased. Else FAIL -> adopt
MSE + risk calibration permanently (no further loss tuning).
    EPS=0.0001 PYTHONPATH=. python rescue_gate.py"""
import csv, json, os, sys
import numpy as np
import torch

sys.path.insert(0, ".")
from revival_gate import get_apis
from autopsy_weighting import rollout_windows
from gate12_matched import build_matched, solve_s, EPS

W = (64, 64); SEEDS = [10, 11, 12]; D = 0; EP = 2000; KS = [1, 3, 5, 10]
LAM = 2.0 * (10 - 1) / 2  # placeholder; achieved ratio asserted below
OUT = f"results/rescue_gate_eps{EPS:g}"; os.makedirs(OUT, exist_ok=True)
api = get_apis(); cwc = api["cwc"]


def simplified_metric(pp, lam):
    n = api["normal_dir"](pp["X"], pp["p_obs"]).astype(np.float64)
    N = len(n)
    M = np.zeros((N, 4, 4))
    P = np.tile(np.eye(2), (N, 1, 1)) + lam * n[:, :, None] * n[:, None, :]
    M[:, :2, :2] = P * (2.0 / (2.0 + lam))       # trace-normalised to 2
    M[:, 2, 2] = M[:, 3, 3] = 1.0
    t = np.stack([-n[:, 1], n[:, 0]], -1)
    rn = float(np.einsum("bi,bij,bj->b", n, M[:, :2, :2], n).mean())
    rt = float(np.einsum("bi,bij,bj->b", t, M[:, :2, :2], t).mean())
    assert abs(rn / rt - 10.0) < 0.05, f"achieved ratio {rn/rt:.3f} != 10"
    assert np.abs(M[:, :2, 2:]).max() == 0.0, "coupling leaked"
    return M


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
    s10, a10 = solve_s(api, ptr, "rot", 10.0)
    Msimp = simplified_metric(ptr, 9.0)   # ratio=(1+lam)/1 => lam=9
    arms = {"A-mse": np.tile(np.eye(4), (len(ptr["X"]), 1, 1)),
            "B-full": build_matched(api, ptr, "rot", s10),
            "C-simplified": Msimp}
    for nm, M in arms.items():
        m_, _, _ = cwc.train(ptr, M, W, EP, seed, D)
        mt = metrics(m_, d["test"], seed + 500, radius)
        for k, v in mt.items():
            rows.append(dict(arm=nm, seed=seed, k=k, **v))
        print(f"  seed{seed} {nm}: k5 q95n {mt[5]['q95_near']:.2e} "
              f"k10 {mt[10]['q95_near']:.2e}", flush=True)
with open(f"{OUT}/rescue.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
    w.writerows(rows)

g = lambda a, s, k, f: next(r[f] for r in rows if r["arm"] == a
                            and r["seed"] == s and r["k"] == k)
print(f"\n{'arm':>13} {'k':>3} {'q95_near(mean)':>15} {'d_vs_A':>8} "
      f"{'rmse_all d':>10} {'fsafe':>7}")
for nm in ("A-mse", "B-full", "C-simplified"):
    for k in KS:
        q = np.mean([g(nm, s, k, "q95_near") for s in SEEDS])
        qa = np.mean([g("A-mse", s, k, "q95_near") for s in SEEDS])
        ra = np.mean([g(nm, s, k, "rmse_all") / g("A-mse", s, k, "rmse_all")
                      - 1 for s in SEEDS])
        fs = np.mean([g(nm, s, k, "false_safe") for s in SEEDS])
        print(f"{nm:>13} {k:>3} {q:>15.3e} {q/qa-1:>+8.1%} {ra:>+10.1%} "
              f"{fs:>7.4f}")
imp = [1 - np.mean([g("C-simplified", s, k, "q95_near")
                    / g("A-mse", s, k, "q95_near") for k in (5, 10)])
       for s in SEEDS]
ok_imp = np.mean(imp) >= 0.10 and sum(x > 0 for x in imp) >= 2
ok_rmse = np.mean([g("C-simplified", s, k, "rmse_all")
                   / g("A-mse", s, k, "rmse_all") - 1
                   for s in SEEDS for k in (5, 10)]) <= 0.05
ok_fs = all(np.mean([g("C-simplified", s, k, "false_safe") for s in SEEDS])
            <= np.mean([g("A-mse", s, k, "false_safe") for s in SEEDS]) + 1e-9
            for k in (5, 10))
v = dict(improvement_per_seed=[float(x) for x in imp],
         imp_ok=bool(ok_imp), rmse_ok=bool(ok_rmse), fs_ok=bool(ok_fs),
         result="PASS" if (ok_imp and ok_rmse and ok_fs) else "FAIL")
json.dump(v, open(f"{OUT}/verdict.json", "w"), indent=2)
print("\n>>> " + ("PASS: simplified loss rescues the idea -> closed-loop "
                  "isolation experiment next"
                  if v["result"] == "PASS" else
                  "FAIL: adopt MSE + risk calibration permanently; "
                  "no further loss tuning"))
print(f"wrote {OUT}/")
