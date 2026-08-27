#!/usr/bin/env python3
"""Closed-loop validation of the offline proxy.

Every conclusion so far rests on

    proxy = 48.5 * rmse_normal + 12.3 * rmse_tangent

whose coefficients were measured by injecting a PERSISTENT directional bias
(Gate DIR).  A learned model's error is state-dependent and roughly zero-mean,
so it need not affect closed-loop violation in the same proportion.  The proxy
enters the reasoning chain very early, so if it is biased, everything after it
needs re-reading.

This script trains each arm, runs closed-loop MPC with the learned model, and
checks whether the closed-loop ranking matches the proxy ranking.

It also reports two distractor-propagation modes.  Arms that zero the distractor
weight ('prop', 'mask') predict the distractor state badly, and the distractor
state is a network input -- so bad distractor prediction can leak into the core
prediction over a horizon.  'model' mode exposes that; 'true' mode removes it.

    python closed_loop_check.py --quick
    python closed_loop_check.py --seeds 3 --n-dist 8 --epochs 500 --n-ep 40
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr

from cmpc2d.augdyn import AugDyn
from cmpc2d.cweight import build_metric
from cmpc2d.data import build_dataset
from cmpc2d.env import Params, f_true, sample_distract, sample_scenario
from cmpc2d.eval import paired_diff, run_episode
from constraint_weight_check import ARMS, evaluate, prepare, train


def closed_loop(model, n_dist, n_ep, seed, z_mode, params=Params, device="cpu"):
    rows = []
    for i in range(n_ep):
        rng = np.random.default_rng(10_000 + seed * 1000 + i)
        scn = sample_scenario(rng, params, jitter=True)
        z0 = sample_distract(rng, 1, n_dist)[0] if n_dist else np.zeros(0)
        dyn = AugDyn(model, z0, n_dist, params, z_mode, device)
        m, _ = run_episode(dyn, scn, params=params, seed=10_000 + i)
        m["episode"] = i
        rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-dist", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-ep", type=int, default=None)
    ap.add_argument("--n-traj", type=int, default=None)
    ap.add_argument("--hidden", default="64,64")
    ap.add_argument("--arms", default="uniform,mask,diag,prop,random")
    ap.add_argument("--z-mode", default="model,true")
    ap.add_argument("--out", default="results/closed_loop")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    epochs = a.epochs or (60 if a.quick else 500)
    n_ep = a.n_ep or (5 if a.quick else 40)
    n_traj = a.n_traj or (15 if a.quick else 60)
    hidden = tuple(int(h) for h in a.hidden.split(","))
    arms = a.arms.split(",")
    zmodes = a.z_mode.split(",")
    seeds = list(range(a.seeds))
    D = a.n_dist
    os.makedirs(a.out, exist_ok=True)

    print(f"D={D} hidden={hidden} epochs={epochs} episodes/seed={n_ep} "
          f"seeds={seeds} arms={arms}")

    # --- reference: true dynamics, to bound what is achievable --------------
    ref = []
    for s in seeds:
        for i in range(n_ep):
            rng = np.random.default_rng(10_000 + s * 1000 + i)
            scn = sample_scenario(rng, Params, jitter=True)
            m, _ = run_episode(lambda x, u: f_true(x, u), scn, seed=10_000 + i)
            ref.append(m["viol_integral"])
    print(f"true-dynamics MPC violation: {np.mean(ref):.6f}\n")

    rows, per_ep = [], []
    for arm in arms:
        for zm in zmodes:
            offl, cl = [], {k: [] for k in ("viol_integral", "viol_freq",
                                            "track_rmse", "objective",
                                            "infeas_rate")}
            for s in seeds:
                d = build_dataset(n_traj=n_traj, seed=s)
                ptr, pte = prepare(d["train"], D, s), prepare(d["test"], D, s + 500)
                M, _ = build_metric(ptr["win_X"], ptr["p_obs"], D, 10, 0.9,
                                    arm, 0.05, seed=s)
                model, _, _ = train(ptr, M, hidden, epochs, s, D, device=a.device)
                offl.append(evaluate(model, pte, D, a.device))
                ep_rows = closed_loop(model, D, n_ep, s, zm, device=a.device)
                for r in ep_rows:
                    r.update(arm=arm, z_mode=zm, seed=s)
                per_ep += ep_rows
                for k in cl:
                    cl[k].append(np.mean([r[k] for r in ep_rows]))
            o = {k: float(np.mean([x[k] for x in offl])) for k in offl[0]}
            r = dict(arm=arm, z_mode=zm, **o,
                     **{f"cl_{k}": float(np.mean(v)) for k, v in cl.items()})
            rows.append(r)
            print(f"  {arm:9} z={zm:6} proxy={o['proxy']:.5f}  "
                  f"cl_viol={r['cl_viol_integral']:.5f}  "
                  f"track={r['cl_track_rmse']:.4f}  "
                  f"infeas={r['cl_infeas_rate']:.3f}")

    with open(f"{a.out}/summary.csv", "w", newline="") as f:
        k = sorted({x for r in rows for x in r})
        w = csv.DictWriter(f, fieldnames=k); w.writeheader(); w.writerows(rows)
    with open(f"{a.out}/episodes.csv", "w", newline="") as f:
        k = sorted({x for r in per_ep for x in r})
        w = csv.DictWriter(f, fieldnames=k); w.writeheader(); w.writerows(per_ep)

    # --- is the proxy a valid instrument? ----------------------------------
    print("\n" + "=" * 78)
    out = {}
    for zm in zmodes:
        sub = [r for r in rows if r["z_mode"] == zm]
        rho, p = spearmanr([r["proxy"] for r in sub],
                           [r["cl_viol_integral"] for r in sub])
        order_proxy = [r["arm"] for r in sorted(sub, key=lambda r: r["proxy"])]
        order_cl = [r["arm"] for r in sorted(sub, key=lambda r: r["cl_viol_integral"])]
        agree = order_proxy == order_cl
        out[zm] = dict(spearman=float(rho), p=float(p),
                       order_proxy=order_proxy, order_closed_loop=order_cl,
                       identical_ranking=agree)
        print(f"  z_mode={zm}:  Spearman(proxy, closed-loop) = {rho:+.3f} (p={p:.3f})")
        print(f"    proxy order : {' < '.join(order_proxy)}")
        print(f"    closed loop : {' < '.join(order_cl)}")
        print(f"    -> {'rankings agree' if agree else 'RANKINGS DIFFER'}")

    # paired test, best arm vs uniform, on the realistic z mode
    zm = zmodes[0]
    base = [r["viol_integral"] for r in per_ep if r["arm"] == "uniform" and r["z_mode"] == zm]
    for arm in arms:
        if arm == "uniform":
            continue
        v = [r["viol_integral"] for r in per_ep if r["arm"] == arm and r["z_mode"] == zm]
        if len(v) == len(base):
            d = paired_diff(base, v)
            out.setdefault("paired", {})[arm] = d
            print(f"  {arm:9} vs uniform (closed loop): {d['mean_diff']:+.5f} "
                  f"CI[{d['ci_lo']:+.5f},{d['ci_hi']:+.5f}] p={d['p']:.3f} "
                  f"dz={d['cohen_dz']:+.2f}")

    valid = all(v["spearman"] > 0.6 for v in out.values() if isinstance(v, dict)
                and "spearman" in v)
    print("\n>>> " + ("Proxy is a usable instrument; earlier offline conclusions "
                      "stand as ranked."
                      if valid else
                      "Proxy does NOT track closed-loop violation. Offline "
                      "conclusions from step 3 onward must be re-read; use "
                      "closed-loop metrics from here on."))
    out["proxy_valid"] = bool(valid)
    out["true_dynamics_violation"] = float(np.mean(ref))
    with open(f"{a.out}/verdict.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"wrote {a.out}/")


if __name__ == "__main__":
    main()
