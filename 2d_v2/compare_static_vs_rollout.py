#!/usr/bin/env python3
"""Static constraint-normal weighting vs. rollout constraint loss (OFFLINE ONLY).

Motivation (from the Tier-1 diagnostics):
  * a tangential error is decision-irrelevant only at k=0 (|dg| ratio ~3500)
  * along the horizon the constraint normal ROTATES, so by k=5 the ratio is ~4
  * static normal weighting only ever sees k=0, which is why it bought 16.6 %
    normal accuracy at the cost of 120 % tangential accuracy -- a losing trade

Prediction under test: `constraint_rollout`, which evaluates g at the predicted
FUTURE states and therefore sees the rotation, improves the horizon constraint
error without the same tangential blow-up.

This script runs NO closed-loop MPC.  It is the cheap go/no-go before paying for
the full Step-3 evaluation.

    python compare_static_vs_rollout.py --quick
    python compare_static_vs_rollout.py --seeds 3
"""

import argparse
import csv
import json
import os

import numpy as np
import torch

from cmpc2d.data import build_dataset
from cmpc2d.env import Params, f_true, g_val, normal_dir, tangent_dir
from cmpc2d.model import eval_errors, rollout_torch, g_torch, train_model


# --- sensitivity of closed-loop violation to each error direction, measured by
# --- Gate DIR (viol_integral per unit directional error).  Used only to turn
# --- offline errors into a comparable proxy; override with --sens if re-measured.
SENS_NORMAL, SENS_TANGENT = 48.5, 12.3


@torch.no_grad()
def horizon_constraint_error(model, data, H=10, params=Params, device="cpu"):
    """Mean |g(x_hat_{t+k}) - g(x_{t+k})| over the horizon, plus near/active split."""
    X = torch.tensor(data["X"].astype(np.float32), device=device)
    WU = torch.tensor(data["win_U"][:, :H].astype(np.float32), device=device)
    WX = torch.tensor(data["win_X"][:, :H].astype(np.float32), device=device)
    OB = torch.tensor(data["p_obs"].astype(np.float32), device=device)
    Xh = rollout_torch(model, X, WU, params)
    ob = OB[:, None, :].expand(-1, H, -1)
    dg = (g_torch(Xh, ob, params) - g_torch(WX, ob, params)).abs().cpu().numpy()
    state_err = (Xh - WX).pow(2).sum(-1).sqrt().cpu().numpy()
    m = data["margin_now"]
    near = m < 0.5
    out = dict(cerr_horizon=float(dg.mean()),
               cerr_k1=float(dg[:, 0].mean()),
               cerr_k5=float(dg[:, min(4, H - 1)].mean()),
               cerr_kH=float(dg[:, -1].mean()),
               rollout_rmse=float(state_err.mean()))
    if near.any():
        out["cerr_horizon_near"] = float(dg[near].mean())
    return out


def proxy_violation(err):
    """Offline stand-in for closed-loop violation, using Gate-DIR sensitivities.

    CAVEAT: this proxy is built from ONE-STEP directional errors, so it is a fair
    instrument for one-step-loss arms (uniform, dir_weighted) but systematically
    unfair to rollout arms, which trade one-step accuracy for horizon constraint
    accuracy on purpose.  Treat `cerr_horizon_near` as the primary offline signal
    for those arms and the proxy as secondary.
    """
    return SENS_NORMAL * err["rmse_normal"] + SENS_TANGENT * err["rmse_tangent"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--out", default="results/static_vs_rollout")
    ap.add_argument("--hidden", default="64,64")
    ap.add_argument("--w-normal", type=float, default=50.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--H", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    n_traj, epochs = (8, 40) if a.quick else (60, 200)
    hidden = tuple(int(h) for h in a.hidden.split(","))
    seeds = list(range(a.seed_offset, a.seed_offset + a.seeds))
    os.makedirs(a.out, exist_ok=True)

    arms = ["uniform", "multistep_mse", "dir_weighted", "constraint_rollout"]
    rows = []
    for s in seeds:
        data = build_dataset(n_traj=n_traj, seed=s)
        print(f"\n--- seed {s} ({len(data['train']['X'])} train transitions) ---")
        for arm in arms:
            model, hist = train_model(
                data["train"], mode=arm, hidden=hidden, epochs=epochs, seed=s,
                w_normal=a.w_normal, w_tangent=1.0, w_vel=1.0,
                beta=a.beta, gamma=a.gamma, H_roll=a.H, device=a.device)
            err = eval_errors(model, data["test"], Params, a.device)
            ch = horizon_constraint_error(model, data["test"], a.H, Params, a.device)
            r = dict(seed=s, arm=arm, **err, **ch,
                     proxy_viol=proxy_violation(err),
                     final_loss=hist[-1]["train_loss"],
                     max_grad=max(h["grad_norm"] for h in hist))
            rows.append(r)
            print(f"  {arm:20s} normal={err['rmse_normal']:.5f} "
                  f"tangent={err['rmse_tangent']:.5f} "
                  f"cerr_H={ch['cerr_horizon']:.5f} proxy={r['proxy_viol']:.4f}")

    keys = sorted({k for r in rows for k in r})
    with open(f"{a.out}/raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    def agg(arm, k):
        return float(np.mean([r[k] for r in rows if r["arm"] == arm]))

    print("\n" + "=" * 88)
    print(f"{'arm':20s}{'normal':>10}{'tangent':>10}{'cerr_H':>11}{'cerr_near':>11}{'proxy':>10}{'vs base':>10}")
    base = agg("uniform", "proxy_viol")
    summary = {}
    for arm in arms:
        p = agg(arm, "proxy_viol")
        summary[arm] = {k: agg(arm, k) for k in
                        ("rmse_normal", "rmse_tangent", "rmse_all",
                         "cerr_horizon", "cerr_horizon_near", "proxy_viol")}
        print(f"{arm:20s}{agg(arm,'rmse_normal'):10.5f}{agg(arm,'rmse_tangent'):10.5f}"
              f"{agg(arm,'cerr_horizon'):11.5f}{agg(arm,'cerr_horizon_near'):11.5f}"
              f"{p:10.4f}{(p-base)/base:+9.1%}")

    cr, dw, un = (summary["constraint_rollout"], summary["dir_weighted"],
                  summary["uniform"])
    ms = summary["multistep_mse"]
    # PRIMARY signal: near-constraint horizon constraint-value error (the target).
    # SECONDARY: tangential blow-up, and the (biased) one-step proxy.
    d_uni = (cr["cerr_horizon_near"] - un["cerr_horizon_near"]) / un["cerr_horizon_near"]
    d_sta = (cr["cerr_horizon_near"] - dw["cerr_horizon_near"]) / dw["cerr_horizon_near"]
    d_ms = (cr["cerr_horizon_near"] - ms["cerr_horizon_near"]) / ms["cerr_horizon_near"]
    verdict = dict(
        cerr_near_vs_uniform=d_uni,
        cerr_near_vs_static=d_sta,
        cerr_near_vs_multistep=d_ms,          # isolates constraint geometry from horizon
        tangential_blowup_static=dw["rmse_tangent"] / un["rmse_tangent"],
        tangential_blowup_rollout=cr["rmse_tangent"] / un["rmse_tangent"],
        proxy_vs_uniform=(cr["proxy_viol"] - base) / base)
    print("\nverdict:", json.dumps(verdict, indent=2, default=float))
    go = (d_uni < -0.05 and d_sta < 0.0 and d_ms < 0.0
          and verdict["tangential_blowup_rollout"] < 1.5)
    print("\n>>> " + (
        "GO: rollout form improves near-constraint horizon error without a "
        "tangential blow-up, and beats BOTH static weighting and plain multistep. "
        "Run the reduced 3-arm Step 3."
        if go else
        "NO-GO / INCONCLUSIVE: check cerr_near_vs_multistep -- if that is >=0 the "
        "gain is coming from multi-step training, not from constraint geometry."))
    with open(f"{a.out}/summary.json", "w") as f:
        json.dump(dict(summary=summary, verdict=verdict, seeds=seeds,
                       config=vars(a)), f, indent=2, default=float)
    print(f"wrote {a.out}/raw.csv and summary.json")


if __name__ == "__main__":
    main()
