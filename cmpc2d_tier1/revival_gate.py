#!/usr/bin/env python3
"""Revival gate for directional weighting (minimal package, pre-registered).

Run from cmpc2d project root (PYTHONPATH=.), env constraint_mpc. Does NOT
modify any existing project file; uses train()'s ckpts snapshots for
per-epoch train/test decomposition.

    # smoke test first (times one training, checks APIs):
    PYTHONPATH=. python revival_gate.py --quick
    # full pilot (overnight):
    PYTHONPATH=. python revival_gate.py

PRE-REGISTERED SPEC (2026-09-07, do not revise after results):
  arms    : mask, prop at achieved kappa targets {1,2,5,10,20,40,74}
            (kappa=1 kept SEPARATE from mask: velocity block differs via
            horizon propagation; script prints the matrix difference)
  seeds   : 0,1,2 (pilot);  widths: 64x64 and 32x32;  D=8; 2000 epochs
  stratum : near-boundary = margin_now <= 25th percentile of the split
            (fixed in advance; quantile is not to be changed post hoc)
  signs   : rho > 0 means safe; delta_rho = rho_hat - rho; positive =
            model optimistic (dangerous side)
  primary : near-boundary delta_rho q95 (and q90), per arm, held-out
  guards  : mean rmse_n worsening <= 5% vs mask; false-safe rate not
            increased vs mask (false-safe = rho_hat>0 & rho<=0; if
            violating test samples are too few, q95 stays primary --
            do NOT alter test data to manufacture events)
  EXIT (revival PASSES if): some kappa in {5,10,20}, on at least one
            width, improves near-boundary q95 by >=10% vs mask, with the
            improvement direction consistent in >=2 of 3 seeds, and both
            guards hold. Then -> 5-seed formal replication (same cells),
            then closed loop. Otherwise the directional line is SEALED as
            a method; fixed-axis / decoupled-head / DxWidth mechanism
            experiments only if a negative-result analysis is written.
"""

import argparse, csv, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, ".")


def get_apis():
    from cmpc2d.cweight import build_metric, weight_report
    from cmpc2d.data import build_dataset
    from cmpc2d.env import NX, Params, f_nominal, normal_dir, tangent_dir
    from cmpc2d.model import ResidualMLP
    import constraint_weight_check as cwc
    return dict(build_metric=build_metric, weight_report=weight_report,
                build_dataset=build_dataset, NX=NX, Params=Params,
                f_nominal=f_nominal, normal_dir=normal_dir,
                tangent_dir=tangent_dir, ResidualMLP=ResidualMLP, cwc=cwc)


def metric_for(api, p0, D, arm, strength, seed, H=10, gamma=0.9, eps=0.05):
    out = api["build_metric"](p0["win_X"], p0["p_obs"], D, H, gamma, arm,
                              eps, seed=seed, strength=strength)
    return out[0] if isinstance(out, tuple) else out


def achieved_kappa(api, p0, D, s, seed):
    M = metric_for(api, p0, D, "prop", s, seed)
    return api["weight_report"](M, p0["X"], p0["p_obs"], D)["normal_over_tangent"], M


def invert_kappas(api, p0, D, targets, seed):
    """Bisection on strength for each target ratio. Returns
    [(target, strength, achieved)] with unreachable targets clamped."""
    lo_s, hi_s = 1e-4, 1.0
    k_lo, _ = achieved_kappa(api, p0, D, lo_s, seed)
    k_hi, _ = achieved_kappa(api, p0, D, hi_s, seed)
    out = []
    for kt in targets:
        if kt <= k_lo:
            out.append((kt, lo_s, k_lo)); continue
        if kt >= k_hi:
            out.append((kt, hi_s, k_hi)); continue
        a, b = lo_s, hi_s
        for _ in range(40):
            m = 0.5 * (a + b)
            km, _ = achieved_kappa(api, p0, D, m, seed)
            if km < kt: a = m
            else: b = m
        km, _ = achieved_kappa(api, p0, D, 0.5 * (a + b), seed)
        out.append((kt, 0.5 * (a + b), km))
    return out


@torch.no_grad()
def decompose(api, model, prep, d_raw, D):
    """Per-sample one-step decomposition on a prepared split.
    Returns dict of arrays: en, et, drho (=rho_hat - rho, radius-free),
    rho_true (needs radius), margin_now."""
    NX = api["NX"]
    pred = model(torch.tensor(prep["Xa"]), torch.tensor(prep["U"])).numpy()
    xhat = api["f_nominal"](prep["X"], prep["U"], api["Params"]) + pred[:, :NX]
    e = xhat[:, :2] - prep["Xn"][:, :2]
    n = api["normal_dir"](prep["X"], prep["p_obs"])
    t = api["tangent_dir"](prep["X"], prep["p_obs"])
    po = prep["p_obs"]
    dhat = np.linalg.norm(xhat[:, :2] - po, axis=1)
    dtru = np.linalg.norm(prep["Xn"][:, :2] - po, axis=1)
    return dict(en=(e * n).sum(-1), et=(e * t).sum(-1), drho=dhat - dtru,
                dist_true=dtru, margin=np.asarray(prep["margin"]))


def stats(dec, radius, near_q=0.25):
    near = dec["margin"] <= np.quantile(dec["margin"], near_q)
    rho_true = dec["dist_true"] - radius
    rho_hat = rho_true + dec["drho"]
    viol = rho_true <= 0
    return dict(
        rmse_n=float(np.sqrt((dec["en"] ** 2).mean())),
        rmse_t=float(np.sqrt((dec["et"] ** 2).mean())),
        bias_rho=float(dec["drho"].mean()),
        q90_near=float(np.quantile(dec["drho"][near], .90)),
        q95_near=float(np.quantile(dec["drho"][near], .95)),
        n_near=int(near.sum()),
        n_viol=int(viol.sum()),
        false_safe=float(np.mean(rho_hat[viol] > 0)) if viol.any() else float("nan"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--widths", default="64,64;32,32")
    ap.add_argument("--kappas", default="1,2,5,10,20,40,74")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--out", default="results/revival_gate")
    a = ap.parse_args()
    api = get_apis(); cwc = api["cwc"]

    seeds = [int(s) for s in a.seeds.split(",")]
    widths = [tuple(int(x) for x in w.split(",")) for w in a.widths.split(";")]
    kappas = [float(k) for k in a.kappas.split(",")]
    epochs = a.epochs
    if a.quick:
        seeds, widths, kappas, epochs = [0], [(64, 64)], [1, 10, 74], 200
    ck = tuple(sorted({max(1, epochs // 8) * i for i in range(1, 9)} | {epochs}))
    os.makedirs(a.out, exist_ok=True)
    print(f"seeds={seeds} widths={widths} kappa_targets={kappas} "
          f"epochs={epochs} ckpts={ck} D={a.D}")
    print("signs: rho>0 safe; drho=rho_hat-rho; positive=optimistic(dangerous). "
          "near = margin_now<=q25 (pre-registered).")

    # ---- kappa inversion + kappa=1 vs mask matrix comparison (seed0 data)
    d0 = api["build_dataset"](n_traj=60, seed=seeds[0])
    p0 = cwc.prepare(d0["train"], a.D, seeds[0])
    inv = invert_kappas(api, p0, a.D, kappas, seeds[0])
    print("kappa inversion (target -> strength -> achieved):")
    for kt, s, ka in inv:
        print(f"  {kt:>6.1f} -> s={s:.5f} -> achieved {ka:.3f}")
    M_mask = metric_for(api, p0, a.D, "mask", 1.0, seeds[0])
    _, M_k1 = achieved_kappa(api, p0, a.D, inv[0][1], seeds[0])
    dmax = float(np.abs(M_mask - M_k1).max())
    print(f"kappa=1 vs mask: max |M_diff| = {dmax:.3e} -> "
          + ("IDENTICAL (kappa=1 anchors mask)" if dmax < 1e-9 else
             "DIFFERENT (kept as separate arms; expected: velocity block "
             "carries propagated structure in prop family)"))

    arm_specs = [("mask", None, None)] + [
        (f"k{kt:g}", s, ka) for kt, s, ka in inv]

    rows, curves = [], []
    t0 = time.time()
    for hidden in widths:
        for seed in seeds:
            d = api["build_dataset"](n_traj=60, seed=seed)
            ptr = cwc.prepare(d["train"], a.D, seed)
            pte = cwc.prepare(d["test"], a.D, seed + 500)
            r = np.linalg.norm(ptr["X"][:, :2] - ptr["p_obs"], axis=1) - ptr["margin"]
            radius, rstd = float(np.median(r)), float(np.std(r))
            if rstd > 1e-6:
                print(f"WARNING seed{seed}: derived radius std {rstd:.2e} -- "
                      "margin formula differs; false_safe/absolute-rho unreliable, "
                      "drho quantiles remain valid")
            for name, s, ka in arm_specs:
                arm, st = ("mask", 1.0) if name == "mask" else ("prop", s)
                M = metric_for(api, ptr, a.D, arm, st, seed)
                model, hist, snaps = cwc.train(ptr, M, hidden, epochs, seed,
                                               a.D, ckpts=ck)
                for ep, sd in sorted(snaps.items()):
                    m2 = api["ResidualMLP"](hidden, n_dist=a.D)
                    m2.load_state_dict(sd); m2.eval()
                    for split, prep in (("train", ptr), ("test", pte)):
                        dec = decompose(api, m2, prep, None, a.D)
                        st_ = stats(dec, radius)
                        curves.append(dict(width=str(hidden), seed=seed,
                                           arm=name, kappa=ka or 0.0,
                                           epoch=ep, split=split, **st_))
                        if ep == epochs and split == "test":
                            np.savez(f"{a.out}/res_{hidden[0]}_{seed}_{name}.npz",
                                     **dec)
                            rows.append(curves[-1])
                el = time.time() - t0
                print(f"  [{el/60:5.1f}m] w={hidden} seed={seed} {name:>5} "
                      f"(kappa {ka if ka else 'mask'}): test rmse_n "
                      f"{rows[-1]['rmse_n'] if rows else float('nan'):.2e} "
                      f"q95_near {rows[-1]['q95_near']:.2e}")

    for fn, data in (("final.csv", rows), ("curves.csv", curves)):
        with open(f"{a.out}/{fn}", "w", newline="") as f:
            keys = sorted({k for r in data for k in r})
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(data)

    # ---- pre-registered exit evaluation (baseline arm = mask) ----------
    print("\n" + "=" * 90)
    print("REVIVAL GATE (baseline = mask; all 6 mid-kappa cells printed in full)")
    verdict = {"passes": False, "cells": []}
    for hidden in widths:
        base = {s: next(r for r in rows if r["width"] == str(hidden)
                        and r["seed"] == s and r["arm"] == "mask") for s in seeds}
        for name in [n for n, _, _ in arm_specs if n != "mask"]:
            kt = float(name[1:])
            cell = [r for r in rows if r["width"] == str(hidden) and r["arm"] == name]
            if not cell:
                continue
            imp = [(base[r["seed"]]["q95_near"] - r["q95_near"])
                   / max(abs(base[r["seed"]]["q95_near"]), 1e-12) for r in cell]
            dn = [r["rmse_n"] / base[r["seed"]]["rmse_n"] - 1 for r in cell]
            fs_ok = all(not (r["false_safe"] > base[r["seed"]]["false_safe"])
                        for r in cell if r["false_safe"] == r["false_safe"])
            rec = dict(width=str(hidden), kappa=kt,
                       q95_improve_mean=float(np.mean(imp)),
                       q95_improve_seeds=[float(i) for i in imp],
                       rmse_n_change_mean=float(np.mean(dn)), fs_guard=fs_ok)
            mid = kt in (5, 10, 20)
            ok = (mid and np.mean(imp) >= 0.10
                  and sum(i > 0 for i in imp) >= 2
                  and np.mean(dn) <= 0.05 and fs_ok)
            rec["passes"] = bool(ok)
            verdict["cells"].append(rec)
            verdict["passes"] |= ok
            tag = " <-- PASS" if ok else ""
            print(f"  w={hidden} kappa={kt:>5.1f}: q95_near improve "
                  f"{np.mean(imp):+.1%} (seeds {['%+.0f%%' % (100*i) for i in imp]}), "
                  f"rmse_n {np.mean(dn):+.1%}, fs_guard={fs_ok}"
                  f"{' [mid-k gate cell]' if mid else ''}{tag}")
    print("\n>>> " + ("REVIVAL: proceed to 5-seed formal replication on the "
                      "passing cell(s), same criteria."
                      if verdict["passes"] else
                      "SEALED per pre-registered exit: no mid-kappa dangerous-"
                      "tail benefit on either width. Mechanism experiments "
                      "(fixed-axis, decoupled heads, DxWidth) only if writing "
                      "the negative-result analysis."))
    with open(f"{a.out}/gate.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"wrote {a.out}/ (final.csv, curves.csv, gate.json, res_*.npz)")


if __name__ == "__main__":
    main()
