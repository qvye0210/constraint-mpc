#!/usr/bin/env python3
"""Decisive check before investing in Step 3.

Two questions, both cheap and both able to kill the line:

  PART A -- is capacity (or data) actually binding?
      Sweep network width and dataset size, training uniform MSE only.
      If test error is flat across widths, there is nothing scarce to
      re-allocate and every direction-aware loss is a no-op by construction.
      The train/test gap says whether we are under- or over-fitting.

  PART B -- is ANY weighting operating point profitable?
      The earlier Gate A used a single point (w_normal = 50) and lost:
      16.6 % normal gain cost 120 % tangential degradation.  But the
      substitution rate is the SLOPE of a curve, not a constant.  This sweeps
      w_normal and computes the local slope at each point:

          substitution = d(tangential error) / |d(normal error)|
          price ratio  = s_normal / s_tangent  (~3.9, from Gate DIR)

      A point is profitable iff substitution < price ratio, equivalently iff
      the proxy violation drops.  If the whole curve sits above the price
      ratio, the method cannot win in this regime at any weighting.

    python capacity_check.py --quick
    python capacity_check.py --seeds 3
"""

import argparse
import csv
import json
import os

import numpy as np

from cmpc2d.data import build_dataset
from cmpc2d.env import Params
from cmpc2d.model import eval_errors, train_model

# Gate DIR sensitivities: closed-loop viol_integral per unit directional error.
SENS_NORMAL, SENS_TANGENT = 48.5, 12.3
PRICE_RATIO = SENS_NORMAL / SENS_TANGENT


def proxy(e):
    return SENS_NORMAL * e["rmse_normal"] + SENS_TANGENT * e["rmse_tangent"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--out", default="results/capacity_check")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--part", default="all", choices=["all", "a", "b", "c"])
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the training budget for parts A and B")
    ap.add_argument("--hidden", default="256,256,128",
                    help="architecture used by part C, e.g. 256,256,128")
    ap.add_argument("--w-list", default=None,
                    help="comma-separated w_normal values, e.g. 1,5,20,50")
    ap.add_argument("--n-traj", type=int, default=None,
                    help="override dataset size")
    ap.add_argument("--epoch-list", default="200,500,1000,2000",
                    help="training budgets swept by part C")
    a = ap.parse_args()

    seeds = list(range(a.seed_offset, a.seed_offset + a.seeds))
    os.makedirs(a.out, exist_ok=True)

    if a.quick:
        widths = [(8, 8), (32, 32), (128, 128)]
        n_trajs = [10, 30]
        epochs = 60
        w_list = [1.0, 2.0, 5.0, 20.0, 50.0]
        cap_for_b = [(16, 16), (64, 64)]
    else:
        widths = [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128), (256, 256, 128)]
        n_trajs = [15, 30, 60]
        epochs = 200
        w_list = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        cap_for_b = [(8, 8), (16, 16), (64, 64), (256, 256, 128)]

    if a.epochs is not None:
        epochs = a.epochs
    if a.w_list:
        w_list = [float(x) for x in a.w_list.split(",")]
    if a.n_traj is not None:
        n_trajs = [a.n_traj]

    cache = {}

    def get(n_traj, s):
        if (n_traj, s) not in cache:
            cache[(n_traj, s)] = build_dataset(n_traj=n_traj, seed=s)
        return cache[(n_traj, s)]

    report = {}

    # ---------------- PART A ----------------
    if a.part in ("all", "a"):
        print("=" * 84)
        print("PART A: is capacity / data binding?  (uniform MSE only)")
        print("=" * 84)
        rows = []
        for n_traj in n_trajs:
            for hid in widths:
                tr, te = [], []
                for s in seeds:
                    d = get(n_traj, s)
                    m, _ = train_model(d["train"], mode="uniform", hidden=hid,
                                       epochs=epochs, seed=s, device=a.device)
                    tr.append(eval_errors(m, d["train"], Params, a.device)["rmse_all"])
                    te.append(eval_errors(m, d["test"], Params, a.device)["rmse_all"])
                r = dict(n_traj=n_traj, hidden="x".join(map(str, hid)),
                         n_params=sum(np.prod(x) for x in zip([6] + list(hid), list(hid) + [4])),
                         train_rmse=float(np.mean(tr)), test_rmse=float(np.mean(te)))
                r["gap"] = r["test_rmse"] / r["train_rmse"]
                rows.append(r)
                print(f"  n_traj={n_traj:3d} hidden={r['hidden']:12s} "
                      f"train={r['train_rmse']:.5f} test={r['test_rmse']:.5f} "
                      f"test/train={r['gap']:.2f}")
        with open(f"{a.out}/part_a_capacity.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

        big = max(n_trajs)
        sub = [r for r in rows if r["n_traj"] == big]
        best, worst = min(r["test_rmse"] for r in sub), max(r["test_rmse"] for r in sub)
        span = worst / best
        binding = bool(span > 1.5)
        report["part_a"] = dict(width_span=float(span), capacity_binding=binding,
                                best_test=float(best), worst_test=float(worst))
        print(f"\n  widest/narrowest test error ratio = {span:.2f}  -> capacity "
              f"{'IS' if binding else 'is NOT'} binding")
        if not binding:
            print("  => nothing scarce to re-allocate; any direction-aware loss is a "
                  "no-op here by construction.")

    # ---------------- PART B ----------------
    if a.part in ("all", "b"):
        print("\n" + "=" * 84)
        print(f"PART B: exchange curve.  profitable iff substitution < {PRICE_RATIO:.2f}")
        print("=" * 84)
        rows = []
        n_traj = max(n_trajs)
        for hid in cap_for_b:
            ref = None
            print(f"\n  hidden={'x'.join(map(str,hid))}")
            print(f"    {'w_n':>7}{'e_normal':>11}{'e_tangent':>11}"
                  f"{'d_n':>9}{'d_t':>9}{'subst':>9}{'proxy':>10}{'vs w=1':>9}")
            for w in w_list:
                es = []
                for s in seeds:
                    d = get(n_traj, s)
                    mode = "uniform" if w == 1.0 else "dir_weighted"
                    m, _ = train_model(d["train"], mode=mode, hidden=hid, epochs=epochs,
                                       seed=s, w_normal=w, w_tangent=1.0, w_vel=1.0,
                                       device=a.device)
                    es.append(eval_errors(m, d["test"], Params, a.device))
                e = {k: float(np.mean([x[k] for x in es])) for k in es[0]}
                p = proxy(e)
                if ref is None:
                    ref, pref = e, p
                dn = (e["rmse_normal"] - ref["rmse_normal"]) / ref["rmse_normal"]
                dt = (e["rmse_tangent"] - ref["rmse_tangent"]) / ref["rmse_tangent"]
                sub = dt / abs(dn) if dn < -1e-9 else float("inf")
                rows.append(dict(hidden="x".join(map(str, hid)), w_normal=w,
                                 e_normal=e["rmse_normal"], e_tangent=e["rmse_tangent"],
                                 d_normal=dn, d_tangent=dt, substitution=sub,
                                 proxy=p, proxy_rel=(p - pref) / pref))
                print(f"    {w:>7.1f}{e['rmse_normal']:>11.5f}{e['rmse_tangent']:>11.5f}"
                      f"{dn:>+9.1%}{dt:>+9.1%}"
                      f"{(f'{sub:.2f}' if np.isfinite(sub) else '  n/a'):>9}"
                      f"{p:>10.4f}{(p-pref)/pref:>+9.1%}")
        with open(f"{a.out}/part_b_exchange.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

        win = [r for r in rows if r["proxy_rel"] < -0.01]
        report["part_b"] = dict(price_ratio=PRICE_RATIO, n_profitable=len(win),
                                best=min(rows, key=lambda r: r["proxy_rel"]))
        print(f"\n  profitable operating points: {len(win)}/{len(rows)}")
        if win:
            b = min(win, key=lambda r: r["proxy_rel"])
            print(f"  best: hidden={b['hidden']} w_normal={b['w_normal']} "
                  f"proxy {b['proxy_rel']:+.1%} (substitution {b['substitution']:.2f})")
            print("  => there IS a profitable regime; worth re-running the gates there.")
        else:
            print("  => the whole exchange curve is unprofitable; direction-aware "
                  "weighting cannot win in this regime at any weight.")

    # ---------------- PART C : does uniform catch up with more training? ------
    if a.part == "c":
        print("=" * 84)
        print("PART C: is the gain a real re-allocation, or just an optimisation effect?")
        print("If uniform closes the gap given a longer budget, the effect is not")
        print("decision-relevant capacity allocation and the method story does not hold.")
        print("=" * 84)
        hid = tuple(int(h) for h in a.hidden.split(","))
        ep_list = [int(x) for x in a.epoch_list.split(",")]
        n_traj = max(n_trajs)
        ws = w_list if a.w_list else [1.0, 20.0, 50.0]
        if 1.0 not in ws:
            ws = [1.0] + ws
        rows = []
        print(f"\n  hidden={'x'.join(map(str,hid))}  n_traj={n_traj}  seeds={seeds}")
        print(f"    {'epochs':>8}{'w_n':>7}{'e_normal':>11}{'e_tangent':>11}"
              f"{'proxy':>10}{'vs w=1':>9}")
        for ep in ep_list:
            ref_p = None
            for w in ws:
                es = []
                for s in seeds:
                    d = get(n_traj, s)
                    mode = "uniform" if w == 1.0 else "dir_weighted"
                    m, _ = train_model(d["train"], mode=mode, hidden=hid, epochs=ep,
                                       seed=s, w_normal=w, w_tangent=1.0, w_vel=1.0,
                                       device=a.device)
                    es.append(eval_errors(m, d["test"], Params, a.device))
                e = {k: float(np.mean([x[k] for x in es])) for k in es[0]}
                p = proxy(e)
                if ref_p is None:
                    ref_p = p
                rows.append(dict(epochs=ep, w_normal=w, e_normal=e["rmse_normal"],
                                 e_tangent=e["rmse_tangent"], proxy=p,
                                 proxy_rel=(p - ref_p) / ref_p))
                print(f"    {ep:>8}{w:>7.1f}{e['rmse_normal']:>11.6f}"
                      f"{e['rmse_tangent']:>11.6f}{p:>10.5f}{(p-ref_p)/ref_p:>+9.1%}")
        with open(f"{a.out}/part_c_convergence.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        best = {ep: min(r["proxy_rel"] for r in rows if r["epochs"] == ep) for ep in ep_list}
        print("\n  best advantage over uniform, by budget:")
        for ep in ep_list:
            print(f"    {ep:>6} epochs : {best[ep]:+.1%}")
        shrink = best[ep_list[-1]] > best[ep_list[0]] * 0.5
        report["part_c"] = dict(best_by_epochs={str(k): v for k, v in best.items()},
                                advantage_shrinks=bool(shrink))
        print("\n  => " + ("ADVANTAGE SHRINKS with budget: likely an optimisation "
                           "effect, not capacity re-allocation. The method story "
                           "does not hold as stated."
                           if shrink else
                           "ADVANTAGE PERSISTS: consistent with genuine directional "
                           "re-allocation. Re-run Gate A / Gate B at this setting."))

    with open(f"{a.out}/report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nwrote {a.out}/")


if __name__ == "__main__":
    main()
