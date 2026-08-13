#!/usr/bin/env python3
"""Tier-1 gate runner.

    python run_gates.py --quick               # ~minutes, code-path check
    python run_gates.py --full --seeds 3      # real run
    python run_gates.py --gate dir            # single gate

Nothing here overwrites existing experiment results: everything is written under
--out (default results/tier1_gates/).
"""

import argparse
import csv
import json
import os
import time

import numpy as np

from cmpc2d.data import build_dataset, coverage_report
from cmpc2d.env import Params
from cmpc2d.gates import (gate_a_verdict, gate_b_verdict, gate_dir_verdict,
                          run_gate_a, run_gate_b, run_gate_dir)
from cmpc2d.model import train_model
from cmpc2d.mpc import MPCConfig


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def check_health(name, hist):
    """NaN / gradient-explosion / weight-imbalance diagnostics."""
    issues = []
    losses = [h["train_loss"] for h in hist]
    gns = [h["grad_norm"] for h in hist]
    if any(not np.isfinite(l) for l in losses):
        issues.append("non-finite loss")
    if max(gns) > 1e3:
        issues.append(f"grad norm spike {max(gns):.1e}")
    if losses[-1] > losses[0]:
        issues.append("loss did not decrease")
    return dict(name=name, final_loss=losses[-1], max_grad=max(gns),
                issues=";".join(issues) if issues else "none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--gate", default="all", choices=["all", "dir", "b", "a"])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--out", default="results/tier1_gates")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hidden", default="64,64")
    args = ap.parse_args()
    quick = args.quick or not args.full

    cfgq = dict(n_traj=8, epochs=40, n_ep_dir=4, n_ep_b=8,
                eps_list=(-0.1, -0.05, 0.0, 0.05, 0.1)) if quick else \
           dict(n_traj=60, epochs=200, n_ep_dir=30, n_ep_b=50,
                eps_list=(-0.1,-0.05,-0.02,-0.01,0.0,0.01,0.02,0.05,0.1))
    hidden = tuple(int(h) for h in args.hidden.split(","))

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    report = dict(mode="quick" if quick else "full", cfg=cfgq, hidden=hidden,
                  seeds=args.seeds)
    print(f"[cfg] {report['mode']} hidden={hidden} seeds={args.seeds}")

    # ---------------- Gate DIR (no training) ----------------
    if args.gate in ("all", "dir"):
        print("\n=== Gate DIR: directional sanity ===")
        rows = []
        for s in range(args.seeds):
            r = run_gate_dir(eps_list=cfgq["eps_list"], n_ep=cfgq["n_ep_dir"], seed=s)
            for x in r:
                x["seed"] = s
            rows += r
        write_csv(f"{args.out}/gate_dir_raw.csv", rows)
        v = gate_dir_verdict(rows)
        report["gate_dir"] = v
        print(f"  |eps|={v['eps']}  normal(worst eps={v['eps_normal_worst']})={v['normal']:.4f}"
              f"  tangent={v['tangent']:.4f}"
              f"  ratio={v['ratio']:.2f}  -> {'PASS' if v['passed'] else 'FAIL'}")

    # ---------------- data + model (needed by Gate B / A) ----------------
    if args.gate in ("all", "b", "a"):
        print("\n=== Building dataset ===")
        data = build_dataset(n_traj=cfgq["n_traj"], seed=0, verbose=True)
        cov = {k: coverage_report(data[k]) for k in ("train", "val", "test")}
        report["coverage"] = cov
        print("  coverage(train):", {k: round(v, 3) for k, v in cov["train"].items()})

    health = []
    # ---------------- Gate B ----------------
    if args.gate in ("all", "b"):
        print("\n=== Gate B: violation attribution ===")
        rows_all = []
        for s in range(args.seeds):
            model, hist = train_model(data["train"], mode="uniform", hidden=hidden,
                                      epochs=cfgq["epochs"], seed=s, device=args.device)
            health.append(check_health(f"gate_b_seed{s}", hist))
            rt, rl = run_gate_b(model, n_ep=cfgq["n_ep_b"], seed=s)
            for r in rt: r.update(arm="true_dyn", seed=s)
            for r in rl: r.update(arm="learned_dyn", seed=s)
            rows_all += rt + rl
        write_csv(f"{args.out}/gate_b_raw.csv", rows_all)
        rt = [r for r in rows_all if r["arm"] == "true_dyn"]
        rl = [r for r in rows_all if r["arm"] == "learned_dyn"]
        v = gate_b_verdict(rt, rl)
        report["gate_b"] = v
        print(f"  true={v['true_mean']:.5f}  learned={v['learned_mean']:.5f}"
              f"  diff CI=[{v['ci_lo']:.4f},{v['ci_hi']:.4f}]"
              f"  -> {'PASS' if v['passed'] else 'FAIL'}")

    # ---------------- Gate A ----------------
    if args.gate in ("all", "a"):
        print("\n=== Gate A: capacity trade-off ===")
        rows = []
        for s in range(args.seeds):
            res = run_gate_a(data, hidden=hidden, epochs=cfgq["epochs"], seed=s,
                             device=args.device)
            health.append(check_health(f"gate_a_uniform_seed{s}", res["hist"][0]))
            health.append(check_health(f"gate_a_dirw_seed{s}", res["hist"][1]))
            v = gate_a_verdict(res)
            v["seed"] = s
            rows.append({**v, **{f"uni_{k}": x for k, x in res["uniform"].items()},
                         **{f"dir_{k}": x for k, x in res["dir_weighted"].items()}})
            print(f"  seed {s}: normal {v['normal_rel_change']:+.1%}  "
                  f"tangent {v['tangent_rel_change']:+.1%}  "
                  f"-> {'PASS' if v['passed'] else 'FAIL'}")
        write_csv(f"{args.out}/gate_a_raw.csv", rows)
        report["gate_a"] = dict(
            normal_rel_change=float(np.mean([r["normal_rel_change"] for r in rows])),
            tangent_rel_change=float(np.mean([r["tangent_rel_change"] for r in rows])),
            passed=bool(np.mean([r["passed"] for r in rows]) > 0.5))

    write_csv(f"{args.out}/health.csv", health)
    report["runtime_sec"] = time.time() - t0
    with open(f"{args.out}/report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)

    print("\n" + "=" * 60)
    for g in ("gate_dir", "gate_b", "gate_a"):
        if g in report:
            print(f"{g:9s} {'PASS' if report[g]['passed'] else 'FAIL'}")
    bad = [h for h in health if h["issues"] != "none"]
    print(f"health: {len(bad)} issue(s)" + (f" -> {bad}" if bad else ""))
    print(f"runtime {report['runtime_sec']:.1f}s   results in {args.out}/")


if __name__ == "__main__":
    main()
