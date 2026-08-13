#!/usr/bin/env python3
"""Locate the capacity-constrained regime where Gate A can pass.

Gate A is the gate most likely to fail for a *fixable* reason (too much model
capacity / too much data -> re-allocation is a no-op).  This sweeps the knobs.
    python sweep_gate_a.py --quick
"""
import argparse, csv, itertools, json, os
import numpy as np
from cmpc2d.data import build_dataset
from cmpc2d.gates import run_gate_a, gate_a_verdict

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--out", default="results/tier1_gates")
ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--seed-offset", type=int, default=0,
                help="first seed index; use with --seeds 1 to parallelise")
a = ap.parse_args()

if a.quick:
    grid = dict(n_traj=[8], hidden=[(16,16),(64,64)], w_normal=[10.,50.], epochs=[60])
else:
    grid = dict(n_traj=[15,60], hidden=[(16,16),(32,32),(64,64),(256,256,128)],
                w_normal=[3.,10.,30.,100.], epochs=[200])

os.makedirs(a.out, exist_ok=True)
seeds = list(range(a.seed_offset, a.seed_offset + a.seeds))
rows = []
cache = {}
for n_traj, hidden, wn, ep in itertools.product(grid["n_traj"], grid["hidden"],
                                                grid["w_normal"], grid["epochs"]):
    vs = []
    for s in seeds:
        if (n_traj, s) not in cache:
            cache[(n_traj, s)] = build_dataset(n_traj=n_traj, seed=s)
        res = run_gate_a(cache[(n_traj, s)], hidden=hidden, epochs=ep,
                         w_normal=wn, seed=s)
        vs.append(gate_a_verdict(res))
    r = dict(n_traj=n_traj, hidden="x".join(map(str, hidden)), w_normal=wn, epochs=ep,
             seeds="|".join(map(str, seeds)),
             normal_rel=float(np.mean([v["normal_rel_change"] for v in vs])),
             tangent_rel=float(np.mean([v["tangent_rel_change"] for v in vs])),
             passed=bool(np.mean([v["passed"] for v in vs]) > .5))
    rows.append(r)
    print(f"n_traj={n_traj:3d} hid={r['hidden']:12s} w_n={wn:5.0f} | "
          f"normal {r['normal_rel']:+.1%}  tangent {r['tangent_rel']:+.1%}  "
          f"{'PASS' if r['passed'] else 'fail'}")

with open(f"{a.out}/gate_a_sweep.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"\nwrote {a.out}/gate_a_sweep.csv   ({sum(r['passed'] for r in rows)}/{len(rows)} passed)")
