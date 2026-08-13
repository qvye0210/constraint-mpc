#!/usr/bin/env python3
"""Merge per-seed gate outputs produced by parallel --seed-offset runs.

    python merge_results.py results/gates_seed*    --out results/gates_merged
    python merge_results.py results/sweep_seed*    --out results/sweep_merged --sweep
"""

import argparse
import csv
import glob
import json
import os

import numpy as np

from cmpc2d.eval import paired_diff
from cmpc2d.gates import gate_a_verdict, gate_b_verdict, gate_dir_verdict


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        out = []
        for r in csv.DictReader(f):
            d = {}
            for k, v in r.items():
                try:
                    d[k] = float(v)
                except (TypeError, ValueError):
                    d[k] = v
            out.append(d)
        return out


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()

    dirs = sorted({d for pat in a.dirs for d in glob.glob(pat)})
    os.makedirs(a.out, exist_ok=True)
    print(f"merging {len(dirs)} run(s): {dirs}")
    report = dict(source_dirs=dirs)

    if a.sweep:
        rows = [r for d in dirs for r in read_csv(f"{d}/gate_a_sweep.csv")]
        agg = {}
        for r in rows:
            k = (r["n_traj"], r["hidden"], r["w_normal"], r["epochs"])
            agg.setdefault(k, []).append(r)
        out = []
        for k, rs in sorted(agg.items()):
            out.append(dict(n_traj=k[0], hidden=k[1], w_normal=k[2], epochs=k[3],
                            n_seeds=len(rs),
                            normal_rel=float(np.mean([r["normal_rel"] for r in rs])),
                            tangent_rel=float(np.mean([r["tangent_rel"] for r in rs])),
                            pass_frac=float(np.mean([str(r["passed"]) == "True" for r in rs]))))
        write_csv(f"{a.out}/gate_a_sweep_merged.csv", out)
        ok = [r for r in out if r["pass_frac"] > 0.5]
        print(f"  {len(ok)}/{len(out)} configs pass in >half the seeds")
        for r in sorted(ok, key=lambda r: r["normal_rel"])[:5]:
            print(f"   n_traj={r['n_traj']:.0f} hid={r['hidden']} w_n={r['w_normal']:.0f}"
                  f" | normal {r['normal_rel']:+.1%} tangent {r['tangent_rel']:+.1%}")
        report["sweep_pass"] = ok
    else:
        rows = [r for d in dirs for r in read_csv(f"{d}/gate_dir_raw.csv")]
        if rows:
            write_csv(f"{a.out}/gate_dir_raw.csv", rows)
            v = gate_dir_verdict(rows)
            report["gate_dir"] = v
            print(f"  gate_dir  ratio={v['ratio']:.2f}  "
                  f"{'PASS' if v['passed'] else 'FAIL'}  (n={len(rows)})")

        rows = [r for d in dirs for r in read_csv(f"{d}/gate_b_raw.csv")]
        if rows:
            write_csv(f"{a.out}/gate_b_raw.csv", rows)
            rt = [r for r in rows if r["arm"] == "true_dyn"]
            rl = [r for r in rows if r["arm"] == "learned_dyn"]
            v = gate_b_verdict(rt, rl)
            report["gate_b"] = v
            print(f"  gate_b    true={v['true_mean']:.5f} learned={v['learned_mean']:.5f}"
                  f"  {'PASS' if v['passed'] else 'FAIL'}  (n={len(rt)} pairs)")

        rows = [r for d in dirs for r in read_csv(f"{d}/gate_a_raw.csv")]
        if rows:
            write_csv(f"{a.out}/gate_a_raw.csv", rows)
            nr = float(np.mean([r["normal_rel_change"] for r in rows]))
            tr = float(np.mean([r["tangent_rel_change"] for r in rows]))
            pf = float(np.mean([str(r["passed"]) == "True" for r in rows]))
            report["gate_a"] = dict(normal_rel_change=nr, tangent_rel_change=tr,
                                    pass_frac=pf, passed=bool(pf > 0.5), n=len(rows))
            print(f"  gate_a    normal {nr:+.1%} tangent {tr:+.1%}"
                  f"  pass {pf:.0%} of {len(rows)} seed(s)")

    with open(f"{a.out}/merged_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"wrote {a.out}/merged_report.json")


if __name__ == "__main__":
    main()
