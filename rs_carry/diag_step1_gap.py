#!/usr/bin/env python3
"""STEP 1 GATE (of the two pre-registered gates): is the residual dynamics of
the carry task genuinely hard to learn, for the right reason?

Everything upstream of the method depends on this. The gate PASSES only if
  gap        > 5   (test/train MSE at the larger budget: model cannot fit it)
  traj/trans < 5   (the difficulty is NOT distribution shift between trajectories)
and constraint coverage (near+viol) should sit roughly in 10-40%.

The learning target is the residual AFTER an identified linear nominal
(see rscarry/data.py for why). The report includes the linear-explained
fraction; if that is ~99% the old free-space trap has returned and nothing
else in the output means anything.

    python diag_step1_gap.py --quick        (~15 min: 40 traj, 200/800 epochs)
    python diag_step1_gap.py                (~1 h:   120 traj, 200/2000)
"""
import argparse, json, os
import numpy as np
from rscarry.data import build_dataset, coverage, fit_linear_nominal, apply_nominal
from rscarry.model import train, mse

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--n-traj", type=int, default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/step1")
a = ap.parse_args()
n = a.n_traj or (40 if a.quick else 120)
lo, hi = (200, 800) if a.quick else (200, 2000)
os.makedirs(a.out, exist_ok=True)

print(f"collecting {n} trajectories ...")
d = build_dataset(n_traj=n, seed=a.seed)
cov = coverage(d["train"])
print(f"coverage: viol {cov['frac_violating']:.1%}  near {cov['frac_near']:.1%}  "
      f"far {cov['frac_far']:.1%}   residual rms {cov['residual_rms']:.4f}")

# --- linear-trap guard ---------------------------------------------------
tr = d["train"]
Yc = tr["Xn"] - tr["Xn"].mean(0)
lin_expl = 1.0 - (tr["R"]**2).sum() / max((Yc**2).sum(), 1e-12)
print(f"linear nominal explains {lin_expl:.1%} of next-state variance "
      f"(the residual below is the remaining nonlinear part)")

# --- trajectory vs transition split -------------------------------------
ts = tr["trajs"] + d["val"]["trajs"] + d["test"]["trajs"]
W = d["_W"]
def pack(sel):
    X = np.concatenate([t["X"] for t in sel]); U = np.concatenate([t["U"] for t in sel])
    Y = np.concatenate([t["Xn"] for t in sel])
    return dict(X=X, U=U, R=Y - apply_nominal(W, X, U))
n_tr = int(0.8 * len(ts))
A_tr, A_te = pack(ts[:n_tr]), pack(ts[n_tr:])
allp = pack(ts); N = len(allp["X"])
idx = np.random.default_rng(a.seed).permutation(N)
B_tr = {k: v[idx[:int(.8*N)]] for k, v in allp.items()}
B_te = {k: v[idx[int(.8*N):]] for k, v in allp.items()}

out = {}
print(f"\n{'split':>12}{'budget':>8}{'train':>11}{'test':>11}{'gap':>7}")
for lbl, TR, TE in (("trajectory", A_tr, A_te), ("transition", B_tr, B_te)):
    for ep in (lo, hi):
        m = train(TR, epochs=ep, seed=a.seed)
        r = dict(train=mse(m, TR), test=mse(m, TE))
        out[f"{lbl}_{ep}"] = r
        print(f"{lbl:>12}{ep:>8}{r['train']:>11.3e}{r['test']:>11.3e}"
              f"{r['test']/r['train']:>7.1f}")

gap = out[f"trajectory_{hi}"]["test"] / out[f"trajectory_{hi}"]["train"]
ratio = out[f"trajectory_{hi}"]["test"] / out[f"transition_{hi}"]["test"]
passed = bool(gap > 5 and ratio < 5)
print(f"\n  gap={gap:.1f} (need >5)   traj/trans={ratio:.2f} (need <5)")
print(">>> " + ("STEP 1 PASS: dynamics genuinely hard, not an extrapolation "
                "artefact. Run diag_step2_compound.py."
                if passed else
                "STEP 1 FAIL. If gap<=5 the task is still too easy -> raise "
                "payload range or exploration, ONCE. If traj/trans>=5 the "
                "trajectory distribution is too spread -> narrow targets. Do "
                "not iterate more than once on each knob; if it still fails, "
                "this task does not support the method -- stop here."))
json.dump(dict(gap=gap, traj_over_trans=ratio, passed=passed,
               linear_explained=lin_expl, coverage=cov, runs=out),
          open(f"{a.out}/verdict.json", "w"), indent=2, default=float)
print(f"wrote {a.out}/verdict.json")
