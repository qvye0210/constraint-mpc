#!/usr/bin/env python3
"""Cross-objective audit. 3 models (mask, matched-rot k5, k10) x 3 eval
objectives (each family's M), train/test, per-checkpoint curves, block
decomposition. Run: PYTHONPATH=. python cross_audit.py [--width 64,64]"""
import argparse, csv, os, sys
import numpy as np
import torch

sys.path.insert(0, ".")
from revival_gate import get_apis, decompose, stats
from gate12_matched import build_matched, solve_s

ap = argparse.ArgumentParser()
ap.add_argument("--width", default="64,64")
W = tuple(int(x) for x in ap.parse_args().width.split(","))
SEEDS = [0, 1, 2]; EPOCHS = 2000; D = 8
CK = tuple(sorted({250 * i for i in range(1, 9)}))
OUT = f"results/cross_audit_{W[0]}"; os.makedirs(OUT, exist_ok=True)
api = get_apis(); cwc = api["cwc"]
from cmpc2d.cweight import metric_loss

d0 = api["build_dataset"](n_traj=60, seed=SEEDS[0])
p00 = cwc.prepare(d0["train"], D, SEEDS[0])
S5, a5 = solve_s(api, p00, "rot", 5.0)
S10, a10 = solve_s(api, p00, "rot", 10.0)
print(f"doses: k5 s={S5:.5f} ach {a5:.3f} | k10 s={S10:.5f} ach {a10:.3f}")
ARMS = [("mask", 0.0), ("rot5", S5), ("rot10", S10)]


def blocks(err, M):
    e = err.numpy(); Mn = M.numpy() if torch.is_tensor(M) else M
    q = lambda ei, Mi: float(np.einsum("bi,bij,bj->b", ei, Mi, ei).mean())
    pos = q(e[:, :2], Mn[:, :2, :2]); vel = q(e[:, 2:4], Mn[:, 2:4, 2:4])
    cross = 2 * float(np.einsum("bi,bij,bj->b", e[:, :2],
                                Mn[:, :2, 2:4], e[:, 2:4]).mean())
    tot = float(np.einsum("bi,bij,bj->b", e, Mn, e).mean())
    return pos, vel, cross, tot - pos - vel - cross


rows = []
for seed in SEEDS:
    d = api["build_dataset"](n_traj=60, seed=seed)
    ptr = cwc.prepare(d["train"], D, seed)
    pte = cwc.prepare(d["test"], D, seed + 500)
    radius = float(np.median(np.linalg.norm(
        ptr["X"][:, :2] - ptr["p_obs"], axis=1) - ptr["margin"]))
    Ms = {sp: {nm: build_matched(api, pp, "rot", s) for nm, s in ARMS}
          for sp, pp in (("train", ptr), ("test", pte))}
    tens = {sp: {nm: torch.tensor(M.astype(np.float32))
                 for nm, M in Ms[sp].items()} for sp in Ms}
    for nm, s in ARMS:
        model, _, snaps = cwc.train(ptr, Ms["train"][nm], W, EPOCHS, seed, D,
                                    ckpts=CK)
        for ep, sd in sorted(snaps.items()):
            m2 = api["ResidualMLP"](W, n_dist=D); m2.load_state_dict(sd); m2.eval()
            for sp, pp in (("train", ptr), ("test", pte)):
                with torch.no_grad():
                    err = (m2(torch.tensor(pp["Xa"]), torch.tensor(pp["U"]))
                           - torch.tensor(pp["R"]))
                st = stats(decompose(api, m2, pp, None, D), radius)
                rec = dict(model=nm, seed=seed, epoch=ep, split=sp, **st)
                for en_, _s in ARMS:
                    rec[f"L_{en_}"] = float(metric_loss(err, tens[sp][en_]))
                if ep == CK[-1]:
                    p_, v_, c_, r_ = blocks(err, Ms[sp][nm])
                    rec.update(Lpos=p_, Lvel=v_, Lcross=c_, Lrest=r_)
                rows.append(rec)
        print(f"  {nm} seed{seed} done")
with open(f"{OUT}/audit.csv", "w", newline="") as f:
    keys = sorted({k for r in rows for k in r})
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

print("\nCROSS GRID at final epoch (mean over seeds; rows=model, cols=objective)")
for sp in ("train", "test"):
    print(f"-- {sp}")
    for nm, _ in ARMS:
        sel = [r for r in rows if r["model"] == nm and r["split"] == sp
               and r["epoch"] == CK[-1]]
        line = f"  {nm:>6}: " + "  ".join(
            f"L_{en_}={np.mean([r[f'L_{en_}'] for r in sel]):.3e}"
            for en_, _ in ARMS)
        line += (f"  | rmse_n {np.mean([r['rmse_n'] for r in sel]):.2e}"
                 f" q95 {np.mean([r['q95_near'] for r in sel]):.2e}"
                 f" bias {np.mean([r['bias_rho'] for r in sel]):+.1e}"
                 f" | blocks p/v/c/r "
                 + "/".join(f"{np.mean([r[k] for r in sel]):.2e}"
                            for k in ("Lpos", "Lvel", "Lcross", "Lrest")))
        print(line)
print(f"wrote {OUT}/audit.csv (per-epoch curves incl. false_safe)")
