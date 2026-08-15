#!/usr/bin/env python3
"""Do distracting state dimensions restore a usable capacity trade-off?

Background.  `capacity_check.py --part c` showed that on the clean 2D testbed the
advantage of direction-weighted training over uniform MSE vanishes as the
training budget grows (-22.9 % at 200 epochs, +0.0 % at 2000).  The one-step
error kept falling monotonically, i.e. capacity was never actually scarce, so
"re-allocating capacity by decision-relevance" had nothing to re-allocate.

This script tests the fix used by VaGraM (Voelcker et al., ICLR 2022): append
superfluous state dimensions following an independent nonlinear dynamical
system.  Unlike shrinking the network, distractors impose an error floor that
does NOT disappear with a longer budget -- which is the property the premise
needs, and the property that survives a reviewer saying "just train longer".

Two questions, in order:

  Q1  Does the error floor appear?
      Sweep the training budget at each distractor count.  If e_normal for the
      uniform model stops falling, capacity is genuinely binding.

  Q2  Does the advantage survive the budget?
      If the direction-weighted advantage holds at the largest budget, it is a
      real re-allocation.  If it decays as before, distractors did not fix it.

Offline only -- no closed-loop MPC.  Closed-loop would additionally require
propagating the distractor state inside the MPC rollout; that is only worth
building if this check passes.

    python distract_check.py --quick
    python distract_check.py --seeds 3 --n-dist 0,4,8,16 --epoch-list 200,500,1000,2000
"""

import argparse
import csv
import json
import os

import numpy as np
import torch

from cmpc2d.data import build_dataset
from cmpc2d.env import (NU, NX, Params, f_distract, f_nominal, normal_dir,
                        sample_distract, tangent_dir)
from cmpc2d.model import ResidualMLP

SENS_NORMAL, SENS_TANGENT = 48.5, 12.3
PRICE_RATIO = SENS_NORMAL / SENS_TANGENT


def attach_distractors(d, n_dist, seed):
    """Add independent distractor states to every transition.

    The distractors are autonomous and independent of (x, u), so a fresh draw
    per transition is equivalent to reading them off a trajectory, and it keeps
    the data pipeline untouched.
    """
    rng = np.random.default_rng(90_000 + seed)
    z = sample_distract(rng, len(d["X"]), n_dist)
    return z.astype(np.float32), f_distract(z).astype(np.float32)


def train(data, z, zn, mode, w_normal, hidden, epochs, seed, n_dist,
          bs=256, lr=1e-3, device="cpu", params=Params):
    """One-step training on the augmented state; identical for every arm."""
    torch.manual_seed(seed)
    X, U, Xn, OB = (data["X"].astype(np.float32), data["U"].astype(np.float32),
                    data["Xn"].astype(np.float32), data["p_obs"].astype(np.float32))
    R_core = (Xn - f_nominal(X, U, params)).astype(np.float32)
    R = np.concatenate([R_core, zn - z], axis=1) if n_dist else R_core
    Xa = np.concatenate([X, z], axis=1) if n_dist else X

    model = ResidualMLP(hidden, n_dist=n_dist).to(device)
    zin = np.concatenate([Xa, U], axis=1)
    model.in_mu.copy_(torch.tensor(zin.mean(0)))
    model.in_sd.copy_(torch.tensor(zin.std(0) + 1e-6))
    model.out_sd.copy_(torch.tensor(R.std(0) + 1e-8))

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tX = torch.tensor(Xa, device=device)
    tU = torch.tensor(U, device=device)
    tR = torch.tensor(R, device=device)
    nrm = torch.tensor(normal_dir(X, OB).astype(np.float32), device=device)
    tan = torch.tensor(tangent_dir(X, OB).astype(np.float32), device=device)

    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            err = model(tX[idx], tU[idx]) - tR[idx]
            if mode == "uniform":
                loss = (err ** 2).sum(-1).mean()
            else:
                e_pos = err[:, :2]
                en = (e_pos * nrm[idx]).sum(-1)
                et = (e_pos * tan[idx]).sum(-1)
                rest = (err[:, 2:] ** 2).sum(-1)
                loss = (w_normal * en ** 2 + et ** 2 + rest).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        sched.step()
    return model


@torch.no_grad()
def evaluate(model, data, z, zn, n_dist, device="cpu", params=Params):
    X, U, Xn, OB = (data["X"].astype(np.float32), data["U"].astype(np.float32),
                    data["Xn"].astype(np.float32), data["p_obs"].astype(np.float32))
    Xa = np.concatenate([X, z], axis=1) if n_dist else X
    pred = model(torch.tensor(Xa, device=device),
                 torch.tensor(U, device=device)).cpu().numpy()
    e_core = (f_nominal(X, U, params) + pred[:, :NX]) - Xn
    en = (e_core[:, :2] * normal_dir(X, OB)).sum(-1)
    et = (e_core[:, :2] * tangent_dir(X, OB)).sum(-1)
    out = dict(rmse_normal=float(np.sqrt((en ** 2).mean())),
               rmse_tangent=float(np.sqrt((et ** 2).mean())),
               rmse_core=float(np.sqrt((e_core ** 2).sum(-1).mean())))
    if n_dist:
        ed = pred[:, NX:] - (zn - z)
        out["rmse_distract"] = float(np.sqrt((ed ** 2).sum(-1).mean()))
    out["proxy"] = SENS_NORMAL * out["rmse_normal"] + SENS_TANGENT * out["rmse_tangent"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--n-dist", default=None, help="e.g. 0,4,8,16")
    ap.add_argument("--epoch-list", default=None, help="e.g. 200,500,1000,2000")
    ap.add_argument("--w-list", default="1,5,20,50")
    ap.add_argument("--hidden", default="64,64")
    ap.add_argument("--n-traj", type=int, default=None)
    ap.add_argument("--out", default="results/distract_check")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    dists = [int(x) for x in (a.n_dist or ("0,8" if a.quick else "0,4,8,16")).split(",")]
    eps = [int(x) for x in (a.epoch_list or ("60,150" if a.quick else "200,500,1000,2000")).split(",")]
    ws = [float(x) for x in a.w_list.split(",")]
    if 1.0 not in ws:
        ws = [1.0] + ws
    hidden = tuple(int(h) for h in a.hidden.split(","))
    n_traj = a.n_traj or (15 if a.quick else 60)
    seeds = list(range(a.seed_offset, a.seed_offset + a.seeds))
    os.makedirs(a.out, exist_ok=True)

    print(f"hidden={hidden} n_traj={n_traj} seeds={seeds} distractors={dists}")
    data_cache = {s: build_dataset(n_traj=n_traj, seed=s) for s in seeds}

    rows = []
    for D in dists:
        print(f"\n{'='*80}\nD = {D} distractor dimensions\n{'='*80}")
        print(f"  {'epochs':>7}{'w_n':>6}{'e_normal':>12}{'e_tangent':>12}"
              f"{'e_distract':>12}{'proxy':>10}{'vs w=1':>9}")
        for ep in eps:
            ref = None
            for w in ws:
                res = []
                for s in seeds:
                    d = data_cache[s]
                    ztr, zntr = attach_distractors(d["train"], D, s)
                    zte, znte = attach_distractors(d["test"], D, s + 500)
                    m = train(d["train"], ztr, zntr,
                              "uniform" if w == 1.0 else "dir", w, hidden, ep, s, D,
                              device=a.device)
                    res.append(evaluate(m, d["test"], zte, znte, D, a.device))
                e = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
                if ref is None:
                    ref = e["proxy"]
                rel = (e["proxy"] - ref) / ref
                rows.append(dict(n_dist=D, epochs=ep, w_normal=w, proxy_rel=rel, **e))
                print(f"  {ep:>7}{w:>6.1f}{e['rmse_normal']:>12.6f}"
                      f"{e['rmse_tangent']:>12.6f}"
                      f"{e.get('rmse_distract', float('nan')):>12.6f}"
                      f"{e['proxy']:>10.5f}{rel:>+9.1%}")

    with open(f"{a.out}/raw.csv", "w", newline="") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 80)
    print("Q1  does an error floor appear?   (uniform e_normal vs budget)")
    report = {}
    for D in dists:
        v = [next(r["rmse_normal"] for r in rows if r["n_dist"] == D
                  and r["epochs"] == ep and r["w_normal"] == 1.0) for ep in eps]
        floor = v[-1] / v[0]
        print(f"  D={D:<3} " + "  ".join(f"{ep}:{x:.6f}" for ep, x in zip(eps, v))
              + f"   last/first={floor:.2f}"
              + ("  <- saturated" if floor > 0.5 else "  <- still falling"))
        report[f"D{D}_floor_ratio"] = float(floor)

    print("\nQ2  does the advantage survive the budget?")
    ok = []
    for D in dists:
        best = [min(r["proxy_rel"] for r in rows if r["n_dist"] == D and r["epochs"] == ep)
                for ep in eps]
        keeps = best[-1] < -0.02 and best[-1] <= best[0] * 0.5
        ok.append(keeps)
        print(f"  D={D:<3} " + "  ".join(f"{ep}:{x:+.1%}" for ep, x in zip(eps, best))
              + ("   <- PERSISTS" if keeps else "   <- decays"))
        report[f"D{D}_best_by_epoch"] = {str(e): float(b) for e, b in zip(eps, best)}

    report["any_persists"] = bool(any(ok))
    print("\n>>> " + ("SOME D PERSISTS: distractors restore a real trade-off. "
                      "Re-run the gates at that D."
                      if any(ok) else
                      "NO D PERSISTS: distractors do not fix the premise on this "
                      "plant. Move to the 2-link arm with structural "
                      "misspecification (Stribeck friction, unobserved payload)."))
    with open(f"{a.out}/report.json", "w") as f:
        json.dump(dict(report=report, config=vars(a)), f, indent=2, default=float)
    print(f"wrote {a.out}/")


if __name__ == "__main__":
    main()
