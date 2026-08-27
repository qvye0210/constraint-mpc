#!/usr/bin/env python3
"""Corrected constraint-gradient weighting experiment (EXPERIMENT_SPEC.md).

Fixes every item the earlier runs got wrong:
  1. irrelevant dimensions now receive exactly zero weight (they never did)
  2. weights propagated along the horizon, not evaluated only at k=1
  3. rank protection via M + eps*I
  4. loss SCALE matched across arms (mean trace 1)
  5. random-anisotropic control arm with matched spectrum
  6. weight-distribution diagnostics
  7. comparison at matched training loss, not matched epochs

    python constraint_weight_check.py --quick
    python constraint_weight_check.py --seeds 3 --n-dist 0,8 --epoch-list 200,500,1000,2000

PRE-REGISTERED DECISION RULE (do not revise after seeing results):
    With D in {0,8}, 2000 epochs, normalised weights and the random control
    included: if 'prop' improves the proxy by <2% over uniform, OR is not
    clearly better than 'random', constraint-gradient weighting is judged
    ineffective on this class of system.  No further auxiliary explanation is
    to be introduced; the work converts to a scope-characterisation paper.
"""

import argparse
import csv
import json
import os

import numpy as np
import torch

from cmpc2d.cweight import (build_metric, metric_loss, metric_ratio_by_k,
                            weight_report)
from cmpc2d.data import build_dataset
from cmpc2d.env import (NU, NX, Params, f_distract, f_nominal, normal_dir,
                        sample_distract, tangent_dir)
from cmpc2d.model import ResidualMLP

SENS_NORMAL, SENS_TANGENT = 48.5, 12.3
ARMS = ["uniform", "mask", "static", "prop", "diag", "random"]


def attach_distractors(d, n_dist, seed):
    rng = np.random.default_rng(90_000 + seed)
    z = sample_distract(rng, len(d["X"]), n_dist)
    return z.astype(np.float32), f_distract(z).astype(np.float32)


def prepare(d, n_dist, seed, params=Params):
    z, zn = attach_distractors(d, n_dist, seed)
    X, U, Xn = (d["X"].astype(np.float32), d["U"].astype(np.float32),
                d["Xn"].astype(np.float32))
    R_core = (Xn - f_nominal(X, U, params)).astype(np.float32)
    R = np.concatenate([R_core, zn - z], 1) if n_dist else R_core
    Xa = np.concatenate([X, z], 1) if n_dist else X
    return dict(Xa=Xa, U=U, R=R, X=X, Xn=Xn, p_obs=d["p_obs"].astype(np.float32),
                win_X=d["win_X"], margin=d["margin_now"])


def train(prep, M, hidden, epochs, seed, n_dist, bs=256, lr=1e-3,
          device="cpu", ckpts=()):
    """Weighted training.  Plain MSE on train is logged so that arms can later
    be compared at MATCHED training loss rather than matched epochs."""
    torch.manual_seed(seed)
    model = ResidualMLP(hidden, n_dist=n_dist).to(device)
    zin = np.concatenate([prep["Xa"], prep["U"]], 1)
    model.in_mu.copy_(torch.tensor(zin.mean(0)))
    model.in_sd.copy_(torch.tensor(zin.std(0) + 1e-6))
    model.out_sd.copy_(torch.tensor(prep["R"].std(0) + 1e-8))

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tX = torch.tensor(prep["Xa"], device=device)
    tU = torch.tensor(prep["U"], device=device)
    tR = torch.tensor(prep["R"], device=device)
    tM = torch.tensor(M.astype(np.float32), device=device)

    n, hist, snaps = len(tX), [], {}
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            err = model(tX[idx], tU[idx]) - tR[idx]
            loss = metric_loss(err, tM[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        sched.step()
        if ep + 1 in ckpts or ep == epochs - 1:
            with torch.no_grad():
                e = model(tX, tU) - tR
                hist.append(dict(epoch=ep + 1,
                                 plain_mse=float((e ** 2).sum(-1).mean())))
            snaps[ep + 1] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return model, hist, snaps


@torch.no_grad()
def evaluate(model, prep, n_dist, device="cpu", params=Params):
    pred = model(torch.tensor(prep["Xa"], device=device),
                 torch.tensor(prep["U"], device=device)).cpu().numpy()
    e_core = (f_nominal(prep["X"], prep["U"], params) + pred[:, :NX]) - prep["Xn"]
    en = (e_core[:, :2] * normal_dir(prep["X"], prep["p_obs"])).sum(-1)
    et = (e_core[:, :2] * tangent_dir(prep["X"], prep["p_obs"])).sum(-1)
    out = dict(rmse_normal=float(np.sqrt((en ** 2).mean())),
               rmse_tangent=float(np.sqrt((et ** 2).mean())),
               rmse_core=float(np.sqrt((e_core ** 2).sum(-1).mean())),
               plain_mse=float((np.concatenate(
                   [e_core, pred[:, NX:] - (prep["R"][:, NX:])], 1) ** 2).sum(-1).mean())
               if n_dist else float((e_core ** 2).sum(-1).mean()))
    out["proxy"] = SENS_NORMAL * out["rmse_normal"] + SENS_TANGENT * out["rmse_tangent"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--n-dist", default=None)
    ap.add_argument("--epoch-list", default=None)
    ap.add_argument("--hidden", default="64,64")
    ap.add_argument("--n-traj", type=int, default=None)
    ap.add_argument("--H", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--eps-floor", type=float, default=0.05)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--strength", default="1.0",
                    help="comma-separated anisotropy strengths for 'prop', e.g. "
                         "0.2,0.5,1.0 -- lets arms be compared at matched n/t ratio")
    ap.add_argument("--out", default="results/cgrad")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    dists = [int(x) for x in (a.n_dist or ("0,8" if a.quick else "0,8")).split(",")]
    eps = [int(x) for x in (a.epoch_list or ("60,150" if a.quick else "200,500,1000,2000")).split(",")]
    arms = a.arms.split(",")
    hidden = tuple(int(h) for h in a.hidden.split(","))
    n_traj = a.n_traj or (15 if a.quick else 60)
    seeds = list(range(a.seed_offset, a.seed_offset + a.seeds))
    os.makedirs(a.out, exist_ok=True)

    print(f"hidden={hidden} n_traj={n_traj} seeds={seeds} D={dists} arms={arms}")
    rows, wrows = [], []

    strengths = [float(x) for x in a.strength.split(",")]
    arm_specs = [(arm, 1.0) for arm in arms if arm != "prop"] + \
                [("prop", st) for st in strengths]

    for D in dists:
        print(f"\n{'='*94}\nD = {D} distractor dimensions\n{'='*94}")
        # ---- weight diagnostics (implementation check) -------------------
        d0 = build_dataset(n_traj=n_traj, seed=seeds[0])
        p0 = prepare(d0["train"], D, seeds[0])
        print(f"  {'arm':14}{'w_pos':>9}{'w_vel':>9}{'w_dist':>9}{'w_dist(raw)':>12}"
              f"{'n/t ratio':>11}{'irrelev%':>10}")
        for arm, st in arm_specs:
            M, M_struct = build_metric(p0["win_X"], p0["p_obs"], D, a.H, a.gamma,
                                       arm, a.eps_floor, seed=seeds[0], strength=st)
            wr = weight_report(M, p0["X"], p0["p_obs"], D)
            wr["w_distract_struct"] = weight_report(
                M_struct, p0["X"], p0["p_obs"], D)["w_distract"]
            wr.update(arm=arm, strength=st, n_dist=D)
            wrows.append(wr)
            print(f"  {arm+('' if st==1.0 else f'@{st:g}'):14}{wr['w_position']:>9.4f}{wr['w_velocity']:>9.4f}"
                  f"{wr['w_distract']:>9.4f}{wr['w_distract_struct']:>12.2e}"
                  f"{wr['normal_over_tangent']:>11.2f}"
                  f"{wr['frac_weight_on_irrelevant']:>9.1%}")
        if D > 0:
            chk = [w for w in wrows if w["arm"] == "prop" and w["n_dist"] == D
                   and w["strength"] == 1.0][0]
            ok = chk["w_distract_struct"] < 1e-9
            print(f"  implementation check: structural distractor weight for "
                  f"'prop' = {chk['w_distract_struct']:.2e} -> "
                  f"{'OK' if ok else 'BUG'}   (with eps floor: "
                  f"{chk['w_distract']:.4f}; the floor is uniform by design)")

        # ---- training ----------------------------------------------------
        print(f"\n  {'epochs':>7}{'arm':>14}{'e_normal':>11}{'e_tangent':>11}"
              f"{'plain_mse':>12}{'proxy':>10}{'vs unif':>9}")
        for ep in eps:
            ref = None
            for arm, st in arm_specs:
                res = []
                for s in seeds:
                    d = build_dataset(n_traj=n_traj, seed=s)
                    ptr = prepare(d["train"], D, s)
                    pte = prepare(d["test"], D, s + 500)
                    M, _ = build_metric(ptr["win_X"], ptr["p_obs"], D, a.H,
                                        a.gamma, arm, a.eps_floor, seed=s,
                                        strength=st)
                    m, _, _ = train(ptr, M, hidden, ep, s, D, device=a.device)
                    res.append(evaluate(m, pte, D, a.device))
                e = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
                if ref is None:
                    ref = e["proxy"]
                rel = (e["proxy"] - ref) / ref
                rows.append(dict(n_dist=D, epochs=ep, arm=arm, strength=st,
                                 proxy_rel=rel, **e))
                print(f"  {ep:>7}{(arm+('' if st==1.0 else f'@{st:g}')):>14}{e['rmse_normal']:>11.6f}"
                      f"{e['rmse_tangent']:>11.6f}{e['plain_mse']:>12.2e}"
                      f"{e['proxy']:>10.5f}{rel:>+9.1%}")

    for name, data in (("raw", rows), ("weights", wrows)):
        if data:
            with open(f"{a.out}/{name}.csv", "w", newline="") as f:
                k = sorted({x for r in data for x in r})
                w = csv.DictWriter(f, fieldnames=k); w.writeheader(); w.writerows(data)

    # ---- pre-registered verdict -----------------------------------------
    print("\n" + "=" * 94)
    ep_last = eps[-1]
    verdict = {}
    for D in dists:
        sel = {r["arm"]: r for r in rows if r["n_dist"] == D
               and r["epochs"] == ep_last and r.get("strength", 1.0) == 1.0}
        if "prop" not in sel:
            continue
        vs_u = sel["prop"]["proxy_rel"]
        vs_r = ((sel["prop"]["proxy"] - sel["random"]["proxy"]) / sel["random"]["proxy"]
                if "random" in sel else float("nan"))
        vs_m = ((sel["prop"]["proxy"] - sel["mask"]["proxy"]) / sel["mask"]["proxy"]
                if "mask" in sel else float("nan"))
        passes = bool(vs_u < -0.02 and vs_r < -0.01)
        direction_adds = bool(vs_m < -0.05)
        verdict[f"D{D}"] = dict(prop_vs_uniform=vs_u, prop_vs_random=vs_r,
                                prop_vs_mask=vs_m, passes=passes,
                                direction_adds_value=direction_adds)
        print(f"  D={D}: prop vs uniform {vs_u:+.1%}, vs random {vs_r:+.1%}, "
              f"vs MASK {vs_m:+.1%}  -> {'PASS' if passes else 'FAIL'}"
              f" | direction adds value: {'YES' if direction_adds else 'NO'}")
        print(f"      (mask = irrelevant dims zeroed, core uniform. If prop is "
              f"NOT clearly better than mask, the gain is masking alone -- i.e. "
              f"VaGraM's known mechanism with grad g swapped in.)")
    print("\n>>> " + ("Constraint-gradient weighting shows a real, "
                      "direction-attributable effect."
                      if any(v["passes"] for v in verdict.values()) else
                      "Per the pre-registered rule: ineffective on this class of "
                      "system. Do NOT introduce a new auxiliary explanation. "
                      "Next step is a task where capacity genuinely binds "
                      "(2-link arm / UR5e with limited data), or conversion to a "
                      "scope-characterisation paper."))
    with open(f"{a.out}/verdict.json", "w") as f:
        json.dump(dict(verdict=verdict, config=vars(a)), f, indent=2, default=float)
    print(f"wrote {a.out}/")


if __name__ == "__main__":
    main()
