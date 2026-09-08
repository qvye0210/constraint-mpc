#!/usr/bin/env python3
"""Switch audit (experiment 2). D=0, 64x64. From one converged mask
checkpoint per seed, three branches with IDENTICAL batch order and fresh
Adam for all (incl. continue-mask, so optimizer-reset is controlled):
  continue-mask | switch-rotH (matched H=10 rot k10) | switch-posrot
  (position-only local (n,t) k10, isotropic velocity, trace-matched,
   k=1 == mask elementwise -- asserted).
EPS=0.0001 PYTHONPATH=. python switch_audit.py"""
import csv, inspect, os, sys
import numpy as np
import torch

sys.path.insert(0, ".")
from revival_gate import get_apis, decompose, stats, rollout_windows
from gate12_matched import build_matched, solve_s, EPS

W = (64, 64); SEEDS = [0, 1, 2]; D = 0; EP_A = 2000; EP_B = 2000
KAPPA = 10.0; OUT = f"results/switch_audit_eps{EPS:g}"
os.makedirs(OUT, exist_ok=True)
api = get_apis(); cwc = api["cwc"]
NX = api["NX"]


def posrot_metric(api, prep, kappa):
    n = api["normal_dir"](prep["X"], prep["p_obs"])
    t = np.stack([-n[:, 1], n[:, 0]], -1)
    N = len(n); nx = prep["Xa"].shape[1]
    M = np.zeros((N, nx, nx))
    P = (2.0 / (kappa + 1)) * (kappa * n[:, :, None] * n[:, None, :]
                               + t[:, :, None] * t[:, None, :])
    M[:, :2, :2] = P
    M[:, 2, 2] = 1.0; M[:, 3, 3] = 1.0
    from cmpc2d.cweight import _finalise
    return _finalise(M + (EPS / nx) * np.eye(nx), 0.95)


def sig_default(name, fb):
    p = inspect.signature(cwc.train).parameters
    return p[name].default if name in p and p[name].default is not inspect._empty else fb


BS, LR = sig_default("bs", 256), sig_default("lr", 1e-3)
J = lambda err, M: float(np.einsum("bi,bij,bj->b", err, M, err).mean())


def branch_train(ckpt, prep_tr, prep_te, Ms_tr, Ms_te, M_train, seed, tag, rows):
    m = api["ResidualMLP"](W, n_dist=D); m.load_state_dict(ckpt)
    theta0 = torch.cat([v.flatten() for v in ckpt.values()])
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP_B)
    Xa, U, R = (torch.tensor(prep_tr[k]) for k in ("Xa", "U", "R"))
    Mt = torch.tensor(M_train.astype(np.float32))
    rng = np.random.default_rng(777 + seed)          # same for all branches
    g0 = None
    for ep in range(EP_B):
        idx = rng.permutation(len(Xa))
        for i in range(0, len(idx), BS):
            b = idx[i:i + BS]
            e = m(Xa[b], U[b]) - R[b]
            loss = torch.einsum("bi,bij,bj->b", e, Mt[b], e).mean()
            opt.zero_grad(); loss.backward()
            if g0 is None:
                g0 = float(torch.sqrt(sum((p.grad ** 2).sum()
                                          for p in m.parameters())))
            torch.nn.utils.clip_grad_norm_(m.parameters(), 10)
            opt.step()
        sch.step()
        if (ep + 1) % 100 == 0:
            with torch.no_grad():
                rec = dict(branch=tag, seed=seed, epoch=ep + 1, grad0=g0)
                th = torch.cat([v.flatten() for v in m.state_dict().values()])
                rec["drift"] = float((th - theta0).norm() / theta0.norm())
                for sp, pp, Ms in (("train", prep_tr, Ms_tr),
                                   ("test", prep_te, Ms_te)):
                    err = (m(torch.tensor(pp["Xa"]), torch.tensor(pp["U"]))
                           - torch.tensor(pp["R"])).numpy()
                    for nm, MM in Ms.items():
                        rec[f"J_{nm}_{sp}"] = J(err, MM)
                st = stats(decompose(api, m, prep_te, None, D), RAD[seed])
                rec.update(rmse_n=st["rmse_n"], rmse_t=st["rmse_t"],
                           q95_near=st["q95_near"])
                rows.append(rec)
    return m


rows, RAD = [], {}
for seed in SEEDS:
    d = api["build_dataset"](n_traj=60, seed=seed)
    ptr = cwc.prepare(d["train"], D, seed)
    pte = cwc.prepare(d["test"], D, seed + 500)
    RAD[seed] = float(np.median(np.linalg.norm(
        ptr["X"][:, :2] - ptr["p_obs"], axis=1) - ptr["margin"]))
    s10, a10 = solve_s(api, ptr, "rot", KAPPA)
    M_mask = build_matched(api, ptr, "rot", 0.0)
    M_rotH = build_matched(api, ptr, "rot", s10)
    M_pos = posrot_metric(api, ptr, KAPPA)
    M_pos1 = posrot_metric(api, ptr, 1.0)
    dchk = float(np.abs(M_pos1 - M_mask).max())
    print(f"seed{seed}: rotH achieved {a10:.3f}; posrot k=1 vs mask "
          f"max|diff| {dchk:.2e}")
    assert dchk < 1e-9, "posrot kappa=1 != mask -- construction rejected"
    Ms_tr = dict(mask=M_mask, rotH=M_rotH, pos=M_pos)
    Ms_te = dict(mask=build_matched(api, pte, "rot", 0.0),
                 rotH=build_matched(api, pte, "rot", s10),
                 pos=posrot_metric(api, pte, KAPPA))
    model, _, _ = cwc.train(ptr, M_mask, W, EP_A, seed, D)   # phase A
    ckpt = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for tag, M in (("cont-mask", M_mask), ("sw-rotH", M_rotH),
                   ("sw-posrot", M_pos)):
        mb = branch_train(ckpt, ptr, pte, Ms_tr, Ms_te, M, seed, tag, rows)
        for sp, dd, sz in (("test", d["test"], seed + 500),):
            res = rollout_windows(mb, dd, D, sz, 10, api["Params"])
            near = res["m0"] <= np.quantile(res["m0"], .25)
            for k in (5, 10):
                rows.append(dict(branch=tag, seed=seed, epoch=99000 + k,
                                 q95_near=float(np.quantile(
                                     res["r_rho"][near, k - 1], .95)),
                                 rmse_n=float(np.sqrt(
                                     (res["en"][:, k - 1] ** 2).mean()))))
        print(f"  {tag} seed{seed} done")
with open(f"{OUT}/switch.csv", "w", newline="") as f:
    keys = sorted({k for r in rows for k in r})
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

print("\nFINAL (ep 2000, test, mean over seeds); epoch=990xx rows = "
      "multi-step k=5/10:")
for tag in ("cont-mask", "sw-rotH", "sw-posrot"):
    sel = [r for r in rows if r["branch"] == tag and r.get("epoch") == EP_B]
    print(f"  {tag:>10}: J_mask {np.mean([r['J_mask_test'] for r in sel]):.3e}"
          f"  J_rotH {np.mean([r['J_rotH_test'] for r in sel]):.3e}"
          f"  J_pos {np.mean([r['J_pos_test'] for r in sel]):.3e}"
          f"  rmse_n {np.mean([r['rmse_n'] for r in sel]):.2e}"
          f"  q95 {np.mean([r['q95_near'] for r in sel]):.2e}"
          f"  drift {np.mean([r['drift'] for r in sel]):.3f}"
          f"  grad0 {np.mean([r['grad0'] for r in sel]):.2e}")
print(f"wrote {OUT}/switch.csv (per-100ep curves incl. all J's)")
