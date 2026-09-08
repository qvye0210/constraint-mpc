#!/usr/bin/env python3
"""Final root-cause cell: frozen mask backbone, head-only refits on rotH.
Arms per seed: mask-head (reference) | WLS-rotH (analytic, convex) |
adam-rotH (head-only) | WLS-mask (sanity). Hard check: WLS-rotH train
J_rotH must be <= mask-head's (mask head is in the feasible set).
EPS=0.0001 PYTHONPATH=. python head_audit.py"""
import copy, csv, os, sys
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, ".")
from revival_gate import get_apis, decompose, stats
from autopsy_weighting import rollout_windows
from gate12_matched import build_matched, solve_s, EPS
from switch_audit import posrot_metric

W = (64, 64); SEEDS = [0, 1, 2]; D = 0; EPOCHS = 2000
OUT = f"results/head_audit_eps{EPS:g}"; os.makedirs(OUT, exist_ok=True)
api = get_apis(); cwc = api["cwc"]
J = lambda e, M: float(np.einsum("bi,bij,bj->b", e, M, e).mean())


def last_linear(m):
    return [x for x in m.modules() if isinstance(x, nn.Linear)][-1]


def feats(m, pp):
    ll = last_linear(m); buf = {}
    h = ll.register_forward_hook(lambda mod, i, o: buf.__setitem__("f", i[0].detach()))
    with torch.no_grad():
        m(torch.tensor(pp["Xa"]), torch.tensor(pp["U"]))
    h.remove()
    return buf["f"].numpy().astype(np.float64)


def wls_head(Phi, pred_off, R, M, ridge=1e-9):
    """Solve min_A sum (A p_i - r_i)^T M_i (A p_i - r_i); p includes bias.
    pred_off: residual target already equals R (model predicts residual)."""
    P = np.concatenate([Phi, np.ones((len(Phi), 1))], 1)
    no, h = R.shape[1], P.shape[1]
    G = np.einsum("bkl,bj,bm->kjlm", M, P, P).reshape(no * h, no * h)
    b = np.einsum("bkl,bl,bj->kj", M, R.astype(np.float64), P).reshape(-1)
    a = np.linalg.solve(G + ridge * np.eye(no * h), b)
    return a.reshape(no, h)


def set_head(m, A):
    m2 = copy.deepcopy(m); ll = last_linear(m2)
    with torch.no_grad():
        ll.weight.copy_(torch.tensor(A[:, :-1], dtype=ll.weight.dtype))
        ll.bias.copy_(torch.tensor(A[:, -1], dtype=ll.bias.dtype))
    return m2


rows = []
for seed in SEEDS:
    d = api["build_dataset"](n_traj=60, seed=seed)
    ptr = cwc.prepare(d["train"], D, seed)
    pte = cwc.prepare(d["test"], D, seed + 500)
    rad = float(np.median(np.linalg.norm(
        ptr["X"][:, :2] - ptr["p_obs"], axis=1) - ptr["margin"]))
    s10, _ = solve_s(api, ptr, "rot", 10.0)
    Ms = {sp: dict(mask=build_matched(api, pp, "rot", 0.0),
                   rotH=build_matched(api, pp, "rot", s10),
                   pos=posrot_metric(api, pp, 10.0))
          for sp, pp in (("train", ptr), ("test", pte))}
    base, _, _ = cwc.train(ptr, Ms["train"]["mask"], W, EPOCHS, seed, D)
    Ftr, Fte = feats(base, ptr), feats(base, pte)
    R = ptr["R"].astype(np.float64)

    arms = {"mask-head": base}
    arms["wls-rotH"] = set_head(base, wls_head(Ftr, None, ptr["R"], Ms["train"]["rotH"]))
    arms["wls-mask"] = set_head(base, wls_head(Ftr, None, ptr["R"], Ms["train"]["mask"]))
    m3 = copy.deepcopy(base)
    for p_ in m3.parameters():
        p_.requires_grad_(False)
    ll = last_linear(m3); ll.weight.requires_grad_(True); ll.bias.requires_grad_(True)
    opt = torch.optim.Adam([ll.weight, ll.bias], lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    Xa, U, Rt = (torch.tensor(ptr[k]) for k in ("Xa", "U", "R"))
    Mt = torch.tensor(Ms["train"]["rotH"].astype(np.float32))
    rng = np.random.default_rng(777 + seed)
    for ep in range(EPOCHS):
        idx = rng.permutation(len(Xa))
        for i in range(0, len(idx), 256):
            b_ = idx[i:i + 256]
            e = m3(Xa[b_], U[b_]) - Rt[b_]
            loss = torch.einsum("bi,bij,bj->b", e, Mt[b_], e).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    arms["adam-rotH"] = m3

    for nm, m_ in arms.items():
        rec = dict(arm=nm, seed=seed)
        for sp, pp in (("train", ptr), ("test", pte)):
            with torch.no_grad():
                err = (m_(torch.tensor(pp["Xa"]), torch.tensor(pp["U"]))
                       - torch.tensor(pp["R"])).numpy()
            for on, MM in Ms[sp].items():
                rec[f"J_{on}_{sp}"] = J(err, MM)
        st = stats(decompose(api, m_, pte, None, D), rad)
        rec.update(rmse_n=st["rmse_n"], q95_1=st["q95_near"])
        res = rollout_windows(m_, d["test"], D, seed + 500, 10, api["Params"])
        near = res["m0"] <= np.quantile(res["m0"], .25)
        pref = np.maximum.accumulate(np.abs(res["r_rho"]), axis=1)
        for k in (5, 10):
            rec[f"q95_{k}"] = float(np.quantile(res["r_rho"][near, k - 1], .95))
            rec[f"pref_{k}"] = float(np.quantile(pref[:, k - 1], .90))
        rows.append(rec)
        print(f"  seed{seed} {nm}: J_rotH_tr {rec['J_rotH_train']:.3e} "
              f"q95_1 {rec['q95_1']:.2e} q95_10 {rec['q95_10']:.2e}")
    a, b_ = (next(r for r in rows if r["seed"] == seed and r["arm"] == n)
             for n in ("wls-rotH", "mask-head"))
    assert a["J_rotH_train"] <= b_["J_rotH_train"] * (1 + 1e-6), \
        "BUG: analytic WLS worse than mask head on its own train objective"

with open(f"{OUT}/head.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("\nMEAN over seeds (test):")
for nm in ("mask-head", "wls-mask", "wls-rotH", "adam-rotH"):
    sel = [r for r in rows if r["arm"] == nm]
    print(f"  {nm:>9}: J_rotH_tr {np.mean([r['J_rotH_train'] for r in sel]):.3e}"
          f"  J_rotH_te {np.mean([r['J_rotH_test'] for r in sel]):.3e}"
          f"  J_mask_te {np.mean([r['J_mask_test'] for r in sel]):.3e}"
          f"  q95_1 {np.mean([r['q95_1'] for r in sel]):.2e}"
          f"  q95_5 {np.mean([r['q95_5'] for r in sel]):.2e}"
          f"  q95_10 {np.mean([r['q95_10'] for r in sel]):.2e}"
          f"  pref_10 {np.mean([r['pref_10'] for r in sel]):.2e}")
print(f"wrote {OUT}/head.csv")
