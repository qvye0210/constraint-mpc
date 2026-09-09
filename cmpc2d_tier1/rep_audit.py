#!/usr/bin/env python3
"""Representation 2x2 (final cell, advisor-amended stop line).
{cont-mask, sw-rotH} backbone x {WLS-mask, WLS-rotH} head, D=0, 64x64.
EPS=0.0001 PYTHONPATH=. python rep_audit.py
Readout (pre-registered): rotH-backbone+WLS-mask ~= mask-backbone+WLS-mask
=> features intact, damage readout-level; worse under BOTH heads =>
representation itself degraded (destructive drift)."""
import copy, csv, os, sys
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, ".")
from revival_gate import get_apis, decompose, stats
from autopsy_weighting import rollout_windows
from gate12_matched import build_matched, solve_s, EPS

W = (64, 64); SEEDS = [0, 1, 2]; D = 0; EP = 2000
OUT = f"results/rep_audit_eps{EPS:g}"; os.makedirs(OUT, exist_ok=True)
api = get_apis(); cwc = api["cwc"]
J = lambda e, M: float(np.einsum("bi,bij,bj->b", e, M, e).mean())
LL = lambda m: [x for x in m.modules() if isinstance(x, nn.Linear)][-1]


def feats(m, pp):
    buf = {}; h = LL(m).register_forward_hook(
        lambda mod, i, o: buf.__setitem__("f", i[0].detach()))
    with torch.no_grad():
        m(torch.tensor(pp["Xa"]), torch.tensor(pp["U"]))
    h.remove(); return buf["f"].numpy().astype(np.float64)


def wls(Phi, R, M, sd, ridge=1e-9):
    Rs = R.astype(np.float64) / sd
    Ms = M * sd[None, :, None] * sd[None, None, :]
    P = np.concatenate([Phi, np.ones((len(Phi), 1))], 1)
    no, h = Rs.shape[1], P.shape[1]
    G = np.einsum("bkl,bj,bm->kjlm", Ms, P, P).reshape(no * h, no * h)
    b = np.einsum("bkl,bl,bj->kj", Ms, Rs, P).reshape(-1)
    return np.linalg.solve(G + ridge * np.eye(no * h), b).reshape(no, h)


def with_head(m, A):
    m2 = copy.deepcopy(m); ll = LL(m2)
    with torch.no_grad():
        ll.weight.copy_(torch.tensor(A[:, :-1], dtype=ll.weight.dtype))
        ll.bias.copy_(torch.tensor(A[:, -1], dtype=ll.bias.dtype))
    return m2


def cont_train(ckpt, ptr, M, seed):
    m = api["ResidualMLP"](W, n_dist=D); m.load_state_dict(ckpt)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)
    Xa, U, R = (torch.tensor(ptr[k]) for k in ("Xa", "U", "R"))
    Mt = torch.tensor(M.astype(np.float32))
    rng = np.random.default_rng(777 + seed)
    for ep in range(EP):
        idx = rng.permutation(len(Xa))
        for i in range(0, len(idx), 256):
            b = idx[i:i + 256]
            e = m(Xa[b], U[b]) - R[b]
            loss = torch.einsum("bi,bij,bj->b", e, Mt[b], e).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 10)
            opt.step()
        sch.step()
    return m


rows = []
for seed in SEEDS:
    d = api["build_dataset"](n_traj=60, seed=seed)
    ptr = cwc.prepare(d["train"], D, seed)
    pte = cwc.prepare(d["test"], D, seed + 500)
    rad = float(np.median(np.linalg.norm(
        ptr["X"][:, :2] - ptr["p_obs"], axis=1) - ptr["margin"]))
    s10, _ = solve_s(api, ptr, "rot", 10.0)
    Mm_tr = build_matched(api, ptr, "rot", 0.0)
    Mr_tr = build_matched(api, ptr, "rot", s10)
    Mm_te = build_matched(api, pte, "rot", 0.0)
    Mr_te = build_matched(api, pte, "rot", s10)
    base, _, _ = cwc.train(ptr, Mm_tr, W, EP, seed, D)
    ck = {k: v.detach().clone() for k, v in base.state_dict().items()}
    bks = {"bb-mask": cont_train(ck, ptr, Mm_tr, seed),
           "bb-rotH": cont_train(ck, ptr, Mr_tr, seed)}
    for bn, bm in bks.items():
        Phi = feats(bm, ptr); sd = bm.out_sd.numpy().astype(np.float64)
        for hn, Mh in (("h-mask", Mm_tr), ("h-rotH", Mr_tr)):
            mdl = with_head(bm, wls(Phi, ptr["R"], Mh, sd))
            with torch.no_grad():
                err = (mdl(torch.tensor(pte["Xa"]), torch.tensor(pte["U"]))
                       - torch.tensor(pte["R"])).numpy()
            st = stats(decompose(api, mdl, pte, None, D), rad)
            res = rollout_windows(mdl, d["test"], D, seed + 500, 10,
                                  api["Params"])
            near = res["m0"] <= np.quantile(res["m0"], .25)
            rows.append(dict(backbone=bn, head=hn, seed=seed,
                             J_mask=J(err, Mm_te), J_rotH=J(err, Mr_te),
                             rmse_n=st["rmse_n"], q95_1=st["q95_near"],
                             q95_10=float(np.quantile(
                                 res["r_rho"][near, 9], .95))))
            print(f"  seed{seed} {bn}+{hn}: J_mask {rows[-1]['J_mask']:.3e} "
                  f"q95_1 {rows[-1]['q95_1']:.2e} q95_10 {rows[-1]['q95_10']:.2e}")
with open(f"{OUT}/rep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("\n2x2 MEAN over seeds (test):")
for bn in ("bb-mask", "bb-rotH"):
    for hn in ("h-mask", "h-rotH"):
        sel = [r for r in rows if r["backbone"] == bn and r["head"] == hn]
        print(f"  {bn}+{hn}: J_mask {np.mean([r['J_mask'] for r in sel]):.3e}"
              f"  J_rotH {np.mean([r['J_rotH'] for r in sel]):.3e}"
              f"  rmse_n {np.mean([r['rmse_n'] for r in sel]):.2e}"
              f"  q95_1 {np.mean([r['q95_1'] for r in sel]):.2e}"
              f"  q95_10 {np.mean([r['q95_10'] for r in sel]):.2e}")
print(f"wrote {OUT}/rep.csv")
