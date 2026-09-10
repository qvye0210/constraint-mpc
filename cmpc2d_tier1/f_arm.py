#!/usr/bin/env python3
"""Arm F: blockdiag, ZERO cross terms, spectrum EXACTLY equal to E-Hfull's
(post-finalise, floor included). Eig-to-block assignment pre-registered:
greedy pairing that best matches D-Hbdiag's block traces (printed).
Readout: F ~ E => conditioning; F ~ D => pos-vel cross coupling.
EPS=0.0001 PYTHONPATH=. python f_arm.py   (needs results/comp_audit_*/comp.csv)"""
import copy, csv, os, sys
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, ".")
from revival_gate import get_apis, decompose, stats
from autopsy_weighting import rollout_windows
from gate12_matched import build_matched, solve_s, EPS

W = (64, 64); SEEDS = [0, 1, 2]; D = 0; EP = 2000
OUT = f"results/f_arm_eps{EPS:g}"; os.makedirs(OUT, exist_ok=True)
api = api_ = get_apis(); cwc = api["cwc"]
LL = lambda m: [x for x in m.modules() if isinstance(x, nn.Linear)][-1]


def blocks_from(E):  # D-style blockdiag (for trace targets), comp_audit conv.
    Dm = np.zeros_like(E)
    Dm[:, :2, :2] = E[:, :2, :2]; Dm[:, 2:4, 2:4] = E[:, 2:4, 2:4]
    td = np.trace(Dm, axis1=1, axis2=2)[:, None, None]
    return Dm * (4.0 / td)


def f_metric(api, pp, E):
    lam = np.sort(np.linalg.eigvalsh(E[0]))          # constant across samples
    Dref = blocks_from(E)
    tp = float(np.trace(Dref[0, :2, :2]) / np.trace(Dref[0]) * np.trace(E[0]))
    best, ba = None, None
    import itertools
    for pair in itertools.combinations(range(4), 2):
        s_ = lam[list(pair)].sum()
        if best is None or abs(s_ - tp) < best:
            best, ba = abs(s_ - tp), pair
    pos_l = lam[list(ba)]
    vel_l = lam[[i for i in range(4) if i not in ba]]
    n = api["normal_dir"](pp["win_X"][:, 0], pp["p_obs"])
    t = np.stack([-n[:, 1], n[:, 0]], -1)
    M = np.zeros((len(n), 4, 4))
    for axes, ls, o in ((slice(0, 2), pos_l, 0), (slice(2, 4), vel_l, 2)):
        M[:, o:o+2, o:o+2] = (ls[1] * n[:, :, None] * n[:, None, :]
                              + ls[0] * t[:, :, None] * t[:, None, :])
    lf = np.sort(np.linalg.eigvalsh(M[0]))
    assert np.abs(lf - lam).max() < 1e-10, "spectrum mismatch"
    print(f"  F blocks: pos {pos_l.round(3)} vel {vel_l.round(3)} "
          f"(target pos trace {tp:.3f}, achieved {pos_l.sum():.3f}); "
          f"eig == E: OK")
    return M


def wls(Phi, R, M, sd, ridge=1e-9):
    Rs = R.astype(np.float64) / sd
    Ms = M * sd[None, :, None] * sd[None, None, :]
    P = np.concatenate([Phi, np.ones((len(Phi), 1))], 1)
    no, h = Rs.shape[1], P.shape[1]
    G = np.einsum("bkl,bj,bm->kjlm", Ms, P, P).reshape(no * h, no * h)
    b = np.einsum("bkl,bl,bj->kj", Ms, Rs, P).reshape(-1)
    return np.linalg.solve(G + ridge * np.eye(no * h), b).reshape(no, h)


def feats(m, pp):
    buf = {}; h = LL(m).register_forward_hook(
        lambda mod, i, o: buf.__setitem__("f", i[0].detach()))
    with torch.no_grad():
        m(torch.tensor(pp["Xa"]), torch.tensor(pp["U"]))
    h.remove(); return buf["f"].numpy().astype(np.float64)


def train_from(ckpt, ptr, M, seed):
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
    E = build_matched(api, ptr, "rot", s10)
    A = np.tile(np.eye(4) * 0.25, (len(E), 1, 1))    # trace-1 iso, floor moot
    F = f_metric(api, ptr, E)
    base, _, _ = cwc.train(ptr, np.tile(np.eye(4), (len(E), 1, 1)), W, EP,
                           seed, D)
    ck = {k: v.detach().clone() for k, v in base.state_dict().items()}
    m_ = train_from(ck, ptr, F, seed)
    with torch.no_grad():
        pass
    st = stats(decompose(api, m_, pte, None, D), rad)
    Phi = feats(m_, ptr); sd = m_.out_sd.numpy().astype(np.float64)
    A_head = wls(Phi, ptr["R"], np.tile(np.eye(4), (len(Phi), 1, 1)), sd)
    mh = copy.deepcopy(m_)
    with torch.no_grad():
        LL(mh).weight.copy_(torch.tensor(A_head[:, :-1],
                                         dtype=LL(mh).weight.dtype))
        LL(mh).bias.copy_(torch.tensor(A_head[:, -1], dtype=LL(mh).bias.dtype))
    st2 = stats(decompose(api, mh, pte, None, D), rad)
    res = rollout_windows(m_, d["test"], D, seed + 500, 10, api["Params"])
    near = res["m0"] <= np.quantile(res["m0"], .25)
    rows.append(dict(arm="F-specE", seed=seed, rmse_n=st["rmse_n"],
                     q95_1=st["q95_near"],
                     q95_10=float(np.quantile(res["r_rho"][near, 9], .95)),
                     rep_rmse_n=st2["rmse_n"]))
    print(f"  seed{seed} F: rmse_n {st['rmse_n']:.2e} q95_1 "
          f"{st['q95_near']:.2e} rep_rmse_n {st2['rmse_n']:.2e}")
with open(f"{OUT}/f.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"\nF mean: rmse_n {np.mean([r['rmse_n'] for r in rows]):.2e}  "
      f"q95_1 {np.mean([r['q95_1'] for r in rows]):.2e}  "
      f"rep {np.mean([r['rep_rmse_n'] for r in rows]):.2e}")
print("compare against comp_audit means: D rmse 1.65e-05 / E 1.93e-05 "
      "(per-seed rows in results/comp_audit_*/comp.csv)")
print(f"wrote {OUT}/f.csv")
