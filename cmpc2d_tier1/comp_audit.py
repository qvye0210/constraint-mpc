#!/usr/bin/env python3
"""Component decomposition of rotH. D=0, 64x64, seeds 0-2.
Arms (same phase-A mask ckpt, same batch order, same finalise/trace path):
  A mse | B local-pos(k10) | C H-pos | D H-blockdiag | E H-full(rotH)
Endpoints per arm: direct test metrics; frozen-backbone analytic h-mask
refit (representation quality); plus FREE gradient diagnostics at the
shared checkpoint (component norms, pairwise cosines, neg-cos fraction,
batch variance H1 vs H10).  EPS=0.0001 PYTHONPATH=. python comp_audit.py"""
import copy, csv, os, sys
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, ".")
from revival_gate import get_apis, decompose, stats
from autopsy_weighting import rollout_windows
from gate12_matched import build_matched, solve_s, EPS

W = (64, 64); SEEDS = [0, 1, 2]; D = 0; EP = 2000
OUT = f"results/comp_audit_eps{EPS:g}"; os.makedirs(OUT, exist_ok=True)
api = get_apis(); cwc = api["cwc"]
J = lambda e, M: float(np.einsum("bi,bij,bj->b", e, M, e).mean())
LL = lambda m: [x for x in m.modules() if isinstance(x, nn.Linear)][-1]


def posrot(api, pp, kappa):
    n = api["normal_dir"](pp["X"], pp["p_obs"])
    t = np.stack([-n[:, 1], n[:, 0]], -1)
    M = np.zeros((len(n), 4, 4))
    M[:, :2, :2] = (2 / (kappa + 1)) * (kappa * n[:, :, None] * n[:, None, :]
                                        + t[:, :, None] * t[:, None, :])
    M[:, 2, 2] = M[:, 3, 3] = 1.0
    return M


def finalise(M):
    from cmpc2d.cweight import _finalise
    return _finalise(M + (EPS / 4) * np.eye(4), 0.95)


def arms_for(pp, s10):
    E = build_matched(api, pp, "rot", s10)          # already finalised
    core = E.copy()
    C = np.zeros_like(core); Dm = np.zeros_like(core)
    C[:, :2, :2] = core[:, :2, :2]
    tp = np.trace(C, axis1=1, axis2=2)[:, None, None]
    C = C * (2.0 / tp); C[:, 2, 2] = C[:, 3, 3] = 1.0
    Dm[:, :2, :2] = core[:, :2, :2]; Dm[:, 2:4, 2:4] = core[:, 2:4, 2:4]
    td = np.trace(Dm, axis1=1, axis2=2)[:, None, None]
    Dm = Dm * (4.0 / td)
    A = np.tile(np.eye(4), (len(E), 1, 1))
    return {"A-mse": finalise(A), "B-local": finalise(posrot(api, pp, 10.0)),
            "C-Hpos": finalise(C), "D-Hbdiag": finalise(Dm), "E-Hfull": E}


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


def grad_diag(base, ptr, E, seed):
    """Backbone grads of component losses at shared ckpt, 20 batches."""
    comps = {}
    Mpp = np.zeros_like(E); Mpp[:, :2, :2] = E[:, :2, :2]
    Mvv = np.zeros_like(E); Mvv[:, 2:4, 2:4] = E[:, 2:4, 2:4]
    Mpv = E - Mpp - Mvv
    Ms = {"mse": np.tile(np.eye(4), (len(E), 1, 1)), "p_H": Mpp,
          "v_H": Mvv, "pv": Mpv, "p_1": posrot(api, ptr, 10.0) * 0}
    Ms["p_1"][:, :2, :2] = posrot(api, ptr, 10.0)[:, :2, :2]
    bb = [p for n, p in base.named_parameters() if "net.4" not in n]
    Xa, U, R = (torch.tensor(ptr[k]) for k in ("Xa", "U", "R"))
    rng = np.random.default_rng(99)
    G = {k: [] for k in Ms}
    for _ in range(20):
        b = rng.choice(len(Xa), 256, replace=False)
        for k, M in Ms.items():
            e = base(Xa[b], U[b]) - R[b]
            L = torch.einsum("bi,bij,bj->b", e,
                             torch.tensor(M[b].astype(np.float32)), e).mean()
            g = torch.autograd.grad(L, bb, retain_graph=False,
                                    allow_unused=True)
            G[k].append(torch.cat([x.flatten() for x in g if x is not None]
                                  ).numpy())
    out = {}
    for k, gs in G.items():
        gs = np.stack(gs); mu = gs.mean(0)
        out[k] = dict(norm=float(np.linalg.norm(mu)),
                      bvar=float(np.mean(np.linalg.norm(gs - mu, axis=1) ** 2)
                                 / max(np.linalg.norm(mu) ** 2, 1e-30)))
    cos = lambda a, b_: float((a @ b_) / (np.linalg.norm(a)
                                          * np.linalg.norm(b_) + 1e-30))
    mus = {k: np.stack(v).mean(0) for k, v in G.items()}
    pairs = [("p_H", "v_H"), ("p_H", "pv"), ("v_H", "pv"),
             ("p_H", "mse"), ("p_1", "mse"), ("p_1", "p_H")]
    negfrac = {}
    for a, b_ in pairs:
        cs = [cos(x, y) for x, y in zip(G[a], G[b_])]
        negfrac[f"{a}~{b_}"] = (float(np.mean(cs)), float(np.mean(
            np.array(cs) < 0)))
    print(f"  [grad seed{seed}] " + " ".join(
        f"{k}:|g|{v['norm']:.1e},bv{v['bvar']:.2f}" for k, v in out.items()))
    print("               cos(mean,negfrac): " + " ".join(
        f"{k}:{v[0]:+.2f}/{v[1]:.2f}" for k, v in negfrac.items()))


rows = []
for seed in SEEDS:
    d = api["build_dataset"](n_traj=60, seed=seed)
    ptr = cwc.prepare(d["train"], D, seed)
    pte = cwc.prepare(d["test"], D, seed + 500)
    rad = float(np.median(np.linalg.norm(
        ptr["X"][:, :2] - ptr["p_obs"], axis=1) - ptr["margin"]))
    s10, _ = solve_s(api, ptr, "rot", 10.0)
    AR_tr, AR_te = arms_for(ptr, s10), arms_for(pte, s10)
    for nm, M in AR_tr.items():
        w_ = np.linalg.eigvalsh(M[0])
        print(f"  seed{seed} {nm}: trace {np.trace(M[0]):.3f} "
              f"eig[{w_.min():.3f},{w_.max():.3f}]")
    base, _, _ = cwc.train(ptr, AR_tr["A-mse"], W, EP, seed, D)
    ck = {k: v.detach().clone() for k, v in base.state_dict().items()}
    grad_diag(base, ptr, AR_tr["E-Hfull"], seed)
    for nm in AR_tr:
        m_ = train_from(ck, ptr, AR_tr[nm], seed)
        with torch.no_grad():
            err = (m_(torch.tensor(pte["Xa"]), torch.tensor(pte["U"]))
                   - torch.tensor(pte["R"])).numpy()
        st = stats(decompose(api, m_, pte, None, D), rad)
        Phi = feats(m_, ptr); sd = m_.out_sd.numpy().astype(np.float64)
        mh = copy.deepcopy(m_); llw = wls(Phi, ptr["R"], AR_tr["A-mse"], sd)
        with torch.no_grad():
            LL(mh).weight.copy_(torch.tensor(llw[:, :-1],
                                             dtype=LL(mh).weight.dtype))
            LL(mh).bias.copy_(torch.tensor(llw[:, -1],
                                           dtype=LL(mh).bias.dtype))
        st2 = stats(decompose(api, mh, pte, None, D), rad)
        res = rollout_windows(m_, d["test"], D, seed + 500, 10, api["Params"])
        near = res["m0"] <= np.quantile(res["m0"], .25)
        rows.append(dict(arm=nm, seed=seed, J_mask=J(err, AR_te["A-mse"]),
                         rmse_n=st["rmse_n"], q95_1=st["q95_near"],
                         q95_10=float(np.quantile(res["r_rho"][near, 9], .95)),
                         rep_rmse_n=st2["rmse_n"], rep_q95_1=st2["q95_near"]))
        print(f"  seed{seed} {nm}: rmse_n {st['rmse_n']:.2e} "
              f"q95_1 {st['q95_1'] if 'q95_1' in st else st['q95_near']:.2e} "
              f"rep_rmse_n {st2['rmse_n']:.2e}")
with open(f"{OUT}/comp.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("\nMEAN over seeds (test):  [rep_* = frozen backbone + analytic h-mask]")
for nm in ("A-mse", "B-local", "C-Hpos", "D-Hbdiag", "E-Hfull"):
    sel = [r for r in rows if r["arm"] == nm]
    print(f"  {nm:>9}: rmse_n {np.mean([r['rmse_n'] for r in sel]):.2e}"
          f"  q95_1 {np.mean([r['q95_1'] for r in sel]):.2e}"
          f"  q95_10 {np.mean([r['q95_10'] for r in sel]):.2e}"
          f"  rep_rmse_n {np.mean([r['rep_rmse_n'] for r in sel]):.2e}"
          f"  rep_q95_1 {np.mean([r['rep_q95_1'] for r in sel]):.2e}")
print(f"wrote {OUT}/comp.csv")
