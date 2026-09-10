#!/usr/bin/env python3
"""Prefix-risk gate (phase 1, offline). See spec in repo discussion.
  PYTHONPATH=. python prefix_risk_gate.py --selftest
  PYTHONPATH=. python prefix_risk_gate.py --quick
  PYTHONPATH=. nohup python -u prefix_risk_gate.py > prefix_risk.log 2>&1 &
Conventions: D=0, rho>0 safe, d=rho_hat-rho>0 dangerous optimism, H=10,
episode-level splits via seed offsets {0,+1000,+2000,+3000}, dynamics =
plain MSE (frozen; hash-asserted), risk head gets no dynamics gradients.
Pre-registered PASS criteria in verdict() — fixed before final test."""
import argparse, csv, hashlib, json, os, sys
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, ".")
from revival_gate import get_apis
from autopsy_weighting import rollout_windows, traj_boundaries

H = 10; SEEDS = [0, 1, 2]; W = (64, 64)
OUT = "results/prefix_risk_gate"; os.makedirs(OUT, exist_ok=True)
api = get_apis(); cwc = api["cwc"]
HS = [1, 3, 5, 10]


def md5(sd):
    b = b"".join(v.numpy().tobytes() for v in sd.values())
    return hashlib.md5(b).hexdigest()


def dyn_train(seed, epochs):
    d = api["build_dataset"](n_traj=60, seed=seed)
    prep = cwc.prepare(d["train"], 0, seed)
    M = np.tile(np.eye(4), (len(prep["X"]), 1, 1))
    m, _, _ = cwc.train(prep, M, W, epochs, seed, 0)
    m.eval()
    [p.requires_grad_(False) for p in m.parameters()]
    r = np.linalg.norm(prep["X"][:, :2] - prep["p_obs"], axis=1) - prep["margin"]
    assert np.std(r) < 1e-6, "margin convention broken"
    return m, float(np.median(r))


def windows(model, seed_gen, radius, tag):
    """Labels+features for the 'train' half of build_dataset(seed_gen)."""
    d = api["build_dataset"](n_traj=60, seed=seed_gen)["train"]
    res = rollout_windows(model, d, 0, seed_gen, H, api["Params"])
    X, U, Xn = d["X"], d["U"], d["Xn"]
    ids = traj_boundaries(X, Xn)
    starts = [i for i in range(len(X) - H) if ids[i] == ids[i + H]]
    assert len(starts) == len(res["en"]), "window enumeration mismatch"
    d_opt = np.maximum(res["r_rho"], 0.0)                    # [d]_+
    Sn = np.maximum.accumulate(d_opt, axis=1)                # prefix max, (N,H)
    S2 = np.maximum.accumulate(np.sqrt(res["en"] ** 2 + res["et"] ** 2), 1)
    rho_hat = res["dist_true"] + res["r_rho"] - radius
    rho_true = res["dist_true"] - radius
    Useq = np.stack([U[np.array(starts) + k] for k in range(H)], 1)
    F = np.concatenate([X[starts], Useq.reshape(len(starts), -1),
                        rho_hat], 1).astype(np.float32)      # 4+20+H = 34
    return dict(F=F, Sn=Sn.astype(np.float32), S2=S2.astype(np.float32),
                rho_hat=rho_hat, rho_true=rho_true, m0=res["m0"],
                eid=np.array([f"{tag}:{i}" for i in ids[starts]]))


class RiskHead(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 64), nn.SiLU(),
                                 nn.Linear(64, 64), nn.SiLU(),
                                 nn.Linear(64, H))

    def forward(self, x):                                    # nonneg, monotone
        return torch.cumsum(nn.functional.softplus(self.net(x)), dim=-1)


def pinball(q, y, tau=0.95):
    e = y - q
    return torch.mean(torch.maximum(tau * e, (tau - 1) * e))


def fit_head(Ftr, Str, seed, epochs):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
    h = RiskHead(Ftr.shape[1]); torch.manual_seed(seed)
    opt = torch.optim.Adam(h.parameters(), lr=1e-3)
    X = torch.tensor((Ftr - mu) / sd); Y = torch.tensor(Str)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        idx = rng.permutation(len(X))
        for i in range(0, len(idx), 256):
            b = idx[i:i + 256]
            loss = pinball(h(X[b]), Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    h.eval(); return h, (mu, sd)


def bounds_for(method, head_norm, cal, test, S_key):
    """Return (N_test, H) bound array, calibrated on cal split."""
    if method.startswith("global"):
        c = np.quantile(cal[S_key], 0.95, axis=0)            # per-h fixed
        return np.tile(c, (len(test["F"]), 1))
    h, (mu, sd) = head_norm
    with torch.no_grad():
        qc = h(torch.tensor((cal["F"] - mu) / sd)).numpy()
        qt = h(torch.tensor((test["F"] - mu) / sd)).numpy()
    ch = np.quantile(cal[S_key] - qc, 0.95, axis=0)          # per-h offset
    return np.maximum(0.0, qt + ch)


def boot(vals, eids, fn, B=1000, seed=0):
    rng = np.random.default_rng(seed); u = np.unique(eids); st = []
    for _ in range(B):
        pick = rng.choice(u, len(u), replace=True)
        idx = np.concatenate([np.where(eids == x)[0] for x in pick])
        st.append(fn(vals[idx]))
    return float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))


def selftest():
    print("selftest:")
    # sign: predicted CLOSER to obstacle than truth -> rho_hat<rho -> d<0 (pessimism)
    # predicted FARTHER (optimism) -> d>0 dangerous. Manual construction:
    c = np.zeros(2); p_true = np.array([1.0, 0.0]); p_hat = np.array([1.2, 0.0])
    r = 0.5
    d = (np.linalg.norm(p_hat - c) - r) - (np.linalg.norm(p_true - c) - r)
    assert d > 0, "optimism sign wrong"
    print("  sign convention OK (farther-than-true => d>0 dangerous)")
    x = np.random.randn(5, 34).astype(np.float32)
    hh = RiskHead(34)
    q = hh(torch.tensor(x)).detach().numpy()
    assert (q >= -1e-7).all() and (np.diff(q, axis=1) >= -1e-7).all()
    print("  head nonneg + monotone in h OK")
    a = np.maximum.accumulate(np.maximum(np.array([[0.1, -0.2, 0.3]]), 0), 1)
    assert (np.diff(a) >= 0).all(); print("  prefix target monotone OK")
    e1 = {f"a:{i}" for i in range(3)}; e2 = {f"b:{i}" for i in range(3)}
    assert not (e1 & e2); print("  split namespacing disjoint OK")
    z = np.zeros((2, H)); assert np.maximum.accumulate(np.maximum(z, 0), 1).max() == 0
    print("  zero error -> zero risk OK")
    m, _ = dyn_train(0, 30)
    h0 = md5(m.state_dict())
    F = np.random.randn(64, 34).astype(np.float32)
    S = np.abs(np.random.randn(64, H)).astype(np.float32)
    fit_head(F, S, 0, 3)
    assert md5(m.state_dict()) == h0; print("  dynamics hash unchanged OK")
    assert F.dtype == np.float32; print("  float32 OK")
    h1, _ = fit_head(F, S, 7, 3); h2, _ = fit_head(F, S, 7, 3)
    for a_, b_ in zip(h1.parameters(), h2.parameters()):
        assert torch.allclose(a_, b_)
    print("  same-seed reproducible OK\nSELFTEST PASS")


def main(quick=False):
    EPD = 300 if quick else 2000
    EPH = 60 if quick else 500
    seeds = [0] if quick else SEEDS
    manifest = dict(quick=quick, ep_dyn=EPD, ep_head=EPH, seeds=seeds,
                    splits={"dyn": "s", "risk": "s+1000",
                            "cal": "s+2000", "test": "s+3000"})
    rows, per_ep = [], []
    for seed in seeds:
        m, radius = dyn_train(seed, EPD)
        torch.save(m.state_dict(), f"{OUT}/dyn_seed{seed}.pt")
        manifest[f"dyn_md5_{seed}"] = md5(m.state_dict())
        sp = {k: windows(m, seed + o, radius, f"s{seed}+{o}")
              for k, o in (("risk", 1000), ("cal", 2000), ("test", 3000))}
        eall = [set(v["eid"]) for v in sp.values()]
        assert not (eall[0] & eall[1] or eall[1] & eall[2] or eall[0] & eall[2])
        h_n = fit_head(sp["risk"]["F"], sp["risk"]["Sn"], seed, EPH)
        assert md5(m.state_dict()) == manifest[f"dyn_md5_{seed}"], \
            "dynamics changed by risk training"
        h_2 = fit_head(sp["risk"]["F"], sp["risk"]["S2"], seed + 77, EPH)
        te = sp["test"]
        near_thr = float(np.quantile(sp["cal"]["m0"], 0.25))   # test-independent
        near = te["m0"] <= near_thr
        B = {"global-normal": bounds_for("global", None, sp["cal"], te, "Sn"),
             "global-2d": bounds_for("global", None, sp["cal"], te, "S2"),
             "conditional-2d": bounds_for("cond", h_2, sp["cal"], te, "S2"),
             "conditional-normal": bounds_for("cond", h_n, sp["cal"], te, "Sn")}
        Skey = {"global-normal": "Sn", "global-2d": "S2",
                "conditional-2d": "S2", "conditional-normal": "Sn"}
        nviol_total = 0
        for nm, bd in B.items():
            S = te[Skey[nm]]
            for h in HS:
                j = h - 1
                cov = (S[:, j] <= bd[:, j])
                acc = (np.min(te["rho_hat"][:, :h], 1) > bd[:, j])
                viol = (np.min(te["rho_true"][:, :h], 1) <= 0)
                nviol_total = max(nviol_total, int(viol.sum()))
                fsafe = float(np.mean(viol[acc])) if acc.any() else 0.0
                lo, hi = boot(cov.astype(float), te["eid"], np.mean)
                rows.append(dict(seed=seed, method=nm, h=h,
                                 coverage=float(cov.mean()),
                                 cov_lo=lo, cov_hi=hi,
                                 cov_near=float(cov[near].mean()),
                                 w_mean=float(bd[:, j].mean()),
                                 w_med=float(np.median(bd[:, j])),
                                 w_p90=float(np.quantile(bd[:, j], .90)),
                                 acc_rate=float(acc.mean()),
                                 false_safe=fsafe,
                                 n_viol=int(viol.sum())))
        manifest[f"n_viol_seed{seed}"] = nviol_total
        np.savez(f"{OUT}/predictions_seed{seed}.npz",
                 **{f"bd_{k}": v for k, v in B.items()},
                 Sn=te["Sn"], S2=te["S2"], m0=te["m0"])
        print(f"seed{seed} done (test windows {len(te['F'])}, "
              f"violation windows {nviol_total})")
    with open(f"{OUT}/metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=2)

    # ---------- pooled table + pre-registered verdict ----------
    def pool(nm, h, k):
        return float(np.mean([r[k] for r in rows
                              if r["method"] == nm and r["h"] == h]))
    print(f"\n{'method':>19} {'h':>3} {'cov':>6} {'cov_near':>8} "
          f"{'w_mean':>9} {'acc':>6} {'fsafe':>7}")
    for nm in B:
        for h in (5, 10):
            print(f"{nm:>19} {h:>3} {pool(nm, h, 'coverage'):>6.3f} "
                  f"{pool(nm, h, 'cov_near'):>8.3f} "
                  f"{pool(nm, h, 'w_mean'):>9.2e} "
                  f"{pool(nm, h, 'acc_rate'):>6.3f} "
                  f"{pool(nm, h, 'false_safe'):>7.4f}")
    nv = min(manifest[f"n_viol_seed{s}"] for s in seeds)
    verdict = {"criteria": "see spec section 8", "insufficient": nv < 10}
    if nv < 10:
        print(f"\nINSUFFICIENT EVENTS (min violation windows {nv} < 10) "
              "-- false-safe criterion not assessable, PASS not allowed")
        verdict["result"] = "INSUFFICIENT EVENTS"
    else:
        ok_cov = all(pool("conditional-normal", h, "coverage") >= 0.93
                     and pool("conditional-normal", h, "coverage") <= 0.99
                     for h in (5, 10))
        narrow = []
        for s in seeds:
            g = np.mean([r["w_mean"] for r in rows if r["seed"] == s
                         and r["method"] == "global-normal" and r["h"] in (5, 10)])
            c = np.mean([r["w_mean"] for r in rows if r["seed"] == s
                         and r["method"] == "conditional-normal" and r["h"] in (5, 10)])
            narrow.append(1 - c / g)
        ok_nar = np.mean(narrow) >= 0.10 and sum(x > 0 for x in narrow) >= 2
        ok_fs = all(pool("conditional-normal", h, "false_safe")
                    <= pool("global-normal", h, "false_safe") + 1e-9
                    for h in (5, 10))
        ok_2d = all(pool("conditional-normal", h, "w_mean")
                    < pool("conditional-2d", h, "w_mean") for h in (5, 10))
        verdict.update(coverage_ok=bool(ok_cov),
                       narrowing=[float(x) for x in narrow],
                       narrowing_ok=bool(ok_nar), false_safe_ok=bool(ok_fs),
                       beats_2d=bool(ok_2d),
                       result="PASS" if all([ok_cov, ok_nar, ok_fs, ok_2d])
                       else "FAIL")
        print("\n" + (">>> PASS: proceed to risk-based hold length"
                      if verdict["result"] == "PASS" else
                      ">>> FAIL: conditional risk head adds no value "
                      f"(cov {ok_cov} narrow {ok_nar} "
                      f"fs {ok_fs} beats2d {ok_2d})"))
    json.dump(verdict, open(f"{OUT}/verdict.json", "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for nm in B:
            ax.scatter([pool(nm, h, "w_mean") for h in HS],
                       [pool(nm, h, "coverage") for h in HS], label=nm)
        ax.set_xlabel("mean bound width"); ax.set_ylabel("coverage")
        ax.legend(); fig.savefig(f"{OUT}/coverage_width.png", dpi=130,
                                 bbox_inches="tight")
        d0 = np.load(f"{OUT}/predictions_seed{seeds[0]}.npz")
        fig2, ax2 = plt.subplots()
        ax2.scatter(d0["m0"], d0["bd_conditional-normal"][:, 9], s=4)
        ax2.set_xlabel("start margin"); ax2.set_ylabel("cond-normal bound h=10")
        fig2.savefig(f"{OUT}/risk_vs_clearance.png", dpi=130,
                     bbox_inches="tight")
    except Exception as e:
        print(f"(plots skipped: {e})")
    print(f"wrote {OUT}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else main(quick=a.quick)
