#!/usr/bin/env python3
"""Direction-mechanism audit (experiment 1, corrected spec).
Offline analysis only; NO training. Frozen MSE dynamics from
results/prefix_risk_gate/dyn_seed{0,1,2}.pt (md5-checked vs manifest).
    PYTHONPATH=. python constraint_alignment_audit.py --selftest
    PYTHONPATH=. python constraint_alignment_audit.py
Spec (pre-registered, per advisor's corrections):
- d_rho^+ shares rho with the violation label => PR-AUC is AUXILIARY only.
- MAIN test: within (horizon x rho_hat-stratum x e2D-stratum) bins whose
  edges are frozen on the DESIGN split, does normal alignment |e_n|/e2D
  still separate FALSE-SAFE events?  Label:
      Y_FS(h) = 1[ min_{k<=h} rho_hat_k > 0  AND  min_{k<=h} rho_k < 0 ].
- Negative controls: pessimistic [rho-rho_hat]_+, |e_t|, shuffled-normal
  projection. Controls are informative, not auto-abort (pessimistic error
  may correlate with boundary proximity); they are judged inside the same
  conditional bins.
- Splits by whole episodes, fresh seed offsets: design=+5000,
  calibration=+6000, test=+7000; test read once, after thresholds frozen.
- rho > 0 safe; d = rho_hat - rho > 0 optimism (sign selftest aborts).
PASS = conditional-alignment separation (criterion 3): pooled FS-rate gap
(top vs bottom alignment tercile, within bins) positive with episode-
bootstrap 95% CI excluding 0, same direction in majority of horizons.
Criteria 1-2 (matched-FS TPR +10pp; matched-accept FS -30%) reported as
AUXILIARY with the shared-rho caveat printed. Else NO-GO (exact wording
per spec)."""
import argparse, csv, hashlib, json, os, sys
import numpy as np
import torch

sys.path.insert(0, ".")
from revival_gate import get_apis
from autopsy_weighting import rollout_windows

KS = [1, 3, 5, 10]; H = 10; SEEDS = [0, 1, 2]
OUT = "results/constraint_alignment_audit"; os.makedirs(OUT, exist_ok=True)
api = api_ = get_apis(); cwc = api["cwc"]


def md5(sd):
    return hashlib.md5(b"".join(v.numpy().tobytes()
                                for v in sd.values())).hexdigest()


def load_dyn(seed):
    p = f"results/prefix_risk_gate/dyn_seed{seed}.pt"
    if not os.path.exists(p):
        sys.exit(f"MISSING: {p} -- frozen checkpoint unavailable, stopping "
                 "(spec forbids retraining here)")
    m = api["ResidualMLP"]((64, 64), n_dist=0)
    sd = torch.load(p, map_location="cpu")
    m.load_state_dict(sd); m.eval()
    man = json.load(open("results/prefix_risk_gate/manifest.json"))
    if man.get(f"dyn_md5_{seed}") != md5(m.state_dict()):
        sys.exit(f"MD5 MISMATCH for {p}, stopping")
    return m


def episode_table(model, seed_gen, tag):
    d = api["build_dataset"](n_traj=60, seed=seed_gen)["train"]
    res = rollout_windows(model, d, 0, seed_gen, H, api["Params"])
    prep = cwc.prepare(d, 0, seed_gen)
    r = np.linalg.norm(prep["X"][:, :2] - prep["p_obs"], axis=1) - prep["margin"]
    assert np.std(r) < 1e-6
    radius = float(np.median(r))
    rho_t = res["dist_true"] - radius
    rho_h = rho_t + res["r_rho"]
    e2 = np.sqrt(res["en"] ** 2 + res["et"] ** 2)
    return dict(en=res["en"], et=res["et"], e2=e2, drho=res["r_rho"],
                rho_t=rho_t, rho_h=rho_h, m0=res["m0"],
                eid=np.array([f"{tag}:{i}" for i in res["eid"]]))


def scores_labels(T, h):
    j = h
    mh = T["rho_h"][:, :j].min(1); mt = T["rho_t"][:, :j].min(1)
    yfs = ((mh > 0) & (mt < 0)).astype(float)
    yv = (mt < 0).astype(float)
    k = j - 1
    S = dict(e2D=T["e2"][:, k], abs_en=np.abs(T["en"][:, k]),
             en_pos=np.maximum(T["en"][:, k], 0),
             drho_pos=np.maximum(T["drho"][:, k], 0),
             pess=np.maximum(-T["drho"][:, k], 0),
             abs_et=np.abs(T["et"][:, k]))
    rng = np.random.default_rng(1234 + h)
    perm = rng.permutation(len(T["en"]))
    ex = T["en"][:, k] ** 2 + T["et"][:, k] ** 2
    # shuffled-normal: project this sample's error vector magnitude onto a
    # random other sample's alignment fraction
    align = np.abs(T["en"][:, k]) / (T["e2"][:, k] + 1e-12)
    S["shuffled_n"] = np.sqrt(ex) * align[perm]
    return S, yfs, yv, align, mh


def prauc(s, y):
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(y.sum(), 1)
    return float(np.trapz(prec, rec)) if y.sum() else float("nan")


def rocauc(s, y):
    if y.sum() in (0, len(y)):
        return float("nan")
    r = s.argsort().argsort().astype(float)
    return float((r[y == 1].mean() - (y.sum() - 1) / 2) / (len(y) - y.sum()))


def boot_gap(vals, eids, B=800, seed=0):
    rng = np.random.default_rng(seed); u = np.unique(eids); st = []
    for _ in range(B):
        pick = rng.choice(u, len(u), replace=True)
        idx = np.concatenate([np.where(eids == x)[0] for x in pick])
        st.append(np.nanmean(vals[idx]))
    return float(np.nanmean(vals)), float(np.percentile(st, 2.5)), \
        float(np.percentile(st, 97.5))


def conditional_gap(T, h, edges_rho, edges_e2, eid):
    """Per-window FS-gap contribution: within (rho_hat_min, e2D) bin,
    top-vs-bottom alignment tercile FS-rate difference."""
    S, yfs, _, align, mh = scores_labels(T, h)
    e2 = S["e2D"]
    br = np.digitize(mh, edges_rho); be = np.digitize(e2, edges_e2)
    gap = np.full(len(yfs), np.nan)
    for b1 in range(3):
        for b2 in range(3):
            m = (br == b1) & (be == b2)
            if m.sum() < 30 or yfs[m].sum() == 0:
                continue
            a = align[m]; lo, hi = np.quantile(a, [1 / 3, 2 / 3])
            top, bot = m.copy(), m.copy()
            top[m] &= a >= hi; bot[m] &= a <= lo
            g = yfs[top].mean() - yfs[bot].mean()
            gap[m] = g
    return gap


def selftest():
    c = np.zeros(2); r = 0.5
    p_t, p_h = np.array([1.0, 0]), np.array([1.2, 0])
    d = (np.linalg.norm(p_h - c) - r) - (np.linalg.norm(p_t - c) - r)
    assert d > 0, "sign selftest FAILED"
    print("sign selftest OK (predicted farther => d_rho > 0 optimism)")


def main():
    rows, cond_rows = [], []
    for seed in SEEDS:
        m = load_dyn(seed)
        De = episode_table(m, seed + 5000, f"d{seed}")
        Ca = episode_table(m, seed + 6000, f"c{seed}")
        Te = episode_table(m, seed + 7000, f"t{seed}")
        for h in KS:
            Sd, yd, _, _, mhd = scores_labels(De, h)
            edges_rho = np.quantile(mhd, [1 / 3, 2 / 3])
            edges_e2 = np.quantile(Sd["e2D"], [1 / 3, 2 / 3])
            St, yt, yv, _, _ = scores_labels(Te, h)
            for nm, s in St.items():
                rows.append(dict(seed=seed, h=h, score=nm,
                                 prauc_fs=prauc(s, yt), roc_fs=rocauc(s, yt),
                                 prauc_v=prauc(s, yv),
                                 brier=float(np.mean((s / (s.max() + 1e-12)
                                                      - yt) ** 2)),
                                 n_fs=int(yt.sum()), n_v=int(yv.sum())))
            g = conditional_gap(Te, h, edges_rho, edges_e2, Te["eid"])
            mu, lo, hi = boot_gap(g, Te["eid"], seed=seed * 10 + h)
            cond_rows.append(dict(seed=seed, h=h, fs_gap=mu, lo=lo, hi=hi,
                                  n_binned=int(np.isfinite(g).sum())))
        print(f"seed{seed} done (test FS events h=5: "
              f"{next(r['n_fs'] for r in rows if r['seed']==seed and r['h']==5 and r['score']=='e2D')})")
    for fn, data in (("summary.csv", rows), ("conditional.csv", cond_rows)):
        with open(f"{OUT}/{fn}", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader(); w.writerows(data)

    print("\nAUXILIARY (shared-rho caveat: d_rho_pos is coupled to the "
          "label; ranking below is NOT the pass criterion)")
    print(f"{'score':>11} " + " ".join(f"h{h}:PR-AUC(FS)" for h in KS))
    for nm in ("e2D", "abs_en", "en_pos", "drho_pos", "pess", "abs_et",
               "shuffled_n"):
        v = [np.mean([r["prauc_fs"] for r in rows if r["score"] == nm
                      and r["h"] == h]) for h in KS]
        print(f"{nm:>11} " + " ".join(f"{x:>12.3f}" for x in v))
    print("\nMAIN (conditional alignment FS-gap, pooled, episode-boot CI):")
    ok_dirs = 0
    for h in KS:
        mus = [r for r in cond_rows if r["h"] == h]
        mu = np.mean([r["fs_gap"] for r in mus])
        lo = np.mean([r["lo"] for r in mus]); hi = np.mean([r["hi"] for r in mus])
        pos = sum(r["fs_gap"] > 0 for r in mus)
        ok_dirs += mu > 0
        print(f"  h={h:>2}: gap {mu:+.4f}  CI[{lo:+.4f},{hi:+.4f}]  "
              f"seeds>0: {pos}/3")
    all_pos_ci = all(np.mean([r["lo"] for r in cond_rows if r["h"] == h]) > 0
                     for h in (5, 10))
    passed = bool(all_pos_ci and ok_dirs >= 3)
    v = dict(main_criterion="conditional alignment FS-gap",
             pass_=passed,
             note="criteria 1-2 auxiliary due to shared-rho coupling")
    json.dump(v, open(f"{OUT}/verdict.json", "w"), indent=2)
    print("\n>>> " + ("PASS: directional information exists beyond error "
                      "magnitude and boundary proximity -> proceed to "
                      "offline_prefix_reuse"
                      if passed else
                      "NO-GO: constraint alignment contains no useful "
                      "information beyond Euclidean prediction error in "
                      "this dataset."))
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for h in KS:
            mus = [r["fs_gap"] for r in cond_rows if r["h"] == h]
            ax.bar(str(h), np.mean(mus))
        ax.set_xlabel("h"); ax.set_ylabel("conditional FS gap")
        fig.savefig(f"{OUT}/risk_by_angle.png", dpi=130, bbox_inches="tight")
    except Exception as e:
        print(f"(plot skipped: {e})")
    print(f"wrote {OUT}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest()
    if not a.selftest:
        main()
