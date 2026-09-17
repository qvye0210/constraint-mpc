#!/usr/bin/env python3
"""Offline prefix-reuse audit (experiment 2, coverage-matched spec).
Zero training. Frozen MSE dynamics (md5-checked). Splits by whole
episodes, fresh offsets: design=+8000, calibration=+9000, test=+10000;
test read once after everything is frozen.
    PYTHONPATH=. python offline_prefix_reuse.py --selftest
    PYTHONPATH=. python offline_prefix_reuse.py
ONLINE RULE (all triggers identical, no future information):
    accept prefix h  iff  rho_hat_k > q_{s,k}  for all k <= h,
with q from calibration only; stratum s in {near, far} frozen on the
DESIGN split (threshold = q25 of design start-margins), applied to both
normal and 2D arms alike. Arms:
  fixed-h (h in {1,2,3,5,8,10})       — design-split choice reported
  eucl    : q = stratified q95 of prefix-max ||p_hat - p||_2
  lin-n   : q = stratified q95 of prefix-max [n^T(p_hat - p)]_+
  margin  : q = stratified q95 of prefix-max [rho_hat - rho]_+
Matched quantity = target coverage 0.95 for every calibrated arm (same
alpha, same strata, same data). False-safe reported, expected ~0 (prior
audit: no natural FS signal in this task); conclusions are efficiency
claims at matched coverage, NOT safety-discrimination claims. Replanning
numbers are a POTENTIAL/opportunity proxy (open-loop replay), not real
solver-call reduction.
GO (all required): test overall coverage >= 0.93; near coverage >= 0.90;
margin-arm median prefix >= 3; P(h>=3) >= 0.50; margin-arm mean prefix
>= 1.20 x eucl-arm mean prefix. NO-GO wordings per original spec.
"""
import argparse, csv, hashlib, json, os, sys
import numpy as np
import torch

sys.path.insert(0, ".")
from revival_gate import get_apis
from autopsy_weighting import rollout_windows

H = 10; HS = [1, 2, 3, 5, 8, 10]; SEEDS = [0, 1, 2]; ALPHA = 0.95
OUT = "results/offline_prefix_reuse"; os.makedirs(OUT, exist_ok=True)
api = get_apis(); cwc = api["cwc"]


def md5(sd):
    return hashlib.md5(b"".join(v.numpy().tobytes()
                                for v in sd.values())).hexdigest()


def load_dyn(seed):
    p = f"results/prefix_risk_gate/dyn_seed{seed}.pt"
    if not os.path.exists(p):
        sys.exit(f"MISSING {p} -- stopping (no retraining in this gate)")
    m = api["ResidualMLP"]((64, 64), n_dist=0)
    m.load_state_dict(torch.load(p, map_location="cpu")); m.eval()
    man = json.load(open("results/prefix_risk_gate/manifest.json"))
    if man.get(f"dyn_md5_{seed}") != md5(m.state_dict()):
        sys.exit(f"MD5 MISMATCH {p} -- stopping")
    return m


def table(model, seed_gen, tag):
    d = api["build_dataset"](n_traj=60, seed=seed_gen)["train"]
    res = rollout_windows(model, d, 0, seed_gen, H, api["Params"])
    prep = cwc.prepare(d, 0, seed_gen)
    r = np.linalg.norm(prep["X"][:, :2] - prep["p_obs"], axis=1) - prep["margin"]
    assert np.std(r) < 1e-6
    radius = float(np.median(r))
    rho_t = res["dist_true"] - radius
    rho_h = rho_t + res["r_rho"]
    S = dict(
        eucl=np.maximum.accumulate(np.sqrt(res["en"] ** 2 + res["et"] ** 2), 1),
        lin_n=np.maximum.accumulate(np.maximum(res["en"], 0), 1),
        margin=np.maximum.accumulate(np.maximum(res["r_rho"], 0), 1))
    return dict(S=S, rho_t=rho_t, rho_h=rho_h, m0=res["m0"],
                eid=np.array([f"{tag}:{i}" for i in res["eid"]]))


def calibrate(Ca, near_thr):
    q = {}
    nc = Ca["m0"] <= near_thr
    for nm, s in Ca["S"].items():
        qn = np.quantile(s[nc], ALPHA, axis=0)
        qf = np.quantile(s[~nc], ALPHA, axis=0)
        q[nm] = (qn, qf)
    return q


def choose_h(Te, q, near_thr):
    """Online rule: h = max{h: rho_hat_k > q_{s,k} forall k<=h}."""
    out = {}
    near = Te["m0"] <= near_thr
    for nm, (qn, qf) in q.items():
        bound = np.where(near[:, None], qn[None, :], qf[None, :])
        ok = Te["rho_h"] > bound
        h = np.zeros(len(ok), dtype=int)
        run = np.ones(len(ok), dtype=bool)
        for k in range(H):
            run &= ok[:, k]
            h[run] = k + 1
        out[nm] = h
    return out


def cover_stats(Te, hsel, key, near_thr):
    """Coverage: among accepted prefixes (h>=1), prefix error <= bound? We
    report the direct safety-relevant version: within accepted prefix, no
    true violation (rho_t stays > 0) -> covered; FS = accepted & violated."""
    near = Te["m0"] <= near_thr
    acc = hsel >= 1
    viol = np.array([Te["rho_t"][i, :h].min() <= 0 if h >= 1 else False
                     for i, h in enumerate(hsel)])
    cov = np.where(acc, ~viol, True)
    return dict(coverage=float(cov[acc].mean()) if acc.any() else 1.0,
                cov_near=float(cov[acc & near].mean())
                if (acc & near).any() else 1.0,
                fs=float(viol[acc].mean()) if acc.any() else 0.0,
                acc_rate=float(acc.mean()))


def selftest():
    rho_h = np.array([[.5, .5, -.1], [.5, .5, .5]])
    q = {"m": (np.array([.1, .2, .3]), np.array([.1, .2, .3]))}
    Te = dict(rho_h=rho_h, m0=np.array([0., 1.]))
    h = choose_h(Te, q, 0.5)["m"]
    assert list(h) == [2, 3], h
    print("SELFTEST PASS (online prefix rule)")


def main():
    rows = []
    for seed in SEEDS:
        m = load_dyn(seed)
        De = table(m, seed + 8000, f"d{seed}")
        Ca = table(m, seed + 9000, f"c{seed}")
        Te = table(m, seed + 10000, f"t{seed}")
        near_thr = float(np.quantile(De["m0"], 0.25))    # frozen on design
        q = calibrate(Ca, near_thr)
        hsel = choose_h(Te, q, near_thr)
        for nm, h in hsel.items():
            st = cover_stats(Te, h, nm, near_thr)
            rows.append(dict(seed=seed, arm=nm,
                             mean_h=float(h.mean()), med_h=float(np.median(h)),
                             p10_h=float(np.percentile(h, 10)),
                             p90_h=float(np.percentile(h, 90)),
                             p_ge3=float((h >= 3).mean()),
                             p_ge5=float((h >= 5).mean()),
                             pot_reduction=float(1 - 1 / np.maximum(h, 1).mean()),
                             **st))
        for hf in HS:                                     # fixed-h reference
            h = np.full(len(Te["m0"]), hf)
            st = cover_stats(Te, h, None, near_thr)
            rows.append(dict(seed=seed, arm=f"fixed{hf}", mean_h=float(hf),
                             med_h=float(hf), p10_h=hf, p90_h=hf,
                             p_ge3=float(hf >= 3), p_ge5=float(hf >= 5),
                             pot_reduction=float(1 - 1 / hf), **st))
        print(f"seed{seed}: margin mean_h "
              f"{next(r['mean_h'] for r in rows if r['seed']==seed and r['arm']=='margin'):.2f}"
              f" eucl {next(r['mean_h'] for r in rows if r['seed']==seed and r['arm']=='eucl'):.2f}")
    with open(f"{OUT}/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    P = lambda a, k: float(np.mean([r[k] for r in rows if r["arm"] == a]))
    print(f"\n{'arm':>8} {'cov':>6} {'cov_near':>8} {'mean_h':>7} "
          f"{'med_h':>6} {'P(h>=3)':>8} {'pot_red':>8} {'fs':>6}")
    for a in ("eucl", "lin_n", "margin", "fixed3", "fixed5"):
        print(f"{a:>8} {P(a,'coverage'):>6.3f} {P(a,'cov_near'):>8.3f} "
              f"{P(a,'mean_h'):>7.2f} {P(a,'med_h'):>6.1f} "
              f"{P(a,'p_ge3'):>8.2f} {P(a,'pot_reduction'):>8.2f} "
              f"{P(a,'fs'):>6.3f}")
    gain = P("margin", "mean_h") / max(P("eucl", "mean_h"), 1e-9) - 1
    ok = dict(cov=P("margin", "coverage") >= 0.93,
              near=P("margin", "cov_near") >= 0.90,
              med=P("margin", "med_h") >= 3,
              pge3=P("margin", "p_ge3") >= 0.50,
              gain=gain >= 0.20)
    v = dict(gain_vs_eucl=float(gain), **{k: bool(x) for k, x in ok.items()},
             result="PASS" if all(ok.values()) else "NO-GO",
             note="efficiency-at-matched-coverage claim only; FS "
                  "discrimination deferred to closed loop; replanning "
                  "numbers are opportunity proxy")
    json.dump(v, open(f"{OUT}/verdict.json", "w"), indent=2)
    if v["result"] == "PASS":
        print("\n>>> PASS: proceed to closed-loop experiment (port terminal "
              "wall, frozen boundary-rich task for safety testing)")
    elif ok["cov"] and ok["near"] and not (ok["med"] and ok["pge3"]):
        print("\n>>> NO-GO: calibration is safe but too conservative to "
              "reduce replanning.")
    elif not ok["gain"]:
        print("\n>>> NO-GO: constraint alignment is only a metric "
              "substitution with no useful prefix advantage.")
    else:
        print("\n>>> NO-GO: see verdict.json component flags.")
    print(f"wrote {OUT}/  (gain margin-vs-eucl: {gain:+.1%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest()
    if not a.selftest:
        main()
