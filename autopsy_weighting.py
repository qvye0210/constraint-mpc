#!/usr/bin/env python3
"""Autopsy of the 2D loss-weighting result (mask -75.8% / prop +99.5%).

Run from the cmpc2d project root (PYTHONPATH=.), conda env constraint_mpc.

EXP0 (no training):  python autopsy_weighting.py --audit
    Scans results/*/ for raw.csv + verdict.json, recomputes every proxy_rel
    from the raw columns, checks stored vs recomputed, and prints the full
    provenance of the metric (split / formula / aggregation / seeds /
    checkpoint status / held-out-direct confirmation).

EXP1 (retrain, direction-decomposed):
    python autopsy_weighting.py --run --from results/<dir_with_mask_run>
    Retrains uniform/mask/prop with the EXACT config recovered from that
    run's verdict.json (checkpoints were never persisted, so retraining is
    the only faithful reconstruction), then evaluates per-horizon
    RMSE_n(k)/RMSE_t(k), signed margin bias r_rho(k), dangerous positive
    tails, false-safe rate, prefix-max E_rho_h, residual covariance ellipse
    in (n,t), near/far stratification, train vs held-out, with
    EPISODE-level bootstrap CIs.

Pre-registered reading table (advisor's, do not revise after results):
  A  held-out e_n down AND dangerous tail down, e_t up   -> old aggregate
     metric was unfair; tempering (Exp2) justified, switch primary metric
     to normal-projection error.
  B  train e_n down, test e_n up                          -> normal-direction
     overfitting confirmed.
  C  e_n and e_t both up                                  -> extreme weights
     destroyed learning overall; no ellipse ever formed.
  D  e_n ~flat, e_t up a lot                              -> weighting bought
     no safety benefit.
  E  RMSE down but false-safe rate up                     -> mean metrics
     misleading; closed loop may be MORE dangerous.
"""

import argparse, csv, glob, json, os, sys
import numpy as np

SENS_NORMAL, SENS_TANGENT = 48.5, 12.3


# ---------------------------------------------------------------- EXP 0
def audit():
    runs = sorted(glob.glob("results/*/"))
    if not runs:
        print("no results/*/ directories found -- run from cmpc2d project root")
        return
    print("=" * 100)
    print("EXP0: audit of stored aggregate results (NO training performed)")
    print("=" * 100)
    print("""
METRIC PROVENANCE (from constraint_weight_check.py source, fixed at run time):
  * split      : build_dataset(n_traj, seed) -> {'train','test'}; test = held-out
                 trajectories from the same generator, distractors drawn with
                 seed+500 (train uses seed) -> test inputs never seen in training.
  * error      : ONE-STEP DIRECT error, e_core = (f_nominal(X,U)+model(X,U)[:NX]) - Xn,
                 computed on the TEST set. NOT a rollout, NOT closed-loop.
  * decomposition stored: rmse_normal / rmse_tangent already exist per row
                 (en = e.n(X,p_obs), et = e.t(X,p_obs)) -- but only their
                 WEIGHTED SUM was used for the verdict:
                 proxy = 48.5*rmse_normal + 12.3*rmse_tangent
  * aggregation: RMSE over all test transitions, then MEAN over seeds,
                 then proxy_rel = (proxy - proxy_uniform)/proxy_uniform
                 within the same (n_dist, epochs) block.
  * checkpoints: NEVER persisted (train() snaps discarded in the main loop)
                 -> Exp1 must retrain; config below is the registered spec.
""")
    for rd in runs:
        raw_p, ver_p = os.path.join(rd, "raw.csv"), os.path.join(rd, "verdict.json")
        if not os.path.exists(raw_p):
            continue
        rows = list(csv.DictReader(open(raw_p)))
        cfg = {}
        if os.path.exists(ver_p):
            cfg = json.load(open(ver_p)).get("config", {})
        print("-" * 100)
        print(f"RUN {rd}")
        if cfg:
            keys = ["arms", "seeds", "seed_offset", "n_dist", "epoch_list",
                    "hidden", "n_traj", "H", "gamma", "eps_floor"]
            print("  config: " + ", ".join(f"{k}={cfg.get(k)}" for k in keys if k in cfg))
        else:
            print("  config: verdict.json missing -- provenance INCOMPLETE for this run")
        # recompute proxy and proxy_rel from raw columns
        bad_proxy, bad_rel = 0, 0
        groups = {}
        for r in rows:
            key = (r.get("n_dist"), r.get("epochs"))
            groups.setdefault(key, []).append(r)
        for (D, ep), grp in sorted(groups.items()):
            uni = [r for r in grp if r["arm"] == "uniform"]
            if not uni:
                print(f"  D={D} ep={ep}: NO uniform arm -> proxy_rel undefined here")
                continue
            ref = float(uni[0]["proxy"])
            line = [f"  D={D} ep={ep}:"]
            for r in grp:
                p_re = SENS_NORMAL * float(r["rmse_normal"]) + SENS_TANGENT * float(r["rmse_tangent"])
                if abs(p_re - float(r["proxy"])) > 1e-6 * max(1.0, abs(p_re)):
                    bad_proxy += 1
                rel_re = (float(r["proxy"]) - ref) / ref
                if abs(rel_re - float(r["proxy_rel"])) > 1e-6:
                    bad_rel += 1
                line.append(f"{r['arm']} {rel_re:+.1%}")
            print("  ".join(line))
        print(f"  formula check : proxy column {'OK' if bad_proxy == 0 else f'{bad_proxy} MISMATCHES'}"
              f" | proxy_rel column {'OK' if bad_rel == 0 else f'{bad_rel} MISMATCHES'}")
        has = sorted({r["arm"] for r in rows})
        print(f"  arms present  : {has}")
        if "mask" in has and "prop" in has:
            print("  >> this run contains mask+prop: candidate source of -75.8%/+99.5%."
                  "  Use it as --from for EXP1.")
    print("-" * 100)
    print("If no run reproduces -75.8%/+99.5% from its own raw.csv, the numbers were "
          "misread and Exp1 must anchor to what the CSVs actually say.")


# ---------------------------------------------------------------- EXP 1
def local_mask_metric(shape_like, n_dist, finalise):
    """Fallback mask arm: identity on core NX dims, zero on distractors.
    Isotropic within core -> isolates 'dimension masking' from direction."""
    from cmpc2d.env import NX
    N, nx, _ = shape_like
    M = np.zeros((N, nx, nx))
    idx = np.arange(NX)
    M[:, idx, idx] = 1.0
    return finalise(M, 0.95)


def get_metric(arm, p0, D, cfg, seed):
    from cmpc2d import cweight
    try:
        out = cweight.build_metric(p0["win_X"], p0["p_obs"], D, int(cfg["H"]),
                                   float(cfg["gamma"]), arm,
                                   float(cfg["eps_floor"]), seed=seed)
        M = out[0] if isinstance(out, tuple) else out
        return M, "cweight." + arm
    except Exception as e:
        if arm != "mask":
            raise
        N, nx = len(p0["win_X"]), p0["Xa"].shape[1]
        return local_mask_metric((N, nx, nx), D, cweight._finalise), "local_mask_fallback"


def traj_boundaries(X, Xn, tol=1e-6):
    """Episode ids from transition adjacency: new episode where Xn[i] != X[i+1]."""
    d = np.linalg.norm(Xn[:-1] - X[1:], axis=1)
    cut = np.where(d > tol * max(1.0, np.abs(X).max()))[0]
    ids = np.zeros(len(X), dtype=int)
    for c in cut:
        ids[c + 1:] += 1
    return ids


def derive_radius(X, p_obs, margin_now):
    r = np.linalg.norm(X[:, :2] - p_obs, axis=1) - margin_now
    return float(np.median(r)), float(np.std(r))


def rollout_windows(model, d, D, seed, H, params, device="cpu"):
    """Open-loop k-step rollout on contiguous windows of the raw split d.
    Returns per-window arrays: e_n(k), e_t(k), r_rho(k) [rho_hat - rho_true],
    rho_true(k), margin_now of window start, episode id."""
    import torch
    from cmpc2d.env import NX, f_nominal, f_distract, normal_dir, tangent_dir, sample_distract
    X, U, Xn = d["X"].astype(np.float32), d["U"].astype(np.float32), d["Xn"].astype(np.float32)
    p_obs = d["p_obs"].astype(np.float32)
    ids = traj_boundaries(X, Xn)
    rng = np.random.default_rng(90_000 + seed)
    z_all = sample_distract(rng, len(X), D).astype(np.float32) if D else None

    starts = [i for i in range(len(X) - H) if ids[i] == ids[i + H]]
    en = np.zeros((len(starts), H)); et = np.zeros_like(en)
    rr = np.zeros_like(en); rho_t = np.zeros_like(en)
    m0 = np.zeros(len(starts)); eid = np.zeros(len(starts), dtype=int)
    c = p_obs[0] if p_obs.ndim > 1 else p_obs  # obstacle centre (fixed per dataset)

    with torch.no_grad():
        for w, i in enumerate(starts):
            x = X[i].copy(); z = z_all[i].copy() if D else None
            m0[w] = d["margin_now"][i]; eid[w] = ids[i]
            for k in range(H):
                u = U[i + k]
                xin = np.concatenate([x, z]) if D else x
                pred = model(torch.tensor(xin[None]), torch.tensor(u[None])).numpy()[0]
                x = f_nominal(x[None], u[None])[0] + pred[:NX]
                if D:
                    z = f_distract(z[None])[0] + 0.0 * pred[NX:]  # true distractor dyn; model resid unused for truth
                    z = z.astype(np.float32)
                xt = X[i + k + 1] if k < H - 1 else Xn[i + k]
                po = p_obs[i + k] if p_obs.ndim > 1 else p_obs
                n = normal_dir(xt[None], po[None])[0]; t = tangent_dir(xt[None], po[None])[0]
                e = x[:2] - xt[:2]
                en[w, k] = float(e @ n); et[w, k] = float(e @ t)
                rr[w, k] = float(np.linalg.norm(x[:2] - po) - np.linalg.norm(xt[:2] - po))
                rho_t[w, k] = float(np.linalg.norm(xt[:2] - po))  # radius subtracted later
                x = x.astype(np.float32)
    return dict(en=en, et=et, r_rho=rr, dist_true=rho_t, m0=m0, eid=eid)


def boot_ci(vals, eids, fn, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    uids = np.unique(eids)
    stats = []
    for _ in range(B):
        pick = rng.choice(uids, len(uids), replace=True)
        idx = np.concatenate([np.where(eids == u)[0] for u in pick])
        stats.append(fn(vals[idx]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def run(from_dir, arms, B):
    import torch
    from cmpc2d.data import build_dataset
    from cmpc2d.env import Params
    sys.path.insert(0, ".")
    import constraint_weight_check as cwc

    cfg = json.load(open(os.path.join(from_dir, "verdict.json")))["config"]
    D = int(str(cfg["n_dist"]).split(",")[-1])          # use the D of the verdict run
    epochs = int(str(cfg["epoch_list"]).split(",")[-1])  # final registered epoch count
    hidden = tuple(int(h) for h in str(cfg["hidden"]).split(","))
    n_traj = int(cfg["n_traj"]) if cfg.get("n_traj") else 60
    seeds = list(range(int(cfg.get("seed_offset", 0)),
                       int(cfg.get("seed_offset", 0)) + int(cfg["seeds"])))
    H = int(cfg["H"])
    print(f"EXP1 config (from {from_dir}): D={D} epochs={epochs} hidden={hidden} "
          f"n_traj={n_traj} seeds={seeds} H={H} arms={arms}")
    print("NOTE: checkpoints were never persisted; models below are faithful retrains "
          "under the registered config (same seeds, same data, same metric build).")

    out_rows, ell = [], {}
    os.makedirs("results/autopsy", exist_ok=True)
    for arm in arms:
        per_seed = {"train": [], "test": []}
        for s in seeds:
            d = build_dataset(n_traj=n_traj, seed=s)
            ptr = cwc.prepare(d["train"], D, s)
            pte = cwc.prepare(d["test"], D, s + 500)
            M, src = get_metric(arm, ptr, D, cfg, s)
            model, _, _ = cwc.train(ptr, M, hidden, epochs, s, D)
            r_tr, s_tr = derive_radius(ptr["X"], d["train"]["p_obs"][0] if d["train"]["p_obs"].ndim > 1 else d["train"]["p_obs"], ptr["margin"])
            for split, dd, seed_z in (("train", d["train"], s), ("test", d["test"], s + 500)):
                res = rollout_windows(model, dd, D, seed_z, H, Params)
                res["radius"] = r_tr; res["radius_std"] = s_tr; res["metric_src"] = src
                per_seed[split].append(res)
        # ---- aggregate (concatenate windows across seeds; eids offset per seed)
        for split in ("train", "test"):
            en = np.concatenate([r["en"] for r in per_seed[split]])
            et = np.concatenate([r["et"] for r in per_seed[split]])
            rr = np.concatenate([r["r_rho"] for r in per_seed[split]])
            dt = np.concatenate([r["dist_true"] for r in per_seed[split]])
            m0 = np.concatenate([r["m0"] for r in per_seed[split]])
            eid = np.concatenate([r["eid"] + 10_000 * i for i, r in enumerate(per_seed[split])])
            rad = per_seed[split][0]["radius"]; rho_true = dt - rad
            near = m0 < np.median(m0)
            for k in range(H):
                fs = float(np.mean((rr[:, k] > 0) & (rho_true[:, k] + rr[:, k] > 0) & (rho_true[:, k] < 0)))
                lo, hi = boot_ci(np.abs(en[:, k]), eid, lambda v: np.sqrt((v ** 2).mean()), B)
                out_rows.append(dict(
                    arm=arm, split=split, k=k + 1,
                    rmse_n=float(np.sqrt((en[:, k] ** 2).mean())), rmse_n_lo=lo, rmse_n_hi=hi,
                    rmse_t=float(np.sqrt((et[:, k] ** 2).mean())),
                    bias_rho=float(rr[:, k].mean()),
                    rho_q50=float(np.quantile(rr[:, k], .50)),
                    rho_q90=float(np.quantile(rr[:, k], .90)),
                    rho_q95=float(np.quantile(rr[:, k], .95)),
                    false_safe=fs,
                    rmse_n_near=float(np.sqrt((en[near, k] ** 2).mean())),
                    rmse_n_far=float(np.sqrt((en[~near, k] ** 2).mean())),
                ))
            # prefix-max E_rho_h (p90 across windows)
            pref = np.maximum.accumulate(np.abs(rr), axis=1)
            for h in range(H):
                out_rows.append(dict(arm=arm, split=split, k=h + 1,
                                     E_rho_h_p90=float(np.quantile(pref[:, h], .90))))
            if split == "test":
                ell[arm] = np.cov(np.stack([en[:, 0], et[:, 0]]))
        print(f"  {arm}: done  (metric source: {per_seed['test'][0]['metric_src']}, "
              f"radius={per_seed['test'][0]['radius']:.4f} "
              f"+/- {per_seed['test'][0]['radius_std']:.1e} <- must be ~0 or margin "
              f"formula differs and false_safe/absolute-rho rows are unreliable)")

    with open("results/autopsy/decomposed.csv", "w", newline="") as f:
        keys = sorted({k for r in out_rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out_rows)

    # ---- ellipse figure
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        th = np.linspace(0, 2 * np.pi, 200)
        for arm, C in ell.items():
            w_, V = np.linalg.eigh(C)
            pts = V @ (np.sqrt(np.maximum(w_, 0))[:, None] * np.stack([np.cos(th), np.sin(th)]))
            ax.plot(pts[0], pts[1], label=arm)
        ax.set_xlabel("e_n (k=1, held-out)"); ax.set_ylabel("e_t")
        ax.axhline(0, lw=.5, c="gray"); ax.axvline(0, lw=.5, c="gray")
        ax.legend(); ax.set_title("residual covariance ellipse (1-sigma)")
        fig.savefig("results/autopsy/ellipse.png", dpi=150, bbox_inches="tight")
        print("wrote results/autopsy/ellipse.png")
    except Exception as e:
        print(f"(ellipse figure skipped: {e})")

    # ---- automatic reading against the 5-row table (held-out, k=1 and k=H)
    print("\n" + "=" * 100)
    print("VERDICT TABLE INPUTS (held-out; rows are pre-registered, judge manually too)")
    sel = {(r["arm"], r["split"], r["k"]): r for r in out_rows if "rmse_n" in r}
    for k in (1, H):
        u = sel[("uniform", "test", k)]
        print(f"\n  k={k}:  {'arm':8}{'rmse_n':>10}{'d%':>8}{'rmse_t':>10}{'d%':>8}"
              f"{'rho_q90':>10}{'false_safe':>11}")
        for arm in arms:
            r = sel[(arm, "test", k)]
            print(f"        {arm:8}{r['rmse_n']:>10.5f}{(r['rmse_n']/u['rmse_n']-1):>+8.1%}"
                  f"{r['rmse_t']:>10.5f}{(r['rmse_t']/u['rmse_t']-1):>+8.1%}"
                  f"{r['rho_q90']:>10.5f}{r['false_safe']:>11.4f}")
        tr = {arm: sel[(arm, "train", k)] for arm in arms}
        for arm in arms:
            if arm == "uniform":
                continue
            a, b = sel[(arm, "test", k)], tr[arm]
            au, bu = sel[("uniform", "test", k)], tr["uniform"]
            if b["rmse_n"] / bu["rmse_n"] < 0.98 and a["rmse_n"] / au["rmse_n"] > 1.02:
                print(f"        -> row B fires for {arm}: train e_n down, test e_n up "
                      f"(normal-direction overfitting)")
    print("\nfull per-k table: results/autopsy/decomposed.csv "
          "(includes CIs, near/far strata, prefix-max E_rho_h)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--from", dest="from_dir", default=None)
    ap.add_argument("--arms", default="uniform,mask,prop")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    if a.audit:
        audit()
    elif a.run:
        if not a.from_dir:
            sys.exit("--run requires --from results/<dir> (pick the mask+prop run "
                     "identified by --audit) so the config is the registered one")
        run(a.from_dir, a.arms.split(","), a.boot)
    else:
        sys.exit("use --audit first, then --run --from <dir>")
