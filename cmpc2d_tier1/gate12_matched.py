#!/usr/bin/env python3
"""Five-gate root-cause plan: GATE 1 (static checks, no training) then
GATE 2 (matched-family trainings). Run from cmpc2d_tier1, PYTHONPATH=.

MATCHED FAMILY (pre-registered): c_k = [d, (k-1)*dt*d, 0...],
M_raw = sum_k gamma^{k-1} c_k c_k^T, direction d FROZEN within the window:
  rot : d = n(x_{t+1})  (rotates across samples only)
  fx  : d = (1,0)   fy : d = (0,1)
M_raw scaled to trace NX, kappa via the source's strength interpolation
toward isotropic-core, floor/eps and _finalise exactly per mask convention
so that kappa=1 == M_mask ELEMENTWISE. Old prop differs from this family in
three declared ways (within-window rotation, trace/floor convention,
strength frame) -- that difference is the object under test, not a bug.

GATE 1 (abort on any failure, no training):
  a. M_matched(kappa=1) == M_mask  (<1e-9)
  b. all three families: identical eigenvalue spectra per sample and across
     samples; achieved kappa within +-5% of target for each family
  c. margin sign convention verified (margin_now == |p-c|-r, rho>0 safe)
  d. near set (margin_now <= q25) is arm-independent by construction; sizes printed
  e. angle stats: /_(n_t, n_{t+1}) and /_(n_{t+1}, n_{t+k}) medians/p90
GATE 2 (12 trainings, 64x64, seeds 0/1/2):
  arms rot-k5, rot-k10, fx-k10, fy-k10; matched-k1 baseline = mask rows
  from results/revival_gate/final.csv (licensed by gate 1a equality).
  Records final weighted train loss, train/test e_n/e_t, near-q95.
"""
import csv, os, sys
import numpy as np
import torch

sys.path.insert(0, ".")
from revival_gate import get_apis, decompose, stats

WIDTH = (64, 64); SEEDS = [0, 1, 2]; EPOCHS = 2000; D = 8
H, GAMMA, EPS = 10, 0.9, 0.05
ARMS = [("rot", 5.0), ("rot", 10.0), ("fx", 10.0), ("fy", 10.0)]
OUT = "results/gate12_matched"


def build_matched(api, ptr, fam, s):
    from cmpc2d.cweight import _finalise
    NX = api["NX"]; dt = api["Params"].dt
    N = len(ptr["win_X"]); nx = NX + D
    if fam == "rot":
        dvec = api["normal_dir"](ptr["win_X"][:, 0], ptr["p_obs"])
    else:
        dvec = np.tile([1.0, 0.0] if fam == "fx" else [0.0, 1.0], (N, 1))
    M = np.zeros((N, nx, nx))
    for k in range(1, H + 1):
        c = np.zeros((N, nx))
        c[:, :2] = dvec
        c[:, 2:4] = (k - 1) * dt * dvec
        M += (GAMMA ** (k - 1)) * c[:, :, None] * c[:, None, :]
    tr = np.trace(M, axis1=1, axis2=2)[:, None, None]
    M = M * (NX / tr)                                   # trace -> NX (mask conv.)
    core = np.zeros(nx); core[:NX] = 1.0
    iso = np.diag(core)                                 # trace NX
    M = s * M + (1.0 - s) * iso[None]
    return _finalise(M + (EPS / nx) * np.eye(nx), 0.95)


def achieved(api, ptr, fam, s):
    M = build_matched(api, ptr, fam, s)
    if fam == "rot":
        n = api["normal_dir"](ptr["win_X"][:, 0], ptr["p_obs"])
    else:
        n = np.tile([1.0, 0.0] if fam == "fx" else [0.0, 1.0],
                    (len(ptr["win_X"]), 1))
    t = np.stack([-n[:, 1], n[:, 0]], -1)
    nx = M.shape[1]
    pad = lambda v: np.pad(v, ((0, 0), (0, nx - 2)))
    en = np.einsum("bi,bij,bj->b", pad(n), M, pad(n))
    et = np.einsum("bi,bij,bj->b", pad(t), M, pad(t))
    return float((en / np.maximum(et, 1e-12)).mean()), M


def solve_s(api, ptr, fam, target):
    a, b = 0.0, 1.0
    for _ in range(50):
        m = 0.5 * (a + b)
        k, _ = achieved(api, ptr, fam, m)
        if k < target: a = m
        else: b = m
    s = 0.5 * (a + b)
    return s, achieved(api, ptr, fam, s)[0]


def main():
    api = get_apis(); cwc = api["cwc"]
    os.makedirs(OUT, exist_ok=True)
    d0 = api["build_dataset"](n_traj=60, seed=SEEDS[0])
    ptr0 = cwc.prepare(d0["train"], D, SEEDS[0])

    # ---------------- GATE 1 ----------------
    from cmpc2d.cweight import build_metric
    Mm = build_metric(ptr0["win_X"], ptr0["p_obs"], D, H, GAMMA, "mask", EPS)
    M_mask = Mm[0] if isinstance(Mm, tuple) else Mm
    M1 = build_matched(api, ptr0, "rot", 0.0)
    da = float(np.abs(M1 - M_mask).max())
    print(f"gate1a matched(kappa=1) vs mask: max|diff| = {da:.2e}")
    if da > 1e-9:
        sys.exit("ABORT gate1a")
    eigs = {}
    for fam in ("rot", "fx", "fy"):
        _, M = achieved(api, ptr0, fam, 0.7)
        w = np.linalg.eigvalsh(M)
        eigs[fam] = w
        spread = float(np.abs(w - w.mean(0)).max())
        print(f"gate1b {fam}: per-sample eigenvalue spread {spread:.2e}")
        if spread > 1e-9:
            sys.exit("ABORT gate1b (spectrum varies across samples)")
    for fam in ("fx", "fy"):
        dspec = float(np.abs(eigs[fam] - eigs["rot"]).max())
        print(f"gate1b {fam} vs rot spectrum diff {dspec:.2e}")
        if dspec > 1e-9:
            sys.exit("ABORT gate1b (families not spectrum-matched)")
    doses = {}
    for fam, kt in ARMS:
        s, ach = solve_s(api, ptr0, fam, kt)
        ok = abs(ach / kt - 1) <= 0.05
        print(f"gate1b dose {fam} k={kt:g}: s={s:.5f} achieved {ach:.3f} "
              f"-> {'OK' if ok else 'FAIL'}")
        if not ok:
            sys.exit("ABORT gate1b dose")
        doses[(fam, kt)] = s
    r = np.linalg.norm(ptr0["X"][:, :2] - ptr0["p_obs"], axis=1) - ptr0["margin"]
    print(f"gate1c margin convention: radius {np.median(r):.4f} std {np.std(r):.2e} "
          f"(rho = |p-c| - r, rho>0 safe; must be ~0)")
    if np.std(r) > 1e-6:
        sys.exit("ABORT gate1c")
    q25 = np.quantile(ptr0["margin"], 0.25)
    print(f"gate1d near set: margin_now<=q25={q25:.4f}, "
          f"n={int((ptr0['margin'] <= q25).sum())}/{len(ptr0['margin'])} "
          "(dataset-defined, identical for every arm)")
    n_t = api["normal_dir"](ptr0["X"], ptr0["p_obs"])
    ang = lambda a, b: np.degrees(np.arccos(
        np.clip((a * b).sum(-1), -1, 1)))
    n1 = api["normal_dir"](ptr0["win_X"][:, 0], ptr0["p_obs"])
    a01 = ang(n_t, n1)
    print(f"gate1e angle(n_t, n_t+1): median {np.median(a01):.1f} deg, "
          f"p90 {np.quantile(a01, .9):.1f} deg")
    for k in (3, 5, 10):
        nk = api["normal_dir"](ptr0["win_X"][:, k - 1], ptr0["p_obs"])
        ak = ang(n1, nk)
        print(f"gate1e angle(n_t+1, n_t+{k}): median {np.median(ak):.1f} deg, "
              f"p90 {np.quantile(ak, .9):.1f} deg")
    print("GATE 1 PASSED\n")

    # ---------------- GATE 2 ----------------
    ref = {}
    for row in csv.DictReader(open("results/revival_gate/final.csv")):
        if (row["split"] == "test" and int(row["epoch"]) == EPOCHS
                and row["width"] == str(WIDTH) and row["arm"] in ("mask", "k1")):
            ref[(row["arm"], int(row["seed"]))] = row
    rows = []
    for seed in SEEDS:
        d = api["build_dataset"](n_traj=60, seed=seed)
        ptr = cwc.prepare(d["train"], D, seed)
        pte = cwc.prepare(d["test"], D, seed + 500)
        radius = float(np.median(np.linalg.norm(
            ptr["X"][:, :2] - ptr["p_obs"], axis=1) - ptr["margin"]))
        for fam, kt in ARMS:
            M = build_matched(api, ptr, fam, doses[(fam, kt)])
            model, hist, _ = cwc.train(ptr, M, WIDTH, EPOCHS, seed, D)
            from cmpc2d.cweight import metric_loss
            with torch.no_grad():
                err = (model(torch.tensor(ptr["Xa"]), torch.tensor(ptr["U"]))
                       - torch.tensor(ptr["R"]))
                wtl = float(metric_loss(err, torch.tensor(
                    M.astype(np.float32))))
            res = {}
            for split, prep in (("train", ptr), ("test", pte)):
                st = stats(decompose(api, model, prep, None, D), radius)
                res.update({f"{split}_{k}": v for k, v in st.items()})
            rows.append(dict(family=fam, kappa=kt, seed=seed,
                             wtrain_loss=wtl,
                             plain_train_mse=float(hist[-1]["plain_mse"])
                             if hist else float("nan"), **res))
            print(f"  {fam} k={kt:g} seed{seed}: train_n {res['train_rmse_n']:.2e} "
                  f"test_n {res['test_rmse_n']:.2e} q95 {res['test_q95_near']:.2e}")
    with open(f"{OUT}/gate2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 88)
    print("GATE 2 readout (baseline matched-k1 == mask, from final.csv):")
    for seed in SEEDS:
        mk = ref[("mask", seed)]; k1 = ref[("k1", seed)]
        print(f"  seed{seed}: mask(=matched-k1) q95 {float(mk['q95_near']):.2e} "
              f"rmse_n {float(mk['rmse_n']):.2e} | OLD-prop k1 q95 "
              f"{float(k1['q95_near']):.2e} rmse_n {float(k1['rmse_n']):.2e}")
    print("readings (pre-registered): old-k1 >> mask => old structure is a root "
          "cause at this width. fx/fy ~= mask while rot degrades => cross-sample "
          "rotation (gradient conflict). all degrade => neither rotation nor old "
          "structure necessary -> GATE 3 (convergence / whitening).")
    print(f"wrote {OUT}/gate2.csv")


if __name__ == "__main__":
    main()
