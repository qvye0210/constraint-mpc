#!/usr/bin/env python3
"""day2_collect_train.py -- block 2.1. Repo root, env robmpc.
  PYTHONPATH=. python day2_collect_train.py --verify   # plant + nom@10 status
  PYTHONPATH=. python day2_collect_train.py --collect  # 16 episodes (train12/dev4)
  PYTHONPATH=. python day2_collect_train.py --train    # nominal/LS/ridge + eval
  PYTHONPATH=. python day2_collect_train.py --paired   # short paired closed loop
PROTOCOL (frozen before collection; rationale in report):
  Plant = author continuous dynamics, 2% mass mismatch, ONE draw per
  episode, fixed within episode (get_err_dyn_random per episode seed).
  Task has a single path/start/goal -> only legal diversity = mismatch
  draws => dataset labelled CROSS-PARAMETER training/generalisation;
  fixed-parameter identification NOT obtainable here (reported limit).
  Split by episode seed: train 100..111 (12), dev 200..203 (4);
  test seeds 900..909 RESERVED, not run. Behaviour controller: ft.
  Hidden mismatch draw saved as metadata only, never a feature.
  Learned (A+dA, B+dB) enters the CONTROLLER only; plant untouched
  (asserted: controller matrices share no memory with plant objects).
  Normalisation: identity (physical units), recorded in config;
  LS via lstsq (no explicit inverse); ridge alphas frozen {1e-6,1e-4,1e-2},
  chosen on dev only. No intercept (matches current interface; recorded).
OUT = results/day2_dev_data/ (independent dir, nothing overwritten)."""
import argparse, json, os, pickle, sys, time
import numpy as np

sys.path.insert(0, ".")
from exec_interface import (RecordingIntegrator, get_ps, nominal_AB,
                            f_nominal, LinearResidual, ZeroResidual,
                            make_controller, rollout)

OUT = "results/day2_dev_data"; os.makedirs(OUT, exist_ok=True)
TRAIN_SEEDS = list(range(100, 112)); DEV_SEEDS = list(range(200, 204))
TEST_SEEDS_RESERVED = list(range(900, 910))
ALPHAS = [1e-6, 1e-4, 1e-2]; NR_STEPS = 500


def build_sim(ps, controller, seed):
    """Author plant: continuous integrator + per-episode mismatch draw."""
    from aux import TimeStepIntegratorContinuous
    from corridor_simulators.nom import NomSimulator
    from examples.six_dof.world import DemoWorld
    np.random.seed(seed)                      # governs the mismatch draw
    dyn_nom = ps.get_nominal_dynamics()
    dyn_err = ps.get_err_dyn_random()
    integ = RecordingIntegrator(TimeStepIntegratorContinuous(dt=ps.dt))
    integ.inner.set_dyns(dyn_nom, dyn_err)
    world = DemoWorld()
    sim = NomSimulator(controller, integ, world)
    # plant/controller isolation: matrices vs pinocchio dyn objects
    assert not any(controller.A is o or controller.B is o
                   for o in (dyn_nom, dyn_err))
    return sim, integ, world, dyn_err


def path_and_radii(world):
    from aux import interpolate_equidistant
    p = interpolate_equidistant(world.get_demo_path(), delta=0.05)
    return p, world.sdf(p[:, :3])


def run_episode(ps, seed, controller, nr_steps=NR_STEPS):
    sim, integ, world, dyn_err = build_sim(ps, controller, seed)
    path, radii = path_and_radii(world)
    t0 = time.time()
    (status, ts), _ = sim.simulate(path, radii, nr_steps=nr_steps)
    wall = time.time() - t0
    solver_status = getattr(controller, "status", "n/a")
    return dict(records=integ.records, status=status, ts=ts, wall=wall,
                solver_status=str(solver_status))


def cmd_verify(ps):
    """Section 2: mismatch plant + nom@10 with raw solver status."""
    from controllers.nom_controller import NominalController
    import cvxpy as cp
    A, B = nominal_AB(ps)
    cont = make_controller(ps, ZeroResidual())
    # capture raw status per step by wrapping solve
    raw = []
    orig = cont.solve
    def solve_logged():
        try:
            ok = orig()
            raw.append(str(cont.problem.status))
            return ok
        except Exception as e:
            raw.append(f"EXC:{type(e).__name__}")
            raise
    cont.solve = solve_logged
    r = run_episode(ps, seed=1, controller=cont, nr_steps=40)
    print(f"nom on mismatch plant: status={r['status']} ts={r['ts']} "
          f"raw statuses tail: {raw[-3:]}")
    json.dump(dict(term=r["status"], ts=r["ts"], raw_statuses=raw),
              open(f"{OUT}/nom_verify.json", "w"), indent=2)
    n = len(r["records"])
    if n:
        x, u, xn = r["records"][-1]
        print(f"transitions recorded {n}; last has pre/exec/post: "
              f"{x.shape}/{u.shape}/{xn.shape}")


def collect_split(ps, seeds, tag):
    from controllers.common import load_all_controllers
    rows, meta = [], []
    for sd in seeds:
        fcont, rcont, nom_cont = load_all_controllers(ps, horizon_length=20)
        # behaviour controller: ft (author's succeeding method)
        from corridor_simulators.flexible_tube import FlexibleTubeSimulator
        from aux import TimeStepIntegratorContinuous
        from examples.six_dof.world import DemoWorld
        np.random.seed(sd)
        dyn_nom = ps.get_nominal_dynamics(); dyn_err = ps.get_err_dyn_random()
        integ = RecordingIntegrator(TimeStepIntegratorContinuous(dt=ps.dt))
        integ.inner.set_dyns(dyn_nom, dyn_err)
        world = DemoWorld()
        sim = FlexibleTubeSimulator(fcont, integ, world,
                                    aux_controller_steps=1)
        path, radii = path_and_radii(world)
        t0 = time.time()
        (status, ts), _ = sim.simulate(path, radii, nr_steps=NR_STEPS)
        wall = time.time() - t0
        A, B = nominal_AB(ps)
        for i, (x, u, xn) in enumerate(integ.records):
            gt = world.is_collision_free_gt(x[:len(x) // 2][:3])
            rows.append((sd, i, x, u, xn, gt))
        meta.append(dict(seed=sd, status=status, ts=ts, wall=wall,
                         n_trans=len(integ.records),
                         controller="ft"))
        print(f"[{tag}] seed {sd}: {status} ts={ts} "
              f"trans={len(integ.records)} {wall:.1f}s", flush=True)
    A, B = nominal_AB(ps)
    X = np.array([r[2] for r in rows]); U = np.array([r[3] for r in rows])
    Xn = np.array([r[4] for r in rows])
    ep = np.array([r[0] for r in rows]); st = np.array([r[1] for r in rows])
    gt = np.array([r[5] for r in rows])
    Xn_nom = f_nominal(X, U, A, B)
    np.savez(f"{OUT}/{tag}.npz", X=X, U=U, Xn=Xn, ep=ep, step=st,
             gt_free=gt, Xn_nom=Xn_nom, R=Xn - Xn_nom, dt=ps.dt)
    json.dump(meta, open(f"{OUT}/{tag}_meta.json", "w"), indent=2)
    m = len(X) // 2 if X.ndim == 1 else X.shape[1] // 2
    Rr = Xn - Xn_nom
    print(f"[{tag}] eps {len(seeds)} trans {len(X)}; residual |pos| med "
          f"{np.median(np.abs(Rr[:, :6])):.2e}  |vel| med "
          f"{np.median(np.abs(Rr[:, 6:])):.2e}")


def fit_ls(X, U, R, alpha=0.0):
    Z = np.concatenate([X, U], 1)
    if alpha == 0.0:
        W, *_ = np.linalg.lstsq(Z, R, rcond=None)
    else:
        n = Z.shape[1]
        W = np.linalg.solve(Z.T @ Z + alpha * np.eye(n), Z.T @ R)
    dA, dB = W[:X.shape[1]].T, W[X.shape[1]:].T
    return dA, dB


def eval_model(d, dA, dB, A, B, H=10):
    res = LinearResidual(dA, dB) if dA is not None else ZeroResidual()
    pred = f_nominal(d["X"], d["U"], A, B) + res(d["X"], d["U"])
    e = pred - d["Xn"]; m = d["X"].shape[1] // 2
    out = dict(pos_rmse=float(np.sqrt((e[:, :m] ** 2).mean())),
               vel_rmse=float(np.sqrt((e[:, m:] ** 2).mean())))
    # multi-step: within-episode windows, recorded u_exec, own recursion
    errs = []
    for sd in np.unique(d["ep"]):
        i = np.where(d["ep"] == sd)[0]
        if len(i) < H + 1:
            continue
        for s0 in range(0, len(i) - H, H):
            idx = i[s0:s0 + H + 1]
            Xr = rollout(d["X"][idx[0]], d["U"][idx[:-1]], A, B, res)
            errs.append(np.linalg.norm(Xr[-1][:m] - d["Xn"][idx[-2]][:m]))
    out["k10_pos_err"] = float(np.median(errs)) if errs else None
    out["k10_windows"] = len(errs)
    return out


def cmd_train(ps):
    A, B = nominal_AB(ps)
    tr = dict(np.load(f"{OUT}/train.npz")); dv = dict(np.load(f"{OUT}/dev.npz"))
    table = {}
    table["nominal"] = {"train": eval_model(tr, None, None, A, B),
                        "dev": eval_model(dv, None, None, A, B)}
    dA, dB = fit_ls(tr["X"], tr["U"], tr["R"])
    np.savez(f"{OUT}/model_ls.npz", dA=dA, dB=dB, norm="identity")
    table["ls_mse"] = {"train": eval_model(tr, dA, dB, A, B),
                       "dev": eval_model(dv, dA, dB, A, B)}
    best = None
    for a in ALPHAS:
        dAr, dBr = fit_ls(tr["X"], tr["U"], tr["R"], alpha=a)
        m = eval_model(dv, dAr, dBr, A, B)
        if best is None or m["pos_rmse"] < best[1]["pos_rmse"]:
            best = (a, m, dAr, dBr)
    np.savez(f"{OUT}/model_ridge.npz", dA=best[2], dB=best[3],
             alpha=best[0], norm="identity")
    table["ridge"] = {"alpha": best[0],
                      "train": eval_model(tr, best[2], best[3], A, B),
                      "dev": best[1]}
    json.dump(table, open(f"{OUT}/pred_table.json", "w"), indent=2)
    print(json.dumps(table, indent=2))
    # sanity: online vs offline prediction on one (x,u)
    c = make_controller(ps, LinearResidual(dA, dB))
    x0, u0 = tr["X"][0], tr["U"][0]
    off = f_nominal(x0, u0, A, B) + LinearResidual(dA, dB)(x0, u0)
    on = (c.A @ x0 + c.B @ u0)
    print(f"online==offline predictor: {np.max(np.abs(on - off)):.2e}")


def cmd_paired(ps):
    ld = np.load(f"{OUT}/model_ls.npz")
    arms = [("nominal", ZeroResidual()),
            ("ls_mse", LinearResidual(ld["dA"], ld["dB"]))]
    out = {}
    for nm, res in arms:
        cont = make_controller(ps, res)
        r = run_episode(ps, seed=777, controller=cont, nr_steps=80)
        out[nm] = dict(status=r["status"], ts=r["ts"],
                       solver=r["solver_status"], wall=r["wall"])
        print(f"paired {nm}: {out[nm]}")
    json.dump(out, open(f"{OUT}/paired_short.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("verify", "collect", "train", "paired"):
        ap.add_argument(f"--{f}", action="store_true")
    a = ap.parse_args(); ps = get_ps()
    if a.verify: cmd_verify(ps)
    if a.collect:
        collect_split(ps, TRAIN_SEEDS, "train")
        collect_split(ps, DEV_SEEDS, "dev")
        print("reserved test seeds (NOT run):", TEST_SEEDS_RESERVED)
    if a.train: cmd_train(ps)
    if a.paired: cmd_paired(ps)

