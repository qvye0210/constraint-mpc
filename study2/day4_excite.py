#!/usr/bin/env python3
"""day4_excite.py -- wrist-excitation collection + causal verification.
Repo root, env robmpc.
  PYTHONPATH=. python day4_excite.py --collect   # 16 episodes, excitation on
  PYTHONPATH=. python day4_excite.py --verify    # retrain on merged data ->
                                                 # snapshot swap + closed loop
FROZEN (before any result): excitation = piecewise-constant uniform
inputs on wrist channels (dims 3-5) only, amplitude 2.0 (10% of u_lim),
held 5 steps; seeds train 1100-1111, dev 1200-1203; eval task unchanged;
data shared by all methods; gt collision during collection => STOP and
report (no amplitude tuning). Purpose: system-identification coverage of
the task neighbourhood (natural wrist|v| ~0.02; target ~0.2; NOT chasing
the model-induced 0.44 divergence)."""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, ".")
from exec_interface import get_ps, nominal_AB, f_nominal, LinearResidual, \
    ZeroResidual
from day3_solve_diag import build_problem, ALL, run_to_failure

OUT = "results/day4_excite"; os.makedirs(OUT, exist_ok=True)
AMP, HOLD = 2.0, 5
TR = list(range(1100, 1112)); DV = list(range(1200, 1204))


class ExciteRecord:
    def __init__(self, inner, rng, m):
        self.inner, self.rng, self.m = inner, rng, m
        self.records, self.t, self.cur = [], 0, np.zeros(m)

    def solve_time_step(self, x, u):
        if self.t % HOLD == 0:
            self.cur = np.zeros(self.m)
            self.cur[3:6] = self.rng.uniform(-AMP, AMP, 3)
        self.t += 1
        u2 = np.asarray(u, float) + self.cur
        xn = self.inner.solve_time_step(x, u2)
        self.records.append((np.array(x, float).ravel(), u2.copy(),
                             np.array(xn, float).ravel()))
        return xn

    def __getattr__(self, a):
        return getattr(self.inner, a)


def collect(ps, seeds, tag):
    from controllers.common import load_all_controllers
    from corridor_simulators.flexible_tube import FlexibleTubeSimulator
    from aux import TimeStepIntegratorContinuous, interpolate_equidistant
    from examples.six_dof.world import DemoWorld
    rows, meta = [], []
    for sd in seeds:
        fcont, _, _ = load_all_controllers(ps, horizon_length=20)
        np.random.seed(sd)
        dn, de = ps.get_nominal_dynamics(), ps.get_err_dyn_random()
        integ = ExciteRecord(TimeStepIntegratorContinuous(dt=ps.dt),
                             np.random.default_rng(sd + 5000), ps.config_dim)
        integ.inner.set_dyns(dn, de)
        w = DemoWorld()
        sim = FlexibleTubeSimulator(fcont, integ, w, aux_controller_steps=1)
        path = interpolate_equidistant(w.get_demo_path(), delta=0.05)
        (status, ts), _ = sim.simulate(path, w.sdf(path[:, :3]), nr_steps=500)
        coll = sum(0 if w.is_collision_free_gt(r[0][:3]) else 1
                   for r in integ.records)
        if coll:
            sys.exit(f"STOP: gt collision during collection seed {sd} "
                     f"({coll} states) — report, do not retune amplitude")
        wv = max(np.abs(r[0][ps.config_dim + 3:]).max()
                 for r in integ.records)
        rows += integ.records
        meta.append(dict(seed=sd, status=status, ts=ts,
                         n=len(integ.records), wrist_v_max=float(wv)))
        print(f"[{tag}] seed {sd}: {status} ts={ts} wrist|v|max {wv:.3f}",
              flush=True)
    A, B = nominal_AB(ps)
    X = np.array([r[0] for r in rows]); U = np.array([r[1] for r in rows])
    Xn = np.array([r[2] for r in rows])
    np.savez(f"{OUT}/{tag}.npz", X=X, U=U, Xn=Xn,
             R=Xn - f_nominal(X, U, A, B))
    json.dump(meta, open(f"{OUT}/{tag}_meta.json", "w"), indent=2)


def verify(ps):
    A, B = nominal_AB(ps)
    old = np.load("results/day2_dev_data/train.npz")
    new = np.load(f"{OUT}/train.npz")
    X = np.concatenate([old["X"], new["X"]]); U = np.concatenate([old["U"], new["U"]])
    R = np.concatenate([old["R"], new["R"]])
    print(f"merged train: {len(X)} transitions; wrist|v| coverage now "
          f"{np.abs(X[:, ps.config_dim+3:]).max():.3f}")
    Z = np.concatenate([X, U], 1)
    W, *_ = np.linalg.lstsq(Z, R, rcond=None)
    dA, dB = W[:X.shape[1]].T, W[X.shape[1]:].T
    np.savez(f"{OUT}/model_ls_v2.npz", dA=dA, dB=dB, norm="identity")
    snap = json.load(open("results/day3_diag/capture_777.json"))["A"]["snap"]
    print("frozen-snapshot swap, v2 model:",
          build_problem(ps, A + dA, B + dB, snap, ALL))
    res = LinearResidual(dA, dB)
    for sd in (777, 300, 301):
        steps, sn = run_to_failure(ps, sd, "A", res)
        wv = max(np.abs(np.array(s["x"])[ps.config_dim + 3:]).max()
                 for s in steps) if steps else 0.0
        print(f"closed loop seed {sd}: "
              f"{'survived 40 steps' if sn is None else 'failed@'+str(sn['step'])+' raw='+sn['raw_status']}"
              f"  wrist|v|max {wv:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(); ps = get_ps()
    if a.collect:
        collect(ps, TR, "train"); collect(ps, DV, "dev")
    if a.verify:
        verify(ps)
