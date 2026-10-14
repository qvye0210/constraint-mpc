"""
day8_reference_arms.py — author ft / rt vs frozen fixed-margin baseline on the SAME mismatch draws (seeds 900-909).
Repo root, PYTHONPATH=. ; b read from results/day7_margin/b.json and NOT changed.

Protocol = author's: FlexibleTubeSimulator / TubeSimulator / NomSimulator, aux_controller_steps=1, goal_tol 0.01,
nr_steps 500, author world collision check, TimeStepIntegratorContinuous with per-seed (dyn_nom, dyn_err)
taken from episode_ctx so every arm sees the identical physical parameter draw (asserted).
Recorded per arm x seed: status, t2g, executed |v| overshoot, gt collisions on executed states, MPC wall time,
solver-reported time where available.
"""
import argparse, json, os, sys, time
import numpy as np
import cvxpy as cp
sys.path.insert(0, ".")

from exec_interface import get_ps, ZeroResidual
from day5_corrector_check import episode_ctx
from aux import TimeStepIntegratorContinuous, get_linear_double_integrator_discrete_dynamics
from controllers.common import load_all_controllers
from controllers.nom_controller import NominalController
from corridor_simulators.flexible_tube import FlexibleTubeSimulator
from corridor_simulators.tube import TubeSimulator
from corridor_simulators.nom import NomSimulator
from examples.six_dof.world import DemoWorld

OUT = "results/day8_ref"
os.makedirs(OUT, exist_ok=True)
SEEDS = list(range(900, 910))
GOAL_TOL, NR_STEPS = 0.01, 500


class Recorder:
    """Transparent wrapper: same interface as TimeStepIntegratorContinuous, logs executed (x, u, x_next)."""
    def __init__(self, dt, dyn_nom, dyn_err):
        self.base = TimeStepIntegratorContinuous(dt=dt); self.base.set_dyns(dyn_nom, dyn_err)
        self.dt, self.dyn_nom, self.dyn_err = dt, dyn_nom, dyn_err
        self.X, self.U = [], []
    def set_dyns(self, a, b): self.base.set_dyns(a, b); self.dyn_nom, self.dyn_err = a, b
    def get_noise(self, x, u): return self.base.get_noise(x, u)
    def solve_time_step(self, x, u):
        xn = self.base.solve_time_step(x, u)
        self.X.append(np.array(xn, float)); self.U.append(np.array(u, float))
        return xn


def find_dyns(it):
    """Locate (dyn_nom, dyn_err) inside episode_ctx's integrator (possibly a wrapper)."""
    seen, stack = set(), [it]
    while stack:
        o = stack.pop()
        if id(o) in seen or o is None: continue
        seen.add(id(o))
        if hasattr(o, "dyn_err") and getattr(o, "dyn_err") is not None:
            return o.dyn_nom, o.dyn_err
        stack += [v for v in vars(o).values() if hasattr(v, "__dict__")] if hasattr(o, "__dict__") else []
    sys.exit(f"cannot find dyn_err in episode_ctx integrator; attrs = {list(vars(it).keys())}")


def build_nom_b(ps, H, b, ref_cont):
    m = ps.config_dim
    A, B = get_linear_double_integrator_discrete_dynamics(m, dt=ps.dt, method="zoh")
    Q = np.eye(2 * m) * 10; Q[m:, m:] *= .01
    c = NominalController(A, B, Q, np.eye(2 * m) * 1e4, np.eye(m) * 1e-3, H,
                          u_lim=ps.u_amp_nom, v_lim=ps.v_amp_nom, p_amp=np.pi)
    c.use_solver = ref_cont.use_solver
    assert len(c.problem.constraints) == len(ref_cont.problem.constraints), "nom recipe differs from loaded nom"
    assert np.allclose(c.A, ref_cont.A) and np.allclose(c.B, ref_cont.B)
    v_lim = float(ps.v_amp_nom)
    cons = list(c.problem.constraints) + [c.X[m:, 1:] <= v_lim - b, c.X[m:, 1:] >= -(v_lim - b)]
    assert cons[-1].shape == (m, H)
    c.problem = cp.Problem(c.problem.objective, cons)
    return c


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arms", default="ft,rt,nom,nom+b")
    a = ap.parse_args()
    assert not os.path.exists(f"{OUT}/done.flag"), "day8 already run (read-once)"
    ps = get_ps(); m = ps.config_dim; v_lim = float(ps.v_amp_nom)
    b = json.load(open("results/day7_margin/b.json"))["b"]
    c0 = episode_ctx(ps, SEEDS[0], ZeroResidual())
    H = c0["cont"].N
    print(f"config_dim={m} dt={ps.dt} v_lim={v_lim} H={H} FROZEN b={b:.5f}")
    fcont, rcont, ncont = load_all_controllers(ps, horizon_length=H)
    assert fcont is not None and rcont is not None, "ft/rt loader returned None (path patch missing?)"
    nbcont = build_nom_b(ps, H, b, ncont)
    world = DemoWorld()
    conts = {"ft": fcont, "rt": rcont, "nom": ncont, "nom+b": nbcont}
    arms = a.arms.split(",")
    rows = {k: [] for k in arms}

    for seed in SEEDS:
        c = episode_ctx(ps, seed, ZeroResidual())
        dyn_nom, dyn_err = find_dyns(c["it"])
        path, radii = c["path"], c["radii"]
        x0 = np.r_[path[0][:m], np.zeros(m)]
        u_probe = 0.5 * np.ones(m)
        for name in arms:
            rec = Recorder(ps.dt, dyn_nom, dyn_err)
            # same physical draw as day7 for this seed
            assert np.allclose(rec.base.solve_time_step(x0, u_probe), c["it"].solve_time_step(x0, u_probe), atol=1e-12), \
                f"integrator mismatch seed {seed} arm {name}"
            rec.X.clear(); rec.U.clear()
            cont = conts[name]
            if name == "ft":
                sim = FlexibleTubeSimulator(cont, rec, world, aux_controller_steps=1)
            elif name == "rt":
                sim = TubeSimulator(cont, rec, world, aux_controller_steps=1)
            else:
                sim = NomSimulator(cont, rec, world)
            np.random.seed(seed)
            t0 = time.time()
            (status, ts), res = sim.simulate(path, radii, nr_steps=NR_STEPS, goal_tol=GOAL_TOL, verbose=False)
            wall = time.time() - t0
            X = np.array(rec.X) if rec.X else np.zeros((0, 2 * m))
            over = float((np.abs(X[:, m:]).max() - v_lim)) if len(X) else 0.0
            gt_bad = int(sum(not world.is_collision_free_gt(x[:3]) for x in X))
            mpc_t = [r.times[2] for r in res[1:]]  # skip first (compile)
            solver_t = [r.times[3] for r in res[1:] if len(r.times) > 3]
            row = dict(seed=seed, status=status, t=int(ts), n_exec=len(X),
                       max_exec_v_over_limit=over, gt_violations=gt_bad,
                       mpc_wall_med=float(np.median(mpc_t)) if mpc_t else None,
                       mpc_wall_p95=float(np.quantile(mpc_t, .95)) if mpc_t else None,
                       solver_time_med=float(np.median(solver_t)) if solver_t else None,
                       episode_wall=wall)
            rows[name].append(row)
            np.savez(f"{OUT}/traj_{name.replace('+', 'p')}_{seed}.npz", X=X, U=np.array(rec.U))
            print(f"{name:6s} seed {seed}: {status}@{ts} over {over:+.2e} gt_bad {gt_bad} "
                  f"mpc_med {row['mpc_wall_med']} solver_med {row['solver_time_med']}")

    table = {}
    for name, rs_ in rows.items():
        succ = [r for r in rs_ if r["status"] == "success"]
        table[name] = dict(
            success_rate=len(succ) / len(rs_),
            status_counts={s: sum(r["status"] == s for r in rs_) for s in sorted({r["status"] for r in rs_})},
            t2g_median=float(np.median([r["t"] for r in succ])) if succ else None,
            t2g_all=[r["t"] if r["status"] == "success" else None for r in rs_],
            max_exec_v_over_limit=max(r["max_exec_v_over_limit"] for r in rs_),
            gt_violations_total=sum(r["gt_violations"] for r in rs_),
            mpc_wall_med=float(np.median([r["mpc_wall_med"] for r in rs_ if r["mpc_wall_med"]])),
            mpc_wall_p95=float(np.median([r["mpc_wall_p95"] for r in rs_ if r["mpc_wall_p95"]])),
            solver_time_med=float(np.median([r["solver_time_med"] for r in rs_ if r["solver_time_med"]]))
            if any(r["solver_time_med"] for r in rs_) else None)
    json.dump(dict(b=b, seeds=SEEDS, protocol="author simulators, aux=1, goal_tol=0.01, nr_steps=500",
                   arms=table, rows=rows), open(f"{OUT}/reference.json", "w"), indent=1)
    open(f"{OUT}/done.flag", "w").write(time.ctime())
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
