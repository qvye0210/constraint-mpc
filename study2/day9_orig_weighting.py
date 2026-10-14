"""
day9_orig_weighting.py — original directional weighting (matched rotH, gate12_matched.py) vs plain MSE
on the six-DoF LINEAR residual class. Repo root (rob_cvx_mpc_rob_man), PYTHONPATH=. ; env robmpc.
Parent dir cmpc2d_tier1 is appended to sys.path so cmpc2d.cweight._finalise / metric_loss are REUSED, not rewritten.

Formula (verbatim from gate12_matched.build_matched, port to 12-d state, D=0):
  c_k = [d, (k-1)*dt*d]   (position block d, velocity block (k-1)dt d),   k = 1..H0
  M_raw = sum_k GAMMA^(k-1) c_k c_k^T ;  M_raw *= NX/trace ;  M = s*M_raw + (1-s)*I_NX ;
  M = _finalise(M + EPS/NX * I, 0.95)   ;  loss = mean_i err_i^T M_i err_i     (metric_loss)
  direction d frozen within the window; for the corridor arm d = n(x_{t+1}) of the largest-margin ball;
  for the velocity arm d lives on the velocity block: c_k = (A^{k-1})^T [0; e_j] = [0; e_j] (constant).
  kappa dose by bisection on the achieved n^T M n / mean_t t^T M t ratio (tangent = 5 orthonormal directions in 6-d).
Frozen config: H0=10, GAMMA=0.9, kappa=10 ; EPS: 1e-4 (clean-floor verdict) by default, 0.05 = gate12 file value via --eps.

Stages: --check (equivalence + numerical checks), --offline (dev H=20 table), --loop (OPTIONAL 3-episode deployment demo)
"""
import argparse, hashlib, json, os, sys
import numpy as np
sys.path.insert(0, ".")
sys.path.append(os.path.abspath(".."))          # cmpc2d_tier1 (appended: never shadows repo modules)

from aux import get_linear_double_integrator_discrete_dynamics, interpolate_equidistant, index_compute_largets_margin_bubble
from examples.six_dof.world import DemoWorld
from exec_interface import get_ps
from day5_corrector_check import v2_res

OUT = "results/day9_orig"
os.makedirs(OUT, exist_ok=True)
H0, GAMMA, KAPPA = 10, 0.9, 10.0
H_EVAL, NEAR_THR = 20, 0.1
NX = 12; M6 = 6


def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()[:10]


# ------------------------------------------------------------ weights (port of build_matched)
def build_M(dvec_pos, dvec_vel, dt, s, eps, finalise):
    """dvec_pos: (N,6) or None; dvec_vel: (N,6) or None. One of them is the direction carrier."""
    N = len(dvec_pos if dvec_pos is not None else dvec_vel)
    M = np.zeros((N, NX, NX))
    for k in range(1, H0 + 1):
        c = np.zeros((N, NX))
        if dvec_pos is not None:
            c[:, :M6] = dvec_pos; c[:, M6:] = (k - 1) * dt * dvec_pos
        else:
            c[:, M6:] = dvec_vel                            # (A^{k-1})^T [0;e] = [0;e]
        M += (GAMMA ** (k - 1)) * c[:, :, None] * c[:, None, :]
    tr = np.trace(M, axis1=1, axis2=2)[:, None, None]
    M = M * (NX / tr)
    M = s * M + (1.0 - s) * np.eye(NX)[None]
    return finalise(M + (eps / NX) * np.eye(NX), 0.95)


def achieved_kappa(M, d, block):
    """block 0 = position, 1 = velocity. tangent basis: 5 orthonormal vectors orthogonal to d in R^6."""
    ratios = []
    for Mi, di in zip(M, d):
        Q, _ = np.linalg.qr(np.c_[di, np.eye(M6)[:, :5]])
        T = Q[:, 1:6]
        sl = slice(0, M6) if block == 0 else slice(M6, NX)
        Mb = Mi[sl, sl]
        en = di @ Mb @ di
        et = np.mean([t @ Mb @ t for t in T.T])
        ratios.append(en / max(et, 1e-12))
    return float(np.mean(ratios))


def solve_s(dpos, dvel, dt, eps, finalise, block, target=KAPPA):
    a, b = 0.0, 1.0
    for _ in range(50):
        m = 0.5 * (a + b)
        k = achieved_kappa(build_M(dpos, dvel, dt, m, eps, finalise), dpos if dpos is not None else dvel, block)
        a, b = (m, b) if k < target else (a, m)
    s = 0.5 * (a + b)
    return s, achieved_kappa(build_M(dpos, dvel, dt, s, eps, finalise), dpos if dpos is not None else dvel, block)


# ------------------------------------------------------------ fits
def ols(Z, Y):                       # Y (N,12) = B Z^T, Z (N,18); B (12,18); double precision lstsq
    B, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    return B.T


def gls(Z, Y, M):
    """min_B mean_i (B z_i - y_i)^T M_i (B z_i - y_i); normal equations via kron, vec row-major of B (12,18)."""
    N, nz = Z.shape
    G = np.zeros((NX * nz, NX * nz)); r = np.zeros(NX * nz)
    for z, y, Mi in zip(Z, Y, M):
        G += np.kron(Mi, np.outer(z, z)); r += np.kron(Mi @ y, z)
    return np.linalg.solve(G, r).reshape(NX, nz)


def J(B, Z, Y, M):
    E = Z @ B.T - Y
    return float(np.mean(np.einsum("ni,nij,nj->n", E, M, E)))


def corridor_dirs(P, path_c, path_r):
    d = np.zeros_like(P)
    for i, p in enumerate(P):
        j = index_compute_largets_margin_bubble(p, path_c, path_r)
        v = p - path_c[j]; n = np.linalg.norm(v)
        d[i] = v / n if n > 1e-12 else np.eye(M6)[0]
    return d


# ------------------------------------------------------------ data
def load_train(path):
    d = np.load(path)
    Z = np.c_[d["X"], d["U"]]; Y = d["R"] if "R" in d.files else d["Xn"] - d["Xn_nom"]
    return d, Z, Y


def dev_windows(m, H, A, B):
    """Reuses day7 window logic; returns list of (ep_id, X_true (H+1,12), U (H,6), Xnom (H+1,12))."""
    W = []
    for path, cont in (("results/day2_dev_data/dev.npz", "ep"), ("results/day4_excite/dev.npz", None)):
        d = np.load(path); X, U, Xn = d["X"], d["U"], d["Xn"]; n = len(X)
        same = (np.array([d["ep"][i] == d["ep"][i + 1] for i in range(n - 1)] + [False]) if cont
                else np.array([np.allclose(Xn[i], X[i + 1], atol=1e-12, rtol=0) for i in range(n - 1)] + [False]))
        ep_id = (d["ep"] if cont else np.cumsum(np.r_[0, ~same[:-1]]) + 10_000)
        for t in range(n):
            if t + H - 1 >= n or not same[t:t + H - 1].all(): continue
            Xt = np.vstack([X[t], Xn[t:t + H]]); Ut = U[t:t + H]
            Xnom = [X[t]]
            for k in range(H): Xnom.append(A @ Xnom[-1] + B @ Ut[k])
            W.append((int(ep_id[t]), Xt, Ut, np.array(Xnom)))
    return W


def rollout(Bm, x, U, A, B):
    X = [x]
    for u in U:
        z = np.r_[X[-1], u]; X.append(A @ X[-1] + B @ u + Bm @ z)
    return np.array(X)


# ------------------------------------------------------------ stages
def stage_check(a, ps, A, B, finalise, path_c, path_r):
    d, Z, Y = load_train(a.train)
    print(f"train {a.train} md5 {md5(a.train)} N={len(Z)}")
    B_ols = ols(Z, Y)
    res = v2_res(); dA, dB = res.dA, res.dB                     # v2 = MSE arm; pins data+protocol
    B_v2 = np.c_[dA, dB]
    dv = float(np.abs(B_ols - B_v2).max())
    print(f"[data pin] OLS on {a.train} vs v2 model: max|diff| {dv:.2e}")
    out = dict(train=a.train, md5=md5(a.train), N=int(len(Z)), ols_vs_v2=dv, eps=a.eps)
    # --- velocity arm: constant W -> exact equivalence
    for j in range(M6):
        dvel = np.tile(np.eye(M6)[j], (len(Z), 1))
        s, k = solve_s(None, dvel, ps.dt, a.eps, finalise, block=1)
        Mv = build_M(None, dvel, ps.dt, s, a.eps, finalise)
        spread = float(np.abs(Mv - Mv[0]).max())
        Bv = gls(Z, Y, Mv)
        out[f"vel_e{j}"] = dict(s=s, achieved=k, W_spread=spread, B_diff_vs_ols=float(np.abs(Bv - B_ols).max()),
                                J_our=J(Bv, Z, Y, Mv), J_ols_in_our=J(B_ols, Z, Y, Mv))
        print(f"[vel e{j}] achieved kappa {k:.2f} s={s:.4f} W spread {spread:.1e} "
              f"GLS-OLS {out[f'vel_e{j}']['B_diff_vs_ols']:.2e} J_our {out[f'vel_e{j}']['J_our']:.4e} J_ols {out[f'vel_e{j}']['J_ols_in_our']:.4e}")
    # --- corridor arm: sample-varying W
    dpos = corridor_dirs(d["Xn"][:, :M6], path_c, path_r)            # d = n(x_{t+1}) per rotH convention (label use disclosed)
    s, k = solve_s(dpos, None, ps.dt, a.eps, finalise, block=0)
    Mc = build_M(dpos, None, ps.dt, s, a.eps, finalise)
    M0 = build_M(dpos, None, ps.dt, 0.0, a.eps, finalise)             # s=0 -> isotropic ("mask" convention)
    sym = float(np.abs(Mc - Mc.transpose(0, 2, 1)).max())
    ev = np.linalg.eigvalsh(Mc); ranks = (ev > 1e-10 * ev.max(1, keepdims=True)).sum(1)
    B_iso = gls(Z, Y, M0); B_c = gls(Z, Y, Mc)
    out["corr"] = dict(s=s, achieved=k, W_symmetry=sym, eig_min=float(ev.min()), eig_max=float(ev.max()),
                       cond_med=float(np.median(ev[:, -1] / ev[:, 0])), rank_min=int(ranks.min()),
                       W_spread=float(np.abs(Mc - Mc[0]).max()),
                       iso_vs_ols=float(np.abs(B_iso - B_ols).max()), B_diff_vs_ols=float(np.abs(B_c - B_ols).max()),
                       J_our=J(B_c, Z, Y, Mc), J_ols_in_our=J(B_ols, Z, Y, Mc), J_mse_our=J(B_c, Z, Y, np.eye(NX)[None].repeat(len(Z), 0)),
                       J_mse_ols=J(B_ols, Z, Y, np.eye(NX)[None].repeat(len(Z), 0)))
    print(f"[corr] achieved kappa {k:.2f} s={s:.4f} sym {sym:.1e} eig [{ev.min():.2e},{ev.max():.2e}] rank_min {ranks.min()} "
          f"W spread {out['corr']['W_spread']:.2e}")
    print(f"[corr] iso-GLS vs OLS {out['corr']['iso_vs_ols']:.2e} | Our vs OLS {out['corr']['B_diff_vs_ols']:.2e}")
    print(f"[corr] J_our(Our) {out['corr']['J_our']:.4e} <= J_our(OLS) {out['corr']['J_ols_in_our']:.4e} : "
          f"{out['corr']['J_our'] <= out['corr']['J_ols_in_our'] * (1 + 1e-9)}")
    assert out["corr"]["J_our"] <= out["corr"]["J_ols_in_our"] * (1 + 1e-9)
    try:                                                                # cross-check with the 2D loss code
        import torch; from cmpc2d.cweight import metric_loss
        E = torch.tensor(Z @ B_c.T - Y, dtype=torch.float64)
        jl = float(metric_loss(E, torch.tensor(Mc)))
        out["corr"]["metric_loss_crosscheck"] = jl; print(f"[corr] cmpc2d.metric_loss {jl:.4e} vs own {out['corr']['J_our']:.4e}")
    except Exception as e:
        out["corr"]["metric_loss_crosscheck"] = f"skipped: {e}"
    np.savez(f"{OUT}/models.npz", B_ols=B_ols, B_our=B_c, B_iso=B_iso, s=s, dpos=dpos)
    json.dump(out, open(f"{OUT}/config.json", "w"), indent=1)


def stage_offline(a, ps, A, B):
    mod = np.load(f"{OUT}/models.npz"); arms = {"MSE": mod["B_ols"], "Our": mod["B_our"], "nominal": np.zeros((NX, 18))}
    W = dev_windows(M6, H_EVAL, A, B); v_lim = float(ps.v_amp_nom)
    print(f"dev windows H={H_EVAL}: {len(W)} (episodes {len(set(w[0] for w in W))})")
    # near set frozen on NOMINAL rollout, both sides, threshold NEAR_THR
    near_mask = []
    for _, Xt, Ut, Xnom in W:
        mu, ml = v_lim - Xnom[1:, M6:], Xnom[1:, M6:] + v_lim          # (H,6) each
        near_mask.append((mu <= NEAR_THR, ml <= NEAR_THR))
    near_idx = [i for i, (u, l) in enumerate(near_mask) if u.any() or l.any()]
    print(f"near windows: {len(near_idx)}  (episodes {len(set(W[i][0] for i in near_idx))})")
    rows = []
    for name, Bm in arms.items():
        pre = [rollout(Bm, Xt[0], Ut, A, B) for _, Xt, Ut, _ in W]
        row = dict(arm=name)
        for k in (1, 5, 10, 20):
            ep = np.array([np.linalg.norm(p[k, :M6] - Xt[k, :M6]) for p, (_, Xt, _, _) in zip(pre, W)])
            ev = np.array([np.linalg.norm(p[k, M6:] - Xt[k, M6:]) for p, (_, Xt, _, _) in zip(pre, W)])
            row[f"pos_rmse_k{k}"] = float(np.sqrt((ep ** 2).mean())); row[f"vel_rmse_k{k}"] = float(np.sqrt((ev ** 2).mean()))
        # optimistic velocity-margin error per side, prefix-max over the window
        dup, dlo, dnear = [], [], []
        for i, (p, (_, Xt, _, _)) in enumerate(zip(pre, W)):
            du = np.maximum((v_lim - p[1:, M6:]) - (v_lim - Xt[1:, M6:]), 0)     # rho_hat - rho, upper side
            dl = np.maximum((p[1:, M6:] + v_lim) - (Xt[1:, M6:] + v_lim), 0)     # lower side
            dup.append(du.max()); dlo.append(dl.max())
            if i in near_idx:
                u, l = near_mask[i]; dnear.append(max(du[u].max() if u.any() else 0, dl[l].max() if l.any() else 0))
        row.update(prefix_q95_upper_all=float(np.quantile(dup, .95)), prefix_q95_lower_all=float(np.quantile(dlo, .95)))
        # near q95 with episode-cluster bootstrap CI
        dn = np.array(dnear); eps_ = np.array([W[i][0] for i in near_idx])
        if len(dn):
            rng = np.random.default_rng(0); uniq = np.unique(eps_); bs = []
            for _ in range(1000):
                pick = rng.choice(uniq, len(uniq)); sel = np.concatenate([dn[eps_ == e] for e in pick]); bs.append(np.quantile(sel, .95))
            row.update(near_prefix_q95=float(np.quantile(dn, .95)), near_ci_lo=float(np.quantile(bs, .025)), near_ci_hi=float(np.quantile(bs, .975)), n_near=len(dn))
        rows.append(row)
        print(json.dumps(row))
    import csv
    with open(f"{OUT}/offline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[1])); w.writeheader(); w.writerows(rows)
    json.dump(dict(near_idx=near_idx, n_windows=len(W)), open(f"{OUT}/near_frozen.json", "w"))


def stage_loop(a, ps, A, B):
    """OPTIONAL deployment demo (3 episodes, seeds 1000-1002), labelled demo not comparison."""
    from day7_fixed_margin import run_episode
    from exec_interface import LinearResidual
    b = json.load(open("results/day7_margin/b.json"))["b"]; mod = np.load(f"{OUT}/models.npz")
    res = {}
    for name in ("MSE", "Our"):
        Bm = mod["B_ols" if name == "MSE" else "B_our"]
        r_ = LinearResidual(Bm[:, :NX], Bm[:, NX:])          # signature assumed (dA, dB)
        res[name] = [run_episode(ps, sd, r_, b)[0] for sd in (1000, 1001, 1002)]
        print(name, [(x["status"], x["t"]) for x in res[name]])
    json.dump(res, open(f"{OUT}/loop_demo.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true"); ap.add_argument("--offline", action="store_true"); ap.add_argument("--loop", action="store_true")
    ap.add_argument("--train", default="results/day4_excite/train_merged.npz"); ap.add_argument("--eps", type=float, default=1e-4)
    a = ap.parse_args()
    from cmpc2d.cweight import _finalise                                      # reused verbatim from the 2D code
    ps = get_ps(); A, B = get_linear_double_integrator_discrete_dynamics(M6, dt=ps.dt, method="zoh")
    world = DemoWorld(); path_c = interpolate_equidistant(world.get_demo_path(), delta=0.05); path_r = world.sdf(path_c[:, :3])
    if a.check: stage_check(a, ps, A, B, _finalise, path_c, path_r)
    if a.offline: stage_offline(a, ps, A, B)
    if a.loop: stage_loop(a, ps, A, B)
