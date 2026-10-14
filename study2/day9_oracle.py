"""
day9_oracle.py — hindsight (oracle) per-window selection between the two full models (MSE, Our) on the existing
day9 dev windows. No training, no tuning, no closed loop. Repo root, PYTHONPATH=. Writes results/day9_orig/oracle_select.csv.

Per window w and per metric m, each model gets a scalar loss L_m(model, w) computed from its H=20 rollout vs truth.
  oracle  : pick argmin_model L_m(model, w) per window (uses TRUE errors -> not deployable; an upper bound on any selector)
  single  : the better of {MSE, Our} over the whole set, chosen on the same aggregate
Aggregate = q95 over windows for tail metrics, sqrt(mean) for RMSE metrics. Gain = 1 - oracle / best_single.
Supplementary row set adds the nominal model as a third option (not part of the question, reported separately).
Episode-cluster bootstrap (7 episodes) -> exploratory intervals only.
"""
import csv, json
import numpy as np
from day9_orig_weighting import OUT, M6, H_EVAL, dev_windows, rollout, NX
from aux import get_linear_double_integrator_discrete_dynamics, interpolate_equidistant, index_compute_largets_margin_bubble
from examples.six_dof.world import DemoWorld
from exec_interface import get_ps

ps = get_ps(); v_lim = float(ps.v_amp_nom)
A, B = get_linear_double_integrator_discrete_dynamics(M6, dt=ps.dt, method="zoh")
W = dev_windows(M6, H_EVAL, A, B)
near = set(json.load(open(f"{OUT}/near_frozen.json"))["near_idx"])
world = DemoWorld(); pc = interpolate_equidistant(world.get_demo_path(), delta=0.05); pr = world.sdf(pc[:, :3])
mod = np.load(f"{OUT}/models.npz")
MODELS = {"MSE": mod["B_ols"], "Our": mod["B_our"], "nominal": np.zeros((NX, 18))}

# shared corridor geometry (same as day9_close): ball chosen at TRUE state
js = [[index_compute_largets_margin_bubble(Xt[k, :M6], pc, pr) for k in range(1, H_EVAL + 1)] for _, Xt, _, _ in W]

per = {}   # per[model][metric] -> array over windows (nan where metric undefined for that window)
for name, Bm in MODELS.items():
    rows = {k: [] for k in ("vel_prefix_opt", "vel_near_prefix_opt", "corr_prefix_opt", "vel_mse_H20", "pos_mse_H20")}
    for i, (e, Xt, Ut, Xnom) in enumerate(W):
        P = rollout(Bm, Xt[0], Ut, A, B)
        vh, vt = P[1:, M6:], Xt[1:, M6:]
        du = np.maximum(vt - vh, 0)            # upper side: rho_hat - rho = (v_lim - vh) - (v_lim - vt)
        dl = np.maximum(vh - vt, 0)            # lower side
        rows["vel_prefix_opt"].append(max(du.max(), dl.max()))
        if i in near:
            mu, ml = (v_lim - Xnom[1:, M6:]) <= 0.1, (Xnom[1:, M6:] + v_lim) <= 0.1
            rows["vel_near_prefix_opt"].append(max(du[mu].max() if mu.any() else 0, dl[ml].max() if ml.any() else 0))
        else:
            rows["vel_near_prefix_opt"].append(np.nan)
        co = []
        for k in range(1, H_EVAL + 1):
            c, r = pc[js[i][k - 1]], pr[js[i][k - 1]]
            co.append(max((r - np.linalg.norm(P[k, :M6] - c)) - (r - np.linalg.norm(Xt[k, :M6] - c)), 0))
        rows["corr_prefix_opt"].append(max(co))
        rows["vel_mse_H20"].append(float(np.mean((vh - vt) ** 2)))
        rows["pos_mse_H20"].append(float(np.mean((P[1:, :M6] - Xt[1:, :M6]) ** 2)))
    per[name] = {k: np.array(v, float) for k, v in rows.items()}

eps = np.array([w[0] for w in W]); uniq = np.unique(eps)
AGG = {"vel_prefix_opt": "q95", "vel_near_prefix_opt": "q95", "corr_prefix_opt": "q95", "vel_mse_H20": "rmse", "pos_mse_H20": "rmse"}


def agg(kind, v):
    v = v[np.isfinite(v)]
    return float(np.quantile(v, .95)) if kind == "q95" else float(np.sqrt(np.mean(v)))


def evaluate(names, metric, idx):
    kind = AGG[metric]
    L = np.stack([per[n][metric][idx] for n in names])            # (n_models, n_windows)
    singles = {n: agg(kind, L[j]) for j, n in enumerate(names)}
    best_n = min(singles, key=singles.get)
    orc = agg(kind, L.min(0))
    pick = np.argmin(L, 0)
    share = {n: float(np.mean(pick[np.isfinite(L[0])] == j)) for j, n in enumerate(names)}
    return singles, best_n, orc, share


out = []
rng = np.random.default_rng(0)
for names, tag in ((("MSE", "Our"), "two-model (question)"), (("MSE", "Our", "nominal"), "three-model (supplementary)")):
    print(f"\n== {tag} ==")
    print(f"{'metric':22s} {'best single':>12s} {'value':>11s} {'oracle':>11s} {'gain':>7s} {'CI (explor.)':>17s}  pick share")
    for metric in AGG:
        all_idx = np.arange(len(W))
        singles, best_n, orc, share = evaluate(names, metric, all_idx)
        gain = 1 - orc / singles[best_n]
        ue = np.unique(eps[np.isfinite(per[names[0]][metric])]); bs = []
        for _ in range(1000):
            sel = np.concatenate([np.where(eps == e)[0] for e in rng.choice(ue, len(ue))])
            s_, b_, o_, _ = evaluate(names, metric, sel)
            bs.append(1 - o_ / s_[b_])       # best single re-chosen inside each resample (no selection advantage)
        lo, hi = np.quantile(bs, [.025, .975])
        n_used = int(np.isfinite(per[names[0]][metric]).sum())
        print(f"{metric:22s} {best_n:>12s} {singles[best_n]:11.4e} {orc:11.4e} {gain:7.2%} [{lo:6.2%},{hi:6.2%}]  "
              + " ".join(f"{k}:{v:.2f}" for k, v in share.items()) + f"  (n={n_used})")
        out.append(dict(set=tag, metric=metric, aggregate=AGG[metric], best_single=best_n,
                        best_single_value=singles[best_n], **{f"single_{k}": v for k, v in singles.items()},
                        oracle=orc, gain=gain, gain_ci_lo=lo, gain_ci_hi=hi, n_windows=n_used,
                        n_episodes=len(np.unique(eps[np.isfinite(per[names[0]][metric])])),
                        **{f"pick_share_{k}": v for k, v in share.items()}))

fields = sorted({k for r in out for k in r}, key=lambda k: list(out[-1]).index(k) if k in out[-1] else 99)
with open(f"{OUT}/oracle_select.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, restval=""); w.writeheader(); w.writerows(out)
print(f"\nwrote {OUT}/oracle_select.csv — oracle uses true errors: an upper bound on any per-window selector, not a method")
