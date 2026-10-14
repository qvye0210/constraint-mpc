"""
day9_close.py — closing stage of the day9 original-weighting comparison. Repo root, PYTHONPATH=. ; env robmpc.
Reuses results/day9_orig/models.npz, config.json, offline.csv and the same dev windows. No training, no tuning,
no closed loop. Writes corrections.md, split_audit.json, corridor_metrics.csv, verdict.json into results/day9_orig.
Existing files are never overwritten except verdict.json, whose previous version (if any) is kept as verdict_v0.json.
"""
import csv, json, os, shutil, hashlib, time
import numpy as np
from day9_orig_weighting import (OUT, NX, M6, H_EVAL, dev_windows, rollout)
from aux import get_linear_double_integrator_discrete_dynamics, interpolate_equidistant, index_compute_largets_margin_bubble
from examples.six_dof.world import DemoWorld
from exec_interface import get_ps

KS = (1, 20)
NBOOT = 1000


# ------------------------------------------------------------------ 1. corrections
def write_corrections():
    p = f"{OUT}/corrections.md"
    txt = f"""# day9 corrections ({time.strftime('%Y-%m-%d %H:%M')})
Original logs and offline.csv are kept unchanged.

C1. Withdrawn: "Our+b cannot differ from ls+b in closed loop". This holds ONLY for the fixed velocity-direction
    weight arm (W constant, GLS==OLS to 4e-14). The corridor-weighted Our model has different coefficients
    (max|B_our-B_ols| = 8.35e-3); its closed-loop behaviour is NOT established by the analysis.

C2. Withdrawn: "perfect model 121 steps, so the attributable gain is capped at ~3% and the 10% threshold is
    unreachable by construction". nom_star runs against the NOMINAL plant (zero mismatch); it is not a
    performance bound for the mismatched system. A true-model MPC would also be only a reference, without an
    optimality proof. No ceiling claim is made.
"""
    if os.path.exists(p):
        shutil.copy(p, p.replace(".md", f"_{int(time.time())}.md"))
    open(p, "w").write(txt)
    print(txt)


# ------------------------------------------------------------------ 2. split audit
def seg_by_continuity(X, Xn):
    n = len(X)
    same = np.array([np.array_equal(Xn[i], X[i + 1]) for i in range(n - 1)] + [False])
    return np.cumsum(np.r_[0, ~same[:-1]])


def row_keys(X, U, Xn):
    return {hashlib.md5(np.r_[x, u, xn].astype(np.float64).tobytes()).hexdigest() for x, u, xn in zip(X, U, Xn)}


def split_audit():
    srcs = {"train_day2": "results/day2_dev_data/train.npz", "train_day4": "results/day4_excite/train.npz",
            "dev_day2": "results/day2_dev_data/dev.npz", "dev_day4": "results/day4_excite/dev.npz"}
    info, keys = {}, {}
    for tag, path in srcs.items():
        d = np.load(path)
        has_ep = "ep" in d.files
        ep = d["ep"] if has_ep else seg_by_continuity(d["X"], d["Xn"])
        info[tag] = dict(path=path, rows=int(len(d["X"])), has_episode_field=has_ep,
                         episode_ids=sorted(int(e) for e in np.unique(ep)) if has_ep else None,
                         n_episodes=int(len(np.unique(ep))),
                         episode_source="stored ep field" if has_ep else "reconstructed by Xn[i]==X[i+1] continuity (no seed identity)")
        keys[tag] = row_keys(d["X"], d["U"], d["Xn"])
    merged = np.load("results/day4_excite/train_merged.npz")
    info["fit_set"] = dict(path="results/day4_excite/train_merged.npz", rows=int(len(merged["X"])),
                           composition="train_day2 + train_day4 concatenated (made this session); OLS on it == v2 exactly")
    ep_overlap = None
    if info["train_day2"]["has_episode_field"] and info["dev_day2"]["has_episode_field"]:
        ep_overlap = sorted(set(info["train_day2"]["episode_ids"]) & set(info["dev_day2"]["episode_ids"]))
    tr = keys["train_day2"] | keys["train_day4"]; dv = keys["dev_day2"] | keys["dev_day4"]
    dup = len(tr & dv)
    # windows actually used
    ps = get_ps(); A, B = get_linear_double_integrator_discrete_dynamics(M6, dt=ps.dt, method="zoh")
    W = dev_windows(M6, H_EVAL, A, B)
    wep = {}
    for e, *_ in W: wep[e] = wep.get(e, 0) + 1
    near = json.load(open(f"{OUT}/near_frozen.json"))["near_idx"]
    near_eps = sorted({W[i][0] for i in near})
    day4_ids_verified = info["train_day4"]["has_episode_field"] and info["dev_day4"]["has_episode_field"]
    out = dict(sources=info,
               dev_windows=dict(n=len(W), per_episode={str(k): v for k, v in sorted(wep.items())},
                                note="ids >= 10000 are reconstructed day4-dev segments"),
               near_windows=dict(n=len(near), episodes=near_eps),
               overlap=dict(episode_id_overlap_day2=ep_overlap,
                            exact_transition_duplicates_train_vs_dev=dup,
                            day4_seed_identity_verified=day4_ids_verified),
               independence=("no stored-episode overlap (day2) and zero exact (x,u,x_next) duplicates across all splits; "
                             "day4 split relies on the collection registry (train 1100-1111, dev 1200-1203), "
                             "seed identity not stored in files" if (ep_overlap == [] and dup == 0 and not day4_ids_verified)
                             else "independent by stored ids and zero duplicates" if (ep_overlap == [] and dup == 0)
                             else "OVERLAP FOUND - dev results are in-sample"))
    json.dump(out, open(f"{OUT}/split_audit.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    return out, W


# ------------------------------------------------------------------ 3. corridor metrics
def corridor_metrics(W):
    ps = get_ps(); A, B = get_linear_double_integrator_discrete_dynamics(M6, dt=ps.dt, method="zoh")
    world = DemoWorld(); pc = interpolate_equidistant(world.get_demo_path(), delta=0.05); pr = world.sdf(pc[:, :3])
    mod = np.load(f"{OUT}/models.npz"); arms = {"MSE": mod["B_ols"], "Our": mod["B_our"]}
    # shared geometry per window/time, selected on the TRUE state (same for both arms)
    geo = []
    for _, Xt, _, _ in W:
        js = [index_compute_largets_margin_bubble(Xt[k, :M6], pc, pr) for k in range(1, H_EVAL + 1)]
        v1 = Xt[1, :M6] - pc[js[0]]; d_frozen = v1 / np.linalg.norm(v1)       # original rotH: n(x_{t+1}), frozen
        geo.append((js, d_frozen))
    per = {a: dict(en_frozen={k: [] for k in KS}, en_local={k: [] for k in KS}, opt={k: [] for k in KS}, pmax=[]) for a in arms}
    for name, Bm in arms.items():
        for (e, Xt, Ut, _), (js, d0) in zip(W, geo):
            P = rollout(Bm, Xt[0], Ut, A, B)
            opts = []
            for k in range(1, H_EVAL + 1):
                j = js[k - 1]; c, r = pc[j], pr[j]
                pt, ph = Xt[k, :M6], P[k, :M6]
                rho_t, rho_h = r - np.linalg.norm(pt - c), r - np.linalg.norm(ph - c)   # inside-ball margin, >=0 safe
                o = max(rho_h - rho_t, 0.0); opts.append(o)
                if k in KS:
                    dl = (pt - c) / np.linalg.norm(pt - c)
                    per[name]["en_frozen"][k].append((e, float(d0 @ (ph - pt))))
                    per[name]["en_local"][k].append((e, float(dl @ (ph - pt))))
                    per[name]["opt"][k].append((e, o))
            per[name]["pmax"].append((e, max(opts)))
    eps_all = sorted({w[0] for w in W})
    rng = np.random.default_rng(0)

    def stat(kind, vals):
        v = np.array(vals)
        return float(np.sqrt(np.mean(v ** 2))) if kind == "rmse" else float(np.quantile(v, .95))

    rows = []
    specs = [(f"normal_err_rmse_frozen_k{k}", "rmse", "en_frozen", k) for k in KS] + \
            [(f"normal_err_rmse_local_k{k}", "rmse", "en_local", k) for k in KS] + \
            [(f"opt_margin_q95_k{k}", "q95", "opt", k) for k in KS] + [("prefixmax_opt_q95_H20", "q95", "pmax", None)]
    for mname, kind, key, k in specs:
        get = lambda a: per[a][key][k] if k else per[a][key]
        m_all = [v for _, v in get("MSE")]; o_all = [v for _, v in get("Our")]
        pooled_m, pooled_o = stat(kind, m_all), stat(kind, o_all)
        ids_m = np.array([e for e, _ in get("MSE")]); vm = np.array(m_all); vo = np.array(o_all)
        bs = []
        for _ in range(NBOOT):
            pick = rng.choice(eps_all, len(eps_all))
            sel = np.concatenate([np.where(ids_m == e)[0] for e in pick])
            bs.append(stat(kind, vo[sel]) / max(stat(kind, vm[sel]), 1e-300))
        rows.append(dict(scope="pooled", metric=mname, MSE=pooled_m, Our=pooled_o, diff=pooled_o - pooled_m,
                         ratio=pooled_o / pooled_m if pooled_m else float("nan"),
                         ratio_ci_lo=float(np.quantile(bs, .025)), ratio_ci_hi=float(np.quantile(bs, .975)),
                         n_windows=len(vm), n_episodes=len(eps_all)))
        better = 0
        for e in eps_all:
            sm, so = stat(kind, vm[ids_m == e]), stat(kind, vo[ids_m == e]); better += so < sm
            rows.append(dict(scope=f"episode_{e}", metric=mname, MSE=sm, Our=so, diff=so - sm,
                             ratio=so / sm if sm else float("nan"), ratio_ci_lo="", ratio_ci_hi="",
                             n_windows=int((ids_m == e).sum()), n_episodes=1))
        rows[-len(eps_all) - 1]["episodes_Our_better"] = f"{better}/{len(eps_all)}"
    fields = ["scope", "metric", "MSE", "Our", "diff", "ratio", "ratio_ci_lo", "ratio_ci_hi", "n_windows", "n_episodes", "episodes_Our_better"]
    with open(f"{OUT}/corridor_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval=""); w.writeheader(); w.writerows(rows)
    print(f"\n{'metric':34s} {'MSE':>11s} {'Our':>11s} {'ratio':>7s} {'CI(explor.)':>16s} {'ep Our<MSE':>10s}")
    for r in rows:
        if r["scope"] == "pooled":
            print(f"{r['metric']:34s} {r['MSE']:11.4e} {r['Our']:11.4e} {r['ratio']:7.4f} "
                  f"[{r['ratio_ci_lo']:.3f},{r['ratio_ci_hi']:.3f}] {r.get('episodes_Our_better',''):>10s}")
    print("geometry: ball = author's largest-margin bubble on the frozen demo path, selected at the TRUE state, shared by both arms; "
          "controller's per-plan corridor not saved. Frozen normal = n(true x_{t+1}) per rotH protocol; local normal secondary. "
          "No frozen corridor near-definition exists -> all-window metrics only.")
    return rows


# ------------------------------------------------------------------ 4. verdict
def verdict(audit, crow):
    cfg = json.load(open(f"{OUT}/config.json"))
    off = {r["arm"]: r for r in csv.DictReader(open(f"{OUT}/offline.csv"))}
    vel = {k: v for k, v in cfg.items() if k.startswith("vel_e")}
    c = cfg["corr"]
    nm, no = float(off["MSE"]["near_prefix_q95"]), float(off["Our"]["near_prefix_q95"])
    pooled = {r["metric"]: r for r in crow if r["scope"] == "pooled"}
    p = f"{OUT}/verdict.json"
    if os.path.exists(p):
        shutil.copy(p, f"{OUT}/verdict_v0.json")
    v = dict(
        fixed_velocity_weight=dict(
            equivalent_to_MSE=True, scope="frozen constant direction, no per-sample modulation; modulated variants untested",
            max_W_spread=max(x["W_spread"] for x in vel.values()),
            max_GLS_minus_OLS=max(x["B_diff_vs_ols"] for x in vel.values()),
            objective_identical=all(abs(x["J_our"] - x["J_ols_in_our"]) <= 1e-12 * x["J_ols_in_our"] for x in vel.values())),
        corridor_weight_distinct_model=dict(distinct=c["B_diff_vs_ols"] > 1e-9, max_B_diff=c["B_diff_vs_ols"],
                                            W_spread=c["W_spread"], cond_median=c["cond_med"], rank_min=c["rank_min"]),
        own_objective_change=dict(J_our_Our=c["J_our"], J_our_OLS=c["J_ols_in_our"],
                                  relative=(c["J_our"] - c["J_ols_in_our"]) / c["J_ols_in_our"],
                                  J_mse_Our=c.get("J_mse_our"), J_mse_OLS=c.get("J_mse_ols")),
        corridor_direction_metrics={k: dict(MSE=r["MSE"], Our=r["Our"], ratio=r["ratio"],
                                            ci_exploratory=[r["ratio_ci_lo"], r["ratio_ci_hi"]],
                                            episodes_Our_better=r.get("episodes_Our_better"))
                                    for k, r in pooled.items()},
        prespecified_velocity_near_q95=dict(MSE=nm, Our=no, relative_change=(no - nm) / nm, threshold=-0.10,
                                            met=(no - nm) / nm <= -0.10),
        data_independence=audit["independence"],
        sample_sufficiency=dict(dev_episodes=len(audit["dev_windows"]["per_episode"]),
                                near_episodes=len(audit["near_windows"]["episodes"]),
                                note="windows are correlated within episodes; intervals exploratory only"),
        closed_loop_ran=False,
        closed_loop_reason="continuation criterion of this round not met (pre-specified velocity near-q95 not reduced by >=10%); "
                           "NOT a demonstration that closed-loop performance is identical (corridor-weighted Our has different coefficients)",
        corrections="see corrections.md (C1 closed-loop identity scoped to velocity arm; C2 121-step ceiling withdrawn)",
        not_supported=["any closed-loop benefit or harm of corridor-weighted Our",
                       "a general 'directional weighting is ineffective' claim beyond this linear class and frozen construction",
                       "generalisation beyond the 7 dev episodes / single path / single mismatch level",
                       "a performance ceiling for the mismatched system"],
        missing_or_nan=[k for k, r in pooled.items() if not np.isfinite(r["ratio"])])
    json.dump(v, open(p, "w"), indent=1)
    print(json.dumps(v, indent=1))


if __name__ == "__main__":
    write_corrections()
    audit, W = split_audit()
    rows = corridor_metrics(W)
    verdict(audit, rows)
