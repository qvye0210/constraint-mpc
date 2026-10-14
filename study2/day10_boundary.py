"""
day10_boundary.py — "action boundary and decision value" offline check. Repo root, PYTHONPATH=. ; env robmpc.
No training, no task change, no hyper-parameter scan. b = 0.0862 frozen (results/day7_margin/b.json).
Output: results/day10_boundary/{config.json, candidates.csv, samples.csv, episodes.csv, verdict.json, samples/*.npz}

FIXED BEFORE RUNNING (declared, not tuned on results):
  TOL = 1e-6 in each constraint's own unit (rad/s, rad, rad/s^2): CLARABEL feasibility ~1e-8, solve_ivp atol 1e-6
  grid s = 0,1/16,...,1 ; refinement 8 bisections inside the first safe->unsafe interval (width 1/16/2^8 = 2.44e-4)
  first-order FD step DELTA = 1e-2 ; |dm/ds| < 1e-8 treated as zero ; collision excluded from first-order (not smooth)
  "vast majority" for the first-order flag = >= 80% of valid samples within 0.05 of the refined true boundary
  corridor frozen per sample: path_traj = current position repeated N+1 (author's episode-start rule), goal from
  compute_goal_state; the closed loop's shifted-plan corridor is NOT reproduced (declared deviation)
  integrator: author TimeStepIntegratorContinuous (stateless) rebuilt from the episode's (dyn_nom, dyn_err);
  parameter identity verified by replaying recorded dev transitions (max|dx| < 1e-8), else the episode is blocked
  collision: DemoWorld.is_collision_free_gt(x[:3]) (as day7/day8), checked on k=1..H states
"""
import csv, hashlib, json, os, sys, time
import numpy as np
sys.path.insert(0, ".")
from exec_interface import get_ps, ZeroResidual
from day5_corrector_check import v2_res
from day7_fixed_margin import make_ctx
from day8_reference_arms import find_dyns
from aux import compute_goal_state, TimeStepIntegratorContinuous
from examples.six_dof.world import DemoWorld

OUT = "results/day10_boundary"; os.makedirs(f"{OUT}/samples", exist_ok=True)
TOL, NGRID, NREF, DELTA, DZERO, FO_FRAC = 1e-6, 16, 8, 1e-2, 1e-8, 0.80
N_EP, N_STATES = 5, 8
TYPES = ("vel_upper", "vel_lower", "corridor", "input_upper", "input_lower", "pos_box")

ps = get_ps(); m = ps.config_dim; v_lim, u_lim = float(ps.v_amp_nom), float(ps.u_amp_nom)
b = json.load(open("results/day7_margin/b.json"))["b"]
world = DemoWorld()
RES_MSE = v2_res()


# ------------------------------------------------------------------ episodes with verified parameters
def seg_by_continuity(X, Xn):
    same = np.array([np.array_equal(Xn[i], X[i + 1]) for i in range(len(X) - 1)] + [False])
    return np.cumsum(np.r_[0, ~same[:-1]])


def integrator_for(seed):
    c = make_ctx(ps, seed, ZeroResidual(), 0.0)
    it = TimeStepIntegratorContinuous(dt=ps.dt); it.set_dyns(*find_dyns(c["it"]))
    return it


def replay_err(it, X, U, Xn, n=3):
    return max(float(np.abs(np.array(it.solve_time_step(X[i], U[i]), float) - Xn[i]).max()) for i in range(min(n, len(X))))


def pick_episodes():
    eps, blocked = [], []
    d2 = np.load("results/day2_dev_data/dev.npz")
    for e in sorted(np.unique(d2["ep"])):
        idx = np.where(d2["ep"] == e)[0]; idx = idx[np.argsort(d2["step"][idx])]
        it = integrator_for(int(e)); err = replay_err(it, d2["X"][idx], d2["U"][idx], d2["Xn"][idx])
        (eps if err < 1e-8 else blocked).append(dict(source="day2_dev", seed=int(e), rows=idx.tolist(), replay_err=err))
    d4 = np.load("results/day4_excite/dev.npz"); seg = seg_by_continuity(d4["X"], d4["Xn"])
    for g in np.unique(seg):
        idx = np.where(seg == g)[0]
        if len(idx) < N_STATES + 1: blocked.append(dict(source="day4_dev", segment=int(g), reason=f"len {len(idx)}")); continue
        hit = None
        for sd in (1200, 1201, 1202, 1203):
            err = replay_err(integrator_for(sd), d4["X"][idx], d4["U"][idx], d4["Xn"][idx])
            if err < 1e-8: hit = (sd, err); break
        (eps if hit else blocked).append(dict(source="day4_dev", segment=int(g), seed=hit[0] if hit else None,
                                              rows=idx.tolist(), replay_err=hit[1] if hit else None))
    return eps[:N_EP], blocked, eps[N_EP:]


# ------------------------------------------------------------------ rollouts and margins
def roll_true(it, x, U):
    X = [np.array(x, float)]
    for k in range(U.shape[1]): X.append(np.array(it.solve_time_step(X[-1], U[:, k]), float))
    return np.array(X)


def roll_model(x, U, A, B, dA=None, dB=None):
    X = [np.array(x, float)]
    for k in range(U.shape[1]):
        xn = A @ X[-1] + B @ U[:, k]
        if dA is not None: xn = xn + dA @ X[-1] + dB @ U[:, k]
        X.append(xn)
    return np.array(X)


def margins(X, U, cs, rs, xg, collide=True):
    Xk = X[1:]
    mg = dict(vel_upper=v_lim - Xk[:, m:], vel_lower=Xk[:, m:] + v_lim,
              corridor=rs[1:] - np.linalg.norm(Xk[:, :m] - cs[1:], axis=1),
              input_upper=u_lim - U.T, input_lower=U.T + u_lim, pos_box=np.pi - np.abs(Xk[:, :m]))
    coll = np.array([not world.is_collision_free_gt(x[:3]) for x in Xk]) if collide else np.zeros(len(Xk), bool)
    return mg, coll, dict(term_vmax=float(np.abs(X[-1, m:]).max()), goal_dist=float(np.linalg.norm(X[-1] - xg)))


def is_safe(mg, coll):
    return (not coll.any()) and all(float(v.min()) >= -TOL for v in mg.values())


def first_violation(mg, coll):
    out = [f"{t}@k{np.unravel_index(np.argmin(v), v.shape)}" for t, v in mg.items() if v.min() < -TOL]
    if coll.any(): out.append(f"collision@k{int(np.argmax(coll)) + 1}")
    return ";".join(out)


def cost(X, U, xg, Q, R, Qe):
    N = U.shape[1]; J = 0.0
    for i in range(N):
        d = X[i] - X[N]; J += d @ Q @ d + U[:, i] @ R @ U[:, i]
    d = X[N] - xg
    return float(J + d @ Qe @ d)


def search(f):
    """f(s) -> bool safe. Grid then refine first safe->unsafe interval. Returns dict + evaluated points."""
    grid = [i / NGRID for i in range(NGRID + 1)]; sg = [f(s) for s in grid]
    if not sg[0]: return dict(anchor_safe=False), grid, sg
    bad = [i for i, v in enumerate(sg) if not v]
    if not bad: return dict(anchor_safe=True, censored=True, lo=1.0, hi=None, resafe=False), grid, sg
    i = bad[0]; lo, hi = grid[i - 1], grid[i]
    for _ in range(NREF):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) else (lo, mid)
    return dict(anchor_safe=True, censored=False, lo=lo, hi=hi, resafe=any(sg[i + 1:])), grid, sg


# ------------------------------------------------------------------ main
def main():
    assert not os.path.exists(f"{OUT}/done.flag"), "already run"
    t0 = time.time()
    eps, blocked, spare = pick_episodes()
    A, B = None, None
    cfg = dict(b=b, TOL=TOL, NGRID=NGRID, NREF=NREF, DELTA=DELTA, DZERO=DZERO, FO_FRAC=FO_FRAC, v_lim=v_lim, u_lim=u_lim,
               dt=ps.dt, episodes=[{k: v for k, v in e.items() if k != "rows"} for e in eps], blocked=blocked,
               spare_unused=[{k: v for k, v in e.items() if k != "rows"} for e in spare],
               corridor_rule="frozen: current position repeated N+1; goal via compute_goal_state",
               collision="DemoWorld.is_collision_free_gt(x[:3])", fo_criterion=f">= {FO_FRAC:.0%} within 0.05")
    cfg["hash"] = hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
    json.dump(cfg, open(f"{OUT}/config.json", "w"), indent=1)
    print(f"episodes: {[(e['source'], e.get('seed')) for e in eps]}  blocked: {len(blocked)}  cfg {cfg['hash']}")
    d2 = np.load("results/day2_dev_data/dev.npz"); d4 = np.load("results/day4_excite/dev.npz")
    cand_rows, samp_rows = [], []
    for e in eps:
        dsrc = d2 if e["source"] == "day2_dev" else d4
        rows = np.array(e["rows"]); ti = np.unique(np.linspace(0, len(rows) - 2, N_STATES).round().astype(int))
        c0 = make_ctx(ps, e["seed"], ZeroResidual(), 0.0); cb = make_ctx(ps, e["seed"], ZeroResidual(), b)
        assert c0["cont"] is not cb["cont"] and cb["cont"]._n_added == 2 and c0["cont"]._n_added == 0
        it = TimeStepIntegratorContinuous(dt=ps.dt); it.set_dyns(*find_dyns(c0["it"]))
        A, B = c0["cont"].A, c0["cont"].B; Q, R = c0["cont"].Q, c0["cont"].R; Qe = np.eye(2 * m) * 1e4
        for t in ti:
            x = dsrc["X"][rows[t]].astype(float); sid = f"{e['source']}_{e['seed']}_t{t}"
            ptraj = x[:m][None].repeat(c0["cont"].N + 1, axis=0)
            cs, rs = c0["sim"].get_corridor_balls(ptraj, c0["path"], c0["radii"], c0["p_g"])
            xg, _ = compute_goal_state(cs, rs, c0["pt"], c0["p_g"], return_index=True)
            st = {}
            for tag, c in (("Ub", cb), ("U0", c0)):
                c["cont"].set_parameter_values(x, cs.T, rs, xg); ok = c["cont"].solve()
                st[tag] = (ok, getattr(c["cont"], "status", "exception"),
                           c["cont"].U.value.copy() if ok else None, getattr(c["cont"], "loss", np.nan))
            rec = dict(sample=sid, episode=f"{e['source']}_{e['seed']}", seed=e["seed"], t=int(t),
                       status_Ub=st["Ub"][1], status_U0=st["U0"][1])
            if not (st["Ub"][0] and st["U0"][0]):
                rec["excluded"] = "endpoint solve failed"; samp_rows.append(rec); print(sid, "excluded", st["Ub"][1], st["U0"][1]); continue
            Ub, U0 = st["Ub"][2], st["U0"][2]
            # cost-function validation against the solver's own objective (nominal plan of U0)
            Jchk = cost(roll_model(x, U0, A, B), U0, xg, Q, R, Qe); rec["cost_check_rel"] = abs(Jchk - st["U0"][3]) / abs(st["U0"][3])
            cache = {}

            def ev(s):
                if s not in cache:
                    U = Ub + s * (U0 - Ub); Xt = roll_true(it, x, U)
                    mg, coll, term = margins(Xt, U, cs, rs, xg)
                    Xn = roll_model(x, U, A, B); mn, cn, _ = margins(Xn, U, cs, rs, xg)
                    Xm = roll_model(x, U, A, B, RES_MSE.dA, RES_MSE.dB); mm, cm, _ = margins(Xm, U, cs, rs, xg)
                    cache[s] = dict(U=U, Xt=Xt, mg=mg, coll=coll, term=term, J=cost(Xt, U, xg, Q, R, Qe),
                                    safe=is_safe(mg, coll), nom_safe=is_safe(mn, cn), mse_safe=is_safe(mm, cm),
                                    viol=first_violation(mg, coll))
                return cache[s]

            tr, grid, _ = search(lambda s: ev(s)["safe"])
            rec.update({f"true_{k}": v for k, v in tr.items()})
            rec["Ub_true_safe"] = ev(0.0)["safe"]; rec["Ub_true_violation"] = ev(0.0)["viol"]
            for mname in ("nom", "mse"):
                r_, _, _ = search(lambda s: ev(s)[f"{mname}_safe"])
                rec.update({f"{mname}_{k}": v for k, v in r_.items()})
                rec[f"{mname}_false_safe_grid"] = sum(ev(s)[f"{mname}_safe"] and not ev(s)["safe"] for s in grid)
                rec[f"{mname}_false_unsafe_grid"] = sum((not ev(s)[f"{mname}_safe"]) and ev(s)["safe"] for s in grid)
            if tr.get("anchor_safe") and not tr["censored"]:
                rec["true_active"] = ev(tr["hi"])["viol"]
            # next-solve feasibility (nom+b from the true state after the first input), grid points only
            for s in grid:
                q = ev(s)
                if "next_status" not in q:
                    x1 = q["Xt"][1]; Xp = roll_model(x, q["U"], A, B)
                    pt2 = c0["sim"].get_shifted_trajectory(Xp.T)[:, :m]
                    cs2, rs2 = c0["sim"].get_corridor_balls(pt2, c0["path"], c0["radii"], c0["p_g"])
                    xg2, _ = compute_goal_state(cs2, rs2, c0["pt"], c0["p_g"], return_index=True)
                    cb["cont"].set_parameter_values(x1, cs2.T, rs2, xg2); ok = cb["cont"].solve()
                    q["next_status"] = getattr(cb["cont"], "status", "exception") if ok else f"FAIL:{getattr(cb['cont'], 'status', 'exception')}"
            # first-order (true information) prediction from the anchor
            if tr.get("anchor_safe"):
                a0, a1 = ev(0.0), ev(DELTA)
                best, name = np.inf, None
                for ty in TYPES:
                    m0, d_ = a0["mg"][ty].ravel(), (a1["mg"][ty].ravel() - a0["mg"][ty].ravel()) / DELTA
                    for j in np.where(d_ < -DZERO)[0]:
                        sx = (m0[j] + TOL) / (-d_[j])
                        if 0 < sx <= 1 and sx < best: best, name = sx, f"{ty}#{j}"
                rec["fo_s"] = best if np.isfinite(best) else None; rec["fo_active"] = name
                rec["fo_censored"] = not np.isfinite(best)
            # decision value: author cost on true rollouts, feasible candidates among ALL evaluated points
            feas = {s: q for s, q in cache.items() if q["safe"]}
            if rec["Ub_true_safe"] and feas:
                sB = min(feas, key=lambda s: feas[s]["J"])
                Jb, JB = cache[0.0]["J"], feas[sB]["J"]
                rec.update(J_Ub=Jb, J_best=JB, s_best=sB, improvement=(Jb - JB) / Jb,
                           goal_dist_Ub=cache[0.0]["term"]["goal_dist"], goal_dist_best=feas[sB]["term"]["goal_dist"],
                           term_vmax_Ub=cache[0.0]["term"]["term_vmax"], term_vmax_best=feas[sB]["term"]["term_vmax"])
            for s, q in sorted(cache.items()):
                row = dict(sample=sid, s=s, on_grid=s in grid, true_safe=q["safe"], nom_safe=q["nom_safe"], mse_safe=q["mse_safe"],
                           violation=q["viol"], collisions=int(q["coll"].sum()), J_true=q["J"], next_status=q.get("next_status", ""),
                           **{f"min_{ty}": float(q["mg"][ty].min()) for ty in TYPES}, **q["term"])
                cand_rows.append(row)
            np.savez(f"{OUT}/samples/{sid}.npz", s=np.array(sorted(cache)),
                     **{f"true_{ty}": np.array([cache[s]["mg"][ty] for s in sorted(cache)]) for ty in TYPES},
                     coll=np.array([cache[s]["coll"] for s in sorted(cache)]), x=x, cs=cs, rs=rs, xg=xg, Ub=Ub, U0=U0)
            samp_rows.append(rec)
            print(f"{sid}: Ub_safe {rec['Ub_true_safe']} true {tr} | mse {rec.get('mse_lo')}-{rec.get('mse_hi')} "
                  f"fo {rec.get('fo_s')} | impr {rec.get('improvement')}")
    write(cand_rows, samp_rows, eps, cfg, time.time() - t0)


def write(cand_rows, samp_rows, eps, cfg, wall):
    def dump(p, rows):
        keys = []
        for r in rows:
            for k in r:
                if k not in keys: keys.append(k)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
    dump(f"{OUT}/candidates.csv", cand_rows); dump(f"{OUT}/samples.csv", samp_rows)
    width = 1 / NGRID / 2 ** NREF
    valid = [r for r in samp_rows if r.get("true_anchor_safe") and r.get("true_censored") is False]
    def mid(r, p): return None if r.get(f"{p}_censored") in (None, True) or not r.get(f"{p}_anchor_safe") else 0.5 * (r[f"{p}_lo"] + r[f"{p}_hi"])
    errs = []
    for r in valid:
        mt = 0.5 * (r["true_lo"] + r["true_hi"])
        if not r.get("mse_anchor_safe"): errs.append(("mse_anchor_unsafe", np.nan)); continue
        mm = mid(r, "mse"); e = (1.0 - r["true_hi"]) if mm is None else abs(mm - mt)   # censored model: lower bound
        errs.append(("lower_bound" if mm is None else "exact", e))
    big = [e for k, e in errs if np.isfinite(e) and e > 0.05 and e > 2 * width]
    fo = [abs(r["fo_s"] - 0.5 * (r["true_lo"] + r["true_hi"])) if r.get("fo_s") is not None else (1.0 - r["true_hi"]) for r in valid]
    ep_rows = []
    for e in eps:
        key = f"{e['source']}_{e['seed']}"; rs_ = [r for r in samp_rows if r["episode"] == key]
        imp = [r["improvement"] for r in rs_ if "improvement" in r]
        ep_rows.append(dict(episode=key, samples=len(rs_), excluded=sum("excluded" in r for r in rs_),
                            Ub_unsafe=sum(r.get("Ub_true_safe") is False for r in rs_),
                            uncensored=sum(r in valid for r in rs_), censored=sum(r.get("true_censored") is True for r in rs_),
                            resafe=sum(bool(r.get("true_resafe")) for r in rs_),
                            improvement_median=float(np.median(imp)) if imp else None, n_improvement=len(imp)))
    dump(f"{OUT}/episodes.csv", ep_rows)
    n_ep_valid = len({r["episode"] for r in valid})
    enough = len(valid) >= 20 and n_ep_valid >= 4
    ep_imp = [r["improvement_median"] for r in ep_rows if r["improvement_median"] is not None]
    dec_ok = len(ep_imp) >= 5
    v = dict(config_hash=cfg["hash"], wall_s=wall,
             sample_sufficiency=dict(valid_uncensored=len(valid), episodes=n_ep_valid, status="OK" if enough else "INSUFFICIENT"),
             boundary_error=dict(n=len(errs), n_exact=sum(k == "exact" for k, _ in errs), n_lower_bound=sum(k == "lower_bound" for k, _ in errs),
                                 n_mse_anchor_unsafe=sum(k == "mse_anchor_unsafe" for k, _ in errs),
                                 frac_over_5pct=(len(big) / len(errs)) if errs else None,
                                 signal=(len(big) / len(errs) >= 0.20) if (errs and enough) else "INSUFFICIENT",
                                 refinement_width=width),
             decision_value=dict(episode_medians=ep_imp,
                                 median_of_episode_medians=float(np.median(ep_imp)) if ep_imp else None,
                                 episodes_improved=sum(x > 0 for x in ep_imp),
                                 signal=(float(np.median(ep_imp)) >= 0.05 and sum(x > 0 for x in ep_imp) >= 4) if dec_ok else "INSUFFICIENT_EPISODES",
                                 note="diagnostic reference inside the evaluated candidate set; not deployable, not a closed-loop bound"),
             first_order=dict(n=len(fo), frac_within_5pct=(float(np.mean(np.array(fo) <= 0.05)) if fo else None),
                              local_response_sufficient=(float(np.mean(np.array(fo) <= 0.05)) >= FO_FRAC) if fo else "INSUFFICIENT"),
             scope="finite-horizon path-constraint boundary under a frozen corridor; NOT the full MPC feasible set nor recursive feasibility")
    bs, ds, fs = v["boundary_error"]["signal"], v["decision_value"]["signal"], v["first_order"]["local_response_sufficient"]
    v["recommend_small_training"] = (bs is True and ds is True and fs is False)
    json.dump(v, open(f"{OUT}/verdict.json", "w"), indent=1, default=str)
    open(f"{OUT}/done.flag", "w").write(time.ctime())
    print(json.dumps(v, indent=1, default=str))
    print(json.dumps(ep_rows, indent=1))


if __name__ == "__main__":
    main()
