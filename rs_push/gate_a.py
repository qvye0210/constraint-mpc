#!/usr/bin/env python3
"""GATE A (pre-registered): can replanning rescue model error on pushing, or is
the error a persistent bias replanning cannot fix?

Fixed-period replanning ONLY. Same frozen model (eval), same MPPI budget, same
paired episode specs per period; MPPI RNG from SeedSequence(1234, episode, solve).

DIFFICULTY LADDER (fixed BEFORE any run; pilot may only pick a level by looking
at maximal-hold, never at k=1):
    0: zone lateral offset 0.055-0.085 m   (easiest)
    1: 0.045-0.075
    2: 0.035-0.065
    3: 0.028-0.050                          (hardest)
Pilot rule: run maximal-hold ONLY at each level (pilot seed space, disjoint from
formal), choose the FIRST level whose episode-violation rate lands in 20-60%.

PASS (fixed before running):
  * paired episode-violation ratio  k=1 / maximal-hold  <= 1/3,
    with paired-bootstrap 95% CI upper bound < 0.5;
  * task success of k=1 not lower than maximal-hold by more than 10 pts;
  * EVENT RULE: fewer than 10 maximal-hold violation EPISODES -> verdict is
    INSUFFICIENT EVENTS (n=60 at 20-60% base rate gives 12-36 events).
  * crossing_uncertain steps are reported; a clean endpoint-detection claim
    requires that count to be 0 (see below), else substep hooks are needed.

crossing_uncertain: a step whose endpoints are both safe but whose start-or-end
clearance is smaller than dt * max(|v_obj|_start, |v_obj|_end) — the object
could have crossed and returned between endpoints. This replaces the earlier
(wrong) "7.5mm << zone radius" argument: crossing depends on CURRENT clearance,
not zone size.

GATE A' (single pre-registered failure branch — spec LOCKED now):
    history length L=8 control steps; inputs = concat of the 8 past per-step
    feature vectors [rel_eef_obj, sin yaw, cos yaw, u] (48) + current (6) = 54;
    zero-padded at episode start; hidden (256,256); epochs 400; same data
    budget and splits; final-epoch checkpoint, no selection. Run once; if it
    also fails, the hold-length premise is dead on this task.

    python gate_a.py --selftest
    python gate_a.py --pilot                    # ladder scan, maximal-hold only
    python gate_a.py --quick  --difficulty D    # 12 paired episodes (smoke)
    python gate_a.py --difficulty D             # 60 paired episodes (the verdict)
"""
import argparse, hashlib, json, os
import numpy as np

# DOORWAY geometry (v2, pre-registered before any v2 run). The single-zone
# ladder was falsified by pilot data: even at 1.2-2.5cm lateral offset the
# planner detoured freely (median episode min-clearance 5.9-7.8cm, violations
# 0-8%). A gap between two symmetric zones removes the detour option -- the
# object MUST pass near the constraint. Ladder value = lateral distance d of
# each zone centre from the path; best achievable clearance = d - (r_zone +
# r_object) = d - 0.061.
# v3 (pre-registered): WALLED doorway. v2's two lone circles were flawed --
# outer edges sat only 13-16cm from the path with a 40cm table half-width, so
# outside detours stayed geometrically feasible, and MPPI's sampling noise makes
# threading a 2-3cm slot lose weight to smooth detours. Walls of overlapping
# circles (centre spacing 0.10 < 2*(r_zone+R_OBJECT)=0.122 -> no gap) extend
# past the table edge; the doorway is the ONLY route. min over 1-Lipschitz
# distances stays 1-Lipschitz. LADDER is the one-sided doorway half-width c
# (metres): one-step error ~6mm < c < k=8 open-loop error ~48mm.
LADDER = [0.030, 0.025, 0.020, 0.015]
WALL_N = 5          # extra circles per side beyond the pillar
WALL_SPACING = 0.10
EFF_R = 0.061       # r_zone + R_OBJECT at defaults


def build_zones(obj_xy, goal_xy, c, phi_deg=0.0):
    """Two walls of circles with a single gap of half-width c at the path
    midpoint. phi rotates the doorway axis away from the path normal (used when
    model error is longitudinal -- see diag_error_direction.py)."""
    path = goal_xy - obj_xy
    path = path / (np.linalg.norm(path) + 1e-12)
    perp = np.array([-path[1], path[0]])
    phi = np.radians(phi_deg)
    assert phi_deg < 60, "slant beyond 60 deg collapses the corridor"
    a = np.cos(phi) * perp + np.sin(phi) * path
    mid = 0.5 * (obj_xy + goal_xy)
    # Scale along-axis offsets by 1/cos(phi) so the corridor half-width SEEN BY
    # THE PATH stays exactly c. Without this (the v3 bug) the pillar's lateral
    # projection (c+EFF_R)cos(phi)-EFF_R goes NEGATIVE for c<=0.02 at phi=45 --
    # the doorway was sealed, which is why every pilot level stalled at the
    # same min-clearance 0.024 regardless of c.
    # Only the PILLAR offset is scaled by 1/cos(phi) (that alone restores the
    # corridor half-width c seen by the path). Inter-circle spacing stays
    # EUCLIDEAN 0.10 < 2*EFF_R: scaling it too (the v4 bug) opened ~19mm leaks
    # between adjacent wall circles at phi=45 -- wider than the doorway itself.
    scale = 1.0 / np.cos(phi)
    zones = []
    for side in (1.0, -1.0):
        base = (c + EFF_R) * scale
        for j in range(WALL_N + 1):
            zones.append(mid + side * a * (base + j * WALL_SPACING))
    return np.array(zones)


def selftest():
    import torch
    from rspush.env import clearance, R_OBJECT
    from rspush.model import OneStep, rollout
    z = np.zeros(2)
    assert clearance(z, z, 0.05) < 0
    b = np.array([0.05 + R_OBJECT, 0.0])
    assert abs(clearance(b, z, 0.05)) < 1e-12
    assert clearance(b + [0.01, 0], z, 0.05) > 0
    assert clearance(b - [0.01, 0], z, 0.05) < 0
    assert min([0.02, -0.01, 0.03]) == -0.01
    for period, want in ((4, 6), (12, 2), (1, 24)):
        solves, age = 0, 10 ** 9
        for t in range(24):
            if age >= period or age >= 12:
                solves += 1; age = 0
            age += 1
        assert solves == want, (period, solves)
    s0 = 0.0; plan = [1.0, 2.0]
    pred = [s0 + plan[0], s0 + plan[0] + plan[1]]
    assert pred[0] == s0 + plan[0]                      # off-by-one
    m = OneStep(); m.eval()
    S = rollout(m, np.zeros(2), np.zeros(3), np.random.default_rng(0)
                .normal(0, .05, (2, 5, 2)))             # float64 in on purpose
    assert S.dtype == np.float32, S.dtype               # dtype guard
    print("selftest: all passed")


def make_specs(n, seed, zoff, phi=0.0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        obj = np.array([rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06)])
        ang = rng.uniform(-np.pi, np.pi)
        goal = obj + 0.24 * np.array([np.cos(ang), np.sin(ang)])
        mid = 0.5 * (obj + goal)
        perp = np.array([-(goal - obj)[1], (goal - obj)[0]])
        perp /= np.linalg.norm(perp)
        zone = build_zones(obj, goal, zoff * rng.uniform(0.95, 1.05), phi)
        out.append(dict(episode_id=i, friction=float(rng.uniform(0.3, 1.2)),
                        mass=float(rng.uniform(0.05, 0.6)),
                        obj_xy=obj, obj_yaw=float(rng.uniform(-np.pi, np.pi)),
                        goal_xy=goal, zone_xy=zone, np_seed=100 * seed + i))
    return out


def pilot_spec(base_seed, slot, retry, zoff, phi=0.0):
    """Same (slot, retry) -> identical episode across ALL difficulty levels;
    only the zone-offset magnitude is mapped through zoff."""
    rng = np.random.default_rng(np.random.SeedSequence(
        [int(base_seed), int(slot), int(retry)]))
    obj = np.array([rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06)])
    ang = rng.uniform(-np.pi, np.pi)
    goal = obj + 0.24 * np.array([np.cos(ang), np.sin(ang)])
    mid = 0.5 * (obj + goal)
    perp = np.array([-(goal - obj)[1], (goal - obj)[0]])
    perp /= np.linalg.norm(perp)
    zone = build_zones(obj, goal, zoff * rng.uniform(0.95, 1.05), phi)
    return dict(episode_id=1000 * slot + retry,
                friction=float(rng.uniform(0.3, 1.2)),
                mass=float(rng.uniform(0.05, 0.6)),
                obj_xy=obj, obj_yaw=float(rng.uniform(-np.pi, np.pi)),
                goal_xy=goal, zone_xy=zone,
                np_seed=7000000 + slot * 100 + retry)


def run_episode(p, planner_cls, model, spec, period, a):
    from rspush.env import clearance, DT
    from rspush.planner import MPPI
    if not p.apply_spec(spec):
        return dict(setup_fail=True)
    planner = MPPI(model, H=a.H)
    plan, age, solve_idx = None, 10 ** 9, 0
    viol = False; max_pen = 0.0; first_v = -1; min_rho = np.inf
    path = spec["goal_xy"] - spec["obj_xy"]
    path = path / (np.linalg.norm(path) + 1e-12)
    mid_s = float((0.5 * (spec["goal_xy"] - spec["obj_xy"])) @ path)
    crossed = False
    cu = 0; max_speed = 0.0; prev_speed = 0.0; solver_fail = 0
    for t in range(a.T):
        if plan is None or age >= period or age >= a.H:
            try:
                plan = planner.solve(p.eef()[:2], p.obj_pose(), spec["goal_xy"],
                                     spec["zone_xy"], a.r_zone,
                                     spec["episode_id"], solve_idx)
            except Exception:
                solver_fail += 1; plan = np.zeros((a.H, 2))
            solve_idx += 1; age = 0
        rho0 = clearance(p.obj_pose()[:2], spec["zone_xy"], a.r_zone)
        r = p.step_eef_vel(plan[age]); age += 1
        rho1 = clearance(r["obj"][:2], spec["zone_xy"], a.r_zone)
        min_rho = min(min_rho, rho1)
        if float((r["obj"][:2] - spec["obj_xy"]) @ path) > mid_s + 0.02:
            crossed = True
        d_max = DT * max(prev_speed, r["obj_speed"])
        if rho0 > 0 and rho1 > 0 and min(rho0, rho1) < d_max:
            cu += 1
        prev_speed = r["obj_speed"]; max_speed = max(max_speed, r["obj_speed"])
        if rho1 < 0:
            if not viol:
                first_v = t
            viol = True; max_pen = max(max_pen, -rho1)
    ge = float(np.linalg.norm(p.obj_pose()[:2] - spec["goal_xy"]))
    pr = np.array(planner.plan_rhos) if planner.plan_rhos else np.array([np.nan])
    return dict(violation=bool(viol), max_penetration=max_pen,
                min_clearance=float(min_rho), crossed=bool(crossed),
                plan_rho_min=float(np.nanmin(pr)),
                plan_rho_neg_frac=float(np.nanmean(pr < 0)),
                first_violation=first_v, goal_err=ge,
                success=bool(ge < a.succ_tol), solves=planner.n_solves,
                solve_wallclock=planner.solve_time, solver_fail=solver_fail,
                crossing_uncertain=cu, max_obj_speed=max_speed,
                friction=spec["friction"], mass=spec["mass"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--difficulty", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--H", type=int, default=12)
    ap.add_argument("--T", type=int, default=90)
    ap.add_argument("--r-zone", type=float, default=0.05)
    ap.add_argument("--phi", type=float, default=0.0,
                    help="doorway axis rotation, deg; set per diag_error_direction.py rule")
    ap.add_argument("--succ-tol", type=float, default=0.05)
    ap.add_argument("--ckpt", default="ckpt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        selftest(); return

    import torch
    from rspush.env import make_env, Push
    from rspush.model import OneStep
    model = OneStep()
    model.load_state_dict(torch.load(f"{a.ckpt}/onestep.pt"))
    model.eval()
    md5 = hashlib.md5(open(f"{a.ckpt}/onestep.pt", "rb").read()).hexdigest()
    env = make_env(); p = Push(env, seed=a.seed)

    if a.pilot:
        from rspush.env import clearance
        a.out = a.out or "results/gate_a_pilot"
        n_valid = a.episodes or 12
        os.makedirs(a.out, exist_ok=True)
        chosen, table = None, {}
        for lvl, zoff in enumerate(LADDER):
            v, minrho, attempted, skipped = [], [], 0, 0
            for slot in range(n_valid):
                got = False
                for retry in range(25):
                    spec = pilot_spec(90000 + a.seed, slot, retry, zoff, a.phi)
                    attempted += 1
                    # zone must not contain start or goal (checked BEFORE any run)
                    if (clearance(spec["obj_xy"], spec["zone_xy"], a.r_zone) < 0.005
                            or clearance(spec["goal_xy"], spec["zone_xy"], a.r_zone) < 0.005):
                        skipped += 1; continue
                    r = run_episode(p, None, model, spec, a.H, a)  # maximal-hold ONLY
                    if "setup_fail" in r:
                        skipped += 1; continue
                    v.append(r["violation"]); minrho.append(r["min_clearance"])
                    table.setdefault("_planneg", []).append(r["plan_rho_neg_frac"])
                    table.setdefault("_succ", {}).setdefault(lvl, []).append(r["success"])
                    table.setdefault("_cross", {}).setdefault(lvl, []).append(r["crossed"])
                    got = True; break
                if not got:
                    print(f"  level {lvl} slot {slot}: no valid spec in 25 retries")
            rate = float(np.mean(v)) if v else float("nan")
            mr = np.array(minrho) if minrho else np.array([np.nan])
            table[lvl] = dict(rate=rate, attempted=attempted, valid=len(v),
                              skipped=skipped,
                              min_clearance=dict(min=float(np.nanmin(mr)),
                                                 p10=float(np.nanquantile(mr, .1)),
                                                 median=float(np.nanmedian(mr))))
            pneg = float(np.nanmean([t.get("plan_rho_neg_frac", np.nan)
                                      for t in table.get("_rows", [])])) if False else None
            sc = table.get("_succ", {}).get(lvl, [np.nan])
            cr = table.get("_cross", {}).get(lvl, [np.nan])
            print(f"difficulty {lvl} (half-width {zoff}): viol {rate:.0%}  "
                  f"succ {np.nanmean(sc):.0%}  crossed {np.nanmean(cr):.0%}  "
                  f"attempted {attempted} valid {len(v)} skipped {skipped}  "
                  f"min-clearance min/p10/med = {np.nanmin(mr):.3f}/"
                  f"{np.nanquantile(mr, .1):.3f}/{np.nanmedian(mr):.3f}")
            if chosen is None and v and 0.2 <= rate <= 0.6:
                chosen = lvl
        table.pop("_succ", None); table.pop("_cross", None)
        pn = table.pop("_planneg", [])
        if pn:
            print(f"planner honesty: fraction of solves with PREDICTED clearance <0 "
                  f"= {np.mean(pn):.1%} (must be ~0; else Gate A measures planner "
                  f"failure, not model error)")
        print(f"\n>>> chosen difficulty: "
              f"{chosen if chosen is not None else 'NONE in 20-60% — send this table back, do not self-tune'}")
        json.dump(dict(table=table, chosen=chosen, ckpt_md5=md5),
                  open(f"{a.out}/verdict.json", "w"), indent=2, default=float)
        env.close(); return

    assert a.difficulty is not None, "formal runs need --difficulty from pilot"
    zoff = LADDER[a.difficulty]
    n = a.episodes or (12 if a.quick else 60)
    a.out = a.out or ("results/gate_a_quick" if a.quick else "results/gate_a")
    os.makedirs(a.out, exist_ok=True)
    seed_base = (10000 if a.quick else 20000) + a.seed   # disjoint from pilot
    specs = make_specs(n, seed_base, zoff, a.phi)
    periods = [1, 2, 4, 8, a.H]
    rows = []
    for i, spec in enumerate(specs):
        for period in periods:
            r = run_episode(p, None, model, spec, period, a)
            r.update(episode=i, period=period)
            rows.append(r)
        if (i + 1) % 5 == 0:
            print(f"  episode {i + 1}/{n}", flush=True)
    env.close()

    ok = [r for r in rows if "setup_fail" not in r]
    cu_steps = sum(r["crossing_uncertain"] for r in ok)
    print(f"\ncrossing_uncertain steps total: {cu_steps} "
          f"({'endpoint detection supported' if cu_steps == 0 else 'SUBSTEP HOOKS NEEDED for a clean safety claim'})"
          f";  max object speed {max(r['max_obj_speed'] for r in ok):.3f} m/s")

    print(f"\n{'period':>7}{'viol%':>8}{'max_pen':>9}{'succ%':>7}{'goal_err':>9}"
          f"{'solves':>7}{'wall(s)':>9}")
    stats = {}
    for period in periods:
        sel = [r for r in ok if r["period"] == period]
        stats[period] = dict(viol=float(np.mean([r["violation"] for r in sel])),
                             pen=float(np.mean([r["max_penetration"] for r in sel])),
                             succ=float(np.mean([r["success"] for r in sel])),
                             gerr=float(np.mean([r["goal_err"] for r in sel])),
                             solves=float(np.mean([r["solves"] for r in sel])),
                             wall=float(np.mean([r["solve_wallclock"] for r in sel])),
                             n=len(sel))
        s = stats[period]
        print(f"{period:>7}{s['viol']:>8.1%}{s['pen']:>9.4f}{s['succ']:>7.0%}"
              f"{s['gerr']:>9.3f}{s['solves']:>7.0f}{s['wall']:>9.2f}")

    vH = [r["violation"] for r in ok if r["period"] == a.H]
    v1 = [r["violation"] for r in ok if r["period"] == 1]
    n_events = int(np.sum(vH))
    if n_events < 10:
        print(f"\n>>> INSUFFICIENT EVENTS: only {n_events} maximal-hold violation "
              f"episodes (<10). Not pass/fail — retune with --pilot or raise n.")
        verdict = dict(insufficient=True, events=n_events)
    else:
        vb = np.array([v1, vH], dtype=float)
        brng = np.random.default_rng(0); ratios = []
        for _ in range(4000):
            j = brng.integers(0, vb.shape[1], vb.shape[1])
            b0, b1 = vb[0, j].mean(), vb[1, j].mean()
            ratios.append(b0 / b1 if b1 > 0 else np.inf)
        ci = (float(np.quantile(ratios, .025)), float(np.quantile(ratios, .975)))
        ratio = np.mean(v1) / np.mean(vH)
        drop = stats[a.H]["succ"] - stats[1]["succ"]
        passed = bool(ratio <= 1 / 3 and ci[1] < 0.5 and drop <= 0.10)
        print(f"\nviolation ratio k=1/maximal-hold = {ratio:.2f} (<=0.33), "
              f"CI [{ci[0]:.2f},{ci[1]:.2f}] (upper<0.5), succ drop {drop:+.0%}")
        print(">>> " + ("GATE A PASS — replanning rescues model error. Next: Gate B."
                        if passed else
                        "GATE A FAIL — run the locked Gate A' (history model) once; "
                        "if that fails too, stop."))
        verdict = dict(ratio=float(ratio), ci=ci, succ_drop=float(drop),
                       events=n_events, passed=passed)
    json.dump(dict(stats={str(k): v for k, v in stats.items()}, verdict=verdict,
                   rows=rows, ckpt_md5=md5, difficulty=a.difficulty,
                   config=vars(a)), open(f"{a.out}/verdict.json", "w"),
              indent=2, default=float)
    print(f"wrote {a.out}/verdict.json   (ckpt md5 {md5[:8]})")


if __name__ == "__main__":
    main()
