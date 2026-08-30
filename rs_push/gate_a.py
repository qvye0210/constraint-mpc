#!/usr/bin/env python3
"""GATE A (pre-registered): can replanning actually rescue model error on the
pushing task, or is the error a persistent bias replanning cannot fix?

Fixed-period replanning ONLY -- no risk logic, no ensembles, no learned
triggers. Same frozen model, same MPPI settings, same paired episodes for
every period.

    periods {1, 2, 4, 8, H}; H = maximal hold (full plan executed, then resolve)

PASS (written before running):
  * paired episode-violation ratio: every-step (k=1) <= 1/3 of maximal-hold,
    with bootstrap 95% CI upper bound < 0.5;
  * task success rate of k=1 not lower than maximal-hold by more than 10 pts;
  * LOW-BASE-RATE RULE: if maximal-hold episode violation rate < 10%, the
    verdict is INSUFFICIENT EVENTS (not pass/fail) -> rerun --pilot to find a
    friction/mass/zone range where it sits in 20-60%, then rerun. Pilot
    episodes never enter the verdict.

    python gate_a.py --pilot            # difficulty scan, ~10 min
    python gate_a.py --quick            # 12 paired episodes
    python gate_a.py                    # 30 paired episodes (the one that counts)
    python gate_a.py --selftest         # unit tests, no robosuite needed
"""
import argparse, json, os, pickle
import numpy as np


# ---------------- unit tests (run without robosuite) -----------------------
def selftest():
    from rspush.env import clearance, R_OBJECT
    z = np.zeros(2)
    assert clearance(z, z, 0.05) < 0                              # center: violation
    b = np.array([0.05 + R_OBJECT, 0.0])
    assert abs(clearance(b, z, 0.05)) < 1e-12                     # boundary
    assert clearance(b + [0.01, 0], z, 0.05) > 0                  # outside
    assert clearance(b - [0.01, 0], z, 0.05) < 0                  # inside
    seq = [0.02, -0.01, 0.03]
    assert min(seq) == -0.01                                      # worst step = min
    # period counting: T=24, period=4 -> 6 solves; period=H(=12) -> 2
    for period, want in ((4, 6), (12, 2), (1, 24)):
        solves = 0; age = 10 ** 9; plan_len = 12
        for t in range(24):
            if age >= period or age >= plan_len:
                solves += 1; age = 0
            age += 1
        assert solves == want, (period, solves)
    # off-by-one: prediction[0] must pair with the state AFTER plan[0]
    s = 0.0; plan = [1.0, 2.0]; pred = [s + plan[0], s + plan[0] + plan[1]]
    s_after = s + plan[0]
    assert pred[0] == s_after
    print("selftest: all passed")


def run(a):
    import torch
    from rspush.env import make_env, Push, clearance, R_OBJECT, EEF_VMAX, DT
    from rspush.model import OneStep
    from rspush.planner import MPPI

    model = OneStep()
    model.load_state_dict(torch.load(f"{a.ckpt}/onestep.pt"))
    model.eval()

    env = make_env(); p = Push(env, seed=a.seed)
    rng = np.random.default_rng(a.seed + 777)

    def sample_spec(i):
        obj = np.array([rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06)])
        ang = rng.uniform(-np.pi, np.pi)
        goal = obj + 0.24 * np.array([np.cos(ang), np.sin(ang)])
        mid = 0.5 * (obj + goal)
        perp = np.array([-(goal - obj)[1], (goal - obj)[0]])
        perp /= np.linalg.norm(perp)
        zone = mid + perp * rng.uniform(0.035, 0.065) * rng.choice([-1, 1])
        return dict(episode_id=i,
                    friction=float(rng.uniform(*a.friction)),
                    mass=float(rng.uniform(*a.mass)),
                    obj_xy=obj, obj_yaw=float(rng.uniform(-np.pi, np.pi)),
                    goal_xy=goal, zone_xy=zone, r_zone=a.r_zone)

    H = a.H
    periods = [1, 2, 4, 8, H]
    rows = []
    n_ep = a.episodes
    for i in range(n_ep):
        spec = sample_spec(i)
        for period in periods:
            if not p.apply_spec(spec):
                rows.append(dict(episode=i, period=period, setup_fail=True))
                continue
            planner = MPPI(model, H=H)
            plan, age, solve_idx = None, 10 ** 9, 0
            viol = False; max_pen = 0.0; first_v = -1; max_step = 0.0
            solver_fail = 0
            for t in range(a.T):
                if plan is None or age >= period or age >= H:
                    try:
                        plan = planner.solve(p.eef()[:2], p.obj_pose(),
                                             spec["goal_xy"], spec["zone_xy"],
                                             spec["r_zone"], i, solve_idx,
                                             u_init=None if plan is None else
                                             np.vstack([plan[age:], np.zeros((min(age, H), 2))]))
                    except Exception:
                        solver_fail += 1
                        plan = np.zeros((H, 2))
                    solve_idx += 1; age = 0
                r = p.step_eef_vel(plan[age]); age += 1
                rho = clearance(r["obj"][:2], spec["zone_xy"], spec["r_zone"])
                max_step = max(max_step, r["obj_step"])
                if rho < 0:
                    if not viol:
                        first_v = t
                    viol = True; max_pen = max(max_pen, -rho)
            goal_err = float(np.linalg.norm(p.obj_pose()[:2] - spec["goal_xy"]))
            rows.append(dict(episode=i, period=period, violation=bool(viol),
                             max_penetration=max_pen, first_violation=first_v,
                             goal_err=goal_err, success=bool(goal_err < a.succ_tol),
                             solves=planner.n_solves,
                             solve_wallclock=planner.solve_time,
                             solver_fail=solver_fail,
                             max_obj_step=max_step,
                             friction=spec["friction"], mass=spec["mass"]))
        if (i + 1) % 5 == 0:
            print(f"  episode {i + 1}/{n_ep} done", flush=True)
    env.close()

    ok = [r for r in rows if "setup_fail" not in r]
    tunnel_bound = max(r["max_obj_step"] for r in ok)
    print(f"\nanti-tunnel check: max object displacement/step = "
          f"{tunnel_bound:.4f} m vs zone radius {a.r_zone} "
          f"({'OK' if tunnel_bound < a.r_zone / 2 else 'BOUND BROKEN — substep detection needed'})")

    print(f"\n{'period':>7}{'viol%':>8}{'max_pen':>9}{'succ%':>7}"
          f"{'goal_err':>9}{'solves':>7}{'wall(s)':>9}")
    stats = {}
    for period in periods:
        sel = [r for r in ok if r["period"] == period]
        v = np.mean([r["violation"] for r in sel])
        stats[period] = dict(
            viol=float(v),
            pen=float(np.mean([r["max_penetration"] for r in sel])),
            succ=float(np.mean([r["success"] for r in sel])),
            gerr=float(np.mean([r["goal_err"] for r in sel])),
            solves=float(np.mean([r["solves"] for r in sel])),
            wall=float(np.mean([r["solve_wallclock"] for r in sel])))
        s = stats[period]
        print(f"{period:>7}{s['viol']:>8.1%}{s['pen']:>9.4f}{s['succ']:>7.0%}"
              f"{s['gerr']:>9.3f}{s['solves']:>7.0f}{s['wall']:>9.2f}")

    base = stats[H]["viol"]; best = stats[1]["viol"]
    # paired bootstrap on the violation ratio
    vb = np.array([[r["violation"] for r in ok if r["period"] == pd]
                   for pd in (1, H)])  # (2, n_ep)
    n = vb.shape[1]; ratios = []
    brng = np.random.default_rng(0)
    for _ in range(2000):
        j = brng.integers(0, n, n)
        b0, b1 = vb[0, j].mean(), vb[1, j].mean()
        ratios.append(b0 / b1 if b1 > 0 else np.inf)
    ci = (float(np.quantile(ratios, .025)), float(np.quantile(ratios, .975)))

    if a.pilot:
        tgt = "in 20-60% ✓" if 0.2 <= base <= 0.6 else \
            ("too easy — raise friction range upper end / zone offset closer"
             if base < 0.2 else "too hard — soften")
        print(f"\nPILOT: maximal-hold violation {base:.0%} -> {tgt}")
        verdict = dict(pilot=True, base_rate=base)
    elif base < 0.10:
        print(f"\n>>> INSUFFICIENT EVENTS: maximal-hold violation {base:.0%} < 10%."
              " Not pass/fail. Rerun --pilot and retune difficulty.")
        verdict = dict(insufficient=True, base_rate=base)
    else:
        ratio = best / base if base > 0 else np.inf
        succ_drop = stats[H]["succ"] - stats[1]["succ"]
        passed = bool(ratio <= 1 / 3 and ci[1] < 0.5 and succ_drop <= 0.10)
        print(f"\nviolation ratio k=1 / maximal-hold = {ratio:.2f} "
              f"(need <= 0.33), bootstrap CI [{ci[0]:.2f}, {ci[1]:.2f}] "
              f"(upper < 0.5), success drop {succ_drop:+.0%} (<= 10 pts)")
        print(">>> " + ("GATE A PASS: replanning genuinely rescues model error "
                        "on this task. Proceed to Gate B (counterfactual "
                        "trigger-signal test)."
                        if passed else
                        "GATE A FAIL: replanning does not rescue the error -> "
                        "persistent-bias regime. Pre-registered branch: retrain "
                        "the model WITH short history (Gate A'), rerun once. If "
                        "that also fails, the hold-length method premise is dead "
                        "on this task; do not add knobs."))
        verdict = dict(ratio=float(ratio), ci=ci, succ_drop=float(succ_drop),
                       passed=passed, base_rate=float(base))
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(stats={str(k): v for k, v in stats.items()},
                   verdict=verdict, rows=rows, config=vars(a)),
              open(f"{a.out}/verdict.json", "w"), indent=2, default=float)
    print(f"wrote {a.out}/verdict.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--H", type=int, default=12)
    ap.add_argument("--T", type=int, default=90)
    ap.add_argument("--r-zone", type=float, default=0.05)
    ap.add_argument("--succ-tol", type=float, default=0.05)
    ap.add_argument("--friction", type=float, nargs=2, default=(0.3, 1.2))
    ap.add_argument("--mass", type=float, nargs=2, default=(0.05, 0.6))
    ap.add_argument("--ckpt", default="ckpt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/gate_a")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        if a.episodes is None:
            a.episodes = 8 if a.pilot else (12 if a.quick else 30)
        run(a)
