"""
day16_magnitude.py — magnitude measurement only: does a physically motivated, unknown-but-fixed mismatch outside the
nominal model class bring prediction error to the scale of the remaining margin? Repo root, PYTHONPATH=. ; env robmpc.
Task, controller, reference timing, b and geometry = day13 (frozen). No method attached, no b change.
Output: results/day16_magnitude/

LEVELS FIXED BEFORE RUNNING (physical reasoning, not results):
  base mismatch: seed-200 link-mass ratios (the authors' 2% draw), kept in every arm
  payload: point mass on the flange at the tool midpoint (0.05 m along flange z);
           levels = {0.2, 0.5, 1, 2} x wrist link mass (0.1 kg)  -> 0.02, 0.05, 0.10, 0.20 kg  (a real 0.10 m tool)
  damping: viscous joint damping D = c I on the PLANT only (authors' DynPinWrapper damping hook, nominal has none);
           levels c = {0.01, 0.03, 0.1, 0.3} N m s/rad (1.5 decades around 0.1, wrist-inertia time constant ~1 s at 0.1)
Reported per level: task status/PASS, executed tool rho min, violations, open-loop 20-step danger-side rho error q95
and its ratio to the remaining margin (executed rho median), executed |v| max and the fraction of the buffer b consumed,
next-step / open-loop false-safe counts. Ratio >= 0.1 = "signal possible"; below = none at this level.
"""
import json, os, sys
import numpy as np
import pinocchio as pin
sys.path.insert(0, ".")
import day13_reftrack as d13
from day12_tooltask import M, ps, b
from aux import TimeStepIntegratorContinuous, DynPinWrapper, DeltaDynPin
from day7_fixed_margin import make_ctx
from day8_reference_arms import find_dyns
from exec_interface import get_ps, ZeroResidual

OUT = "results/day16_magnitude"; os.makedirs(OUT, exist_ok=True); d13.OUT = OUT
PAYLOAD = [0.0, 0.02, 0.05, 0.10, 0.20]; DAMP = [0.0, 0.01, 0.03, 0.1, 0.3]
V_LIM = float(ps.v_amp_nom); BASE_SEED = 200


class Rec:
    def __init__(self, it): self.it, self.vmax = it, 0.0
    def solve_time_step(self, x, u):
        xn = np.array(self.it.solve_time_step(x, u), float); self.vmax = max(self.vmax, float(np.abs(xn[M:]).max())); return xn


def base_ratios():
    c = make_ctx(ps, BASE_SEED, ZeroResidual(), 0.0); _, de = find_dyns(c["it"])
    return np.array([i.mass for i in de.sampled.model.inertias]) / np.array([i.mass for i in de.nom.model.inertias])


def plant(ratios, payload=0.0, damp=0.0):
    p = get_ps(); p.get_err_dyn_from_ratios(ratios)                       # authors' 2% draw
    model = p.model_gt
    if payload > 0:
        fid = model.getFrameId("link_6"); jid = getattr(model.frames[fid], "parentJoint", None) or model.frames[fid].parent
        model.inertias[jid] = model.inertias[jid] + pin.Inertia(payload, np.array([0.0, 0.0, 0.05]), np.zeros((3, 3)))
    sampled = DynPinWrapper(model, damping=(damp * np.eye(M) if damp > 0 else None))
    de = DeltaDynPin(nom=p.dyn_nom, sampled=sampled, ignore_gravity=p.ignore_gravity)
    it = TimeStepIntegratorContinuous(dt=ps.dt); it.set_dyns(p.dyn_nom, de)
    return it


def main():
    T = json.load(open("results/day13_reftrack/zero.json"))["T"]; ratios = base_ratios()
    rows = []
    for kind, levels in (("payload_kg", PAYLOAD), ("damping_Nms", DAMP)):
        for lv in levels:
            rec = Rec(plant(ratios, payload=lv if kind == "payload_kg" else 0.0, damp=lv if kind == "damping_Nms" else 0.0))
            tag = f"{kind}_{lv:g}"
            r = d13.episode(rec, tag, T)
            margin = r["rho_exec_median"]
            rows.append(dict(kind=kind, level=lv, status=r["status"], task_PASS=r["task_PASS"], steps=r["steps"],
                             lat_max_mm=1e3 * r["lat_max"], ang_max_deg=r["ang_max_deg"],
                             rho_exec_min_mm=1e3 * r["rho_exec_min"], violations=r["exec_violation_steps"],
                             dyn_prefix_q95_mm=1e3 * (r["dyn_prefix_pos_q95"] or 0), err_over_margin=(r["dyn_prefix_pos_q95"] or 0) / margin,
                             v_exec_max=r["v_exec_max"], buffer_used=max(0.0, (r["v_exec_max"] or 0) - (V_LIM - b)) / b,
                             fs_next=r["fs_next_step"], fs_window=r["fs_open_loop_window"], tool_active=r["steps_tool_dual_any_active"]))
    json.dump(rows, open(f"{OUT}/scan.json", "w"), indent=1, default=float)
    print(f"\n{'kind':12s} {'level':>6s} {'status':>10s} {'PASS':>5s} {'lat mm':>7s} {'ang':>6s} {'rho_min mm':>10s} {'viol':>4s} "
          f"{'err q95 mm':>10s} {'err/margin':>10s} {'|v|max':>7s} {'buf used':>8s} {'fsN':>3s} {'fsW':>3s}")
    for r in rows:
        print(f"{r['kind']:12s} {r['level']:6g} {r['status'][:10]:>10s} {str(r['task_PASS']):>5s} {r['lat_max_mm']:7.2f} {r['ang_max_deg']:6.2f} "
              f"{r['rho_exec_min_mm']:10.2f} {r['violations']:4d} {r['dyn_prefix_q95_mm']:10.4f} {r['err_over_margin']:10.4f} "
              f"{r['v_exec_max']:7.3f} {r['buffer_used']:8.3f} {r['fs_next']:3d} {r['fs_window']:3d}")


if __name__ == "__main__":
    main()
