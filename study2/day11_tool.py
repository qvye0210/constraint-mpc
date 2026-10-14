"""
day11_tool.py — "tool-pose-dependent constraint" feasibility check. Repo root, PYTHONPATH=. ; env robmpc.
No training, no MPC change, no SDF retraining, mismatch unchanged. Output: results/day11_tool/

Asset facts (author repo, examples/six_dof/manipulator/demo.urdf + man.py, verified in-script):
  metres / radians; 6 revolute joints (z,y,y,z,y,z); joint 3 at 0.4 m, joint 5 at +0.25 m, joint 6 at +0.025 m.
  flange = frame link_6. Wrist axes 4,5,6 all pass through the link_5 origin (spherical wrist centre).
  existing collision model = meshes of link_2, link_3 only -> NO wrist or tool geometry in the author's check.
  author LIMS (man.py) set wrist ranges to 0 for c-space sampling; URDF limits are +-3.14 -> URDF used for skipping.
Synthetic diagnostic tool (NOT a real tool; no mass/inertia simulated -> not a payload experiment):
  capsule from flange origin along flange-local +z, length 0.10 m, radius 0.01 m.
  rho(q) = min(z_base, z_tip) - z_plane - r_tool - d_safe,  d_safe = 0.01 m.
  plane fixed ONCE from one design configuration (episode 1200, median row) so that rho(q_design) = 0.02 m.
Envelope B (provable containment, derived from the URDF): sphere at the wrist centre with radius
  R = 0.025 + 0.10 + 0.01 = 0.135 m contains the tool for ALL wrist angles; B = z_c - z_plane - R - d_safe.
Fixed before running: TOL = 1e-6 m; switch flag |z_base - z_tip| < 1e-3 m; grad central diff h = 1e-6 rad;
  wrist coverage threshold = 1 deg range per axis; near window = min_k rho(nominal) in [0, 0.02] m.
"""
import csv, json, os, sys
import numpy as np
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from examples.six_dof.manipulator.man import DemoManipulator
from day10_boundary import pick_episodes, roll_model, roll_true, integrator_for, RES_MSE, ps

OUT = "results/day11_tool"; os.makedirs(OUT, exist_ok=True)
L_TOOL, R_TOOL, D_SAFE, RHO_DESIGN = 0.10, 0.01, 0.01, 0.02
R_ENV = 0.025 + L_TOOL + R_TOOL
TOL, SWITCH, HGRAD, WRIST_COV, NEAR = 1e-6, 1e-3, 1e-6, np.deg2rad(1.0), (0.0, 0.02)
PERT, H, M = np.deg2rad(5.0), 20, 6
man = DemoManipulator()
LO, HI = man.model.lowerPositionLimit.copy(), man.model.upperPositionLimit.copy()
from aux import get_linear_double_integrator_discrete_dynamics
A, B = get_linear_double_integrator_discrete_dynamics(M, dt=ps.dt, method="zoh")   # author's nominal model


def fk(q):
    T5, T6 = man.get_link_fk(np.asarray(q, float), links=["link_5", "link_6"])
    pb = T6[:3, 3]; return T5[:3, 3], pb, pb + L_TOOL * T6[:3, 2], T6


ZP = None
def rho_parts(q):
    _, pb, pt, _ = fk(q); zb, zt = pb[2], pt[2]
    return min(zb, zt) - ZP - R_TOOL - D_SAFE, ("base" if zb <= zt else "tip"), abs(zb - zt)
def rho(q): return rho_parts(q)[0]
def env(q): return fk(q)[0][2] - ZP - R_ENV - D_SAFE
def grad(q):
    g = np.zeros(M)
    for j in range(M):
        e = np.zeros(M); e[j] = HGRAD; g[j] = (rho(q + e) - rho(q - e)) / (2 * HGRAD)
    return g


def asset_checks(qd):
    out = {}
    T = man.get_link_fk(np.zeros(M), links=["link_3", "link_5", "link_6"])
    out["q0_link3_z"], out["q0_link5_z"], out["q0_link6_z"] = (float(t[2, 3]) for t in T)
    out["q0_tool_axis"] = T[2][:3, 2].tolist()
    assert np.allclose([out["q0_link3_z"], out["q0_link5_z"], out["q0_link6_z"]], [0.4, 0.65, 0.675], atol=1e-9), out
    moved = []
    for j in range(M):
        e = np.zeros(M); e[j] = 0.1
        moved.append(float(np.abs(fk(qd + e)[3] - fk(qd)[3]).max()))
    out["flange_pose_change_per_joint"] = moved; assert min(moved) > 1e-6, "FK ignores a joint"
    e6 = np.zeros(M); e6[5] = 0.3
    out["rho_invariant_to_joint6"] = abs(rho(qd + e6) - rho(qd)) < 1e-12   # tool on the j6 axis: symmetric by construction
    rng = np.random.default_rng(0); worst_c, worst_b = 0.0, -np.inf
    for _ in range(2000):
        q = qd.copy(); q[3:] = rng.uniform(-np.pi, np.pi, 3)
        c, pb, pt, _ = fk(q)
        worst_c = max(worst_c, np.linalg.norm(pb - c), np.linalg.norm(pt - c))
        worst_b = max(worst_b, env(q) - rho(q))
    out["envelope_max_endpoint_dist"] = float(worst_c); out["envelope_minus_rho_max"] = float(worst_b)
    assert worst_c <= R_ENV - R_TOOL + 1e-12 and worst_b <= 1e-12, "envelope containment failed"
    out["wrist_centre_invariant_to_j4j5j6"] = True
    return out


def figures(qd):
    pts = np.array([t[:3, 3] for t in man.get_link_fk(qd, links=[f"link_{i}" for i in range(7)])])
    c, pb, pt, _ = fk(qd)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(pts[:, 0], pts[:, 2], "o-", c="0.3", label="link frames 0-6")
    ax.plot([pb[0], pt[0]], [pb[2], pt[2]], "-", lw=6, c="tab:orange", alpha=.7, label=f"tool {L_TOOL} m (capsule r={R_TOOL})")
    ax.add_patch(plt.Circle((c[0], c[2]), R_ENV, fill=False, ls="--", label=f"wrist envelope R={R_ENV}"))
    ax.axhline(ZP, c="tab:blue", label=f"plane z={ZP:.4f} (safe side +z)")
    ax.axhline(ZP + R_TOOL + D_SAFE, c="tab:blue", ls=":", label="plane + r + d_safe")
    ax.set_aspect("equal"); ax.set_xlabel("world x [m]"); ax.set_ylabel("world z [m]")
    ax.set_title(f"design config: rho={rho(qd):.4f} m (target {RHO_DESIGN})"); ax.legend(fontsize=7)
    fig.savefig(f"{OUT}/fig1_design_geometry_xz.png", dpi=150); plt.close(fig)
    ang = np.deg2rad(np.linspace(-30, 30, 241))
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, col in zip((3, 4, 5), ("tab:red", "tab:green", "tab:purple")):
        vals = [];  ends = []
        for a in ang:
            q = qd.copy(); q[j] += a; r_, e_, _ = rho_parts(q); vals.append(r_); ends.append(e_)
        ax.plot(np.rad2deg(ang), vals, c=col, label=f"rho, joint {j + 1}")
        sw = [i for i in range(1, len(ends)) if ends[i] != ends[i - 1]]
        ax.plot(np.rad2deg(ang[sw]), np.array(vals)[sw], "kx")
    ax.axhline(env(qd), c="k", ls="--", label="envelope B (wrist-invariant)")
    ax.axvspan(-5, 5, color="0.9"); ax.axhline(0, c="0.5", lw=.8)
    ax.set_xlabel("wrist perturbation [deg] (x = active endpoint switch)"); ax.set_ylabel("margin [m]"); ax.legend(fontsize=7)
    fig.savefig(f"{OUT}/fig2_wrist_sweep_design.png", dpi=150); plt.close(fig)


def main():
    global ZP
    eps, blocked, _ = pick_episodes()
    d2 = np.load("results/day2_dev_data/dev.npz"); d4 = np.load("results/day4_excite/dev.npz")
    src = lambda e: d2 if e["source"] == "day2_dev" else d4
    des = [e for e in eps if e["source"] == "day4_dev"]; chk = [e for e in eps if e["source"] == "day2_dev"]
    assert des and len(chk) >= 4, f"need 1 design + >=4 check episodes: {[(e['source'], e.get('seed')) for e in eps]}"
    de = des[0]; rows = np.array(de["rows"]); qd = src(de)["X"][rows[len(rows) // 2], :M].astype(float)
    _, pb, pt, _ = fk(qd); ZP = float(min(pb[2], pt[2]) - R_TOOL - D_SAFE - RHO_DESIGN)
    checks = asset_checks(qd); figures(qd)
    cfg = dict(tool=dict(kind="SYNTHETIC diagnostic capsule, no mass/inertia", L=L_TOOL, r=R_TOOL, axis="flange-local +z from link_6 origin"),
               d_safe=D_SAFE, plane_z=ZP, design=dict(episode=f"day4_dev_{de['seed']}", row=int(rows[len(rows) // 2]), q=qd.tolist()),
               envelope=dict(centre="link_5 origin (spherical wrist)", R=R_ENV), TOL=TOL, SWITCH=SWITCH, HGRAD=HGRAD,
               wrist_coverage_threshold_rad=WRIST_COV, near=NEAR, perturbation_deg=5, joint_limits="URDF",
               author_LIMS_wrist="0 (c-space sampling), noted", checks=checks,
               episodes=[dict(source=e["source"], seed=e["seed"], replay_err=e["replay_err"]) for e in eps], blocked=blocked)
    json.dump(cfg, open(f"{OUT}/config.json", "w"), indent=1, default=str)
    print(json.dumps(checks, indent=1, default=str))

    # ---------------- geometry
    grows = []
    for e in chk:
        r_ = np.array(e["rows"]); ti = np.unique(np.linspace(0, len(r_) - 1, 8).round().astype(int))
        for t in ti:
            q = src(e)["X"][r_[t], :M].astype(float); r0, end0, gap0 = rho_parts(q); g = grad(q)
            row = dict(episode=e["seed"], t=int(t), q=json.dumps(np.round(q, 6).tolist()), rho=r0, envelope=env(q),
                       active_end=end0, near_switch=gap0 < SWITCH, **{f"grad_q{j + 1}": g[j] for j in range(M)})
            dmax, flips, switches = 0.0, 0, 0
            for j in (3, 4, 5):
                for sgn in (+1, -1):
                    qp = q.copy(); qp[j] += sgn * PERT
                    if not (LO[j] <= qp[j] <= HI[j]): row[f"d_j{j + 1}{'+' if sgn > 0 else '-'}"] = "skip"; continue
                    rp, endp, _ = rho_parts(qp); dr = rp - r0
                    row[f"d_j{j + 1}{'+' if sgn > 0 else '-'}"] = dr
                    dmax = max(dmax, abs(dr)); switches += endp != end0
                    flips += (r0 >= TOL and rp < -TOL) or (r0 < -TOL and rp >= TOL)
            row.update(max_abs_drho=dmax, sensitive=dmax >= 0.005, safety_flips=flips, endpoint_switches=switches)
            grows.append(row)
    n_sens = sum(r["sensitive"] for r in grows); ep_sens = len({r["episode"] for r in grows if r["sensitive"]})
    geometry_status = ("GEOMETRY_SENSITIVE" if n_sens >= 0.2 * len(grows) and ep_sens >= 3 else "NOT_SENSITIVE_AT_5DEG")
    with open(f"{OUT}/geometry_configs.csv", "w", newline="") as f:
        keys = list(dict.fromkeys(k for r in grows for k in r)); w = csv.DictWriter(f, keys, restval=""); w.writeheader(); w.writerows(grows)
    print(f"geometry: {n_sens}/{len(grows)} configs sensitive (>=5 mm), episodes {ep_sens} -> {geometry_status}")

    # ---------------- prediction
    wrows, cover = [], {}
    for e in eps:
        r_ = np.array(e["rows"]); D = src(e); it = integrator_for(e["seed"])
        Xall = D["X"][r_]; cover[e["seed"]] = dict(
            wrist_range_rad=[float(np.ptp(Xall[:, j])) for j in (3, 4, 5)],
            wrist_vel_absmax=[float(np.abs(Xall[:, M + j]).max()) for j in (3, 4, 5)])
        for t in range(len(r_) - H + 1):
            x0 = D["X"][r_[t]].astype(float); U = D["U"][r_[t:t + H]].T.astype(float)
            Xt = roll_true(it, x0, U)
            rec_err = float(np.abs(Xt[1:] - D["Xn"][r_[t:t + H]]).max())
            Xn = roll_model(x0, U, A, B); Xm = roll_model(x0, U, A, B, RES_MSE.dA, RES_MSE.dB)
            rt = np.array([rho(x[:M]) for x in Xt[1:]])
            row = dict(episode=e["seed"], t=int(t), replay_err=rec_err, min_rho_true=float(rt.min()))
            for tag, Xh in (("nom", Xn), ("mse", Xm)):
                rh = np.array([rho(x[:M]) for x in Xh[1:]]); d = rh - rt
                dl = np.array([grad(xt[:M]) @ (xh[:M] - xt[:M]) for xt, xh in zip(Xt[1:], Xh[1:])])
                dw = np.array([rho(np.r_[xt[:3], xh[3:M]]) - rho(xt[:M]) for xt, xh in zip(Xt[1:], Xh[1:])])
                sw = np.array([rho_parts(xt[:M])[1] != rho_parts(xh[:M])[1] for xt, xh in zip(Xt[1:], Xh[1:])])
                tip_err = np.array([np.linalg.norm(fk(xh[:M])[2] - fk(xt[:M])[2]) for xt, xh in zip(Xt[1:], Xh[1:])])
                ps_, ts_ = rh.min() >= TOL, rt.min() >= TOL
                band = abs(rh.min()) <= TOL or abs(rt.min()) <= TOL
                row.update({f"{tag}_min_rho": float(rh.min()), f"{tag}_absd_med": float(np.median(np.abs(d))),
                            f"{tag}_absd_max": float(np.abs(d).max()), f"{tag}_dpos_max": float(np.maximum(d, 0).max()),
                            f"{tag}_prefixmax_dpos": float(np.maximum(d, 0).max()), f"{tag}_tip_err_max": float(tip_err.max()),
                            f"{tag}_dlin_err_max": float(np.abs(dl - d).max()),
                            f"{tag}_dlin_err_at_switch": float(np.abs(dl - d)[sw].max()) if sw.any() else "",
                            f"{tag}_wrist_only_d_max": float(np.abs(dw).max()),
                            f"{tag}_false_safe": bool(ps_ and rt.min() < -TOL and not band),
                            f"{tag}_false_unsafe": bool(rh.min() < -TOL and ts_ and not band), f"{tag}_band": bool(band)})
            row["near"] = NEAR[0] <= row["nom_min_rho"] <= NEAR[1]
            wrows.append(row)
    with open(f"{OUT}/prediction_windows.csv", "w", newline="") as f:
        keys = list(dict.fromkeys(k for r in wrows for k in r)); w = csv.DictWriter(f, keys, restval=""); w.writeheader(); w.writerows(wrows)
    erows = []
    for e in eps:
        for scope in ("all", "near"):
            W = [r for r in wrows if r["episode"] == e["seed"] and (scope == "all" or r["near"])]
            er = dict(episode=e["seed"], scope=scope, n_windows=len(W), **{f"cov_{k}": v for k, v in cover[e["seed"]].items()})
            for tag in ("nom", "mse"):
                if W:
                    er.update({f"{tag}_absd_median": float(np.median([r[f"{tag}_absd_med"] for r in W])),
                               f"{tag}_absd_q95": float(np.quantile([r[f"{tag}_absd_max"] for r in W], .95)),
                               f"{tag}_prefix_q95": float(np.quantile([r[f"{tag}_prefixmax_dpos"] for r in W], .95)),
                               f"{tag}_tip_err_q95": float(np.quantile([r[f"{tag}_tip_err_max"] for r in W], .95)),
                               f"{tag}_wrist_only_q95": float(np.quantile([r[f"{tag}_wrist_only_d_max"] for r in W], .95)),
                               f"{tag}_dlin_err_q95": float(np.quantile([r[f"{tag}_dlin_err_max"] for r in W], .95)),
                               f"{tag}_false_safe": sum(r[f"{tag}_false_safe"] for r in W),
                               f"{tag}_false_unsafe": sum(r[f"{tag}_false_unsafe"] for r in W),
                               f"{tag}_band": sum(r[f"{tag}_band"] for r in W)})
            erows.append(er)
    with open(f"{OUT}/prediction_episodes.csv", "w", newline="") as f:
        keys = list(dict.fromkeys(k for r in erows for k in r)); w = csv.DictWriter(f, keys, restval=""); w.writeheader(); w.writerows(erows)

    near_ep = [er for er in erows if er["scope"] == "near" and er["n_windows"] >= 5]
    wrist_covered = any(max(c["wrist_range_rad"]) >= WRIST_COV for c in cover.values())
    big_ep = [er["episode"] for er in erows if er["scope"] == "near" and er["n_windows"] >= 5 and
              max(er.get("nom_prefix_q95", 0), er.get("mse_prefix_q95", 0)) >= 0.002]
    incons = [(r["episode"], t) for r in wrows if r["near"] for t in ("nom", "mse") if r[f"{t}_false_safe"] or r[f"{t}_false_unsafe"]]
    if len(near_ep) < 4: pstat = "INSUFFICIENT_NEAR_COVERAGE"
    elif len(big_ep) < 3: pstat = "NO_MATERIAL_ERROR"
    elif len(incons) < 20 or len({e for e, _ in incons}) <= 1: pstat = ("ERROR_PRESENT_DECISION_VALUE_NOT_ESTABLISHED" if not incons else "EVENTS_INSUFFICIENT")
    else: pstat = "SIGNAL"
    v = dict(geometry_status=geometry_status, geometry=dict(configs=len(grows), sensitive=n_sens, sensitive_episodes=ep_sens,
                                                             safety_flips=sum(r["safety_flips"] for r in grows),
                                                             endpoint_switches=sum(r["endpoint_switches"] for r in grows),
                                                             checks_passed=True, joint6_invariant=checks["rho_invariant_to_joint6"]),
             prediction_status=pstat, wrist_data_coverage=("COVERED" if wrist_covered else "DATA_DOES_NOT_COVER_WRIST_MOTION"),
             coverage=cover, near_episodes_with_5=len(near_ep), episodes_prefix_q95_ge_2mm=big_ep,
             inconsistent_near_windows=len(incons), inconsistent_episodes=sorted({e for e, _ in incons}),
             not_claimed=["new loss effective", "beats fixed margin", "safety guarantee", "paper contribution", "full-arm collision guarantee"])
    json.dump(v, open(f"{OUT}/verdict.json", "w"), indent=1, default=str)
    print(json.dumps(v, indent=1, default=str))
    for er in erows: print(json.dumps({k: (round(x, 6) if isinstance(x, float) else x) for k, x in er.items()}, default=str))


if __name__ == "__main__":
    main()
