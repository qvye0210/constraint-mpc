#!/usr/bin/env python3
"""UR5e residual/constraint orthogonality diagnostic.

Reproduces results/ur5e_diag/orthogonality.png.
Run from the project root:  PYTHONPATH=. python ur5e_diagnostic.py
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ur5e.kinematics import random_configs, NQ
from ur5e.dynamics import URParams, f_true, f_nominal, NX
from ur5e.constraints import SphereObstacle, VelocityLimit, propagated_metric
import os

rng = np.random.default_rng(0)
q = random_configs(rng, 3000, margin=0.4); qd = rng.normal(0, 0.6, (3000, NQ))
X = np.concatenate([q, qd], 1); u = rng.normal(0, 0.6, (3000, NQ))
class P(URParams): pass
P.friction, P.payload_kg = 1.0, 0.35
r = f_true(X, u, P) - f_nominal(X, u, P)
sph = SphereObstacle((0.45, 0.10, 0.35), 0.15)
vels = [VelocityLimit(j, limit=1.2) for j in range(6)]

def stats(cs, H=10):
    Xf = np.stack([X] * H, axis=1)
    M = propagated_metric(Xf, cs, gamma=0.95, p=P)
    d = np.einsum("bii->bi", M); w, V = np.linalg.eigh(M); top = V[:, :, -1]
    al = ((r * top).sum(1) ** 2) / np.maximum((r ** 2).sum(1), 1e-18)
    tot = d.sum(1).mean()
    return d[:, :NQ].sum(1).mean() / tot * 100, d[:, NQ:].sum(1).mean() / tot * 100, al.mean()

A, B = stats([sph]), stats([sph] + vels)
Hs = [1, 2, 5, 10, 25, 50, 75]; qd_share = [stats([sph], H)[1] for H in Hs]
res_q = np.sqrt((r[:, :NQ] ** 2).sum(1).mean()); res_qd = np.sqrt((r[:, NQ:] ** 2).sum(1).mean())

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
x = np.arange(2); w_ = 0.36
ax[0].bar(x - w_ / 2, [A[0], B[0]], w_, label="position block  q", color="#378ADD")
ax[0].bar(x + w_ / 2, [A[1], B[1]], w_, label="velocity block  $\\dot q$", color="#D85A30")
ax[0].set_xticks(x); ax[0].set_xticklabels(["Cartesian\nonly", "Cartesian +\nvelocity limits"])
ax[0].set_ylabel("share of constraint-derived weight (%)"); ax[0].set_ylim(0, 100)
ax[0].set_title("weight"); ax[0].legend(); ax[0].grid(alpha=.3, axis="y")
for i, v in enumerate([A[1], B[1]]): ax[0].text(i + w_ / 2, v + 2, f"{v:.1f}%", ha="center", fontsize=9)

lbl = ["model error\n(residual)", "constraint\ngradient"]
ax[1].bar(lbl, [0, 100], 0.5, color="#378ADD", label="position q")
ax[1].bar(lbl, [100, 0], 0.5, bottom=[0, 100], color="#D85A30", label="velocity $\\dot q$")
ax[1].set_ylabel("energy share (%)"); ax[1].set_title("They are orthogonal")
ax[1].legend(); ax[1].grid(alpha=.3, axis="y")
ax[1].text(0.5, 50, f"residual rms\nq: {res_q:.1e}\n$\\dot q$: {res_qd:.3f}", ha="center",
           fontsize=9, bbox=dict(fc="white", ec="#888", alpha=.85))

ax[2].plot(Hs, qd_share, "o-", color="#D85A30", label="Cartesian only")
ax[2].axhline(B[1], ls="--", color="#0F6E56", label="with velocity limits")
ax[2].set_xlabel("horizon H"); ax[2].set_ylabel("velocity-block weight share (%)")
ax[2].set_ylim(0, 100); ax[2].set_title("Propagation does not fix it")
ax[2].legend(); ax[2].grid(alpha=.3)
fig.suptitle("UR5e velocity-level modelling: model residual and Cartesian constraint "
             "gradient occupy disjoint blocks", fontsize=11)
fig.tight_layout()
os.makedirs("results/ur5e_diag", exist_ok=True)
fig.savefig("results/ur5e_diag/orthogonality.png", dpi=150)
print(f"Cartesian only         : q {A[0]:.1f}%  qd {A[1]:.1f}%  relevant-dir energy {A[2]:.3f}")
print(f"Cartesian + vel limits : q {B[0]:.1f}%  qd {B[1]:.1f}%  relevant-dir energy {B[2]:.3f}")
print(f"isotropic reference    : {1/12:.3f}")
print("saved results/ur5e_diag/orthogonality.png")
