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

os.makedirs("results/ur5e_diag", exist_ok=True)

# Figure 1: Where the weight lands
fig1, ax1 = plt.subplots(figsize=(6.2, 4.5))

x = np.arange(2)
w_ = 0.36

ax1.bar(
    x - w_ / 2,
    [A[0], B[0]],
    w_,
    label="position block $q$",
    color="#378ADD",
)
ax1.bar(
    x + w_ / 2,
    [A[1], B[1]],
    w_,
    label="velocity block $\\dot q$",
    color="#D85A30",
)

ax1.set_xticks(x)
ax1.set_xticklabels([
    "Cartesian\nonly",
    "Cartesian +\nvelocity limits",
])
ax1.set_ylabel("share of constraint-derived weight (%)")
ax1.set_ylim(0, 100)
ax1.set_title("Where the weight lands")
ax1.legend()
ax1.grid(alpha=0.3, axis="y")

for i, v in enumerate([A[1], B[1]]):
    ax1.text(
        i + w_ / 2,
        v + 2,
        f"{v:.1f}%",
        ha="center",
        fontsize=9,
    )

fig1.tight_layout()
fig1.savefig(
    "results/ur5e_diag/weight_distribution.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig1)


# Figure 2: Horizon sensitivity
fig2, ax2 = plt.subplots(figsize=(6.2, 4.5))

ax2.plot(
    Hs,
    qd_share,
    "o-",
    color="#D85A30",
    label="Cartesian only",
)
ax2.axhline(
    B[1],
    linestyle="--",
    color="#0F6E56",
    label="with velocity limits",
)

ax2.set_xlabel("horizon $H$")
ax2.set_ylabel("velocity-block weight share (%)")
ax2.set_ylim(0, 100)
ax2.set_title("Propagation does not fix it")
ax2.legend()
ax2.grid(alpha=0.3)

fig2.tight_layout()
fig2.savefig(
    "results/ur5e_diag/horizon_sensitivity.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig2)


print(
    f"Cartesian only         : q {A[0]:.1f}%  "
    f"qd {A[1]:.1f}%  relevant-dir energy {A[2]:.3f}"
)
print(
    f"Cartesian + vel limits : q {B[0]:.1f}%  "
    f"qd {B[1]:.1f}%  relevant-dir energy {B[2]:.3f}"
)
print(f"isotropic reference    : {1 / 12:.3f}")

print("saved results/ur5e_diag/weight_distribution.png")
print("saved results/ur5e_diag/horizon_sensitivity.png")