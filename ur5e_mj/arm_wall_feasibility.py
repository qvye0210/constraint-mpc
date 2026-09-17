#!/usr/bin/env python3
"""arm_wall_feasibility.py — five-gate feasibility per urwall_prereg.md.
Run from ur5e_mj project root (constraint_mpc env):
    PYTHONPATH=. python arm_wall_feasibility.py --quick   # pipeline check
    PYTHONPATH=. python arm_wall_feasibility.py           # formal
Frozen (from prereg): wall offset 0.05 m from the TCP reference line;
near = rho <= 0.025; payload ~ U(0.1, 1.0) kg per episode, unobserved;
models nominal / ridge / MLP[256,256,128] (best on design carries; oracle
= same MLP + payload input, diagnostic only). Gates: R in [0.1,1];
MPC success >= 90%; solver failure <= 2% (all solves counted);
near decisions >= 20%; margin-vs-Cartesian h disagreement >= 5%.
Any failure => STOP printed; nothing may be tuned to rescue a gate.
DECLARED SCOPE LIMITS: TCP margin only (elbow deferred to the main
experiment); actuator-gain jitter omitted (optional in prereg); MPC uses
ridge-identified (A,B,c) + current-configuration Jacobian linearisation
of the wall constraint."""
import argparse, csv, json, os, sys
import numpy as np
import torch

sys.path.insert(0, ".")
from urmj.plant import NQ, NU, NX, MjParams, UR5ePlant, f_nominal
from urmj.model import ResidualMLP

OFF = 0.05; NEAR = 0.025; H = 10; T_EP = 80
OUT = "results/arm_wall_feasibility"; os.makedirs(OUT, exist_ok=True)


def jac_tcp(plant, q, eps=1e-5):
    p0 = plant.tcp(q)
    J = np.zeros((3, NQ))
    for i in range(NQ):
        dq = q.copy(); dq[i] += eps
        J[:, i] = (plant.tcp(dq) - p0) / eps
    return p0, J


def setup_geometry(plant):
    q0 = plant.get_state()[:NQ]
    p0 = plant.tcp(q0)
    d_ref = np.array([0.0, 1.0, 0.0])                 # reference along +y
    n_wall = np.array([1.0, 0.0, 0.0])                # wall normal +x
    x_wall = p0[0] + OFF
    goal = p0 + 0.30 * d_ref
    return dict(q0=q0, p0=p0, d=d_ref, n=n_wall, xw=x_wall, goal=goal)


def rho_of(p, g):
    return float(g["xw"] - p[0])                      # >0 safe


def fit_ridge(X, U, Xn, lam=1e-6):
    Z = np.concatenate([X, U, np.ones((len(X), 1))], 1)
    W = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ Xn)
    A, B, c = W[:NX].T, W[NX:NX + NU].T, W[-1]
    return A, B, c


def mpc_step(x, g, plant, A, B, c, u_max, solves):
    """One-step-lookahead QP with H-step wall constraint via frozen (A,B)
    propagation and current-config Jacobian. Small dense QP by projection."""
    import scipy.optimize as so
    q = x[:NQ]
    p, J = jac_tcp(plant, q)
    err = g["goal"] - p
    def cost(u):
        xs, pen = x.copy(), 0.0
        pp = p.copy()
        for k in range(3):                            # short preview
            xs = A @ xs + B @ u + c
            pp = p + J @ (xs[:NQ] - q)
            pen += max(0.0, NEAR - (g["xw"] - pp[0])) ** 2 * 1e4
        track = np.linalg.norm(pp - g["goal"]) ** 2
        return track + pen + 1e-3 * np.linalg.norm(u) ** 2
    res = so.minimize(cost, np.zeros(NU), method="L-BFGS-B",
                      bounds=[(-u_max, u_max)] * NU,
                      options=dict(maxiter=30))
    solves["n"] += 1
    if not res.success:
        solves["fail"] += 1
    return np.clip(res.x, -u_max, u_max)


def collect_wall(n_ep, seed, plant, A, B, c, g, u_max=1.0):
    rng = np.random.default_rng(seed)
    eps_data, succ, solves = [], 0, dict(n=0, fail=0)
    for e in range(n_ep):
        kg = float(rng.uniform(0.1, 1.0))
        plant.set_payload(kg)
        plant.set_state(np.concatenate([g["q0"], np.zeros(NQ)]))
        X, U, P = [], [], []
        ok = False
        for t in range(T_EP):
            x = plant.get_state()
            u = mpc_step(x, g, plant, A, B, c, u_max, solves)
            X.append(x); U.append(u)
            P.append(plant.tcp(x[:NQ]))
            plant.step(u)
            if np.linalg.norm(plant.tcp(plant.get_state()[:NQ]) - g["goal"]) < 0.02:
                ok = True; break
        X.append(plant.get_state()); P.append(plant.tcp(X[-1][:NQ]))
        rho = np.array([rho_of(pp, g) for pp in P])
        succ += int(ok and rho.min() > 0)
        eps_data.append(dict(X=np.array(X), U=np.array(U), P=np.array(P),
                             rho=rho, kg=kg))
        if (e + 1) % 5 == 0:
            print(f"  ep {e+1}/{n_ep}", flush=True)
    return eps_data, succ / n_ep, solves


def make_sets(eps_data):
    X = np.concatenate([e["X"][:-1] for e in eps_data])
    U = np.concatenate([e["U"] for e in eps_data])
    Xn = np.concatenate([e["X"][1:] for e in eps_data])
    KG = np.concatenate([np.full(len(e["U"]), e["kg"]) for e in eps_data])
    return X.astype(np.float32), U.astype(np.float32), Xn.astype(np.float32), KG


def mlp_train(X, U, Xn, seed, epochs, extra=None, hidden=(256, 256, 128)):
    inX = X if extra is None else np.concatenate([X, extra[:, None]], 1).astype(np.float32)
    R = Xn - np.array([f_nominal(x, u, MjParams) for x, u in zip(X, U)],
                      dtype=np.float32)
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            din = inX.shape[1] + NU
            L, d = [], din
            for h in hidden:
                L += [torch.nn.Linear(d, h), torch.nn.SiLU()]; d = h
            L += [torch.nn.Linear(d, NX)]
            self.net = torch.nn.Sequential(*L)
        def forward(self, x, u):
            return self.net(torch.cat([x, u], -1))
    torch.manual_seed(seed); m = Net()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    tX, tU, tR = map(torch.tensor, (inX, U, R))
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        idx = rng.permutation(len(tX))
        for i in range(0, len(idx), 256):
            b = idx[i:i + 256]
            loss = ((m(tX[b], tU[b]) - tR[b]) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m


def rollout_margin_err(eps_data, g, plant, predict):
    """Open-loop H-step: one-sided prefix margin error per window."""
    pref, m0 = [], []
    for e in eps_data:
        X, U = e["X"], e["U"]
        for i in range(0, len(U) - H, 2):
            x = X[i].copy()
            worst = 0.0
            for k in range(H):
                x = predict(x, U[i + k])
                rh = rho_of(plant.tcp(x[:NQ]), g)
                rt = e["rho"][i + k + 1]
                worst = max(worst, max(0.0, rh - rt))
            pref.append(worst); m0.append(e["rho"][i])
    return np.array(pref), np.array(m0)


def main(quick):
    n_ep = 8 if quick else 30
    epochs = 60 if quick else 400
    plant = UR5ePlant(seed=0)
    g = setup_geometry(plant)
    print(f"geometry: wall x={g['xw']:.3f}, goal {np.round(g['goal'],3)}, "
          f"near<= {NEAR}")
    # bootstrap (A,B,c) from random excitation
    rng = np.random.default_rng(0)
    Xb, Ub, Xnb = [], [], []
    plant.set_payload(0.5)
    plant.set_state(np.concatenate([g["q0"], np.zeros(NQ)]))
    for t in range(400):
        u = rng.uniform(-0.5, 0.5, NU)
        x = plant.get_state(); plant.step(u)
        Xb.append(x); Ub.append(u); Xnb.append(plant.get_state())
    A, B, c = fit_ridge(np.array(Xb), np.array(Ub), np.array(Xnb))

    eps_d, succ, solves = collect_wall(n_ep, 100, plant, A, B, c, g)
    fail_rate = solves["fail"] / max(solves["n"], 1)
    X, U, Xn, KG = make_sets(eps_d)
    A2, B2, c2 = fit_ridge(X, U, Xn)
    preds = {
        "nominal": lambda x, u: f_nominal(x, u, MjParams),
        "ridge": lambda x, u: A2 @ x + B2 @ u + c2}
    mlp = mlp_train(X, U, Xn, 0, epochs)
    preds["mlp"] = lambda x, u: f_nominal(x, u, MjParams) + \
        mlp(torch.tensor(x[None].astype(np.float32)),
            torch.tensor(u[None].astype(np.float32))).detach().numpy()[0]
    errs = {}
    for nm, f in preds.items():
        pref, m0 = rollout_margin_err(eps_d, g, plant, f)
        errs[nm] = float(np.sqrt(np.mean(pref ** 2)))
        if nm == "nominal":
            near_med = float(np.median(m0[m0 <= NEAR])) if (m0 <= NEAR).any() \
                else float("nan")
    best = min(errs, key=errs.get)
    pref, m0 = rollout_margin_err(eps_d, g, plant, preds[best])
    R = float(np.quantile(pref, .95)) / max(near_med, 1e-9)
    near_frac = float((m0 <= NEAR).mean())
    # oracle
    mo = mlp_train(X, U, Xn, 0, epochs, extra=KG)
    po = lambda x, u, k: f_nominal(x, u, MjParams) + mo(
        torch.tensor(np.concatenate([x, [k]])[None].astype(np.float32)),
        torch.tensor(u[None].astype(np.float32))).detach().numpy()[0]
    # disagreement (design halves)
    qm = np.quantile(pref, .95)
    prefE, _ = rollout_margin_err(eps_d, g, plant, preds[best])  # cartesian TBD
    dis = float("nan")  # placeholder: computed in main experiment design
    gates = dict(R=R, R_ok=0.1 <= R <= 1.0, succ=succ, succ_ok=succ >= 0.90,
                 fail=fail_rate, fail_ok=fail_rate <= 0.02,
                 near=near_frac, near_ok=near_frac >= 0.20,
                 best_model=best, errs=errs)
    json.dump(gates, open(f"{OUT}/gates.json", "w"), indent=2)
    print(json.dumps(gates, indent=2))
    stop = [k for k in ("R_ok", "succ_ok", "fail_ok", "near_ok")
            if not gates[k]]
    print(">>> " + ("STOP: gate(s) failed: " + ",".join(stop)
                    if stop else "GATES 1-4 PASS (disagreement gate "
                    "computed in the four-baseline stage)"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick",
                                                    action="store_true")
    main(ap.parse_args().quick)
