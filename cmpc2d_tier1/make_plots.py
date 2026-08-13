#!/usr/bin/env python3
"""Plots for the Tier-1 gates.   python make_plots.py --out results/tier1_gates"""

import argparse
import collections
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [{k: (float(v) if _num(v) else v) for k, v in r.items()}
                for r in csv.DictReader(f)]


def _num(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def plot_gate_dir(rows, out):
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["direction"], r["eps"])].append(r["viol_integral"])
    dirs = sorted({d for d, _ in agg})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for d in dirs:
        eps = sorted({e for dd, e in agg if dd == d})
        mu = [np.mean(agg[(d, e)]) for e in eps]
        se = [np.std(agg[(d, e)], ddof=1) / max(1, np.sqrt(len(agg[(d, e)]))) for e in eps]
        ax[0].errorbar(eps, mu, yerr=se, marker="o", capsize=3, label=d)
    ax[0].set_xlabel(r"injected model error $\epsilon$ (signed)")
    ax[0].set_ylabel("violation integral")
    ax[0].set_title("Gate DIR: does direction matter?")
    ax[0].legend(); ax[0].grid(alpha=.3)

    eps_abs = sorted({abs(e) for _, e in agg if e != 0})
    ratio = []
    for e in eps_abs:
        n = max(np.mean(agg[("normal", s * e)]) for s in (1, -1) if ("normal", s * e) in agg)
        t = max(np.mean(agg[("tangent", s * e)]) for s in (1, -1) if ("tangent", s * e) in agg)
        ratio.append(n / (t + 1e-9))
    ax[1].plot(eps_abs, ratio, "o-")
    ax[1].axhline(3.0, ls="--", c="r", label="pass threshold")
    ax[1].set_xlabel(r"$|\epsilon|$"); ax[1].set_ylabel("normal / tangential")
    ax[1].set_title("directional selectivity"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{out}/gate_dir.png", dpi=140); plt.close(fig)


def plot_gate_b(rows, out):
    arms = ["true_dyn", "learned_dyn"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for i, key in enumerate(["viol_integral", "viol_max", "track_rmse"]):
        data = [[r[key] for r in rows if r["arm"] == a] for a in arms]
        ax[i].boxplot(data, tick_labels=arms)
        ax[i].set_title(key); ax[i].grid(alpha=.3)
    fig.suptitle("Gate B: violation attribution")
    fig.tight_layout(); fig.savefig(f"{out}/gate_b.png", dpi=140); plt.close(fig)


def plot_gate_a(rows, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    un = [np.mean([r["uni_rmse_normal"] for r in rows]),
          np.mean([r["uni_rmse_tangent"] for r in rows])]
    dw = [np.mean([r["dir_rmse_normal"] for r in rows]),
          np.mean([r["dir_rmse_tangent"] for r in rows])]
    ax.bar(x - .18, un, .36, label="uniform")
    ax.bar(x + .18, dw, .36, label="normal-weighted")
    ax.set_xticks(x); ax.set_xticklabels(["normal-dir error", "tangential-dir error"])
    ax.set_ylabel("RMSE"); ax.set_title("Gate A: capacity trade-off")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{out}/gate_a.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/tier1_gates")
    a = ap.parse_args()
    made = []
    r = read_csv(f"{a.out}/gate_dir_raw.csv")
    if r: plot_gate_dir(r, a.out); made.append("gate_dir.png")
    r = read_csv(f"{a.out}/gate_b_raw.csv")
    if r: plot_gate_b(r, a.out); made.append("gate_b.png")
    r = read_csv(f"{a.out}/gate_a_raw.csv")
    if r: plot_gate_a(r, a.out); made.append("gate_a.png")
    print("wrote:", made)
