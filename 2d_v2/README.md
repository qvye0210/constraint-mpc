# Tier-1: 2D point mass + circular obstacle — three gates

Purpose: decide whether the "decision-relevant prediction error" idea has a
mechanism to exploit **before** building the five-arm comparison.
Each gate can kill the line.

## Why this system

The 1D double integrator with `|p| <= 2` cannot test the hypothesis. There
`dg/dx_{t+k} ≈ [1, k·dt]`, so a constraint-rollout loss is *algebraically* a
re-weighted multi-step MSE — no decision-irrelevant direction exists.

Here `g(x) = r_safe − ||p − p_obs||` is nonlinear: the constraint normal rotates
with the state, so the tangential direction is genuinely irrelevant to first
order. That is the minimal change that lets the hypothesis be true *or false*.

## Two design decisions found during the smoke test

1. **Zero-order hold on the re-solve (`Params.resolve_every = 5`).**
   With per-step re-solving and exact state feedback, only the *one-step* model
   error leaks into closed loop; violations were ~5e-4 (0.05 % of `r_safe`) and
   the multi-step hypothesis was untestable. Holding the control for 5 steps
   raises violations to ~0.18 and makes horizon propagation observable.
   (Also thematically consistent with the event-triggered re-solve work.)

2. **The sign of the injected error matters in Gate DIR.**
   `normal_dir` points along `dg/dp`, i.e. *towards* the obstacle. `eps > 0`
   makes the model pessimistic and violations go **down**. Only `eps < 0`
   (optimistic) tests the hypothesis. Both signs are swept.

## Layout

```
cmpc2d/env.py     dynamics (drag = plant/model mismatch), constraint geometry,
                  normal_dir / tangent_dir, scenario sampling
cmpc2d/mpc.py     single-shooting SLSQP MPC, pluggable batched dyn_fn,
                  hard constraints + soft fallback (infeasibility flagged)
cmpc2d/eval.py    closed-loop rollout, metrics, paired bootstrap / effect size
cmpc2d/data.py    nominal-MPC data collection, TRAJECTORY-level splits,
                  stores future windows + margins (reused by Step 3)
cmpc2d/model.py   residual MLP (learns the mismatch, as planned for the UR5e),
                  direction-decomposed loss, normal/tangential error diagnostics
cmpc2d/gates.py   the three gates + verdicts
run_gates.py      driver (--quick / --full)
sweep_gate_a.py   locates the capacity-constrained regime
make_plots.py     figures
```

## Commands

```bash
python run_gates.py --quick                 # ~35 s, code-path check
python run_gates.py --full --seeds 3        # real run, single process
python sweep_gate_a.py --seeds 3            # regime search
python make_plots.py --out results/tier1_gates
```

### Parallel across seeds

`--seeds N` means "run seeds 0..N-1". To split across processes use
`--seeds 1 --seed-offset s`, which runs seed `s` only. Each seed gets its own
dataset (`build_dataset(seed=s)`) as well as its own model init, so the runs are
genuinely independent — without `--seed-offset` every process would repeat
seed 0 and produce identical output.

```bash
mkdir -p logs
for s in 0 1 2; do
  python run_gates.py --full --seeds 1 --seed-offset $s \
      --out results/gates_seed$s > logs/gates_seed$s.log 2>&1 &
done; wait
python merge_results.py "results/gates_seed*" --out results/gates_merged

for s in 0 1 2; do
  python sweep_gate_a.py --seeds 1 --seed-offset $s \
      --out results/sweep_seed$s > logs/sweep_seed$s.log 2>&1 &
done; wait
python merge_results.py "results/sweep_seed*" --out results/sweep_merged --sweep

python make_plots.py --out results/gates_merged
```

`merge_results.py` pools the raw per-episode rows and recomputes the verdicts on
the combined sample — it does not average per-seed verdicts. Add
`torch.set_num_threads(1)` if the processes contend for cores.

Outputs go to `results/tier1_gates/`: `gate_*_raw.csv` (per-episode / per-seed
raw), `report.json` (verdicts), `health.csv` (NaN / grad-norm / loss-decrease
checks), and three PNGs. Nothing outside that directory is touched.

## Gates and pass criteria

| Gate | Question | Pass criterion |
|---|---|---|
| DIR | does a decision-irrelevant direction exist? | violation(normal) / violation(tangential) > 3 |
| B | is violation caused by model error? | true-dyn ≈ 0, learned-dyn ≫ 0, paired CI excludes 0 |
| A | is there a capacity trade-off? | normal error ↓ > 5 % **and** tangential error ↑ > 5 % |

Gate A failing is the *expected* fixable failure: with enough capacity/data,
re-allocation is a no-op. `sweep_gate_a.py` searches `n_traj × hidden × w_normal`
for a regime where the trade-off is real. If none exists, that must be reported
— the idea is a no-op in this regime and the paper needs an explicit
capacity-constrained setting justified up front.

## Smoke-test status (1 CPU core, quick mode)

| Gate | Result |
|---|---|
| DIR | **PASS** — normal 5.04 vs tangential 1.30, ratio 3.9 (at eps = −0.1) |
| B | **PASS** — true 0.00000 vs learned 0.30507, CI [0.224, 0.392] |
| A | fail at `w_normal=10`; **PASS** at `hidden=64x64, w_normal=50` (normal −8.8 %, tangential +11.8 %) |

These are quick-mode numbers on tiny data — they confirm the code paths and the
qualitative mechanism, **not** the scientific conclusion. Re-run with `--full`.

## Note on `resolve_every`

It is a first-class knob. Sweeping it (1, 2, 3, 5, 8) is worth a figure: it
controls how much horizon-propagated model error reaches the closed loop, and
therefore how much room the method has. At `resolve_every = 1` the ceiling is
near zero, which is itself a useful scope statement for the paper.

---

## Diagnostics: why static normal weighting failed

Measured on the collected data (`--- see diag output ---`):

| quantity | k=0 | k=1 | k=5 | k=10 |
|---|---|---|---|---|
| \|dg\| ratio normal:tangent | ~3500 (1/eps, i.e. 2nd order) | 22.0 | 4.1 | 2.0 |
| constraint-normal rotation (near-constraint) | 0° | 6.9° | 33.2° | 59.3° |

A tangential error is decision-irrelevant **only instantaneously**. Along the
horizon the constraint normal rotates, so tangential error leaks into the normal
direction. With `resolve_every = 5` the closed loop operates at k≈5, where the
ratio is ≈4 — which is exactly the 3.9 measured by Gate DIR. The "disappointing"
Gate DIR number is fully explained by horizon propagation.

This is why `dir_weighted` (a STATIC projection at k=0) bought 16.6 % normal
accuracy for 120 % tangential degradation: it optimises the k=0 geometry while
the closed loop lives at k≈5.

`constraint_rollout` evaluates `g` at the predicted FUTURE states, so it sees
the rotation. `compare_static_vs_rollout.py` tests that prediction offline.

```bash
python compare_static_vs_rollout.py --quick            # ~2 min
python compare_static_vs_rollout.py --seeds 3          # real run
```

Primary metric is `cerr_horizon_near` (near-constraint horizon constraint-value
error). `multistep_mse` is in the arm list to separate the contribution of
multi-step training from the contribution of constraint geometry: if
`cerr_near_vs_multistep >= 0`, any gain is coming from horizon training alone
and the constraint projection is doing nothing.

The `proxy_viol` column is built from ONE-STEP directional errors and is biased
against rollout arms by construction — read it as secondary.
