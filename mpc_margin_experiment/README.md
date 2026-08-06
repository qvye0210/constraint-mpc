# MPC Constraint-Margin Decision-Impact Diagnostic

Mechanism-validation experiment (not a proposed learning method) testing:

> Under matched dynamics-model prediction-error magnitude, do near-constraint
> MPC scenarios show greater downstream decision impact than far-from-constraint
> scenarios?

System: 1D discrete double integrator, finite-horizon tracking MPC (CVXPY + OSQP).
Oracle model = true dynamics. Perturbed model = true dynamics + small constant
additive bias of controlled norm, applied across multiple directions and
magnitudes. Downstream impact measured primarily via first-action discrepancy
Δu0, plus active-set change, infeasibility, objective gaps, and true-dynamics
replay/violation.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate   # optional
pip install -r requirements.txt
```

Note: `pip install cvxpy osqp` can take a minute or two the first time
(compiles/downloads several solver backends: OSQP, CLARABEL, SCS, HiGHS).

## Run

```bash
# 1. Run correctness tests first (should take well under a minute)
python -m pytest tests/ -v

# 2. Smoke test (~50 scenarios, sanity check, ~3 minutes on a typical laptop)
python run_experiment.py --mode smoke --outdir results_smoke

# 3. Full run (~600 scenarios x 8 bias directions x 3 magnitudes = ~14,400
#    perturbed MPC solves + 600 oracle solves; expect roughly 12x the smoke
#    runtime, i.e. ~30-40 minutes depending on machine. Each MPC solve is a
#    small QP (horizon ~18) so this is CPU-bound but easily parallelizable
#    if you want to speed it up further -- see "Speeding it up" below.)
python run_experiment.py --mode full --outdir results
```

Outputs land in the chosen `--outdir`:
- `sample_level_dataset.csv` -- full oracle/perturbed pair-level dataset
- `oracle_scenarios.csv` -- oracle-only scenario data (array columns JSON-encoded)
- `experiment_config.json`, `run_meta.json` -- full reproducibility record
- `stat_*.csv` -- all statistical test results (near/far comparisons, rate
  comparisons, saturation-confound checks, supplementary outcomes)
- `stat_correlations_and_regressions.json` -- Spearman correlations + diagnostic
  linear/logistic regression comparisons
- `group_summary_fixed.csv`, `group_summary_quantile.csv`
- `plots/` -- all required PNG + PDF figures
- `summary.md` -- the final Markdown report with the scientific verdict

## Project layout

```
run_experiment.py     # orchestrator (--mode smoke|full)
src/
  config.py            # all experiment configuration (dataclasses -> JSON)
  dynamics.py           # true discrete double-integrator dynamics
  bias.py                # bias directions + norm-matched bias vectors
  mpc_solver.py           # CVXPY QP: oracle & perturbed MPC, margins, duals, active set
  scenarios.py             # reproducible scenario sampling
  dataset.py                # oracle + perturbed-pair dataset generation
  grouping.py                 # near/far grouping (fixed threshold + quantile)
  analysis.py                  # bootstrap CIs, Mann-Whitney U, Cliff's delta,
                                 # Spearman, diagnostic regressions
  plots.py                     # all required figures
  report.py                     # generates results/summary.md
tests/
  test_correctness.py           # 16 correctness tests (see below)
results/ or results_smoke/      # generated outputs (not checked in)
```

## What the tests check

Dynamics matrix correctness; bias vectors have exactly the requested norm and
that norm is preserved across all directions; oracle and perturbed MPC share
an identical cost/constraint structure (verified by solving with a zero-bias
"perturbed" model and checking it reproduces the oracle exactly); state-margin
values are computed directly from bound - state, independent of dual values;
dual variables are nonnegative and stored in separate arrays from slacks;
true-model replay of a perturbed control sequence genuinely does not use the
bias term (checked against a control replay that DOES reuse the same bias);
one-step prediction-error magnitude is matched exactly across bias directions;
scenario sampling and full dataset generation are reproducible under a fixed
seed; infeasible QPs are flagged via `status`/`solve_success` rather than
raising; active-set membership requires both small slack AND a meaningfully
positive dual, verified constraint-by-constraint.

## Important caveat found during the smoke run

Near-constraint scenarios (defined by the oracle's minimum state-constraint
margin) turned out, in this configuration, to coincide heavily with scenarios
where the oracle's very first control action u0 is already saturated at the
input bound (a "corner solution"). A saturated u0 tends to stay saturated
under a small model bias in both the oracle and perturbed solve, which
mechanically suppresses Δu0 for reasons unrelated to state-constraint
proximity. The code tracks this explicitly (`oracle_u0_saturated`) and the
report includes: (a) a saturation-rate-by-group table, (b) a near/far
re-comparison restricted to non-saturated scenarios, and (c) supplementary
continuous outcomes (|objective_diff|, true-model max violation) that are not
gated by u0 saturation the same way. Read `summary.md`'s "Confound check"
section before trusting the raw Δu0 near/far numbers at face value -- this
was a genuine, unplanned finding, not a bug, and is exactly the kind of thing
this diagnostic experiment is meant to surface.

## Speeding it up (optional)

The `--mode full` run is embarrassingly parallel across scenarios (each
oracle + its perturbed sweep is independent). If you want it faster, the
easiest lever is reducing `sampling.n_scenarios_full` in `src/config.py`
before running (e.g. 300 instead of 600) -- the statistical tests will still
run, just with wider confidence intervals. Parallelizing `generate_dataset`
across processes would also help but isn't implemented here to keep the code
simple and single-threaded/reproducible.
