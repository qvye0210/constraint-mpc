# Stage 3: Dual-Weighted Constraint Rollout Learning

Tests: does directly reducing the model's prediction error on the FUTURE
constraint value (margin), instead of upweighting the full one-step
dynamics loss for near-constraint samples, more effectively reduce
closed-loop constraint violations under a learned-dynamics MPC?

## What's reused from existing code (unchanged)

- `src/config.py`, `src/mpc_solver.py` -- stage 1's linear-model MPC
  (cost/bounds/horizon untouched) is used AS-IS as the nominal/noisy
  data-collection controller.
- `stage2_margin_weighting/generate_data.py` -- `true_step` (the nonlinear
  true plant, post stage-2's wall-clip correction) and `POS_BOUND` /
  `INPUT_BOUND`, reused unchanged.
- The margin-weighting formula and constants from stage 2
  (`1 + 4*exp(-margin/0.2)`, clipped `[0.5, 5.0]`, mean-normalized) are
  reused exactly for `margin_weighted_mse`.
- The stage-2 successive-linearization + DPP-QP pattern for closed-loop
  learned-dynamics MPC is reused, kept as a self-contained copy in
  `stage3/mpc_learned.py` (stage2 and stage3 each have their own
  `models.py` with different network classes, so a literal cross-package
  import would be fragile).

Nothing in `stage1`, `results`, `results_smoke`, or `stage2_margin_weighting/`
is modified. This experiment lives entirely in its own directory and its
own `data/checkpoints/results`.

## Run

```bash
pip install -r requirements.txt
cd stage3_dual_weighted_rollout

python run_experiment.py --quick          # tiny sanity run, ~1-2 min
python run_experiment.py                   # full spec run
python run_experiment.py --skip-tuning      # full run, skip beta tuning (uses config.py default beta=1.0)
```

Single entry point. Re-running (non-`--quick`) auto-archives the previous
`data/`, `checkpoints/`, `results/` into `archive_<timestamp>/` first, so
nothing is silently overwritten.

## Files

```
config.py            # all experiment configuration (H, gamma, beta, weight
                       # formula constants, MPC-eval episode settings, seeds)
generate_data.py       # closed-loop trajectory collection: stage-1 linear
                         # MPC (nominal + noisy) applied to stage-2's true
                         # nonlinear plant; records the MPC's own H-step-
                         # ahead predicted margin/dual as privileged data
models.py                # ResidualMLP (3->256->256->128->2, SiLU), Normalizer
train.py                   # windowing + all 4 training methods + sanity checks
tune_beta.py                 # small-scale, validation-only beta grid search
mpc_learned.py                 # learned-dynamics MPC (per-step NN linearization
                                 # + DPP QP, reusing stage-1 cost/bounds/horizon)
evaluate_prediction.py          # offline metrics + 3 figures
evaluate_mpc.py                   # closed-loop paired-episode evaluation + 1 figure
run_experiment.py                   # orchestrator; single entry point
```

## The four methods

1. **`baseline_mse`**: `L = L_dyn` (plain one-step MSE).
2. **`margin_weighted_mse`**: `L = w(margin_t) * L_dyn`, reusing stage 2's
   exact formula, based on the CURRENT-step margin only.
3. **`constraint_rollout`**: `L = L_dyn + beta * L_con`, where `L_con =
   sum_{k=1..H} gamma^(k-1) * [g(x_hat_{t+k}) - g(x_{t+k})]^2`, `g(x) =
   pos_bound - |p|` (the constraint margin itself). Unweighted across
   horizon steps (equal treatment of every k, just discounted by gamma).
4. **`dual_weighted_constraint_rollout`**: same as (3) but each horizon
   step's squared constraint-value error is additionally weighted by
   `w_{t,k} = clip(1 + alpha_m*exp(-m_{t,k}/tau) + alpha_lambda*lambda_{t,k}/
   (lambda_bar+eps), 0.5, 5.0)`, batch-mean-normalized (mean computed over
   all `B*H` entries in the current mini-batch, so it becomes exactly 1 for
   that batch). `m_{t,k}` and `lambda_{t,k}` are the data-collection MPC's
   OWN horizon-predicted margin/dual for step `k` at the time the window
   started (privileged information, recorded once during `generate_data.py`,
   not recomputed by the learned model). `lambda_bar` is a fixed constant
   (mean of all recorded MPC duals in the training set, analogous to a
   normalizer statistic, fit on train only).

All four are trained on the exact same set of H-step windows (same
starting timestep `t` per trajectory, same mini-batch order per seed,
independent of method), the exact same initial weights per seed
(`torch.manual_seed(seed)` called identically before construction), the
same optimizer/epochs/patience, and the same early-stopping metric
(unweighted one-step validation MSE for all four).

The rollout in methods 3-4 uses the TRUE recorded control sequence (open-
loop / teacher-forcing on actions, not on states) and gradients are
backpropagated through the full H-step chain (no `detach()` between steps).

## Sanity checks (task requirement 7)

`train.py` checks, every batch: loss/grad finiteness (aborts that
method/seed with a printed message if NaN/Inf is hit rather than silently
continuing), gradient-norm clipping (`grad_clip_norm=10.0`) with the max
norm seen logged per run, and per-batch weight min/max logged for both
weighted methods. All of this is written to `checkpoints/fairness_report.json`
and printed during training. `generate_data.py` also reports the MPC
data-collection infeasibility rate.

## Judgment

Implemented in `run_experiment.py::judge()`. Four checks against
`baseline_mse`: (1) future constraint-value error improves for the full
method in every seed; (2) tracking RMSE and cumulative objective don't
degrade by more than 10% in any seed; (3) closed-loop violation_rate
improves with a statistically significant (Wilcoxon p<0.05) paired
difference; (4) the violation-rate improvement direction is consistent
across every seed. If (1) holds but (3)/(4) don't, the verdict is
explicitly `proxy_improved_no_control_benefit` (task requirement: report
this failure mode explicitly rather than folding it into a generic
"not supported"). Full verdict space: `supported` (4/4) /
`proxy_improved_no_control_benefit` / `mixed` / `not_supported` /
`inconclusive` (fewer than 2 seeds, e.g. `--quick`).
