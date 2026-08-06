# Stage 2: Constraint-Margin-Weighted Dynamics Learning (Quick Proof-of-Concept)

Tests: does weighting the dynamics-learning loss by constraint margin reduce
near-constraint one-step/rollout prediction error (without hurting overall
error much) and reduce closed-loop constraint violations under a
learned-dynamics MPC, compared to (a) an unweighted baseline and (b) a
random-weight control that uses the identical weight *distribution* but
scrambled so it no longer correlates with margin?

Reuses stage 1's `src/config.py` (MPC horizon=18, dt=0.1, cost weights,
position bound +/-2.0, input bound +/-0.5) unchanged. Stage 1's true
dynamics were fully linear, so per the task spec a small fixed
unmodeled nonlinearity was added for stage 2's TRUE plant only (see
`generate_data.true_step`); stage 1 files/results were not touched.

## Run

```bash
pip install -r requirements.txt      # adds torch to stage-1's deps
cd stage2_margin_weighting

python run_experiment.py --quick      # ~1-2 min sanity check (1 seed, tiny data)
python run_experiment.py              # full spec (3 seeds, ~10k transitions, 24 episodes)
```

Single command, no manual steps. Full run is expected to finish well under
an hour on CPU (offline training: 3 methods x 3 seeds x <=100 epochs of a
tiny 3->32->32->2 MLP; closed-loop eval: 3x3x24 episodes x 40 steps, each
step = one DPP-cached QP solve, ~10-30ms).

## Files

```
generate_data.py        # nonlinear true plant, trajectory-based transition generation
models.py                # shared MLP architecture + normalizer (identical across methods)
train.py                  # trains baseline / random_weight / margin_weighted, identical
                            # conditions except per-sample loss weight; fairness checks
mpc_learned.py              # learned-dynamics MPC: per-step NN linearization + DPP QP
                              # (reuses stage-1 cost/bounds/horizon, A/B/c as cp.Parameter)
evaluate_prediction.py        # one-step + 15-step rollout RMSE, offline figures (torch-batched)
evaluate_mpc.py                 # closed-loop episodes (fixed set shared across methods/seeds),
                                  # violation/objective/tracking metrics, paired stats, figures
run_experiment.py                 # orchestrator; single entry point; writes results/report.md
data/, checkpoints/, results/       # generated artifacts (not checked in)
```

## Method summary

- **Baseline**: uniform loss weight = 1.
- **Random-weight control**: exactly the margin-weighted weight *values*,
  shuffled within the training set so they no longer correspond to margin
  (checked via an assertion that the sorted weight multisets are identical).
- **Margin-weighted**: `w = clip(1 + 4*exp(-margin/0.2), 0.5, 5.0)`,
  mean-normalized on the training set.

All three share: identical MLP init per seed (`torch.manual_seed(seed)`
called identically before construction), identical optimizer (Adam, lr=1e-3),
identical mini-batch order per seed (separate fixed-seed RNG controls batch
permutation, independent of method), identical max epochs (100) / patience
(10), and early stopping on **unweighted** validation MSE for all three
(so the stopping criterion itself doesn't favor the weighted method).

## Closed-loop MPC with a learned nonlinear model -- key simplification

Stage 1's MPC is a convex QP built around a fixed linear (A, B). To control
with a genuinely nonlinear learned model without adding solver
differentiation or a complex network, `mpc_learned.py` linearizes the
model (first-order Taylor expansion via `torch.autograd.functional.jacobian`)
around the current state and previously applied control at *every*
closed-loop step, and freezes that local affine model across the horizon
for that one QP solve (successive-linearization / real-time-iteration
style). The QP problem structure is built once (with A, B, c as
`cp.Parameter`) and reused via CVXPY's DPP support for every solve across
every episode/method/seed.

## Fairness checks performed

Automated in `train.py` (`fairness_report.json`) and `run_experiment.py`:
identical parameter counts across methods; identical data split; identical
init seeds; identical optimizer/steps; identical closed-loop episodes
(same initial states/references/disturbance sequences, generated once,
independent of method and seed); normalization statistics fit on the
training set only; early stopping uses unweighted validation loss for all
methods; no extra noise added to near-constraint region during data
generation; margin-weighted and random-weight weight *distributions* are
asserted identical (only the sample-to-weight assignment differs).

## Judgment rule

Implements the task's section-8 rule in `run_experiment.py::judge()`:
5 checks (near-RMSE improves every seed; overall RMSE doesn't degrade
>5%; at least one closed-loop metric improves; random-weight does NOT
match margin-weighted's near-RMSE improvement; consistency across seeds)
-> `supported` (5/5) / `mixed` (3-4/5) / `not_supported` (<3/5, enough
seeds) / `inconclusive` (too few seeds, e.g. `--quick` mode).
