# Stage 2: Constraint-Margin-Weighted Dynamics Learning -- Report

**Mode:** quick smoke test  |  **Seeds:** [101]  |  **Runtime:** 4.3s

## Data

- 1200 transitions from 60 trajectories (train/val/test = 960/120/120)

- near-constraint fraction (margin<0.4): overall 22.5%, train 18.6%, val 42.5%, test 33.3%

## Offline prediction results (mean +/- std across seeds)


| method          |   ('overall_rmse', 'mean') |   ('overall_rmse', 'std') |   ('near_rmse', 'mean') |   ('near_rmse', 'std') |   ('far_rmse', 'mean') |   ('far_rmse', 'std') |   ('rollout15_rmse', 'mean') |   ('rollout15_rmse', 'std') |   ('rollout15_near_rmse', 'mean') |   ('rollout15_near_rmse', 'std') |
|:----------------|---------------------------:|--------------------------:|------------------------:|-----------------------:|-----------------------:|----------------------:|-----------------------------:|----------------------------:|----------------------------------:|---------------------------------:|
| baseline        |                  0.0731831 |                       nan |                0.110678 |                    nan |              0.0436291 |                   nan |                     0.342768 |                         nan |                          0.253431 |                              nan |
| margin_weighted |                  0.0735924 |                       nan |                0.109633 |                    nan |              0.0459244 |                   nan |                     0.349174 |                         nan |                          0.23292  |                              nan |
| random_weight   |                  0.0733002 |                       nan |                0.110277 |                    nan |              0.0444294 |                   nan |                     0.344112 |                         nan |                          0.243819 |                              nan |



## Closed-loop MPC results (mean +/- std across seeds)


| method          |   ('tracking_rmse', 'mean') |   ('tracking_rmse', 'std') |   ('cumulative_objective', 'mean') |   ('cumulative_objective', 'std') |   ('violation_rate', 'mean') |   ('violation_rate', 'std') |   ('max_violation', 'mean') |   ('max_violation', 'std') |   ('infeasibility_rate', 'mean') |   ('infeasibility_rate', 'std') |   ('saturation_rate', 'mean') |   ('saturation_rate', 'std') |
|:----------------|----------------------------:|---------------------------:|-----------------------------------:|----------------------------------:|-----------------------------:|----------------------------:|----------------------------:|---------------------------:|---------------------------------:|--------------------------------:|------------------------------:|-----------------------------:|
| baseline        |                     1.26275 |                        nan |                            248.078 |                               nan |                            0 |                         nan |                           0 |                        nan |                                0 |                             nan |                         0.825 |                          nan |
| margin_weighted |                     1.26399 |                        nan |                            248.13  |                               nan |                            0 |                         nan |                           0 |                        nan |                                0 |                             nan |                         0.75  |                          nan |
| random_weight   |                     1.26309 |                        nan |                            248.089 |                               nan |                            0 |                         nan |                           0 |                        nan |                                0 |                             nan |                         0.8   |                          nan |



## Paired statistical comparison (margin_weighted vs baseline / random_weight)


```json
{
  "margin_weighted_vs_baseline": {
    "violation_rate": {
      "mean_diff": 0.0,
      "ci_low": 0.0,
      "ci_high": 0.0,
      "relative_improvement_pct": NaN,
      "wilcoxon_stat": 0.0,
      "wilcoxon_p": 1.0,
      "n_pairs": 4
    },
    "cumulative_objective": {
      "mean_diff": 0.05255574905187199,
      "ci_low": 0.0,
      "ci_high": 0.15766724715561597,
      "relative_improvement_pct": -0.02118521280992315,
      "wilcoxon_stat": 0.0,
      "wilcoxon_p": 1.0,
      "n_pairs": 4
    },
    "tracking_rmse": {
      "mean_diff": 0.001240737010548873,
      "ci_low": 0.0,
      "ci_high": 0.003722211031646619,
      "relative_improvement_pct": -0.09825652497126314,
      "wilcoxon_stat": 0.0,
      "wilcoxon_p": 1.0,
      "n_pairs": 4
    },
    "max_violation": {
      "mean_diff": 0.0,
      "ci_low": 0.0,
      "ci_high": 0.0,
      "relative_improvement_pct": NaN,
      "wilcoxon_stat": 0.0,
      "wilcoxon_p": 1.0,
      "n_pairs": 4
    }
  },
  "margin_weighted_vs_random_weight": {
    "violation_rate": {
      "mean_diff": 0.0,
      "ci_low": 0.0,
      "ci_high": 0.0,
      "relative_improvement_pct": NaN,
      "wilcoxon_stat": 0.0,
      "wilcoxon_p": 1.0,
      "n_pairs": 4
    },
    "cumulative_objective": {
      "mean_diff": 0.04126304499312505,
      "ci_low": 0.0,
      "ci_high": 0.12378913497937516,
      "relative_improvement_pct": -0.016632368753784554,
      "wilcoxon_stat": 0.0,
      "wilcoxon_p": 1.0,
      "n_pairs": 4
    },
    "tracking_rmse": {
      "mean_diff": 0.0009055318223080006,
      "ci_low": 0.0,
      "ci_high": 0.0027165954669240017,
      "relative_improvement_pct": -0.07169190315101781,
      "wilcoxon_stat": 0.0,
      "wilcoxon_p": 1.0,
      "n_pairs": 4
    },
    "max_violation": {
      "mean_diff": 0.0,
      "ci_low": 0.0,
      "ci_high": 0.0,
      "relative_improvement_pct": NaN,
      "wilcoxon_stat": 0.0,
      "wilcoxon_p": 1.0,
      "n_pairs": 4
    }
  }
}
```


## Judgment (section 8 decision rule)


- near_rmse_improved_all_seeds: PASS

- overall_rmse_not_worse_5pct: PASS

- at_least_one_closedloop_metric_improved: FAIL

- random_weight_does_not_match_margin: FAIL

- consistent_across_seeds: PASS


**3/5 checks passed.**


### Verdict: `inconclusive`


## Objective diagnosis (if not fully supported)

Candidate explanations (see task section 8 categories) to check against the numbers above:

- baseline near-zero error? compare baseline near_rmse to the overall scale of dp/dv

- near-constraint sample sufficiency: see data near-fraction above

- prediction improved but control did not: compare near_rmse vs violation_rate rows

- margin not decision-aware enough: compare margin_weighted vs random_weight rows

- insufficient constraint activation in closed loop: check violation_rate for baseline

- variance too large: check std columns above relative to the mean differences


## Known simplifications / deviations from the literal spec

- Closed-loop MPC with a nonlinear learned model is implemented via per-step linearization of the NN (frozen across the horizon for that solve), reusing the stage-1 QP structure with A,B,c promoted to cp.Parameter -- not full nonlinear MPC or solver differentiation, consistent with the 'no solver differentiation, no complex network' instruction.

- Stage-1 true dynamics were fully linear, so the specified nonlinearity was added for stage 2's true plant only (see generate_data.py); stage-1 results/config files were left untouched.

- Near-constraint coverage (~20-30%) was reached via uniform initial-position sampling plus natural wall-bounce dynamics -- NOT via extra noise or region-specific treatment, per the fairness requirement.
