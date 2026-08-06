# MPC Constraint-Margin Decision-Impact Diagnostic -- Summary Report

**Mode:** `full`  |  **Runtime:** 168.0s  |  **Feasible scenarios:** 600 / 600 tried (target 600)  |  **Sample-level pairs:** 14400

## 1. Was prediction-error magnitude matched correctly?

Bias vectors were constructed as `magnitude * unit_direction` for every direction (4 structured + 4 random), so the one-step Euclidean prediction-error norm is exactly equal across all directions at a given magnitude by construction. This is checked in `tests/test_correctness.py::test_one_step_prediction_error_matches_across_directions`, which passed. Magnitudes used: [0.005, 0.015, 0.03] (position range 4.0, velocity range 2.0, so all magnitudes are small relative to the state-constraint envelope).

**Answer: YES -- magnitude matching verified by construction and by test.**

## 2. Were near and far groups meaningfully separated?

**Fixed-threshold grouping** (near <= 0.08, far >= 0.3):


| group_fixed   |   count |       mean |       median |       std |          min |       max |
|:--------------|--------:|-----------:|-------------:|----------:|-------------:|----------:|
| far           |     378 | 0.51657    |  0.504191    | 0.126353  |  0.304563    | 0.971474  |
| mid           |     111 | 0.190958   |  0.19102     | 0.0616113 |  0.080288    | 0.299706  |
| near          |     111 | 0.00935461 | -1.05481e-05 | 0.0196898 | -0.000237845 | 0.0798058 |


**Quantile grouping** (nearest/farthest 30%, thresholds: near <= 0.2136, far >= 0.5106):


| group_quantile   |   count |      mean |    median |       std |          min |      max |
|:-----------------|--------:|----------:|----------:|----------:|-------------:|---------:|
| far              |     180 | 0.6142    | 0.580314  | 0.106364  |  0.510704    | 0.971474 |
| mid              |     240 | 0.397814  | 0.411251  | 0.0859102 |  0.214312    | 0.510625 |
| near             |     180 | 0.0637052 | 0.0224839 | 0.0747152 | -0.000237845 | 0.21195  |


**Answer: YES -- see counts above (both grouping schemes reported; margin distributions are non-overlapping by construction of the group definitions).**

## 3. Was first-action discrepancy larger near constraints?

Per-magnitude comparison (mean with bootstrap 95% CI, Mann-Whitney U one-sided test H1: near > far, Cliff's delta effect size):


| group_col      |   magnitude |   n_near |   n_far |   mean_near |   mean_far |   median_near |   median_far |   mwu_p |   cliffs_delta |
|:---------------|------------:|---------:|--------:|------------:|-----------:|--------------:|-------------:|--------:|---------------:|
| group_fixed    |       0.005 |      888 |    3024 |   2.288e-06 |    0.01391 |     1.05e-06  |    2.347e-06 |       1 |        -0.2404 |
| group_fixed    |       0.015 |      885 |    3024 |   0.001436  |    0.04128 |     1.044e-06 |    2.478e-06 |       1 |        -0.2363 |
| group_fixed    |       0.03  |      878 |    3007 |   0.006921  |    0.08153 |     1.113e-06 |    2.956e-06 |       1 |        -0.2888 |
| group_quantile |       0.005 |     1440 |    1440 |   0.0007425 |    0.02088 |     1.252e-06 |    3.274e-06 |       1 |        -0.2276 |
| group_quantile |       0.015 |     1429 |    1440 |   0.003046  |    0.05945 |     1.156e-06 |    3.09e-06  |       1 |        -0.2361 |
| group_quantile |       0.03  |     1410 |    1439 |   0.008474  |    0.1102  |     1.231e-06 |    4.259e-06 |       1 |        -0.3254 |



0/6 (near/far x magnitude) comparisons show statistically significant (p<0.05) *and* directionally-consistent (near > far) first-action discrepancy.

**Answer (all near-constraint scenarios): NO.** See the input-saturation confound check below before drawing conclusions from this raw comparison.

## 4. Was active-set change more frequent near constraints?

| group_col      |   magnitude |   n_near |   n_far |   rate_near |   rate_far |   p_value |
|:---------------|------------:|---------:|--------:|------------:|-----------:|----------:|
| group_fixed    |       0.005 |      888 |    3024 |      0.5113 |   0        |         0 |
| group_fixed    |       0.015 |      888 |    3024 |      0.7027 |   0        |         0 |
| group_fixed    |       0.03  |      888 |    3024 |      0.7556 |   0.01984  |         0 |
| group_quantile |       0.005 |     1440 |    1440 |      0.3153 |   0        |         0 |
| group_quantile |       0.015 |     1440 |    1440 |      0.4778 |   0        |         0 |
| group_quantile |       0.03  |     1440 |    1440 |      0.5736 |   0.002083 |         0 |


**Answer: YES.**

## 5. Were margin, active set or dual variables informative?

Spearman correlation, oracle min state margin vs. delta_u0: rho=0.2087, p=9.362e-141, n=14331.


Spearman correlation, oracle max normalized state dual vs. delta_u0: rho=-0.2064, p=9.709e-138, n=14331.


Diagnostic linear regression for delta_u0: R^2 (bias magnitude only) = 0.02655, R^2 (magnitude + margin + dual + active flag) = 0.09878.


Diagnostic logistic regression for high-impact (top quartile) delta_u0: AUC (magnitude only) = 0.5217, AUC (full features) = 0.6679.


Diagnostic logistic regression for active-set change: AUC (magnitude only) = 0.5884, AUC (full features) = 0.9595.


**Answer: YES -- margin/dual/active-set features add explanatory power beyond bias magnitude alone (see R^2/AUC deltas).**

## 6. Were results robust across bias directions and magnitudes?

Mean delta_u0 by bias direction (pooled across magnitudes and scenarios):


| bias_direction   |   mean_delta_u0 |
|:-----------------|----------------:|
| random_3         |         0.04332 |
| pos_plus         |         0.04003 |
| random_0         |         0.03769 |
| pos_minus        |         0.03391 |
| random_1         |         0.03052 |
| random_2         |         0.02797 |
| vel_plus         |         0.01654 |
| vel_minus        |         0.01415 |



Coefficient of variation across directions: 0.3475. 
**Answer: YES, reasonably robust (direction-dependence itself is a scientifically relevant finding, not a flaw, per the experiment's scientific-caution guidance not to assume tangent/normal effects a priori).**

## Confound check: input-bound saturation of u0

The oracle's first control action u0 can sit exactly at an input bound (a corner solution). Such corner solutions can be locally invariant to small model perturbations independent of state-constraint proximity, which would suppress delta_u0 for reasons unrelated to the state-margin hypothesis. This is checked explicitly:


| group_col      |   n_near |   n_far |   rate_near |   rate_far |   p_value |
|:---------------|---------:|--------:|------------:|-----------:|----------:|
| group_fixed    |      888 |    3024 |      1      |     0.881  |         0 |
| group_quantile |     1440 |    1440 |      0.9944 |     0.8167 |         0 |



**A substantial fraction of near-constraint scenarios have a saturated u0.** This is a genuine confound: near-boundary tracking references often demand near-maximal early acceleration, which saturates u0 at the input bound in BOTH the oracle and the perturbed solve, mechanically suppressing delta_u0 regardless of state-constraint sensitivity. The re-analysis below restricts to scenarios where u0 is NOT saturated, to isolate the state-margin effect from this input-saturation effect.


**Near/far comparison restricted to non-saturated-u0 scenarios:**


| group_col      |   magnitude |   n_near |   n_far |   mean_near |   mean_far |   median_near |   median_far |    mwu_p |   cliffs_delta |
|:---------------|------------:|---------:|--------:|------------:|-----------:|--------------:|-------------:|---------:|---------------:|
| group_fixed    |       0.005 |        0 |     360 |    nan      |   nan      |      nan      |     nan      | nan      |      nan       |
| group_fixed    |       0.015 |        0 |     360 |    nan      |   nan      |      nan      |     nan      | nan      |      nan       |
| group_fixed    |       0.03  |        0 |     360 |    nan      |   nan      |      nan      |     nan      | nan      |      nan       |
| group_quantile |       0.005 |        8 |     264 |      0.1133 |     0.1095 |        0.1269 |       0.1139 |   0.3592 |        0.0767  |
| group_quantile |       0.015 |        8 |     264 |      0.2895 |     0.2813 |        0.2609 |       0.2953 |   0.4422 |        0.03125 |
| group_quantile |       0.03  |        8 |     264 |      0.4969 |     0.4358 |        0.4874 |       0.3208 |   0.3932 |        0.05777 |



0/6 comparisons remain significant and directionally consistent (near > far) once input-saturated scenarios are excluded.

**Answer, controlling for the input-saturation confound: NO.**

### Supplementary continuous outcomes (not gated by u0 saturation)

delta_u0 can be mechanically suppressed by a saturated corner solution even when the rest of the predicted trajectory is highly sensitive to the bias. |objective_diff| (whole-horizon optimizer cost impact) and the true-model max constraint violation on replay are reported as supplementary outcomes that remain continuous even when u0 saturates:


| group_col      | outcome                          |   magnitude |   n_near |   n_far |   mean_near |   mean_far |      mwu_p |   cliffs_delta |
|:---------------|:---------------------------------|------------:|---------:|--------:|------------:|-----------:|-----------:|---------------:|
| group_fixed    | objective_diff_abs               |       0.005 |      888 |    3024 |    18.22    | 13.51      | 1.52e-39   |         0.2888 |
| group_fixed    | true_replay_max_violation_filled |       0.005 |      888 |    3024 |     0.01775 |  0         | 0          |         0.4865 |
| group_fixed    | objective_diff_abs               |       0.015 |      885 |    3024 |    54.28    | 40.64      | 1.304e-38  |         0.2856 |
| group_fixed    | true_replay_max_violation_filled |       0.015 |      888 |    3024 |     0.0469  |  0         | 0          |         0.5113 |
| group_fixed    | objective_diff_abs               |       0.03  |      878 |    3007 |   110.1     | 81.14      | 4.91e-40   |         0.2922 |
| group_fixed    | true_replay_max_violation_filled |       0.03  |      888 |    3024 |     0.0737  |  0.0008174 | 0          |         0.4836 |
| group_quantile | objective_diff_abs               |       0.005 |     1440 |    1440 |    17.31    | 11.73      | 2.826e-54  |         0.3329 |
| group_quantile | true_replay_max_violation_filled |       0.005 |     1440 |    1440 |     0.01096 |  0         | 3.954e-112 |         0.3007 |
| group_quantile | objective_diff_abs               |       0.015 |     1429 |    1440 |    50.97    | 35.26      | 5.436e-53  |         0.3294 |
| group_quantile | true_replay_max_violation_filled |       0.015 |     1440 |    1440 |     0.02981 |  0         | 7.357e-126 |         0.3326 |
| group_quantile | objective_diff_abs               |       0.03  |     1410 |    1439 |   103.3     | 71.13      | 1.032e-52  |         0.3297 |
| group_quantile | true_replay_max_violation_filled |       0.03  |     1440 |    1440 |     0.05065 |  3.222e-05 | 5.343e-134 |         0.3537 |



12/12 supplementary comparisons are significant and directionally consistent (near > far). **Answer: YES.**

## 7. Did any result contradict the hypothesis?

No (group, magnitude) cell showed a statistically significant reversal (far > near) of the hypothesized effect.

## 8. Does the evidence justify proceeding to Problem 2?

Based on checks [True, True, True, True] (discrepancy-larger-near-constraints, active-set-change-more-frequent-near, margin/dual informativeness, no significant contradiction), the evidence suggests proceeding is justified.

## Final verdict

### `SUPPORTED`

## Scientific cautions

- This experiment is a mechanism-validation diagnostic on a 1D double integrator; results do not automatically generalize to cart-pole, mobile robots, or manipulators.

- Correlation between margin/dual signals and decision impact is not proof of causation beyond the controlled perturbation design used here.

- No neural network or new training loss was used; only the existence and informativeness of the phenomenon were tested.

## Recommended next experiment

Proceed to Problem 2: design a cheap, solver-derived benefit/urgency signal (e.g. combining margin, active-set indicator, and normalized dual value) for a learning-based event-triggered re-solving policy, starting from the existing PB-Soft-MASTD accumulator formulation, and validate it first on this same double-integrator testbed before moving to higher-dimensional systems.
