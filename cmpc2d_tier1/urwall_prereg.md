# Pre-registration: UR5e wall-following feasibility gate

**Status.** 2D point-mass studies are closed with two retained findings:
(i) calibrated constraint-normal error bounds are ~50% narrower than
Euclidean bounds at matched coverage; (ii) this did not convert into
longer safe plan-reuse prefixes, because model error (1e-4 scale) is
negligible against per-step margin dynamics (1e-1 scale). The 2D testbed
is out of regime; no further 2D variants (noise, wall spacing, weakened
models) will be run.

**Conditional hypothesis under test.** Constraint-normal calibration
yields practical plan-reuse benefit in high-dimensional systems, where
aggregate Euclidean error contains many constraint-irrelevant directions
(expected width dilution ~sqrt(12) vs ~sqrt(2) in 2D).

**Task (frozen).** UR5e in MuJoCo, free space, no contact. End-effector
tracks a reference parallel to a planar wall to a goal; safety margin
rho = wall-normal clearance of the end-effector (and elbow link), rho>0
safe, grad rho constant (L=1). Wall offset fixed before any result at
0.05 m from the reference line. near = rho <= 0.5 x wall offset.

**Randomisation (frozen).** Per-episode wrist payload ~ U(0.1, 1.0) kg,
fixed within an episode, NOT observed by the model (matches handling of
objects of unknown mass). Optional actuator-gain jitter +/-10%. No other
error sources may be added later.

**Models (all three trained; best on design set carries forward).**
1. Nominal analytic/identified model; 2. ridge-regression residual;
3. MSE MLP residual [256, 256, 128], inputs (q, qdot, u) 18-d, outputs
12-d, 3 seeds. Choosing a weaker model to inflate error is prohibited.
Diagnostic oracle arm: identical MLP with payload observed as input
(scopes the claim: unknown-load specificity vs generic 6-DoF benefit).

**Splits.** Whole episodes; design / calibration / test collected with
disjoint seeds; test read once, after all thresholds are frozen.

**Feasibility gates (any failure stops the direction; no parameter may
be adjusted to rescue a gate).**
1. Regime: R = q95( max_{k<=10} [rho_hat_k - rho_k]_+ ) / median(rho |
   near), computed on design with the best model. R < 0.1: the 99.4%-
   linearity red flag has returned — stop. R > 1: model too poor,
   prefixes degenerate — stop. Pass iff 0.1 <= R <= 1.
2. Every-step MPC success rate >= 90% (design).
3. Solver failure rate <= 2% (all emergency/retry solves counted).
4. Near-boundary decisions >= 20% of all decisions.
5. Decision disagreement: margin trigger and the strongest Cartesian
   trigger select different h in >= 5% of decisions. Below 5%: stop —
   the comparison cannot resolve the hypothesis on this platform.

**If all gates pass: main experiment (pre-registered here).** Four
calibrated triggers under one online rule (accept h iff rho_hat_k >
frozen stratified q95 for all k <= h): joint-space Euclidean;
per-link/EEF Cartesian Euclidean; first-order normal (grad-rho^T e);
exact margin [rho_hat - rho]_+. Pass requires, on independent test
episodes: safety coverage not below the Cartesian baseline; task success
drop <= 5 pp; margin-trigger mean prefix >= 1.2 x Cartesian, or >= 15%
fewer total solver calls at matched safety. The decisive comparison is
exact margin vs per-link Cartesian.

**Reporting.** Negative outcomes at any gate are reported as such,
including the 2D closure above. British English throughout.
