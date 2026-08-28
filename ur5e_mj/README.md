# UR5e (MuJoCo) testbed — constraint-aware dynamics learning

Replaces the planar testbed, which was severely over-parameterised: with a fixed
architecture the test error fell monotonically from 7.8e-5 to 1.3e-5 and never
bottomed out. With no scarce capacity there is nothing to re-allocate, so every
weighting experiment run there measured the wrong thing.

Random noise does not fix that — unbiased noise leaves the conditional mean
unchanged, so weighted and unweighted losses share an optimum. What creates real
competition is a large amount of learnable-but-not-fully-learnable structure,
which full rigid-body dynamics with friction and an unobserved payload provides.

## Setup

```bash
pip install mujoco torch numpy
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git menagerie
# or: export UR5E_XML=/path/to/scene.xml
```

Uses the official DeepMind Menagerie UR5e model. Kinematics and the constraint
Jacobian come from MuJoCo (`mj_jacSite`), not from a hand-written DH table — the
analytic table disagreed with the official model by ~1 mm in position and 1.2e-3
in the Jacobian.

## Interface

Mirrors the real UR5e: input is a joint VELOCITY command, and the model's own PD
position actuator acts as the internal servo, driven by a target that advances at
the commanded velocity (the servoj construction). Writing a separate torque servo
on top fights that actuator and diverges. Verified: commanding 0.3 rad/s on joint
1 gives 0.298.

```
state x = (q, qd) in R^12 ,  input u = qd_cmd in R^6 ,  dt = 0.02 s
```

## Run order

```bash
python capacity_check.py --n-traj 150 --seeds 3      # Step 0: regime check
python compare_arms.py   --n-traj 150 --seeds 3      # Step 1: four arms
```

**Step 0 must pass before Step 1 means anything.** It checks that test error
bottoms out while train error keeps falling. Measured here: floor ratio 0.85,
train/test gap 8.5x — the limit comes from limited DATA, not from an artificially
small network, which is both the real robot's condition and much harder to
dismiss with "just train longer".

## Arms

| arm | what it isolates |
|---|---|
| `uniform` | plain MSE baseline |
| `mask` | irrelevant dimensions zeroed, rest **uniform** — "stop spending capacity where the constraint cannot reach", with no directional structure |
| `prop` | propagated constraint-gradient metric `M = Σ γ^{k-1} c_k c_kᵀ` |
| `random` | same spectrum as `prop`, random orientation — without it a positive result cannot be attributed to direction rather than to anisotropy |

**The comparison that decides the contribution is `prop` vs `mask`**, not `prop`
vs `uniform`. Both stop spending capacity where the constraint cannot reach; only
`prop` also allocates by direction. If `prop` is not clearly better than `mask`,
the gain is masking alone — VaGraM's established mechanism with `grad g` swapped
in for `grad V`.

## Pre-registered decision rule

Written before running, to stop the pattern of inventing a new explanation after
each negative result:

> `prop` must beat `uniform` by >5% AND beat `mask` by >5% on near-constraint
> directional error, at every training budget. If any fails, the directional part
> is judged ineffective on this system: report the scope result (masking works,
> direction does not) with the mechanism analysis, and do not add a new auxiliary
> explanation.

## Two implementation points that cost weeks on the planar testbed

1. **Weight scale normalisation.** Weights are normalised to mean trace 1 so arms
   share a loss scale. Without it, "better allocation" is confounded with
   "different effective learning rate".
2. **Block-wise rank protection.** A single global floor `eps*trace(M)/NX*I` is
   uniform across dimensions while the structural weight is not; it exceeded the
   velocity-block structure by ~9x and flattened the rank-1 direction there —
   exactly where the residual lives. The floor is now scaled to each block's own
   trace.

## Constraint coverage

`coverage()` must be checked after any change to the start distribution. With the
obstacle at (0.45, 0.10, 0.35) the margin never fell below 0.106 and 100% of
samples were "far", making every constraint-related number meaningless.
`DEFAULT_OBS` is now the centre of the reachable TCP cloud; at `T=80,
attract=0.9` this gives ~18% near and ~28% violating.

## Smoke-test status

Both scripts run end to end. A 1-seed, 40-trajectory, 200-epoch smoke run gave
`mask` −13.4% and `prop` +124% against uniform, i.e. the same ordering as the
planar testbed. That is one seed at the smallest budget and is not a result —
run the full configuration before drawing any conclusion.
