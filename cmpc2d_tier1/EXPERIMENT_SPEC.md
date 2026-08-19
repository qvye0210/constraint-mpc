# Corrected experiment specification — constraint-gradient weighting

Status: **spec only, not implemented.** Written after reading VaGraM (Voelcker et
al., ICLR 2022, arXiv:2204.01464) and finding that the earlier runs did not test
the mechanism they claim.

---

## Why the previous runs do not count

`distract_check.py` up-weighted the constraint-normal component but left every
other output dimension at weight 1:

```python
rest = (err[:, 2:] ** 2).sum(-1)        # velocity AND all D distractor dims, weight 1
loss = (w_normal * en**2 + et**2 + rest).mean()
```

VaGraM works because the value gradient w.r.t. irrelevant dimensions goes to
zero, so those dimensions stop consuming capacity. We never gave the distractors
a low weight. With D=16 the loss contained sixteen unit-weight distractor terms
that kept dominating capacity allocation regardless of `w_normal`.

Under the constraint geometry the correct weight for a distractor dimension is
exactly zero, since dg/dz = 0. We used 1.

**Consequence:** the conclusion "capacity scarcity is not sufficient, the
residual must align with the constraint geometry" is an artefact of this
implementation flaw, not a finding. It must not be carried into the paper.

---

## Required changes

### 1. Propagated constraint-gradient weighting (the actual method)

Weight the error by the constraint gradient propagated along the horizon:

```
c_k = (A^{k-1})^T * grad g(x_{t+k})
M   = sum_k gamma^{k-1} * c_k c_k^T
L   = e^T M e        (+ the L_dyn floor below)
```

Distractor dimensions get exactly zero weight at every k. Velocity gets non-zero
weight only through propagation (dg/dv = 0 at k=0), which is why the static k=0
form was degenerate.

### 2. Rank-deficiency protection

`c_k c_k^T` is rank-1 per k; the tangential direction lies in its nullspace and
is left completely unconstrained, so the model can drift arbitrarily there.
VaGraM hits the same problem and solves it with a Cauchy-Schwarz upper bound
that yields a positive-definite diagonal matrix (their Sec. 3.2). Keep `L_dyn`
or add `eps * I`, and justify the choice in the paper citing their analysis.

### 3. Weight scale normalisation — the biggest uncontrolled variable so far

`w_normal = 50` changed the total loss scale as well as its shape, and Adam is
sensitive to that. We cannot currently separate "better allocation" from
"different effective learning rate", and this contaminates every earlier result
including the −27 % that looked promising.

Normalise weights to batch mean 1 so that uniform and weighted arms have
matched loss magnitude.

### 4. Random-anisotropic control arm

A weight matrix with the **same spectrum** as `M` but **random orientation**.

This answers the attribution question that nothing else can: is the gain from
the constraint direction, or merely from the loss being anisotropic? Without
this arm a positive result cannot be attributed. Cheap; mandatory.

### 5. Weight-distribution analysis

The task originally assigned by the supervisor, now answerable:

- distribution of `w` across the dataset
- weight on distractor dimensions (must be ~0 — also serves as an
  implementation check; non-zero means a bug)
- normal/tangential weight ratio as a function of `k`
- how the ratio tracks the measured `sin(theta_k)` decay

### 6. Compare at matched training loss, not matched epochs

Different losses converge at different rates, so fixed-epoch comparison reads
"converges faster" as "performs better" — exactly the trap `capacity_check.py
--part c` exposed. Report the full budget curve; where possible compare at
matched training-loss levels.

---

## Pre-registered decision rule

Written before running, to stop the pattern of generating a new auxiliary
explanation after each negative result:

> With D in {0, 8}, 2000 epochs, normalised weights, and the random-anisotropic
> control included: if propagated constraint-gradient weighting improves the
> proxy by less than 2 % over uniform, **or** is not clearly better than the
> random-anisotropic control, then constraint-gradient weighting is judged
> ineffective on this class of system. No further auxiliary explanation is to be
> introduced; the work converts to a scope-characterisation paper.

---

## Carried-forward results that remain valid

| result | status |
|---|---|
| `sin(theta_k)` relevance-decay law (0.92–0.96 agreement) | solid, measured, independent of the flaw |
| Gate DIR sign asymmetry (optimistic vs pessimistic error) | solid |
| Gate B attribution (true 8.2e-10 vs learned 0.0132) | solid |
| Part C budget decay | valid as measured, but see scope note below |
| "capacity scarcity is not sufficient" | **retracted** — implementation flaw |

**Scope note on Part C.** VaGraM operates in *online* MBRL where the model is
retrained every round with a limited gradient budget, so under-convergence is
its normal operating condition and an optimisation-speed gain is a real
practical benefit there. This project trains offline once and deploys into MPC,
where training to convergence is normal. Part C is therefore a statement about
transferring these losses to the offline-MPC setting, not a challenge to their
result. Frame it that way or not at all.
