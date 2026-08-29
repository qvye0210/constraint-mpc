# rs_carry — carry-unknown-payload task (robosuite 1.4.1, UR5e, JOINT_VELOCITY)

Task: scripted grasp of a cube whose mass is randomised per episode (0.1–1.0 kg,
never observed by any model), then a recorded transport phase along a corridor
passing a virtual sphere obstacle. Payload acts through CONTACT, so the
controller's gravity compensation does not cancel it — the effect stays in the
residual, same as on the real robot (speedj equivalent).

## Order
```bash
python smoke_test.py            # FIRST — validates every API assumption
python diag_step1_gap.py --quick
python diag_step2_compound.py --quick
# full versions without --quick before trusting a PASS
```

## The two gates (pre-registered)
| gate | question | PASS |
|---|---|---|
| step 1 | residual genuinely hard to learn, for the right reason? | gap>5 AND traj/trans<5 |
| step 2 | does rollout compound constraint error systematically? | corr>0.2 AND err(10)/err(1)>3 |

Method work (direct constraint predictor + MPC) starts only after both PASS.
Each gate allows ONE knob adjustment on failure (step1: payload/exploration or
target spread; step2: none — a fail is a fail). No further iteration.

## Notes
- Nominal model = ridge-identified linear model on the train split; the learning
  target is the残 nonlinear part. `linear_explained` is printed as a guard: if it
  is ~99% the free-space trap has returned.
- Author could not execute robosuite 1.4.1 end-to-end (needs Python<=3.11);
  smoke_test.py is the contract. Diagnostics logic was tested on synthetic data.
