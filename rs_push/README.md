# rs_push — Gate A: does replanning rescue model error in pushing?

Order:
```bash
python gate_a.py --selftest        # no robosuite needed
python collect_train.py --quick    # data + FROZEN model checkpoint
python collect_train.py            # FULL registered budget for the formal ckpt
python gate_a.py --pilot           # ladder scan, maximal-hold only, disjoint seeds
python gate_a.py --quick           # 12 paired episodes
python gate_a.py --difficulty D    # 60 paired episodes — the verdict that counts
```
Pre-registered criteria are printed by the script and stored in verdict.json.
Periods {1,2,4,8,H=12}; maximal-hold = resolve every H steps (legal periodic MPC).
Same frozen model (eval only), same MPPI budget, same paired episode specs
(friction/mass/zone recorded but never given to model or planner); MPPI RNG
seeded by (episode_id, solve_idx). Violations measured as positive clearance
rho = ||obj-zone|| - (r_zone+r_obj) each control step; per-step object
displacement bound is logged to justify no substep tunneling (cap 7.5mm << 5cm).
Low-base-rate rule, paired bootstrap CI, and task-success guard are built in.
Gate A' (history-input retrain) is the single pre-registered branch on failure.
