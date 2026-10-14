"""day20: accept piecewise-Kb models (Kb_seg chosen by progress) so day25 cells can be run in closed loop.
Arm name = the model tag (e.g. directed_b20_n1); loaded from results/day25/<path>/model_<tag>.npz."""
d = open("day20_run.py").read()
if "Kb_seg" in d: raise SystemExit("already patched")
old = ('        if isinstance(mdl, dict) and "Ka" in mdl:          # learned linear residual, frozen at the plan like the oracle\n'
       '            E, w = np.eye(M) + mdl["Kb"], mdl["Ka"] @ np.r_[qbar_pos[k], W["xbar"][k, M:]]')
new = ('        if isinstance(mdl, dict) and "Kb_seg" in mdl:       # piecewise input gain (day25), segment by plan progress\n'
       '            _s = min(mdl["Kb_seg"].shape[0] - 1, int(d12.TASK.sigma(fk(qbar_pos[k])[1]) * mdl["Kb_seg"].shape[0]))\n'
       '            E, w = np.eye(M) + mdl["Kb_seg"][_s], mdl["Ka"] @ np.r_[qbar_pos[k], W["xbar"][k, M:]]\n'
       '        elif isinstance(mdl, dict) and "Ka" in mdl:        # learned linear residual, frozen at the plan like the oracle\n'
       '            E, w = np.eye(M) + mdl["Kb"], mdl["Ka"] @ np.r_[qbar_pos[k], W["xbar"][k, M:]]')
assert old in d, "fill anchor"; d = d.replace(old, new, 1)
old = '            extra = {a: dict(np.load(f"results/day24/{name}/model_{a}.npz")) for a in ARMS if a in ("mse1", "mseN", "cerr", "mse1_cl")}'
new = old + '\n            extra.update({a: dict(np.load(f"results/day25/{name}/model_{a}.npz")) for a in ARMS if a.startswith(("random_", "directed_"))})'
assert old in d, "loader anchor"; d = d.replace(old, new, 1)
open("day20_run.py", "w").write(d); print("day20 patched")
