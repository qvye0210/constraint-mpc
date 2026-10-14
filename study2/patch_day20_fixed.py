"""day20: fixed-payload mode and loading of day27 models.  PAYLOAD=m,c forces every episode's plant to that payload;
arms mse27/shrink27/cerr27/ours27 load results/day27/model_<name>.npz (piecewise Kb_seg, Ka)."""
d = open("day20_run.py").read()
if "PAYLOAD" in d: raise SystemExit("already patched")
old = '        dr = d17.draw(seed); plant = d17.make_model(seed, dr["m"], dr["c"])'
new = ('        dr = d17.draw(seed)\n'
       '        if os.environ.get("PAYLOAD"):                                   # fixed unknown payload protocol (day27)\n'
       '            _m, _c = (float(v) for v in os.environ["PAYLOAD"].split(",")); dr = dict(seed=seed, m=_m, c=_c)\n'
       '        plant = d17.make_model(seed, dr["m"], dr["c"])')
assert old in d, "plant anchor"; d = d.replace(old, new, 1)
old = '            mse = dict(np.load(f"results/day23_mse/{name}/mse_model.npz")) if "mse" in ARMS else None'
new = old + '\n            extra27 = {a: dict(np.load(f"results/day27/model_{a[:-2]}.npz")) for a in ARMS if a.endswith("27")}'
assert old in d, "loader anchor"; d = d.replace(old, new, 1)
old = '                        "mse_trueW": dict(mse, true=plant, swap="w") if mse else None, **extra}[arm]'
new = '                        "mse_trueW": dict(mse, true=plant, swap="w") if mse else None, **extra, **extra27}[arm]'
assert old in d, "dict anchor"; d = d.replace(old, new, 1)
open("day20_run.py", "w").write(d); print("day20 patched")
