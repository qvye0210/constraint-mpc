"""Piecewise task-space metric: instead of one constant Q from the path midpoint, split the reference path into
TS_SEGS segments, build one controller per segment from that segment's midpoint configuration, and let the driver
pick the controller by the current progress. Standard successive-linearisation practice; the only new number is
TS_SEGS, fixed before running."""
import re, sys

# ---------------------------------------------------------------- day17: allow an explicit metric config
s = open("day17_oracle.py").read()
if "TS_QREP_OVERRIDE" in s:
    sys.exit("day17 already patched")
old = "            _LJ = taskspace_LJ(np.array([TS_QREP]))[0]"
new = "            _LJ = taskspace_LJ(np.array([TS_QREP]))[0]        # TS_QREP is set per segment by the driver"
assert old in s, "metric anchor not found"
s = s.replace(old, new, 1)
s = s.replace("TS_QREP = None          # representative reference configuration, set by build_ref",
              "TS_QREP = None          # representative reference configuration, set by the driver (per segment)\n"
              "TS_QREP_OVERRIDE = True # marker: the driver owns TS_QREP", 1)
open("day17_oracle.py", "w").write(s)
print("day17 patched")

# ---------------------------------------------------------------- day20: per-segment controllers
d = open("day20_run.py").read()
if "TS_SEGS" in d:
    sys.exit("day20 already patched")

d = d.replace('RELIN = os.environ.get("RELIN") == "1"',
              'RELIN = os.environ.get("RELIN") == "1"\n'
              'TS_SEGS = int(os.environ.get("TS_SEGS", "4"))   # task-space metric segments (1 = single constant metric)',
              1)

# build a controller per segment in main(), pass the list to episode()
old = '''    d17.TS_QREP = ref["Qp"][len(ref["Qp"]) // 2].copy()          # metric reference config of THIS path
    ctrl = d17.Ctrl(d17.B0)'''
new = '''    nseg = TS_SEGS if os.environ.get("TS17") else 1
    ctrls, qreps = [], []
    for i in range(nseg):                                        # one metric (and controller) per path segment
        lo, hi = i / nseg, (i + 1) / nseg
        mid = int(round(0.5 * (lo + hi) * (len(ref["Qp"]) - 1)))
        d17.TS_QREP = ref["Qp"][mid].copy(); qreps.append(mid)
        ctrls.append(d17.Ctrl(d17.B0))
    ctrl = ctrls'''
assert old in d, "controller construction anchor not found"
d = d.replace(old, new, 1)

# episode(): choose the controller by progress
old = "def episode(ref, mdl_plant, mdl_ctrl, ctrl, tag):"
new = "def episode(ref, mdl_plant, mdl_ctrl, ctrls, tag):"
assert old in d; d = d.replace(old, new, 1)
old = "        W = window(ref, x, ts)\n        qbar = ptraj[1:].copy()"
new = ("        W = window(ref, x, ts)\n"
       "        seg = min(len(ctrls) - 1, int(d12.TASK.sigma(fk(x[:M])[1]) * len(ctrls)))\n"
       "        ctrl = ctrls[seg]\n"
       "        qbar = ptraj[1:].copy()")
assert old in d, "loop anchor not found"
d = d.replace(old, new, 1)
d = d.replace('             solve_ms_med=float(np.median(solve_ms)), solve_ms_p99=float(np.quantile(solve_ms, .99)))',
              '             solve_ms_med=float(np.median(solve_ms)), solve_ms_p99=float(np.quantile(solve_ms, .99)),\n'
              '             n_segments=len(ctrls))', 1)
d = d.replace('          f"R17={os.environ.get(\'R17\', \'1e-3\')} TMPC={NHOLD * int(round(ps.dt * 1000))}ms RELIN={int(RELIN)} arms={ARMS}")',
              '          f"R17={os.environ.get(\'R17\', \'1e-3\')} TMPC={NHOLD * int(round(ps.dt * 1000))}ms RELIN={int(RELIN)} "\n'
              '          f"arms={ARMS} metric_segments={len(ctrls)} at path idx {qreps}")', 1)
open("day20_run.py", "w").write(d)
print("day20 patched")
