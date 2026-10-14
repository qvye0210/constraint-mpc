"""Add a task-space tracking objective to day17 (opt-in via TS17=1). day13 and the default day17 behaviour
are untouched, so the day13 regression assert still holds."""
import re, sys

# ---------------------------------------------------------------- day17_oracle.py
s = open("day17_oracle.py").read()
if "TS17" in s:
    sys.exit("day17 already patched")

helper = '''
TS_TOL_POS, TS_TOL_ANG = 5e-3, np.sin(np.deg2rad(3.0))      # the acceptance tolerances, used as normalisers
TS_RU = 1e-6            # rule: input cost at |u| = u_lim is <= 2% of one stage at tolerance (declared, not tuned)


def _tool_out(q):
    pb, pt, a = fk(q)
    return np.r_[pt, a]


def taskspace_LJ(qref):
    """L * d(tip, axis)/dq at each reference stage; L normalises each channel by its acceptance tolerance."""
    L = np.diag(np.r_[np.full(3, 1.0 / TS_TOL_POS), np.full(3, 1.0 / TS_TOL_ANG)])
    out = []
    for q in qref:
        J = np.zeros((6, M))
        for j in range(M):
            e = np.zeros(M); e[j] = 1e-6
            J[:, j] = (_tool_out(q + e) - _tool_out(q - e)) / 2e-6
        out.append(L @ J)
    return np.array(out)

'''
i = s.index("# ------------------------------------------------------------------ controller")
s = s[:i] + helper.lstrip("\n") + "\n" + s[i:]

# --- Ctrl: extra parameters + alternative objective
old_obj = re.search(r"        obj = sum\(cp\.quad_form\(X\[:M, k\] - self\.Qref\[:, k - 1\].*?\n(.*?)\n", s, re.S)
assert old_obj, "objective block not found"
block = s[old_obj.start():old_obj.end()]
new_block = '''        self.LJ = [cp.Parameter((6, M)) for _ in range(N)] if os.environ.get("TS17") else None
        if self.LJ is not None:                      # task-space tracking: the objective now measures what the
            obj = sum(cp.sum_squares(self.LJ[k - 1] @ (X[:M, k] - self.Qref[:, k - 1])) for k in range(1, N + 1)) + \\
                  sum(cp.quad_form(U[:, k], TS_RU * np.eye(M)) for k in range(N))
        else:
''' + "\n".join("    " + l if l.strip() else l for l in block.rstrip("\n").split("\n")) + "\n"
s = s[:old_obj.start()] + new_block + s[old_obj.end():]

# --- episode: fill LJ every replan
anchor = "        qref = d13.q_ref_at(np.arange(ts + 1, ts + N + 1), T); ctrl.Qref.value = qref.T"
assert anchor in s, "qref anchor not found"
s = s.replace(anchor, anchor + '''
        if ctrl.LJ is not None:
            for _k, _v in enumerate(taskspace_LJ(qref)): ctrl.LJ[_k].value = _v''', 1)
open("day17_oracle.py", "w").write(s)
print("day17 patched")

# ---------------------------------------------------------------- day19_oracle_debug.py
d = open("day19_oracle_debug.py").read()
if "LJ" not in d:
    a = '    ctrl.x0.value = x; ctrl.cs.value = W["cs"]; ctrl.rs.value = W["rs"]; ctrl.Qref.value = W["qref"].T'
    assert a in d
    d = d.replace(a, a + '''
    if getattr(ctrl, "LJ", None) is not None:
        for _k, _v in enumerate(d17.taskspace_LJ(W["qref"])): ctrl.LJ[_k].value = _v''', 1)
    open("day19_oracle_debug.py", "w").write(d)
    print("day19 patched")
