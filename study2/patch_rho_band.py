"""Working-distance band: the tool constraint so far had only a lower bound (rho >= 0), so a controller that knows
the true payload can lift the tool to unload the wrist at no cost. Real process tolerances are two-sided, so the
constraint becomes rho in [RHO_MIN, RHO_MAX] = 10 mm +/- 5 mm (same 5 mm tolerance as the lateral one).
Both bounds use the same linearisation G, h. Applied to every arm; opt-in via RHO_BAND=1 so older results stand."""
import sys

s = open("day17_oracle.py").read()
if "RHO_BAND" in s:
    sys.exit("day17 already patched")

# constant
s = s.replace("TS_QREP = None", "RHO_BAND_HALF = 5e-3     # working-distance tolerance, same as the lateral one\nTS_QREP = None", 1)

# Ctrl: upper-bound parameter + constraint
old = "        self.G, self.h, self.Qb = cp.Parameter((M, N)), cp.Parameter(N), cp.Parameter((M, N))"
assert old in s, "parameter anchor"
s = s.replace(old, old + "\n        self.h_up = cp.Parameter(N)", 1)
old = "        self.tool = cp.sum(cp.multiply(self.G, X[:M, 1:]), axis=0) >= self.h; cons.append(self.tool)"
assert old in s, "tool constraint anchor"
new = ("        self.tool = cp.sum(cp.multiply(self.G, X[:M, 1:]), axis=0) >= self.h; cons.append(self.tool)\n"
       "        if os.environ.get(\"RHO_BAND\"):        # two-sided working distance\n"
       "            cons.append(cp.sum(cp.multiply(self.G, X[:M, 1:]), axis=0) <= self.h_up)")
s = s.replace(old, new, 1)
open("day17_oracle.py", "w").write(s)
print("day17 patched")

# ---------------------------------------------------------------- day20 driver: fill the upper bound
d = open("day20_run.py").read()
if "h_up" in d:
    sys.exit("day20 already patched")
old = "    ctrl.G.value, ctrl.h.value, ctrl.Qb.value = G.T, (G * qbar_pos).sum(1) - r0, qbar_pos.T"
assert old in d, "fill anchor"
new = (old + "\n"
       "    if os.environ.get(\"RHO_BAND\"):\n"
       "        band = d17.RHO_BAND_HALF\n"
       "        ctrl.h.value = (G * qbar_pos).sum(1) - r0 + (d12.RHO_DESIGN - band)   # rho >= design - band\n"
       "        ctrl.h_up.value = (G * qbar_pos).sum(1) - r0 + (d12.RHO_DESIGN + band)  # rho <= design + band")
d = d.replace(old, new, 1)
d = d.replace('          f"arms={ARMS} metric_segments={len(ctrls)} at path idx {qreps}")',
              '          f"arms={ARMS} metric_segments={len(ctrls)} at path idx {qreps} "\n'
              '          f"rho_band={os.environ.get(\'RHO_BAND\', \'0\')}")', 1)
# record how far above the band the tool went
d = d.replace('             violations=int((tm["rho"] < -TOL).sum()),',
              '             violations=int((tm["rho"] < -TOL).sum()),\n'
              '             rho_max=float(tm["rho"].max()),\n'
              '             band_high_steps=int((tm["rho"] > d12.RHO_DESIGN + d17.RHO_BAND_HALF + TOL).sum()),', 1)
open("day20_run.py", "w").write(d)
print("day20 patched")
