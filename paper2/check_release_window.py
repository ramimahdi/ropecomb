"""Where does the rigid peak-to-mean come from? Full curve vs truncated release."""

import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from ropecomb import simulate_rigid
from dualarray_ropecomb import load_case

case7 = load_case("k7")
g = case7.ratio_fn()

rr = simulate_rigid(case7.M, case7.m, case7.v0, g, max_heavy_dist=case7.d_max,
                    max_time=0.5, max_target_acc=3e4, dt=1e-5)
Fp = np.asarray(rr["force_target"], float)
n = len(Fp)

print("frozen designs_16471.json rigid p2m  : %.2f" % case7.pR)
print("samples in the force history          : %d\n" % n)
print("  %-34s %8s" % ("window", "peak/mean"))
print("  " + "-" * 46)
for label, frac in [("full curve, no truncation", 1.000),
                    ("first 99.9%", 0.999), ("first 99.5%", 0.995),
                    ("first 99.0%", 0.990), ("first 98%", 0.980), ("first 95%", 0.950)]:
    c = Fp[:max(1, int(n * frac))]
    print("  %-34s %8.2f" % (label, c.max() / c.mean()))
print("\n  last 10 samples of the force history (N):")
print("   ", np.round(Fp[-10:], 0))
print("  peak occurs at sample %d of %d (%.2f%% through)" % (Fp.argmax(), n, 100.0 * Fp.argmax() / n))
