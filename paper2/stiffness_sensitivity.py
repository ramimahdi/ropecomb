"""Is the compliant peak-to-mean sensitive to the assumed rope stiffness?

Paper 1 Appendix B derives k = EA/L = (E/sigma) F_break / L and notes that E/sigma
lies between about 34 and 66 across UHMWPE, steel wire rope, carbon fibre and PBO,
so working strain is of order 1% whatever the material is. It reports 1.6e4 N/m for
its worked case and states the result is insensitive to stiffness over that range.
This re-tests that on the two dual-array designs of Section 9.

    python3 stiffness_sensitivity.py
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from ropecomb import simulate_compliant
from dualarray_ropecomb import load_case

cases = [load_case("k7"), load_case("k9")]

print("compliant peak-to-mean against assumed rope stiffness")
print("(paper 1 Appendix B: k = (E/sigma) F_break / L, E/sigma in [34, 66])\n")
print("  %-22s %s" % ("stiffness (N/m)", "  ".join("%8s" % s for s in ("k=7 design", "k=9 design"))))
print("  " + "-" * 50)

rows = []
for kr in (8000, 10000, 12000, 16471, 20000, 30000):
    vals = []
    for c in cases:
        g = c.ratio_fn()
        ee = simulate_compliant(c.M, c.m, c.v0, g, k_rope=kr,
                                max_heavy_dist=c.d_max, pretension=c.F, dt=2e-5)
        Fc = np.asarray(ee["force_target"], float)
        cut = max(1, int(len(Fc) * 0.995))
        vals.append(Fc[:cut].max() / c.F)
    rows.append((kr, vals))
    mark = "   <- used here" if kr == 16471 else ""
    print("  %-22s %8.2f  %8.2f%s" % (format(kr, ","), vals[0], vals[1], mark))

a = [r[1][0] for r in rows]
b = [r[1][1] for r in rows]
print("\n  spread over a 3.75x range of stiffness: k=7 %.2f to %.2f, k=9 %.2f to %.2f"
      % (min(a), max(a), min(b), max(b)))
