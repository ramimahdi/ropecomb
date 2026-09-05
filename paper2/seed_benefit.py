"""What does the two-stage seed of Section 8 buy over random restarts?

Fits the k=7, seven-member-per-array design at lambda = 0.03 two ways: once from
the single-array seed (Algorithm 1 steps 3-4, then the staged solve), and by
random restarts of the same joint objective. Reports the running best of the
restarts so the seeded fit can be placed against them.

    python3 seed_benefit.py
"""

import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from ropecomb import target_profile, fit_array
from dualarray_ropecomb.fit import fit_dual_staged, fit_dual
from dualarray_ropecomb.ordering import verify_interleaving

M, m, v0, F, VSTOP = 1000.0, 1.0, 10.0, 3169.6, 4.0
K, N, W, LAM, RESTARTS = 7, 7, 2.11, 0.03, 40

spec = target_profile(M, m, v0, VSTOP, F=F, n=400)
dd, Gs, dmax = spec.d, spec.G, spec.d_max

# Algorithm 1 step 3: the single-array solution that seeds both arrays.
best = None
for sd in range(12):
    fit_res = fit_array(spec, N, K, seed=sd, max_width=W)
    if best is None or fit_res.rms < best[2]:
        best = (np.asarray(fit_res.R), np.asarray(fit_res.s), fit_res.rms)
R_star, s_star, _ = best

t0 = time.time()
seeded = fit_dual_staged(spec, R_star=R_star, s_star=s_star, k=K, width_budget=W,
                         lam_balance=LAM, lam_width=50.0)
t_seeded = time.time() - t0
print("seeded, one fit:  rms %.4f  imbalance %.1f%%   %.2f s"
      % (seeded.rms, 100 * seeded.imbalance, t_seeded))

print("\nrandom restarts of the same objective, running best:")
run, t0 = np.inf, time.time()
for i in range(RESTARTS):
    r = fit_dual(spec, N_lo=N, N_hi=N, k=K, width_budget=W,
                 lam_balance=LAM, lam_width=50.0, n_starts=1, seed=1000 + i)
    if r is None:
        continue
    ok, _ = verify_interleaving(r.raw)
    if ok and r.rms < run:
        run = r.rms
        print("   after %2d restarts: rms %.4f" % (i + 1, run))
t_rand = time.time() - t0

print("\n%d restarts: %.2f s total, %.3f s each; seeded fit %.2f s"
      % (RESTARTS, t_rand, t_rand / RESTARTS, t_seeded))
print("seeded fit is %s the best of %d random restarts (%.4f vs %.4f)"
      % ("better than" if seeded.rms < run else "not better than",
         RESTARTS, seeded.rms, run))
