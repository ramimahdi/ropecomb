"""
What does the two-stage seed of Section 8 buy over random restarts?

Fits the k=7, seven-member-per-array design at lambda = 0.03 two ways: once from
the single-array seed (Algorithm 1 steps 3-4, then the staged solve), and by
random restarts of the same joint objective. Reports the running best of the
restarts so the seeded fit can be placed against them.

    python3 seed_benefit.py
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
sys.path.insert(0, HERE)
from semisym_fit import (ideal_target, fit_semisym, fit_semisym_staged,
                         verify_interleaving)
from ropecomb_fit4 import fit_array

M, m, v0, F, VSTOP = 1000., 1., 10., 3169.6, 4.0
K, N, W, LAM, RESTARTS = 7, 7, 2.11, 0.03, 40

dd, Gs, dmax = ideal_target(M, m, v0, F, VSTOP, n=400,
                            code_dir=os.path.join(HERE, "..", "code"))

# Algorithm 1 step 3: the single-array solution that seeds both arrays.
best = None
for sd in range(12):
    R, s, _, rms = fit_array(dd, Gs, N, K, dmax, seed=sd, R_bounds=(0.05, 8.),
                             width_budget=W, overshoot_weight=0.0)
    if best is None or rms < best[2]:
        best = (R, s, rms)
R_star, s_star, _ = best

t0 = time.time()
seeded = fit_semisym_staged(dd, Gs, R_star, s_star, K, dmax, W,
                            lam_balance=LAM, lam_width=50.)
t_seeded = time.time() - t0
print("seeded, one fit:  rms %.4f  imbalance %.1f%%   %.2f s"
      % (seeded["rms"], 100 * seeded["imbalance"], t_seeded))

print("\nrandom restarts of the same objective, running best:")
run, t0 = np.inf, time.time()
for i in range(RESTARTS):
    r = fit_semisym(dd, Gs, N, N, K, dmax, W, lam_balance=LAM,
                    n_starts=1, seed=1000 + i)
    if r is None:
        continue
    ok, _ = verify_interleaving(r)
    if ok and r["rms"] < run:
        run = r["rms"]
        print("   after %2d restarts: rms %.4f" % (i + 1, run))
t_rand = time.time() - t0

print("\n%d restarts: %.2f s total, %.3f s each; seeded fit %.2f s"
      % (RESTARTS, t_rand, t_rand / RESTARTS, t_seeded))
print("seeded fit is %s the best of %d random restarts (%.4f vs %.4f)"
      % ("better than" if seeded["rms"] < run else "not better than",
         RESTARTS, seeded["rms"], run))
