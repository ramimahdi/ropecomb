"""Algorithm 1, on a reduced grid so it finishes in about a minute.

The published run sweeps N = 2..16 against k in {1,2,3,5,7,9} with 50 restarts
per cell and the best 5 simulated: 4,500 fits and 450 simulations per case,
roughly twenty minutes. This example narrows the grid to show the mechanics.
Widen N_range, k_values and restarts to reproduce the paper.

Run:  python examples/04_algorithm1.py
"""
from ropecomb import best, design, target_profile

M, m = 1000.0, 1.0
spec = target_profile(M, m, v0=10.0, v_end=4.0, stroke_time=0.1)
print(spec, "\n")

cands = design(M, m, target=spec, N_range=range(7, 11), k_values=(3.0, 5.0, 7.0),
               restarts=8, keep=2)

print(f"simulated {len(cands)} candidates; "
      f"{sum(c.admissible for c in cands)} admissible "
      f"(peak-to-mean <= 1.3 and exit >= 99% of ideal)\n")
print("top 8 by score, * marks inadmissible:")
for c in sorted(cands, key=lambda c: -c.score)[:8]:
    print("  ", c)

pick = best(cands)
print(f"\nAlgorithm 1 selects: N={pick.N_eff}, k={pick.k:.0f}, "
      f"{pick.width:.2f} m of array")
print(f"Pruning removed members in "
      f"{sum(c.pruned > 0 for c in cands)} of {len(cands)} candidates.")
