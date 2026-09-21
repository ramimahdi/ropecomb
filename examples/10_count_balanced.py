"""Balance by member count: what the fit cannot do, and what the counts can.

Late in the stroke every engaged member saturates at a ratio contribution of 2,
so each array's ratio tends to twice its member count and the reaction imbalance
tends to |n1 N1 - n2 N2| / (n1 N1 + n2 N2) whatever the spans are. With equal
member counts at odd k that floor is the mirror residual 1/k, and no amount of
fitting gets below it. Choosing counts in the ratio N1 : N2 = n2 : n1 removes it.

This example fits the equal-count and count-balanced designs at k = 7 side by
side and reports the reaction imbalance of each, read out of the simulation
rather than from the geometry alone.

Run:  python examples/10_count_balanced.py        (about a minute)
"""

import numpy as np

from dualarray_ropecomb import (CASES, count_balanced_counts, count_floors,
                                fit_dual_staged, imbalance_trace, net_ratio_fn,
                                pitch_moment)
from ropecomb import fit_array, simulate_compliant, target_profile

K = 7
c = CASES["k7"]
spec = target_profile(c.M, c.m, c.v0, 4.0, F=c.F, n=400)
G_terminal = float(spec.G[-1])
print(spec, "\n")

floors = count_floors(G_terminal, K)
balanced = count_balanced_counts(K, G_terminal)
print(f"terminal ratio {G_terminal:.1f} at k = {K}")
print(f"  member floors                     {floors[0]} and {floors[1]}")
print(f"  smallest counts that balance      {balanced[0]} and {balanced[1]}")
print(f"  equal counts leave at the end     {100 / K:.1f}%   (the mirror residual 1/k)\n")

# --- equal counts, seeded from a single-array fit -------------------------
single = fit_array(spec, c.N, K, seed=0, max_width=c.width_budget)
equal = fit_dual_staged(spec, R_star=single.R, s_star=single.s, k=K,
                        width_budget=c.width_budget, lam_balance=0.03)

# --- count-balanced, seeded by the merged method --------------------------
count = fit_dual_staged(spec, k=K, N_lo=balanced[0], N_hi=balanced[1],
                        width_budget=c.width_budget, lam_balance=0.03,
                        restarts=8, seed=0)

print(f"{'':<34}{'equal counts':>16}{'count-balanced':>16}")
print(f"{'members, leading : trailing':<34}"
      f"{f'{len(equal.R_lo)} : {len(equal.R_hi)}':>16}{f'{len(count.R_lo)} : {len(count.R_hi)}':>16}")
print(f"{'array widths (m)':<34}"
      f"{f'{equal.width_lo:.2f}/{equal.width_hi:.2f}':>16}{f'{count.width_lo:.2f}/{count.width_hi:.2f}':>16}")
print(f"{'fit residual (rms ratio)':<34}{equal.rms:>16.3f}{count.rms:>16.3f}")
print(f"{'terminal ratio reached':<34}{equal.terminal:>16.1f}{count.terminal:>16.1f}")

rows = []
for tag, fit in (("equal", equal), ("count", count)):
    geo = imbalance_trace(fit, d=spec.d, tension=spec.F)
    g = net_ratio_fn(fit.R_lo, fit.s_lo, fit.n_lo, fit.R_hi, fit.s_hi, fit.n_hi)
    run = simulate_compliant(c.M, c.m, c.v0, g, c.k_rope,
                             pretension=c.F, max_heavy_dist=c.d_max, dt=2e-5)
    dyn = imbalance_trace(fit, run)
    mom = pitch_moment(fit, d=spec.d, tension=spec.F)
    f = run["force_target"]
    cut = max(2, int(0.995 * len(f)))
    rows.append(dict(geo=geo, dyn=dyn, mom=mom, pC=float(f[:cut].max() / c.F)))

print(f"{'peak reaction imbalance, geometry':<34}"
      + "".join(f"{100*r['geo']['eps_geometric']:>15.2f}%" for r in rows))
print(f"{'  at the end of the stroke':<34}"
      + "".join(f"{100*r['geo']['eps_terminal']:>15.2f}%" for r in rows))
print(f"{'peak reaction imbalance, simulated':<34}"
      + "".join(f"{100*r['dyn']['eps_peak']:>15.2f}%" for r in rows))
print(f"{'  in kN of the carriage reaction':<34}"
      + "".join(f"{r['dyn']['dF_peak']/1e3:>9.1f} of {r['dyn']['F_carriage_peak']/1e3:>3.0f}" for r in rows))
print(f"{'peak guide moment (kN m)':<34}"
      + "".join(f"{r['mom']['moment_peak']/1e3:>16.1f}" for r in rows))
print(f"{'  equivalent offset (mm)':<34}"
      + "".join(f"{1000*r['mom']['offset_equivalent']:>16.0f}" for r in rows))
print(f"{'peak-to-mean payload force':<34}"
      + "".join(f"{r['pC']:>16.2f}" for r in rows))

print("\nThe counts carry the end of the stroke, where the fit cannot: the")
print("count-balanced design leaves a fraction of a percent at full engagement,")
print("against the mirror residual the equal-count pair cannot escape.")
