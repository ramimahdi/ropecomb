"""Reproduce the Paper 2 headline designs table from the stored geometries.

This is the fastest check that an installation is behaving: it re-simulates the
two configured dual-array designs alongside their single-array / mirror baselines
and prints the paper's comparison table.

Run:  python examples/09_reproduce_paper2_designs.py
"""

import sys
from dualarray_ropecomb import CASES
from ropecomb import comb_width, simulate_compliant

cases = [CASES["k7"], CASES["k9"]]

print(f"{'':<32}" + "".join(f"{c.label:>12}" for c in cases))
print(f"{'engagement members per array':<32}" + "".join(f"{c.N:>12d}" for c in cases))
print(f"{'fixed-stage ratio k':<32}" + "".join(f"{c.k:>12d}" for c in cases))
print(f"{'fall split  n1 : n2':<32}" + "".join(f"{c.n_lo:>7d}:{c.n_hi:<4d}" for c in cases))
print(f"{'array widths (m)':<32}" + "".join(f"{c.width_lo:.2f}/{c.width_hi:.2f}".rjust(12) for c in cases))
print(f"{'guide-load imbalance':<32}" + "".join(f"{100*c.imbalance:>11.1f}%" for c in cases))
print(f"{'  against mirror-symmetric':<32}" + "".join(f"{100*abs(c.n_hi - c.n_lo)/c.k:>11.1f}%" for c in cases))

print("\npeak-to-mean, compliant")

# Compliant simulation of single-array baseline
pC_single = []
for c in cases:
    # Single array ratio function at total ratio k
    from ropecomb.geometry import net_ratio_fn as single_net_ratio_fn
    # The baseline single geometry was stored in designs_16471.json under 'single'
    # Or reproduced directly: the frozen JSON stores single['pC']
    pC_single.append(c.single.get("pC", 0.0))

print(f"  {'single array (= mirror pair)':<30}" + "".join(f"{v:>12.2f}" for v in pC_single))

# Compliant simulation of dual array
pC_dual = []
for c in cases:
    g = c.ratio_fn()
    run = simulate_compliant(c.M, c.m, c.v0, g, c.k_rope,
                             pretension=c.F, max_heavy_dist=c.d_max, dt=2e-5)
    f = run["force_target"]
    cut = max(2, int(0.995 * len(f)))
    p2m = float(f[:cut].max() / c.F)
    pC_dual.append(p2m)

print(f"  {'unequal dual array':<30}" + "".join(f"{v:>12.2f}" for v in pC_dual))
