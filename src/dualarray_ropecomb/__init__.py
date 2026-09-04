"""DualArray RopeComb: dual-array variable mechanical advantage transmissions.

Reference implementation for the paper "Dual-Array Variable Mechanical
Advantage Transmissions with Guide-Load Balancing for Mechanical Launchers".

Quick start
-----------
    from dualarray_ropecomb import load_case, net_ratio_fn, balance_ratio
    from ropecomb import simulate_compliant, peak_to_mean

    d = load_case("k7")
    print(balance_ratio(n1=d.n_lo, n2=d.n_hi))  # 4:3
    print(d.width_lo, d.width_hi)                # 1.95 m, 1.98 m
    print(d.imbalance)                           # 0.081

    g = net_ratio_fn(d)
    run = simulate_compliant(d.M, d.m, d.v0, g, d.k_rope, pretension=d.F, max_heavy_dist=d.d_max)
    cut = max(2, int(0.995 * len(run["force_target"])))
    pC = run["force_target"][:cut].max() / d.F   # 1.12
"""

from .cases import CASES, DualCase, load_case
from .fit import (DualFitResult, R_FLOOR, encode, fit_dual, fit_dual_staged,
                  seed_two_stage)
from .geometry import (comb_width, dual_net_ratio, member_floor, net_ratio_fn)
from .ordering import interleaved_slots, verify_interleaving
from .parity import energy_index, parity_summary, stage_ratio
from .weighting import (BalanceRatio, balance_ratio, falls, guide_imbalance,
                        structural_imbalance)

__version__ = "2.0.0"

__all__ = [
    "load_case", "CASES", "DualCase",
    "balance_ratio", "BalanceRatio", "falls", "structural_imbalance", "guide_imbalance",
    "dual_net_ratio", "net_ratio_fn", "comb_width", "member_floor",
    "fit_dual_staged", "fit_dual", "DualFitResult", "seed_two_stage", "encode", "R_FLOOR",
    "interleaved_slots", "verify_interleaving",
    "energy_index", "stage_ratio", "parity_summary",
    "__version__",
]
