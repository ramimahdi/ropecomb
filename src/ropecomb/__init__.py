"""RopeComb: a configurable rope-and-pulley transmission for mechanical launch.

Reference implementation for the paper "The RopeComb: A Configurable
Rope-and-Pulley Transmission for Impedance-Matched Mechanical Launch".

Quick start
-----------
    from ropecomb import target_profile, fit_array, net_ratio_fn
    from ropecomb import simulate_rigid, simulate_compliant, peak_to_mean

    spec = target_profile(M=1000.0, m=1.0, v0=10.0, v_end=4.0, stroke_time=0.1)
    fit  = fit_array(spec, N=9, k=5.0, seed=0)
    g    = net_ratio_fn(fit.R, fit.s, k=5.0)
    run  = simulate_compliant(1000.0, 1.0, 10.0, g, 16471.0,
                              pretension=spec.F, max_heavy_dist=spec.d_max)
    print(run["summary"]["target_final_speed"], peak_to_mean(run["force_target"]))

Modules
-------
geometry   closed-form array ratio
target     step 1, deceleration to target profile
fit        step 2, least-squares geometry fit
rigid      inextensible-member simulator, and the ideal profile
compliant  elastic pre-tensioned-member simulator
metrics    the paper's reporting conventions
design     Algorithm 1 end to end
cases      the three published designs
"""

from .cases import CANONICAL, Case, load
from .compliant import rope_stiffness, simulate_compliant, stiffness_sweep
from .design import (Candidate, NOMINAL_STIFFNESS, best, design, evaluate)
from .fit import FitResult, fit_array
from .geometry import (array_ratio, comb_width, member_floor, net_ratio_fn,
                       slope_jumps)
from .metrics import fraction_of_ideal, peak_to_mean
from .rigid import G_ACC, simulate_ideal, simulate_rigid
from .target import TargetSpec, solve_design_force, target_profile

__version__ = "1.0.0"

__all__ = [
    "array_ratio", "net_ratio_fn", "comb_width", "member_floor", "slope_jumps",
    "target_profile", "solve_design_force", "TargetSpec",
    "fit_array", "FitResult",
    "simulate_rigid", "simulate_ideal", "G_ACC",
    "simulate_compliant", "rope_stiffness", "stiffness_sweep",
    "peak_to_mean", "fraction_of_ideal",
    "design", "evaluate", "best", "Candidate", "NOMINAL_STIFFNESS",
    "CANONICAL", "Case", "load",
    "__version__",
]
