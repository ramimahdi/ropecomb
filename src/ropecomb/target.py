"""Step 1 of the design procedure: turn a deceleration into a target profile.

The central methodological point of the paper. The ideal profile is not defined
until one fixes HOW MUCH the source mass is asked to give up, and that choice
should be made by stating the deceleration and solving for the force, not by
picking a force and seeing what comes out.

Under an arbitrary force the least-squares landscape is multi-modal and the fit
residual stops predicting anything useful; restarts scatter and the objective
sprouts penalty terms to control them. Under a target specified by deceleration
the landscape is effectively unimodal and a plain squared residual suffices.
All the tuning machinery earlier drafts of this work carried was compensation
for a mis-specified target.
"""

from __future__ import annotations

import numpy as np

from .rigid import simulate_ideal

__all__ = ["solve_design_force", "target_profile", "TargetSpec"]


class TargetSpec:
    """The output of step 1: everything a fit needs to describe its target."""

    __slots__ = ("F", "d", "G", "d_max", "ideal_exit", "ideal_run")

    def __init__(self, F, d, G, d_max, ideal_exit, ideal_run):
        self.F = F                    #: design payload force, N
        self.d = d                    #: sample displacements, m
        self.G = G                    #: target net ratio at those displacements
        self.d_max = d_max            #: braking distance, m
        self.ideal_exit = ideal_exit  #: payload exit velocity of the ideal run
        self.ideal_run = ideal_run    #: the full ideal simulation

    @property
    def full_scale(self):
        """Range of the target ratio; residuals are quoted relative to this."""
        return float(np.ptp(self.G))

    def __repr__(self):
        return (f"TargetSpec(F={self.F:,.1f} N, d_max={self.d_max:.4f} m, "
                f"terminal ratio {self.G[-1]:.1f}:1, "
                f"ideal exit {self.ideal_exit:,.1f} m/s)")


def solve_design_force(M, m, v0=10.0, v_end=4.0, stroke_time=0.1, *, dt=1e-5,
                       lo=None, hi=None, iters=60):
    """Bisect for the uniform payload force that produces a stated deceleration.

    Finds F such that the ideal run, terminated when the source reaches
    ``v_end``, takes ``stroke_time`` seconds.

    The published specification is 10 m/s down to 4 m/s over 0.1 s, which
    surrenders 84% of the source's kinetic energy. Stopping at 4 m/s is the
    hardest deceleration that leaves the tension member continuously loaded in
    all three canonical cases; below it, traction is lost before the end of the
    stroke.

    Bracket note: the default bracket is deliberately wide, 0.005*M to 100*M.
    An earlier version of this work used 3*M and silently clipped the 1,000:1
    case, producing an entire wrong sweep.
    """
    lo = 0.005 * M if lo is None else float(lo)
    hi = 100.0 * M if hi is None else float(hi)

    def elapsed(F):
        run = simulate_ideal(M, m, v0, F, min_heavy_speed=v_end,
                             max_time=0.8, dt=dt)
        return run["summary"]["elapsed"]

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if elapsed(mid) > stroke_time:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def target_profile(M, m, v0=10.0, v_end=4.0, stroke_time=0.1, *, F=None,
                   n=400, dt=1e-5):
    """Step 1 in full: solve for the force, then sample the target ratio.

    Returns a :class:`TargetSpec`. Pass ``F`` to skip the bisection if the
    design force is already known.
    """
    if F is None:
        F = solve_design_force(M, m, v0, v_end, stroke_time, dt=dt)
    run = simulate_ideal(M, m, v0, F, min_heavy_speed=v_end, max_time=0.6, dt=dt)
    d_max = run["summary"]["heavy_travel"]
    d = np.linspace(0.0, d_max, n)
    G = np.interp(d, run["heavy_dist"], run["gear"])
    return TargetSpec(float(F), d, G, float(d_max),
                      float(run["summary"]["target_final_speed"]), run)
