"""Step 2 of the design procedure: fit an array to a target profile.

Both the array ratio and the target are functions of source displacement, so
determining the geometry is the minimisation

    min over {R_i, s_i}  of  INTEGRAL | k G_array(d) - G*(d) |^2 dd

over 2N parameters. The objective is smooth and differentiable except at the
engagement points, and bounded least squares solves it in seconds.

THE OBJECTIVE CARRIES NO ADDITIONAL TERMS. It is the plain squared residual: no
weighting, no one-sided terms, no curvature or smoothness penalties. Earlier
drafts carried all of those; none is needed once the target is specified by
deceleration rather than by a chosen force.

No dynamic simulation enters the fit. The dynamics are used afterwards, to
select among fitted candidates and to verify what the chosen geometry does.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .geometry import array_ratio

__all__ = ["fit_array", "FitResult"]


class FitResult:
    """A fitted geometry and its residual."""

    __slots__ = ("R", "s", "rms", "rms_pct", "solution")

    def __init__(self, R, s, rms, rms_pct, solution):
        self.R = R                #: span half-widths, sorted by engagement order
        self.s = s                #: engagement offsets, ascending
        self.rms = rms            #: RMS ratio error, absolute
        self.rms_pct = rms_pct    #: the same as a percentage of target full scale
        self.solution = solution  #: the raw scipy result

    def __repr__(self):
        return f"FitResult(N={len(self.R)}, residual={self.rms_pct:.2f}% of full scale)"


def fit_array(target, N, k, *, seed=0, R_bounds=(0.05, 8.0), max_width=20.0,
              width_weight=50.0, init_span_range=(0.2, 0.8),
              init_jitter=(0.6, 1.4), max_nfev=20000):
    """Fit N members so that k * G_array(d) tracks the target ratio.

    Parameters
    ----------
    target : TargetSpec
        From :func:`ropecomb.target.target_profile`.
    N : int
        Number of members to fit.
    k : float
        Fixed second-stage ratio.
    seed : int
        Random restart index. The landscape is effectively unimodal under a
        well-specified target, but restarts still explore different total array
        widths, which is what the selection step trades against.
    R_bounds : (float, float)
        Bounds on span half-width, metres. The lower bound is the physical span
        floor: a span must clear the sheave diameter, the member and working
        clearance. 0.05 m, i.e. a 10 cm span, is used throughout the paper and
        IMPROVES the designs rather than costing performance.
    max_width : float
        A one-sided guard on total array width, 2*sum(R). At the published
        scales this is never active: the canonical arrays are 1.9 to 2.8 m wide
        against a 20 m guard, so the penalty contributes exactly zero. It exists
        only to stop a pathological restart running away.

    Initialisation
    --------------
    Span half-widths are drawn in two stages so that the TOTAL array width
    varies from restart to restart. Drawing every R_i independently would make
    the total concentrate near its mean by the central limit theorem: at N=16
    it would sit within a few percent of 0.5N every time and wide arrays would
    never be sampled at all. Instead a per-restart mean span is drawn first,
    then per-member jitter about it.
    """
    d, G_star, d_max = target.d, target.G, target.d_max
    rng = np.random.default_rng(seed)
    scale = np.ptp(G_star) if np.ptp(G_star) > 0 else 1.0

    def unpack(p):
        return np.exp(p[:N]), p[N:]

    def residual(p):
        R, s = unpack(p)
        err = (k * array_ratio(d, R, s) - G_star) / scale
        if max_width is not None:
            over = max(0.0, 2.0 * float(np.sum(R)) - max_width) / max_width
            err = np.concatenate([err, [width_weight * over]])
        return err

    lo = np.concatenate([np.full(N, np.log(R_bounds[0])), np.zeros(N)])
    hi = np.concatenate([np.full(N, np.log(R_bounds[1])), np.full(N, d_max)])

    r_lo, r_hi = R_bounds
    w = rng.uniform(*init_span_range)
    jitter = rng.uniform(*init_jitter, size=N)
    R0 = np.clip(0.5 * w * jitter, r_lo * 1.001, r_hi * 0.999)
    p0 = np.concatenate([np.log(R0), np.sort(rng.uniform(0.0, 0.85 * d_max, N))])
    p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)

    sol = least_squares(residual, p0, bounds=(lo, hi), max_nfev=max_nfev)

    R, s = unpack(sol.x)
    order = np.argsort(s)
    R, s = R[order], s[order]
    rms = float(np.sqrt(np.mean((k * array_ratio(d, R, s) - G_star) ** 2)))
    return FitResult(R, s, rms, 100.0 * rms / scale, sol)
