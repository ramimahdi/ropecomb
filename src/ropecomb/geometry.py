"""Array geometry: the closed-form displacement ratio of a RopeComb array.

A carriage carrying N engagement members descends past N+1 fixed supports. Member
i sits over a span of half-width R_i and first contacts the tension member once
the carriage has descended s_i. Deflecting the member to depth D draws in

    Y(D) = 2 (sqrt(R^2 + D^2) - R)

of length, so its instantaneous contribution to the displacement ratio is
dY/dD = 2D / sqrt(R^2 + D^2), which rises from zero towards an asymptote of 2.
Summing over the array and multiplying by a fixed second stage of ratio k gives

    G(d) = k * SUM_i  2 D_i / sqrt(R_i^2 + D_i^2),      D_i = max(0, d - s_i)

This is exact and closed form. There is no integration anywhere in this module,
which is what makes fitting a smooth least-squares problem rather than a search.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["array_ratio", "net_ratio_fn", "comb_width", "member_floor",
           "slope_jumps"]


def array_ratio(d, R, s):
    """Array ratio G_array(d), excluding any fixed stage.

    Parameters
    ----------
    d : array_like, shape (n,)
        Carriage displacements at which to evaluate the ratio, in metres.
    R : array_like, shape (N,)
        Span half-widths, metres. The span a member descends into is 2*R wide.
    s : array_like, shape (N,)
        Engagement offsets, metres. Carriage displacement at which each member
        first touches the tension member.

    Returns
    -------
    ndarray, shape (n,)
    """
    d = np.asarray(d, dtype=float)
    R = np.asarray(R, dtype=float)
    s = np.asarray(s, dtype=float)
    D = np.maximum(d[:, None] - s[None, :], 0.0)
    return np.sum(2.0 * D / np.sqrt(R[None, :] ** 2 + D ** 2), axis=1)


def net_ratio_fn(R, s, k=1.0):
    """Return a scalar callable g(d) giving the NET ratio, array times stage.

    The simulators call this once per timestep, so it is written as a plain
    Python loop over members rather than a vectorised call: for N of order ten
    that is faster than constructing an array per step.
    """
    Rs = [float(x) for x in R]
    ss = [float(x) for x in s]
    k = float(k)

    def g(d):
        total = 0.0
        for Ri, si in zip(Rs, ss):
            Di = d - si
            if Di > 0.0:
                total += 2.0 * Di / math.sqrt(Ri * Ri + Di * Di)
        return k * total

    return g


def comb_width(R):
    """Total array width, 2*sum(R), in metres.

    This is the array's footprint on the carriage and, per section 7.3 of the
    paper, the governing practical constraint at low mass ratio.
    """
    return float(2.0 * np.sum(np.asarray(R, dtype=float)))


def member_floor(G_final, k):
    """Smallest member count that can reach G_final: N >= G_final / (2k).

    Each member contributes at most 2 to the array ratio. This bounds the
    TARGET, not useful performance: designs that merely meet the floor track
    the profile badly and lose traction early.
    """
    return int(math.ceil(float(G_final) / (2.0 * float(k))))


def slope_jumps(R):
    """Slope discontinuity 2/R_i injected as each member engages.

    Every engaged member is individually concave, so the array approximates a
    convex target only through these staggered upward steps in dG/dd. They are
    also the origin of the force ripple.
    """
    return 2.0 / np.asarray(R, dtype=float)
