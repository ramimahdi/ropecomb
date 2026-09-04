"""Kinematics and closed-form displacement ratios for dual-array RopeCombs.

The displacement ratio of each individual array has the closed form:

    G_bank(d) = SUM_i  2 D_i / sqrt(R_i^2 + D_i^2),   D_i = max(0, d - s_i)

The net ratio produced by the two arrays sharing the fixed stage is:

    G_net(d) = n_lo * G_lo(d) + n_hi * G_hi(d)

Because this net ratio is also closed-form and piece-wise smooth, target
matching and dynamic integration remain exact and fast.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence, Union

import numpy as np
from ropecomb.geometry import array_ratio, comb_width, member_floor

__all__ = ["dual_net_ratio", "net_ratio_fn", "comb_width", "member_floor", "array_ratio"]


def dual_net_ratio(d: Union[float, np.ndarray],
                   R_lo: Sequence[float], s_lo: Sequence[float], n_lo: int,
                   R_hi: Sequence[float], s_hi: Sequence[float], n_hi: int) -> Union[float, np.ndarray]:
    """Calculate the exact net ratio G_net(d) = n_lo*G_lo(d) + n_hi*G_hi(d).

    Parameters
    ----------
    d : float or array-like
        Carriage / source displacement, metres.
    R_lo, s_lo : sequence of float
        Span half-widths (m) and engagement offsets (m) for the lead array.
    n_lo : int
        Fall count for the lead array (typically floor(k/2)).
    R_hi, s_hi : sequence of float
        Span half-widths (m) and engagement offsets (m) for the second array.
    n_hi : int
        Fall count for the second array (typically ceil(k/2)).

    Returns
    -------
    float or np.ndarray
        Net transmission ratio at displacement(s) d.
    """
    g_lo = array_ratio(d, R_lo, s_lo)
    g_hi = array_ratio(d, R_hi, s_hi)
    return n_lo * g_lo + n_hi * g_hi


def net_ratio_fn(*args, **kwargs) -> Callable[[float], float]:
    """Return a fast callable g(d) evaluating the dual-array net ratio.

    Accepts either:
      1. A case object having attributes (R_lo, s_lo, n_lo, R_hi, s_hi, n_hi)
         e.g., net_ratio_fn(case)
      2. Explicit arguments:
         net_ratio_fn(R_lo, s_lo, n_lo, R_hi, s_hi, n_hi)
    """
    if len(args) == 1 and hasattr(args[0], "R_lo"):
        c = args[0]
        R_lo, s_lo, n_lo = c.R_lo, c.s_lo, c.n_lo
        R_hi, s_hi, n_hi = c.R_hi, c.s_hi, c.n_hi
    elif len(args) == 6:
        R_lo, s_lo, n_lo, R_hi, s_hi, n_hi = args
    elif "case" in kwargs:
        c = kwargs["case"]
        R_lo, s_lo, n_lo = c.R_lo, c.s_lo, c.n_lo
        R_hi, s_hi, n_hi = c.R_hi, c.s_hi, c.n_hi
    else:
        R_lo = kwargs["R_lo"]
        s_lo = kwargs["s_lo"]
        n_lo = kwargs["n_lo"]
        R_hi = kwargs["R_hi"]
        s_hi = kwargs["s_hi"]
        n_hi = kwargs["n_hi"]

    pieces = [
        (list(map(float, R_lo)), list(map(float, s_lo)), float(n_lo)),
        (list(map(float, R_hi)), list(map(float, s_hi)), float(n_hi)),
    ]

    def f(d: float) -> float:
        d_val = float(d)
        total = 0.0
        for R, s, mu in pieces:
            b = 0.0
            for Ri, si in zip(R, s):
                Di = d_val - si
                if Di > 0.0:
                    b += 2.0 * Di / math.sqrt(Ri * Ri + Di * Di)
            total += mu * b
        return total

    return f
