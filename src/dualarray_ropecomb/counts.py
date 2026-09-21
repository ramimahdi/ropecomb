"""Balance by member count: the discrete counterpart of the balance condition.

The balance condition G1 : G2 = n2 T2 : n1 T1 is a condition on the two ratio
PROFILES, and a fit can satisfy it through the middle of the stroke. It cannot
satisfy it at the end. Late in the stroke every engaged member tends to its
asymptote of 2, so

    G_j(d) -> 2 N_j,

and the reaction imbalance tends to a value the geometry has no say in:

    eps_terminal = |n1 N1 - n2 N2| / (n1 N1 + n2 N2).

Exact balance at the end of the stroke therefore requires

    n1 N1 = n2 N2,        i.e.   N1 : N2 = n2 : n1,

the array serving the fewer falls carrying proportionally more members. With
equal member counts this reduces to the mirror residual |n1 - n2| / k, which is
1/k at odd k and is why an equal-count pair cannot be driven to zero imbalance
by fitting alone.

For the 1,000:1 target, whose terminal ratio is 79.3, the member floors at
k = 9 are (5, 4) and already stand in the required ratio 5 : 4, so the minimal
nine-member machine is also the balanced one. At k = 7 the floors are (7, 5),
which leave 2.4% at the end; the smallest counts at or above them satisfying the
condition are (8, 6).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from .weighting import falls

__all__ = ["terminal_imbalance", "satisfies_count_condition", "count_floors",
           "count_balanced_counts", "count_ladder"]


def terminal_imbalance(n_lo: int, N_lo: int, n_hi: int, N_hi: int) -> float:
    """Reaction imbalance the saturated arrays leave, |n1 N1 - n2 N2| / (n1 N1 + n2 N2).

    This is the floor the fit cannot get below, whatever the spans. It is zero
    exactly when the count condition holds.
    """
    a = int(n_lo) * int(N_lo)
    b = int(n_hi) * int(N_hi)
    return abs(a - b) / float(a + b)


def satisfies_count_condition(n_lo: int, N_lo: int, n_hi: int, N_hi: int,
                              tol: float = 0.0) -> bool:
    """True when n1 N1 == n2 N2, or within a relative tolerance of it.

    ``tol`` is a relative tolerance on the terminal imbalance, so tol=0.02
    accepts a pair that leaves 2% at the end of the stroke.
    """
    return terminal_imbalance(n_lo, N_lo, n_hi, N_hi) <= tol


def count_floors(G_terminal: float, k: int) -> Tuple[int, int]:
    """Member floors per array, N_j >= G*(d_max) / (4 n_j).

    Each engaged member contributes at most 2 to its array ratio, and with the
    fall-count weighting each array carries half the weighted target, so the
    floor carries a factor 4 rather than 2. Returns (N_lo_min, N_hi_min) for
    the leading (fewer falls) and trailing arrays.
    """
    n_lo, n_hi = falls(k)
    return (int(math.ceil(float(G_terminal) / (4.0 * n_lo))),
            int(math.ceil(float(G_terminal) / (4.0 * n_hi))))


def count_balanced_counts(k: int, G_terminal: Optional[float] = None, *,
                          N_min: Optional[Tuple[int, int]] = None,
                          tol: float = 0.0,
                          max_total: int = 60) -> Tuple[int, int]:
    """Smallest counts (N_lo, N_hi) at or above the floors that balance the end.

    Parameters
    ----------
    k : int
        Fixed-stage ratio. The fall split is taken from :func:`falls`.
    G_terminal : float, optional
        Terminal target ratio, used to set the member floors via
        :func:`count_floors`. Omit it and pass ``N_min`` instead.
    N_min : (int, int), optional
        Explicit floors, overriding ``G_terminal``.
    tol : float
        Relative tolerance on the terminal imbalance. Zero demands n1N1 = n2N2
        exactly.
    max_total : int
        Give up beyond this many members in total.

    Returns
    -------
    (N_lo, N_hi)

    Examples
    --------
    At k = 9 with the 1,000:1 terminal ratio of 79.3 the floors are (5, 4),
    which already balance: the minimal design is nine members. At k = 7 the
    smallest balanced pair at or above the floors is (8, 6).
    """
    n_lo, n_hi = falls(k)
    if N_min is None:
        if G_terminal is None:
            raise ValueError("pass either G_terminal or N_min")
        N_min = count_floors(G_terminal, k)
    lo_min, hi_min = int(N_min[0]), int(N_min[1])

    best = None
    for total in range(lo_min + hi_min, max_total + 1):
        for N_lo in range(lo_min, total - hi_min + 1):
            N_hi = total - N_lo
            if N_hi < hi_min:
                continue
            if satisfies_count_condition(n_lo, N_lo, n_hi, N_hi, tol):
                # among equal totals prefer the pair closest to exact balance
                eps = terminal_imbalance(n_lo, N_lo, n_hi, N_hi)
                if best is None or eps < best[0]:
                    best = (eps, N_lo, N_hi)
        if best is not None:
            return best[1], best[2]
    raise ValueError(f"no counts satisfying the condition within {max_total} members at k={k}")


def count_ladder(k: int, N_min: Tuple[int, int], n_rows: int = 8) -> List[dict]:
    """Enumerate count pairs by total, with the terminal imbalance each leaves.

    Useful for choosing between a minimal machine and a balanced one: the rows
    show what the extra members buy at the end of the stroke.
    """
    n_lo, n_hi = falls(k)
    lo_min, hi_min = int(N_min[0]), int(N_min[1])
    rows: List[dict] = []
    total = lo_min + hi_min
    while len(rows) < n_rows:
        candidates = []
        for N_lo in range(lo_min, total - hi_min + 1):
            N_hi = total - N_lo
            if N_hi < hi_min:
                continue
            candidates.append((terminal_imbalance(n_lo, N_lo, n_hi, N_hi), N_lo, N_hi))
        if candidates:
            eps, N_lo, N_hi = min(candidates)
            rows.append(dict(total=total, N_lo=N_lo, N_hi=N_hi,
                             terminal_imbalance=eps,
                             balanced=bool(eps == 0.0)))
        total += 1
    return rows
