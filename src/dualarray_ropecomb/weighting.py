"""Weighting rules, fall counts, and the transverse guide-load balance condition.

In a dual-array RopeComb with fixed-stage ratio k, the two movable elements
carry n1 and n2 falls of rope, with n1 + n2 = k. The net mechanical advantage
seen by the payload is fall-count weighted:

    G_net(d) = n1 * G1(d) + n2 * G2(d)

Balancing the transverse forces on the central carriage guide requires:

    G1 : G2  =  n2 * T2 : n1 * T1

When rope tensions along both paths are equal (T1 = T2), this simplifies to:

    G1 : G2  =  n2 : n1

Because the array driving FEWER falls must take the LARGER ratio, the lead
array carries n_lo = floor(k/2) falls and takes ratio G1 > G2.
"""

from __future__ import annotations

import math
from typing import Tuple

__all__ = ["falls", "BalanceRatio", "balance_ratio", "structural_imbalance", "guide_imbalance"]


class BalanceRatio:
    """Transverse load balance ratio G1 : G2 = n2*T2 : n1*T1."""

    def __init__(self, n1: int, n2: int, T1: float = 1.0, T2: float = 1.0):
        self.n1 = int(n1)
        self.n2 = int(n2)
        self.T1 = float(T1)
        self.T2 = float(T2)
        self.val = (self.n2 * self.T2) / (self.n1 * self.T1)

    def __float__(self) -> float:
        return self.val

    def __repr__(self) -> str:
        if self.T1 == self.T2:
            return f"BalanceRatio({self.n2}:{self.n1} = {self.val:.4f})"
        return f"BalanceRatio({self.val:.4f}, n1={self.n1}, n2={self.n2}, T1={self.T1}, T2={self.T2})"

    def __str__(self) -> str:
        if self.T1 == self.T2:
            g = math.gcd(self.n2, self.n1)
            return f"{self.n2 // g}:{self.n1 // g}"
        return f"{self.val:.4f}"

    def __eq__(self, other) -> bool:
        if isinstance(other, BalanceRatio):
            return math.isclose(self.val, other.val, rel_tol=1e-9)
        if isinstance(other, (int, float)):
            return math.isclose(self.val, float(other), rel_tol=1e-9)
        return False


def falls(k: int) -> Tuple[int, int]:
    """Split fixed stage ratio k into lower and higher fall counts (n_lo, n_hi).

    The array with fewer falls leads:
        n_lo = floor(k / 2)
        n_hi = ceil(k / 2) = (k + 1) // 2
    """
    k = int(k)
    n_lo = k // 2
    n_hi = (k + 1) // 2
    return n_lo, n_hi


def balance_ratio(n1: int, n2: int, T1: float = 1.0, T2: float = 1.0) -> BalanceRatio:
    """Calculate the ratio G1 : G2 required to balance guide rail transverse load.

    Parameters
    ----------
    n1, n2 : int
        Fall counts served by array 1 (lead) and array 2 (second).
    T1, T2 : float, optional
        Path tensions in rope 1 and 2. Defaults to 1.0 (equal tensions).

    Returns
    -------
    BalanceRatio
        Ratio G1/G2. Displays as 'n2:n1' when printed.
    """
    return BalanceRatio(n1, n2, T1, T2)


def structural_imbalance(k: int) -> float:
    """Inherent structural imbalance |n_hi - n_lo| / k for a mirror-symmetric pair at stage ratio k."""
    n_lo, n_hi = falls(k)
    return abs(n_hi - n_lo) / float(k)


def guide_imbalance(net_ratio, g_lo, g_hi, n_lo: int, n_hi: int) -> float:
    """Fractional peak lateral guide-load imbalance across the stroke.

        max | n_lo * G_lo(d) - n_hi * G_hi(d) | / max(G_net(d))
    """
    import numpy as np
    diff = np.abs(n_lo * np.asarray(g_lo) - n_hi * np.asarray(g_hi))
    denom = max(float(np.max(net_ratio)), 1e-9)
    return float(np.max(diff) / denom)
