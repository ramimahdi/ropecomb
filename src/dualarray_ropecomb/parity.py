"""Stage parity, reeving rules, and parasitic energy index p/k^2.

Why the fixed-ratio stage naturally produces an ODD k:
Let p be the total number of sheaves carried on the travelling block(s).
The stage ratio k depends on the anchoring:

    Stationary end to a TRAVELLING block -> k = 2p + 1 (odd)
    Stationary end to the FRAME          -> k = 2p     (even)

The payload takes the one unpaired part. Because sheaves contribute pairs of
parts, an odd k is what a stage with the payload on a free end naturally
produces.

Stage parasitic energy:
Travelling blocks translate at v_payload / k, so kinetic energy held in p
sheaves scales as:

    E_stage ~ p * m_sheave * (v_payload / k)^2  ~  p / k^2
"""

from __future__ import annotations

from typing import Dict

__all__ = ["energy_index", "stage_ratio", "parity_summary"]


def energy_index(p: int, k: float) -> float:
    """Stage parasitic energy per unit sheave mass: p / k^2.

    Parameters
    ----------
    p : int
        Total number of sheaves carried on travelling blocks.
    k : float
        Overall fixed stage ratio.
    """
    return float(p) / float(k * k)


def stage_ratio(p: int, grounded_to_travelling: bool = True) -> int:
    """Fixed-stage ratio k for p sheaves on travelling blocks.

    Parameters
    ----------
    p : int
        Number of travelling sheaves.
    grounded_to_travelling : bool
        True if the dead-end attaches to a travelling block (k = 2p + 1, odd),
        False if anchored to frame (k = 2p, even).
    """
    return 2 * p + 1 if grounded_to_travelling else 2 * p


def parity_summary(p: int) -> Dict[str, float]:
    """Compare odd stage k=2p+1 with even stage k=2p for p travelling sheaves."""
    k_odd = 2 * p + 1
    k_even = 2 * p
    return {
        "p": p,
        "k_odd": k_odd,
        "k_even": k_even,
        "e_odd": energy_index(p, k_odd),
        "e_even": energy_index(p, k_even),
        "ratio_advantage": k_odd / float(k_even),
    }
