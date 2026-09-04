"""Engagement ordering and interleaving verification.

The array driving fewer falls leads. Because G is continuous at engagement and
its slope steps, the transverse imbalance drifts toward whichever array engaged
last at a rate set by that array's fall count. Leading with the lower-fall array
makes the excursion both smaller and faster to close.

Interleaving constraint:
    s(lo_1) < s(hi_1) < s(lo_2) < s(hi_2) < ...
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np

__all__ = ["interleaved_slots", "verify_interleaving"]


def interleaved_slots(N_lo: int, N_hi: int, low_leads: bool = True) -> List[int]:
    """Generate alternating array slot assignments (0 for lead/lo, 1 for second/hi).

    Parameters
    ----------
    N_lo : int
        Number of engagement members on the lead array.
    N_hi : int
        Number of engagement members on the second array.
    low_leads : bool
        True if the lower-fall array engages first (0, 1, 0, 1...).

    Returns
    -------
    list of int
        Array identifier (0 or 1) for each engagement in ascending order.
    """
    total = N_lo + N_hi
    lead = 0 if low_leads else 1
    other = 1 if low_leads else 0
    nl, no = (N_lo, N_hi) if low_leads else (N_hi, N_lo)

    order: List[int] = []
    i_l, i_o = 0, 0
    while len(order) < total:
        if i_l < nl and (i_o >= no or len(order) % 2 == 0):
            order.append(lead)
            i_l += 1
        elif i_o < no:
            order.append(other)
            i_o += 1
        else:
            order.append(lead)
            i_l += 1
    return order


def verify_interleaving(solution_or_offsets: Union[dict, Tuple[Sequence[float], Sequence[float]]],
                        order: Sequence[int] = None) -> Tuple[bool, str]:
    """Verify that engagement offsets strictly alternate as intended.

    Parameters
    ----------
    solution_or_offsets : dict or tuple of sequences
        Either a fit solution dictionary with 's_lo', 's_hi', 'order' keys,
        or a tuple (s_lo, s_hi).
    order : sequence of int, optional
        Expected order. If None and input is a dict, extracted from dict.
        Otherwise defaults to interleaved_slots(len(s_lo), len(s_hi)).

    Returns
    -------
    ok : bool
        True if the actual offsets follow the expected order.
    pattern : str
        String of 'L' and 'H' showing the achieved sequence.
    """
    if isinstance(solution_or_offsets, dict):
        s_lo = solution_or_offsets["s_lo"]
        s_hi = solution_or_offsets["s_hi"]
        if order is None:
            order = solution_or_offsets.get("order")
    else:
        s_lo, s_hi = solution_or_offsets

    if order is None:
        order = interleaved_slots(len(s_lo), len(s_hi))

    events = sorted([(float(s), 0) for s in s_lo] + [(float(s), 1) for s in s_hi])
    actual = [tag for _, tag in events]
    pattern = "".join("L" if tag == 0 else "H" for tag in actual)
    ok = (actual == list(order))
    return ok, pattern
