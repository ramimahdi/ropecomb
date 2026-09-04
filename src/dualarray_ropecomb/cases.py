"""The configured dual-array designs of the paper at 16,471 N/m rope stiffness.

Stored in designs_16471.json and loaded here for reproducible simulation:
  - 'k7': N=7 members per array, k=7 stage, n1:n2 = 3:4
  - 'k9': N=5 members per array, k=9 stage, n1:n2 = 4:5
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from .geometry import net_ratio_fn

__all__ = ["DualCase", "CASES", "load_case"]

_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "designs_16471.json")


@dataclass
class DualCase:
    name: str
    label: str
    N: int
    k: int
    n_lo: int
    n_hi: int
    width_budget: float
    width_lo: float
    width_hi: float
    imbalance: float
    terminal: float
    pR: float
    pC: float
    vC: float
    R_lo: List[float] = field(repr=False)
    s_lo: List[float] = field(repr=False)
    R_hi: List[float] = field(repr=False)
    s_hi: List[float] = field(repr=False)
    single: Dict[str, Any] = field(repr=False)
    meta: Dict[str, Any] = field(repr=False)

    @property
    def M(self) -> float:
        return float(self.meta["M"])

    @property
    def m(self) -> float:
        return float(self.meta["m"])

    @property
    def v0(self) -> float:
        return float(self.meta["v0"])

    @property
    def F(self) -> float:
        return float(self.meta["F"])

    @property
    def d_max(self) -> float:
        return float(self.meta["d_max"])

    @property
    def k_rope(self) -> float:
        return float(self.meta["k_rope"])

    def ratio_fn(self) -> Callable[[float], float]:
        """Return the fall-count weighted net ratio function g(d)."""
        return net_ratio_fn(self)

    def spans_cm(self) -> Dict[str, List[int]]:
        """Span widths (2*R) in engagement order, centimetres."""
        return {
            "lead": [round(200 * r) for _, r in sorted(zip(self.s_lo, self.R_lo))],
            "second": [round(200 * r) for _, r in sorted(zip(self.s_hi, self.R_hi))],
        }


def _load_all() -> Dict[str, DualCase]:
    with open(_DATA_PATH, "r") as fh:
        raw = json.load(fh)
    meta = raw.get("_meta", {})
    cases = {}
    for tag in ("k7", "k9"):
        if tag not in raw:
            continue
        entry = raw[tag]
        dual = entry["dual"]
        label = f"<{entry['N']},{entry['k']}>"
        case = DualCase(
            name=tag,
            label=label,
            N=entry["N"],
            k=entry["k"],
            n_lo=entry["n_lo"],
            n_hi=entry["n_hi"],
            width_budget=entry["width_budget"],
            width_lo=dual["width_lo"],
            width_hi=dual["width_hi"],
            imbalance=dual["imbalance"],
            terminal=dual["terminal"],
            pR=dual["pR"],
            pC=dual["pC"],
            vC=dual["vC"],
            R_lo=dual["R_lo"],
            s_lo=dual["s_lo"],
            R_hi=dual["R_hi"],
            s_hi=dual["s_hi"],
            single=entry.get("single", {}),
            meta=meta,
        )
        cases[tag] = case
    return cases


CASES: Dict[str, DualCase] = _load_all()


def load_case(name: str) -> DualCase:
    """Fetch a configured dual-array case by key: 'k7' or 'k9'."""
    if name not in CASES:
        raise KeyError(f"{name!r} not found. Available cases: {list(CASES.keys())}")
    return CASES[name]
