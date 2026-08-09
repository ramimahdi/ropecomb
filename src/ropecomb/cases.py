"""The three canonical designs reported in section 5.3 of the paper.

Each was produced by Algorithm 1 at 50 restarts per cell with the top 5
simulated, then selected as the highest-scoring admissible candidate. The
geometries are stored here so results can be reproduced without re-running the
twenty-minute grid search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .geometry import net_ratio_fn

__all__ = ["Case", "CANONICAL", "load"]


@dataclass
class Case:
    label: str
    M: float
    m: float
    v0: float
    v_end: float
    k: float
    F: float
    d_max: float
    R: List[float] = field(repr=False)
    s: List[float] = field(repr=False)
    expect: Dict[str, float] = field(repr=False)

    @property
    def N(self):
        return len(self.R)

    def ratio_fn(self):
        return net_ratio_fn(self.R, self.s, self.k)

    def spans_cm(self):
        """Span widths in engagement order, centimetres."""
        return [round(2 * r * 100) for _, r in sorted(zip(self.s, self.R))]


CANONICAL: Dict[str, Case] = {
    "100to1": Case(
        label="100:1", M=100.0, m=1.0, v0=10.0, v_end=4.0,
        k=2.0, F=999.0, d_max=0.8541214,
        R=[0.323822, 0.416875, 0.284636, 0.154325, 0.080336, 0.05, 0.05, 0.05],
        s=[0.012008, 0.264113, 0.494876, 0.65513, 0.748233, 0.798006, 0.822737,
           0.827804],
        expect=dict(width=2.820, exit_rigid=89.82, exit_compliant=100.25,
                    p2m_rigid=2.047, p2m_compliant=1.123, residual_pct=0.57)),
    "1000to1": Case(
        label="1,000:1", M=1000.0, m=1.0, v0=10.0, v_end=4.0,
        k=5.0, F=3169.6, d_max=0.8541739,
        R=[0.245859, 0.313891, 0.234921, 0.139188, 0.077318, 0.05, 0.05, 0.05,
           0.05],
        s=[0.011899, 0.226126, 0.44068, 0.605092, 0.710225, 0.77045, 0.80248,
           0.818989, 0.818997],
        expect=dict(width=2.422, exit_rigid=284.76, exit_compliant=318.16,
                    p2m_rigid=2.074, p2m_compliant=1.092, residual_pct=0.78)),
    "10000to1": Case(
        label="10,000:1", M=10000.0, m=1.0, v0=10.0, v_end=4.0,
        k=9.0, F=10033.8, d_max=0.8541708,
        R=[0.122351, 0.145927, 0.127425, 0.097121, 0.069024, 0.050238, 0.05,
           0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        s=[0.012119, 0.153535, 0.309894, 0.451219, 0.564599, 0.648136, 0.702785,
           0.739035, 0.764895, 0.786679, 0.799947, 0.799954, 0.801266],
        expect=dict(width=1.924, exit_rigid=885.90, exit_compliant=1004.78,
                    p2m_rigid=2.097, p2m_compliant=1.190, residual_pct=1.48)),
}


def load(name):
    """Fetch a canonical case by key: '100to1', '1000to1' or '10000to1'."""
    if name not in CANONICAL:
        raise KeyError(f"{name!r} not found. Available: {list(CANONICAL)}")
    return CANONICAL[name]
