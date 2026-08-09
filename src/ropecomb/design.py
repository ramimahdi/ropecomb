"""Algorithm 1: the complete design procedure of section 4.7.

Nothing in it is tuned. The only quantities a designer supplies are the masses,
the entry velocity and the deceleration to be achieved.

    1  SOLVE THE TARGET
       bisect on F until the ideal run, terminated at v_h = v_end, takes time T
       -> design force F, braking distance d_max, target profile G*(d)

    2  GRID SEARCH
       for N = 2 .. 16, for k in {1,2,3,5,7,9}, for restart = 1 .. 50:
           fit {R,s} by bounded least squares; no other terms in the objective
       keep the 5 restarts of lowest residual

    3  EVALUATE each kept candidate
       simulate rigid and compliant, prune members that never engage,
       re-simulate, take peak-to-mean over the first 99.5% of the trace

    4  SELECT
       admissible: p2m_compliant <= 1.3 AND v_compliant >= 0.99 * v_ideal
       score = (55 - N_eff - k - W)/50 * v_rigid * v_compliant
                                       / sqrt(p2m_r + p2m_c)
       return the highest-scoring admissible candidate

Two properties are worth stating explicitly. No dynamic simulation enters the
fit, which is why 4,500 fits per case are affordable and only the 450 survivors
are ever simulated. And the grid is not a refinement of a default: member count
and fixed-stage ratio are not separable, so searching them jointly is necessary
rather than merely thorough.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .compliant import simulate_compliant
from .fit import fit_array
from .geometry import comb_width, net_ratio_fn
from .metrics import peak_to_mean
from .rigid import simulate_rigid
from .target import target_profile

__all__ = ["Candidate", "evaluate", "design", "DEFAULT_N_RANGE",
           "DEFAULT_K_VALUES", "NOMINAL_STIFFNESS"]

DEFAULT_N_RANGE = tuple(range(2, 17))
DEFAULT_K_VALUES = (1.0, 2.0, 3.0, 5.0, 7.0, 9.0)
NOMINAL_STIFFNESS = 16471.0     # N/m, the 2 mm UHMWPE member of the worked case


@dataclass
class Candidate:
    """One fitted, pruned and simulated geometry."""

    N_fit: int
    N_eff: int
    k: float
    width: float
    rms: float
    rms_pct: float
    pruned: int
    exit_rigid: float
    exit_compliant: float
    p2m_rigid: float
    p2m_compliant: float
    peak_over_design: float
    admissible: bool
    score: float
    R: np.ndarray = field(repr=False)
    s: np.ndarray = field(repr=False)

    def ratio_fn(self):
        return net_ratio_fn(self.R, self.s, self.k)

    def __str__(self):
        flag = " " if self.admissible else "*"
        return (f"N={self.N_eff:<3d} k={self.k:<3.0f} W={self.width:5.2f} m  "
                f"exit {self.exit_rigid:7.1f} / {self.exit_compliant:7.1f} m/s  "
                f"p2m {self.p2m_rigid:5.3f} / {self.p2m_compliant:5.3f}  "
                f"resid {self.rms_pct:5.2f}%  score {self.score:9.1f}{flag}")


def _score(N_eff, k, width, v_rigid, v_compliant, p2m_r, p2m_c):
    """The width-penalised score of section 4.7.

    Rewards exit velocity in both models, penalises peak-to-mean force in both,
    and discounts cost through N_eff + k + W: member count, fixed-stage ratio
    and array width, the three quantities that set what the machine costs to
    build. Width enters in metres with unit weight, which at these scales makes
    a metre of array about as expensive as one engagement member.

    Its exact form matters less than it appears. Across the 1,000:1 grid,
    scoring on the rigid model alone and scoring on both agree on the winner,
    and the top ten candidates by either measure lie within 0.3% in exit
    velocity. What the score mainly does is break ties among designs that have
    already saturated the ideal.
    """
    return ((55.0 - N_eff - k - width) / 50.0
            * v_rigid * v_compliant / math.sqrt(p2m_r + p2m_c))


def evaluate(target, R, s, k, M, m, *, v0=10.0, k_rope=NOMINAL_STIFFNESS,
             p2m_limit=1.3, exit_limit=0.99, dt_rigid=1e-5, dt_compliant=2e-5,
             max_target_acc=3e4):
    """Step 3 and 4 for one geometry: prune, simulate, score.

    Pruning is evaluated on the LONGER of the rigid and compliant strokes.
    A compliant output member lets the carriage travel further, so members that
    look redundant in the rigid model do engage in the real machine.
    """
    R = np.asarray(R, dtype=float)
    s = np.asarray(s, dtype=float)
    d_max = target.d_max
    F = target.F

    g0 = net_ratio_fn(R, s, k)
    reach_rigid = simulate_rigid(M, m, v0, g0, max_heavy_dist=d_max,
                                 max_target_acc=max_target_acc,
                                 dt=dt_rigid)["summary"]["heavy_travel"]
    reach_comp = simulate_compliant(M, m, v0, g0, k_rope, pretension=F,
                                    max_heavy_dist=d_max,
                                    dt=dt_compliant)["summary"]["heavy_travel"]

    keep = s < max(reach_rigid, reach_comp)
    Rp, sp = R[keep], s[keep]
    if len(Rp) < 2:
        return None

    gp = net_ratio_fn(Rp, sp, k)
    rigid = simulate_rigid(M, m, v0, gp, max_heavy_dist=d_max,
                           max_target_acc=max_target_acc, dt=dt_rigid)
    comp = simulate_compliant(M, m, v0, gp, k_rope, pretension=F,
                              max_heavy_dist=d_max, dt=dt_compliant)

    from .geometry import array_ratio
    rms = float(np.sqrt(np.mean((k * array_ratio(target.d, Rp, sp) - target.G) ** 2)))

    v_r = float(rigid["summary"]["target_final_speed"])
    v_c = float(comp["summary"]["target_final_speed"])
    p2m_r = peak_to_mean(rigid["force_target"])
    p2m_c = peak_to_mean(comp["force_target"])
    W = comb_width(Rp)
    N_eff = int(len(Rp))

    return Candidate(
        N_fit=int(len(R)), N_eff=N_eff, k=float(k), width=W,
        rms=rms, rms_pct=100.0 * rms / target.full_scale,
        pruned=int(len(R) - N_eff),
        exit_rigid=v_r, exit_compliant=v_c,
        p2m_rigid=p2m_r, p2m_compliant=p2m_c,
        peak_over_design=float(np.max(comp["force_target"]) / F),
        admissible=bool(p2m_c <= p2m_limit and v_c >= exit_limit * target.ideal_exit),
        score=_score(N_eff, k, W, v_r, v_c, p2m_r, p2m_c),
        R=Rp, s=sp)


def design(M, m=1.0, *, v0=10.0, v_end=4.0, stroke_time=0.1,
           N_range=DEFAULT_N_RANGE, k_values=DEFAULT_K_VALUES,
           restarts=50, keep=5, k_rope=NOMINAL_STIFFNESS, target=None,
           progress=None):
    """Run Algorithm 1 end to end and return every simulated candidate.

    Sort the result by ``.score`` over the candidates with ``.admissible`` true
    to get the design the procedure selects.

    This is the full grid: 15 member counts x 6 stage ratios x ``restarts``
    fits, of which ``keep`` per cell are simulated. At the published settings
    that is 4,500 fits and 450 simulations per case, of order twenty minutes.
    Narrow ``N_range`` and ``k_values`` for a quick look.

    Parameters
    ----------
    progress : callable, optional
        Called as progress(N, k, candidates_so_far) after each grid cell.
    """
    if target is None:
        target = target_profile(M, m, v0, v_end, stroke_time)

    out = []
    for N in N_range:
        for k in k_values:
            fits = []
            for seed in range(restarts):
                try:
                    fits.append(fit_array(target, N, k, seed=seed))
                except Exception:
                    continue
            if not fits:
                continue
            fits.sort(key=lambda f: f.rms)
            for f in fits[:keep]:
                cand = evaluate(target, f.R, f.s, k, M, m, v0=v0, k_rope=k_rope)
                if cand is not None:
                    out.append(cand)
            if progress is not None:
                progress(N, k, out)
    return out


def best(candidates):
    """The highest-scoring admissible candidate, or None."""
    ok = [c for c in candidates if c.admissible]
    return max(ok, key=lambda c: c.score) if ok else None
