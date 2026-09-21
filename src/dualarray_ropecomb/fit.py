"""Optimization and staged fitting for dual-array RopeComb transmissions.

The joint optimization of two arrays with interleaving constraints and
transverse load balancing is non-convex. The staged solve resolves this in
three phases:
  Phase 0: Seed offsets from a single-array fit, with the second array's offsets
           placed at the midpoints of the lead array (exact interleaving).
  Phase 1: Freeze engagement offsets and solve for span widths only. This prevents
           offsets from collapsing onto each other before widths adapt.
  Phase 2: Solve widths and offsets jointly.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import least_squares

from ropecomb.geometry import array_ratio
from ropecomb.target import TargetSpec

from .geometry import dual_net_ratio
from .ordering import interleaved_slots
from .weighting import falls, guide_imbalance

__all__ = [
    "R_FLOOR", "encode", "seed_two_stage", "seed_merged", "jitter_seed",
    "fit_dual_staged", "fit_dual", "DualFitResult"
]

R_FLOOR = 0.05  # 10 cm minimum span (5 cm half-width) per paper section 5.8


def _softplus(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x > 30.0, x, np.log1p(np.exp(np.clip(x, -50.0, 30.0))))


def _softplus_inv(y: np.ndarray) -> np.ndarray:
    y = np.maximum(np.asarray(y, dtype=float), 1e-9)
    return np.where(y > 30.0, y, np.log(np.expm1(y)))


def _unpack(p: np.ndarray, N_lo: int, N_hi: int, d_max: float,
            order: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unpack unconstrained vector p into (R_lo, s_lo, R_hi, s_hi) preserving ordering."""
    n = N_lo + N_hi
    gaps = _softplus(p[:n])
    cum = np.cumsum(gaps)
    s_norm = cum / max(cum[-1], 1e-9)

    val = max(-60.0, min(60.0, float(p[n])))
    s_span = float(1.0 / (1.0 + math.exp(-val)))
    s_all = s_norm * (s_span * d_max)

    R_all = R_FLOOR + _softplus(p[n + 1: n + 1 + n])

    R_lo, s_lo = [], []
    R_hi, s_hi = [], []
    for o, Ri, si in zip(order, R_all, s_all):
        if o == 0:
            R_lo.append(Ri)
            s_lo.append(si)
        else:
            R_hi.append(Ri)
            s_hi.append(si)

    return (np.asarray(R_lo, dtype=float), np.asarray(s_lo, dtype=float),
            np.asarray(R_hi, dtype=float), np.asarray(s_hi, dtype=float))


def encode(R_lo: Sequence[float], s_lo: Sequence[float],
           R_hi: Sequence[float], s_hi: Sequence[float],
           d_max: float, order: Sequence[int]) -> np.ndarray:
    """Invert _unpack: map (R, s) per array into free parameter vector p."""
    n = len(R_lo) + len(R_hi)
    s_all = np.empty(n)
    R_all = np.empty(n)
    il, ih = 0, 0
    for j, o in enumerate(order):
        if o == 0:
            s_all[j], R_all[j] = s_lo[il], R_lo[il]
            il += 1
        else:
            s_all[j], R_all[j] = s_hi[ih], R_hi[ih]
            ih += 1

    idx = np.argsort(s_all)
    s_all = s_all[idx]
    R_all = R_all[idx]

    span = min(0.999, max(1e-3, float(s_all[-1] / d_max)))
    cum = s_all / max(s_all[-1], 1e-9)
    gaps = np.diff(np.concatenate([[0.0], cum]))
    p = np.concatenate([
        _softplus_inv(gaps),
        [math.log(span / (1.0 - span))],
        _softplus_inv(R_all - R_FLOOR + 1e-6)
    ])
    return p


def seed_two_stage(R_star: Sequence[float], s_star: Sequence[float],
                   d_max: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate two-stage seed from a single-array solution (R*, s*).

    Leading array: single-array solution as-is.
    Second array:  same span widths, offsets at midpoints of leading array.
    """
    order = np.argsort(s_star)
    R = np.asarray(R_star, dtype=float)[order]
    s = np.asarray(s_star, dtype=float)[order]
    mid = np.empty_like(s)
    mid[:-1] = 0.5 * (s[:-1] + s[1:])
    mid[-1] = s[-1] + 0.5 * (d_max - s[-1])
    return R.copy(), s.copy(), R.copy(), mid


def seed_merged(target, N_lo: int, N_hi: int, k: int, *,
                width_budget: float, width_relief: float = 3.0,
                seed: int = 0, max_nfev: int = 20000
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Merged seed: fit N_lo + N_hi members at half the stage ratio, then deal them.

    A dual array at stage ratio k with two arrays of N_lo and N_hi members has,
    late in the stroke, about the reach of ONE array of N_lo + N_hi members at
    stage ratio k/2. So a single-array fit at that reduced ratio is a feasible
    starting point for the pair, and dealing its members alternately in
    engagement order splits it into two arrays that already interleave.

    The leading array takes the first member and any excess, so the members the
    leading array carries beyond alternation engage LAST. Dealing them first
    costs a factor of six in peak imbalance, and dealing them in the middle two
    to five; the rule is not cosmetic.

    The width cap is relaxed by ``width_relief`` for the seed fit only, because
    the merged array is the two real arrays laid end to end and must not be
    squeezed into one array's budget.

    Parameters
    ----------
    target : TargetSpec
    N_lo, N_hi : int
        Members on the leading (fewer falls) and trailing arrays.
    k : int
        The FULL stage ratio; the seed fit uses k/2.
    seed : int
        Restart index passed to the single-array fit.

    Returns
    -------
    (R_lo, s_lo, R_hi, s_hi, order)
        The dealt geometry and the slot assignment it realises.
    """
    from ropecomb.fit import fit_array

    N = int(N_lo) + int(N_hi)
    merged = fit_array(target, N, k / 2.0, seed=seed,
                       max_width=width_relief * float(width_budget),
                       max_nfev=max_nfev)

    order = interleaved_slots(int(N_lo), int(N_hi))
    R = np.asarray(merged.R, dtype=float)
    s = np.asarray(merged.s, dtype=float)
    idx = np.argsort(s)
    R, s = R[idx], s[idx]

    R_lo, s_lo, R_hi, s_hi = [], [], [], []
    for slot, Ri, si in zip(order, R, s):
        if slot == 0:
            R_lo.append(Ri); s_lo.append(si)
        else:
            R_hi.append(Ri); s_hi.append(si)
    return (np.asarray(R_lo), np.asarray(s_lo),
            np.asarray(R_hi), np.asarray(s_hi), order)


def jitter_seed(R_lo, s_lo, R_hi, s_hi, d_max: float, *, sigma: float = 0.3,
                rng: Optional[np.random.Generator] = None):
    """Perturb a dealt seed log-normally, for restarts that stay feasible.

    The seed fit is effectively unique: at half the stage ratio and with a
    generous width cap, independent random starts land on the same geometry to
    within a few millimetres. Restarting from fresh random starts therefore
    explores nothing, and restarting from the seed unperturbed explores nothing
    either. Perturbing spans and engagement gaps by lognormal(0, sigma) keeps the
    ordering and the feasibility of the seed while moving it.
    """
    rng = np.random.default_rng() if rng is None else rng

    def one(Rx, sx):
        o = np.argsort(sx)
        Rx = np.asarray(Rx, dtype=float)[o]
        sx = np.asarray(sx, dtype=float)[o]
        gaps = np.diff(np.concatenate([[0.0], sx])) * rng.lognormal(0.0, sigma, len(sx))
        s_new = np.cumsum(gaps)
        s_new = s_new * min(1.0, 0.98 * d_max / max(float(s_new[-1]), 1e-9))
        return np.maximum(R_FLOOR, Rx * rng.lognormal(0.0, sigma, len(Rx))), s_new

    (R_lo2, s_lo2), (R_hi2, s_hi2) = one(R_lo, s_lo), one(R_hi, s_hi)
    return R_lo2, s_lo2, R_hi2, s_hi2


def _residuals(p: np.ndarray, d: np.ndarray, G_star: np.ndarray,
               N_lo: int, N_hi: int, n_lo: int, n_hi: int,
               d_max: float, max_width: float,
               lam_balance: float, lam_width: float,
               order: Sequence[int]) -> np.ndarray:
    R_lo, s_lo, R_hi, s_hi = _unpack(p, N_lo, N_hi, d_max, order)
    g_lo = array_ratio(d, R_lo, s_lo)
    g_hi = array_ratio(d, R_hi, s_hi)
    scale = max(float(G_star.max()), 1e-9)

    r_fit = (n_lo * g_lo + n_hi * g_hi - G_star) / scale
    r_bal = np.sqrt(lam_balance) * (n_lo * g_lo - n_hi * g_hi) / scale
    w_lo = 2.0 * float(np.sum(R_lo))
    w_hi = 2.0 * float(np.sum(R_hi))
    r_w = np.sqrt(lam_width) * np.array([
        max(0.0, w_lo - max_width),
        max(0.0, w_hi - max_width)
    ]) / max(max_width, 1e-9)
    return np.concatenate([r_fit, r_bal, r_w])


class DualFitResult:
    """Result of a dual-array least-squares fit."""

    def __init__(self, data: Dict[str, Any]):
        self.rms: float = float(data["rms"])
        self.imbalance: float = float(data["imbalance"])
        self.R_lo: np.ndarray = np.asarray(data["R_lo"])
        self.s_lo: np.ndarray = np.asarray(data["s_lo"])
        self.R_hi: np.ndarray = np.asarray(data["R_hi"])
        self.s_hi: np.ndarray = np.asarray(data["s_hi"])
        self.width_lo: float = float(data["width_lo"])
        self.width_hi: float = float(data["width_hi"])
        self.n_lo: int = int(data["n_lo"])
        self.n_hi: int = int(data["n_hi"])
        self.k: int = int(data["k"])
        self.order: List[int] = list(data["order"])
        self.terminal: float = float(data["terminal"])
        self.net: np.ndarray = np.asarray(data["net"])
        self.raw: Dict[str, Any] = data

    def __repr__(self) -> str:
        return (f"DualFitResult(k={self.k}, rms={self.rms:.4f}, "
                f"imbalance={100*self.imbalance:.1f}%, "
                f"widths={self.width_lo:.2f}/{self.width_hi:.2f}m)")

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]


def fit_dual_staged(target_or_d: Union[TargetSpec, Sequence[float]],
                    G_star: Optional[Sequence[float]] = None,
                    R_star: Sequence[float] = None,
                    s_star: Sequence[float] = None,
                    k: int = 7,
                    d_max: Optional[float] = None,
                    width_budget: float = 2.11,
                    lam_balance: float = 1.0,
                    lam_width: float = 50.0,
                    jitter: float = 0.0,
                    seed: int = 0,
                    N_lo: Optional[int] = None,
                    N_hi: Optional[int] = None,
                    restarts: int = 1,
                    jitter_sigma: float = 0.3) -> DualFitResult:
    """Execute the three-phase staged fit for a dual-array transmission.

    Parameters
    ----------
    target_or_d : TargetSpec or array-like
        Either a TargetSpec from ropecomb.target_profile, or displacement array d.
    G_star : array-like, optional
        Target net ratio. Required if target_or_d is an array.
    R_star, s_star : sequence of float
        Single-array fit solution from step 1 used to seed the dual array.
    k : int
        Fixed-stage ratio.
    d_max : float, optional
        Maximum carriage displacement.
    width_budget : float
        Maximum span width (2*sum(R)) allowed per array, metres.
    lam_balance : float
        Weight on the lateral guide-load imbalance penalty.
    lam_width : float
        Weight on the width budget constraint penalty.
    jitter : float
        Standard deviation of a Gaussian perturbation on the encoded seed. Kept
        for compatibility; ``restarts`` with ``jitter_sigma`` is the method the
        paper uses.
    seed : int
        Random seed.
    N_lo, N_hi : int, optional
        Members on the leading and trailing arrays. Give these, with a
        TargetSpec, to seed by the merged method of :func:`seed_merged`, which
        is what unequal member counts require: the arrays then need not have the
        same number of members, and the excess on the leading array engages
        last. Omit them to seed from a supplied single-array fit (R_star,
        s_star) with equal counts, as before.
    restarts : int
        Number of starts. The first is the dealt seed itself; the rest are
        log-normal perturbations of it (:func:`jitter_seed`), because the seed
        fit is unique to within a few millimetres and fresh random starts would
        explore nothing. The best by fit residual is returned.
    jitter_sigma : float
        Log-normal sigma for those perturbations.

    Examples
    --------
    Count-balanced at k = 7, eight members leading and six trailing::

        spec = target_profile(1000.0, 1.0, 10.0, 4.0, 0.1)
        fit = fit_dual_staged(spec, k=7, N_lo=8, N_hi=6,
                              width_budget=2.11, lam_balance=0.03, restarts=31)
    """
    if isinstance(target_or_d, TargetSpec):
        target = target_or_d
        d = target_or_d.d
        G_star = target_or_d.G
        d_max = target_or_d.d_max
    else:
        target = None
        d = np.asarray(target_or_d, dtype=float)
        G_star = np.asarray(G_star, dtype=float)
        d_max = float(d_max if d_max is not None else d[-1])

    n_lo, n_hi = falls(k)

    if N_lo is not None or N_hi is not None:
        if N_lo is None or N_hi is None:
            raise ValueError("give both N_lo and N_hi, or neither")
        if target is None:
            raise ValueError("the merged seed needs a TargetSpec, not raw (d, G_star)")
        N_l, N_h = int(N_lo), int(N_hi)
        R_l, s_l, R_h, s_h, order = seed_merged(target, N_l, N_h, k,
                                                width_budget=width_budget, seed=seed)
    else:
        if R_star is None or s_star is None:
            raise ValueError("give either (N_lo, N_hi) or a single-array seed (R_star, s_star)")
        N_l = N_h = len(R_star)
        order = interleaved_slots(N_l, N_h)
        R_l, s_l, R_h, s_h = seed_two_stage(R_star, s_star, d_max)

    n = N_l + N_h
    args = (d, G_star, N_l, N_h, n_lo, n_hi, d_max, width_budget, lam_balance, lam_width, order)
    rng = np.random.default_rng(seed)

    def solve_from(p0: np.ndarray):
        head = p0[:n + 1].copy()

        def res_widths(q):
            return _residuals(np.concatenate([head, q]), *args)

        # Phase 1: freeze offsets, solve widths only
        s1 = least_squares(res_widths, p0[n + 1:], method="lm", max_nfev=3000)
        p1 = np.concatenate([head, s1.x])
        # Phase 2: joint solve
        return least_squares(_residuals, p1, method="lm", max_nfev=4000, args=args)

    best = None
    for start in range(max(1, int(restarts))):
        if start == 0:
            Rl, sl, Rh, sh = R_l, s_l, R_h, s_h
        else:
            Rl, sl, Rh, sh = jitter_seed(R_l, s_l, R_h, s_h, d_max,
                                         sigma=jitter_sigma, rng=rng)
        p0 = encode(Rl, sl, Rh, sh, d_max, order)
        if jitter > 0.0:
            p0 = p0 + rng.normal(0.0, jitter, p0.shape)
        try:
            sol = solve_from(p0)
        except Exception:
            continue

        R_lo, s_lo, R_hi, s_hi = _unpack(sol.x, N_l, N_h, d_max, order)
        g_lo = array_ratio(d, R_lo, s_lo)
        g_hi = array_ratio(d, R_hi, s_hi)
        net = n_lo * g_lo + n_hi * g_hi
        cost = float(2.0 * sol.cost)
        if best is not None and cost >= best["cost"]:
            continue
        rms = float(np.sqrt(np.mean((net - G_star) ** 2)))
        best = {
            "cost": cost,
            "rms": rms,
            "imbalance": guide_imbalance(net, g_lo, g_hi, n_lo, n_hi),
            "R_lo": R_lo, "s_lo": s_lo, "R_hi": R_hi, "s_hi": s_hi,
            "width_lo": 2.0 * float(np.sum(R_lo)),
            "width_hi": 2.0 * float(np.sum(R_hi)),
            "n_lo": n_lo, "n_hi": n_hi, "k": k, "order": order,
            "terminal": float(net[-1]), "net": net,
            "g_lo": g_lo, "g_hi": g_hi, "solution": sol,
            "N_lo": N_l, "N_hi": N_h, "restart": start,
        }

    if best is None:
        raise RuntimeError("every start failed to converge")
    return DualFitResult(best)


def fit_dual(target_or_d: Union[TargetSpec, Sequence[float]],
             G_star: Optional[Sequence[float]] = None,
             N_lo: int = 7, N_hi: int = 7,
             k: int = 7,
             d_max: Optional[float] = None,
             width_budget: float = 2.11,
             lam_balance: float = 1.0,
             lam_width: float = 50.0,
             n_starts: int = 20,
             seed: int = 0) -> Optional[DualFitResult]:
    """Multi-start free optimization for dual-array geometry."""
    if isinstance(target_or_d, TargetSpec):
        d = target_or_d.d
        G_star = target_or_d.G
        d_max = target_or_d.d_max
    else:
        d = np.asarray(target_or_d, dtype=float)
        G_star = np.asarray(G_star, dtype=float)
        d_max = float(d_max if d_max is not None else d[-1])

    n_lo, n_hi = falls(k)
    order = interleaved_slots(N_lo, N_hi)
    n = N_lo + N_hi
    rng = np.random.default_rng(seed)
    best = None

    args = (d, G_star, N_lo, N_hi, n_lo, n_hi, d_max, width_budget, lam_balance, lam_width, order)

    for _ in range(n_starts):
        p0 = np.concatenate([
            rng.normal(-0.5, 0.8, n),
            [rng.normal(1.5, 0.8)],
            rng.normal(math.log(math.expm1(max(width_budget / max(n, 1), 0.02))), 0.7, n),
        ])
        try:
            sol = least_squares(_residuals, p0, method="lm", max_nfev=4000, args=args)
        except Exception:
            continue
        R_lo, s_lo, R_hi, s_hi = _unpack(sol.x, N_lo, N_hi, d_max, order)
        w_lo, w_hi = 2.0 * float(np.sum(R_lo)), 2.0 * float(np.sum(R_hi))
        if max(w_lo, w_hi) > width_budget * 1.02:
            continue

        g_lo = array_ratio(d, R_lo, s_lo)
        g_hi = array_ratio(d, R_hi, s_hi)
        net = n_lo * g_lo + n_hi * g_hi
        rms = float(np.sqrt(np.mean((net - G_star) ** 2)))
        imb = guide_imbalance(net, g_lo, g_hi, n_lo, n_hi)

        if best is None or rms < best["rms"]:
            best = {
                "rms": rms, "imbalance": imb,
                "R_lo": R_lo, "s_lo": s_lo, "R_hi": R_hi, "s_hi": s_hi,
                "width_lo": w_lo, "width_hi": w_hi,
                "n_lo": n_lo, "n_hi": n_hi, "k": k,
                "order": order, "terminal": float(net[-1]),
                "net": net, "g_lo": g_lo, "g_hi": g_hi, "solution": sol
            }

    return DualFitResult(best) if best is not None else None
