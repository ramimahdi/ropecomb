"""What the guide actually carries: reaction imbalance and pitch moment.

Each fold is symmetric about its member, so the transverse components of the
rope tension cancel and every member pushes on the carriage ALONG the guide.
The reaction is axial. Array j applies

    F_j(d) = n_j G_j(d) T(d),

with T the output-member tension, and the two arrays lie on opposite sides of
the guide, so the carriage carries their sum and the guide carries what fails to
cancel:

    F_carriage = F_1 + F_2,        dF = |F_1 - F_2|.

:func:`imbalance_trace` reads both out of a simulation run, which is the honest
version of the quantity: the geometric figure max|n1 G1 - n2 G2| / max(G_net)
assumes a uniform tension, whereas a run has the tension the compliant member
actually delivers, including its ripple.

Because the arrays act at their members' positions along the carriage, the guide
bearings also see a pitch moment

    M(d) = SUM_2 x_i f_i - SUM_1 x_i f_i,

reacted as a couple M/h over a bearing spacing h. It equals e * dF only where
the two arrays' centres of pressure coincide. The positions x_i are free -- the
array ratio is a sum over members and does not depend on the order in which the
spans are laid along the array -- so :func:`pitch_moment` takes a placement.
The paper's designs are drawn, and this module defaults to, engagement order
outward from the anchor at the guide: rope enters each array at its outboard end
and flows toward the anchor, so that order passes the least rope through the
contacts and minimises both bearing revolutions and rope bending cycles.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from ropecomb.geometry import array_ratio

__all__ = ["imbalance_trace", "member_positions", "pitch_moment"]


def _geometry_of(obj) -> Dict[str, Any]:
    """Accept a DualCase, a DualFitResult or a plain dict of the four arrays."""
    if isinstance(obj, dict):
        g = obj
    else:
        g = {k: getattr(obj, k) for k in ("R_lo", "s_lo", "R_hi", "s_hi", "n_lo", "n_hi")}
    return {k: (np.asarray(g[k], dtype=float) if k.startswith(("R", "s")) else int(g[k]))
            for k in ("R_lo", "s_lo", "R_hi", "s_hi", "n_lo", "n_hi")}


def imbalance_trace(design, run: Optional[Dict[str, Any]] = None, *,
                    d: Optional[Sequence[float]] = None,
                    tension: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """Reaction imbalance through the stroke, from a simulation or from geometry.

    Parameters
    ----------
    design : DualCase, DualFitResult or dict
        Anything carrying R_lo, s_lo, R_hi, s_hi, n_lo, n_hi.
    run : dict, optional
        A run from ``simulate_compliant`` or ``simulate_rigid``. Its
        ``heavy_dist`` and ``force_target`` give the displacement samples and
        the member tension, so the reaction is reported in newtons.
    d, tension : array-like, optional
        Use instead of ``run`` to evaluate at chosen displacements, with a
        uniform tension if ``tension`` is omitted.

    Returns
    -------
    dict with
        d, F_lo, F_hi, F_carriage, dF   : traces (N, or dimensionless if no tension)
        eps                             : dF / max(F_carriage), the trace
        eps_peak, eps_mean              : peak and mean of eps
        d_at_peak                       : displacement of the worst point
        dF_peak, F_carriage_peak        : peak values, N
        eps_terminal                    : the imbalance left at the end of the stroke
        eps_geometric                   : the uniform-tension figure the fit balances
        mirror_residual                 : |n1 - n2| / k, what an equal-count mirror pair leaves

    Notes
    -----
    ``eps_geometric`` is what the synthesis objective penalises and what the
    paper's tables quote for a fitted geometry. ``eps_peak`` from a compliant
    run differs from it by the tension ripple, so the two are reported side by
    side rather than one being presented as the other.
    """
    g = _geometry_of(design)
    n_lo, n_hi = g["n_lo"], g["n_hi"]

    if run is not None:
        d_arr = np.asarray(run["heavy_dist"], dtype=float)
        T = np.asarray(run["force_target"], dtype=float)
        if d is not None:
            raise ValueError("pass either run or d, not both")
    else:
        if d is None:
            raise ValueError("pass a run, or displacements d")
        d_arr = np.asarray(d, dtype=float)
        T = (np.ones_like(d_arr) if tension is None
             else np.broadcast_to(np.asarray(tension, dtype=float), d_arr.shape))

    G_lo = array_ratio(d_arr, g["R_lo"], g["s_lo"])
    G_hi = array_ratio(d_arr, g["R_hi"], g["s_hi"])

    F_lo = n_lo * G_lo * T
    F_hi = n_hi * G_hi * T
    F_c = F_lo + F_hi
    dF = np.abs(F_lo - F_hi)

    peak_c = float(F_c.max()) if F_c.size else 0.0
    eps = dF / max(peak_c, 1e-12)
    net = n_lo * G_lo + n_hi * G_hi
    eps_geo = float(np.max(np.abs(n_lo * G_lo - n_hi * G_hi)) / max(float(net.max()), 1e-12))

    i_peak = int(np.argmax(dF)) if dF.size else 0
    k = n_lo + n_hi
    return dict(
        d=d_arr, F_lo=F_lo, F_hi=F_hi, F_carriage=F_c, dF=dF, eps=eps,
        eps_peak=float(eps.max()) if eps.size else 0.0,
        eps_mean=float(eps.mean()) if eps.size else 0.0,
        d_at_peak=float(d_arr[i_peak]) if d_arr.size else 0.0,
        dF_peak=float(dF.max()) if dF.size else 0.0,
        F_carriage_peak=peak_c,
        eps_terminal=float(eps[-1]) if eps.size else 0.0,
        eps_geometric=eps_geo,
        mirror_residual=abs(n_hi - n_lo) / float(k),
        has_tension=run is not None or tension is not None,
    )


def member_positions(R: Sequence[float], s: Sequence[float],
                     order: Optional[Sequence[int]] = None) -> np.ndarray:
    """Position of each member along the array, metres from the guide axis.

    Spans are laid contiguously, so a member sits at the centre of its own span
    plus the widths of everything between it and the guide. ``order`` gives the
    member indices from the guide outward; the default is engagement order, the
    rope-flow order the paper's drawings use.
    """
    R = np.asarray(R, dtype=float)
    s = np.asarray(s, dtype=float)
    perm = np.argsort(s) if order is None else np.asarray(order, dtype=int)
    Rp = R[perm]
    centres = np.cumsum(2.0 * Rp) - Rp
    x = np.empty_like(centres)
    x[perm] = centres
    return x


def pitch_moment(design, run: Optional[Dict[str, Any]] = None, *,
                 d: Optional[Sequence[float]] = None,
                 tension: Optional[Sequence[float]] = None,
                 order_lo: Optional[Sequence[int]] = None,
                 order_hi: Optional[Sequence[int]] = None,
                 bearing_spacing: Optional[float] = None) -> Dict[str, Any]:
    """Pitch moment on the guide, and the bearing couple that reacts it.

    Returns the trace M(d) = SUM_2 x_i f_i - SUM_1 x_i f_i, its peak, the
    equivalent offset (peak moment divided by peak carriage reaction) and, given
    a bearing spacing, the couple each bearing carries. The centres of pressure
    of the two arrays are returned as well: they coincide only when the moment
    reduces to e * dF.
    """
    g = _geometry_of(design)
    n_lo, n_hi = g["n_lo"], g["n_hi"]

    if run is not None:
        d_arr = np.asarray(run["heavy_dist"], dtype=float)
        T = np.asarray(run["force_target"], dtype=float)
    else:
        if d is None:
            raise ValueError("pass a run, or displacements d")
        d_arr = np.asarray(d, dtype=float)
        T = (np.ones_like(d_arr) if tension is None
             else np.broadcast_to(np.asarray(tension, dtype=float), d_arr.shape))

    x_lo = member_positions(g["R_lo"], g["s_lo"], order_lo)
    x_hi = member_positions(g["R_hi"], g["s_hi"], order_hi)

    def per_member(R, s):
        D = np.maximum(d_arr[:, None] - np.asarray(s)[None, :], 0.0)
        return 2.0 * D / np.sqrt(np.asarray(R)[None, :] ** 2 + D ** 2)

    f_lo = n_lo * per_member(g["R_lo"], g["s_lo"]) * T[:, None]
    f_hi = n_hi * per_member(g["R_hi"], g["s_hi"]) * T[:, None]

    M_lo = (f_lo * x_lo[None, :]).sum(axis=1)
    M_hi = (f_hi * x_hi[None, :]).sum(axis=1)
    M = M_hi - M_lo
    F_c = f_lo.sum(axis=1) + f_hi.sum(axis=1)
    peak_c = float(F_c.max()) if F_c.size else 0.0

    with np.errstate(invalid="ignore", divide="ignore"):
        xbar_lo = np.where(f_lo.sum(axis=1) > 0, M_lo / np.maximum(f_lo.sum(axis=1), 1e-12), np.nan)
        xbar_hi = np.where(f_hi.sum(axis=1) > 0, M_hi / np.maximum(f_hi.sum(axis=1), 1e-12), np.nan)

    out = dict(d=d_arr, moment=M, moment_peak=float(np.abs(M).max()) if M.size else 0.0,
               F_carriage=F_c, F_carriage_peak=peak_c,
               offset_equivalent=(float(np.abs(M).max() / peak_c) if peak_c > 0 else 0.0),
               xbar_lo=xbar_lo, xbar_hi=xbar_hi,
               x_lo=x_lo, x_hi=x_hi)
    if bearing_spacing:
        out["bearing_couple_peak"] = out["moment_peak"] / float(bearing_spacing)
    return out
