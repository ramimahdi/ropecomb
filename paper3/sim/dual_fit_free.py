"""Staged dual-array solve with the cross-array engagement order as an option.

interleave=True  : the semisym_fit parametrisation. One merged sequence of positive gaps is dealt to the two arrays by
                   the fixed slot pattern lo, hi, lo, hi, ..., lo, so the engagement order is a hard constraint.
interleave=False : each array has its own positive gaps and its own stroke fraction, so the two arrays' offsets are
                   independent and members may cross. Nothing else changes: the seed is still the interleaved deal of
                   the merged single-array fit, and phase 1 still optimises the widths with all offsets frozen, which
                   favours the seeded interleaving without enforcing it.
Objective, floors and penalties are those of semisym_fit._residuals (fit / scale, sqrt(lambda) x balance, width
penalty above L_bank with lam_width, R_FLOOR 0.05 m).
"""
import numpy as np
from scipy.optimize import least_squares
from semisym_fit import falls, bank_ratio, encode, _unpack, _residuals, _softplus, _softplus_inv, R_FLOOR, interleaved_slots


def unpack_free(p, N_lo, N_hi, d_max):
    def arr(q, N):
        gaps = _softplus(q[:N]) + 1e-9; cum = np.cumsum(gaps)
        span = 1.0 / (1.0 + np.exp(-np.clip(q[N], -60, 60)))
        return d_max * span * cum / cum[-1]
    s_lo = arr(p[:N_lo + 1], N_lo); s_hi = arr(p[N_lo + 1:N_lo + N_hi + 2], N_hi)
    R = R_FLOOR + _softplus(p[N_lo + N_hi + 2:])
    return R[:N_lo], s_lo, R[N_lo:], s_hi


def encode_free(R_lo, s_lo, R_hi, s_hi, d_max):
    def arr(s):
        s = np.sort(np.asarray(s, float)); last = max(s[-1], 1e-6)
        span = min(0.999, max(1e-3, last / d_max)); cum = s / last
        gaps = np.diff(np.concatenate([[0.0], cum]))
        return np.concatenate([_softplus_inv(gaps), [np.log(span / (1 - span))]])
    o_lo, o_hi = np.argsort(s_lo), np.argsort(s_hi)
    R_all = np.concatenate([np.asarray(R_lo, float)[o_lo], np.asarray(R_hi, float)[o_hi]])
    return np.concatenate([arr(s_lo), arr(s_hi), _softplus_inv(R_all - R_FLOOR + 1e-6)])


def residuals_free(p, dd, Gstar, N_lo, N_hi, n_lo, n_hi, d_max, L_bank, lam_balance, lam_width):
    R_lo, s_lo, R_hi, s_hi = unpack_free(p, N_lo, N_hi, d_max)
    g_lo = bank_ratio(dd, R_lo, s_lo); g_hi = bank_ratio(dd, R_hi, s_hi)
    scale = max(Gstar.max(), 1e-9)
    r_fit = (n_lo * g_lo + n_hi * g_hi - Gstar) / scale
    r_bal = np.sqrt(lam_balance) * (n_lo * g_lo - n_hi * g_hi) / scale
    w_lo, w_hi = 2.0 * R_lo.sum(), 2.0 * R_hi.sum()
    r_w = np.sqrt(lam_width) * np.array([max(0.0, w_lo - L_bank), max(0.0, w_hi - L_bank)]) / max(L_bank, 1e-9)
    return np.concatenate([r_fit, r_bal, r_w])


def staged_solve(R_l, s_l, R_h, s_h, k, W, lam, dd, Gs, d_max, interleave=True, lam_width=50.0, nfev=(3000, 4000), order=None):
    """Phase 1: widths with every offset frozen. Phase 2: joint. Returns ((R_lo, s_lo, R_hi, s_hi), objective).
    With interleave=True the slot pattern is `order` (0 = leading array, 1 = the other), default lo, hi, ..., lo."""
    n_lo, n_hi = falls(k); N_lo, N_hi = len(R_l), len(R_h); n = N_lo + N_hi
    if interleave:
        order = interleaved_slots(N_lo, N_hi) if order is None else np.asarray(order)
        p0 = encode(R_l, s_l, R_h, s_h, d_max, order); nhead = n + 1
        f = lambda p: _residuals(p, dd, Gs, N_lo, N_hi, n_lo, n_hi, d_max, W, lam, lam_width, order)
        unpack = lambda p: _unpack(p, N_lo, N_hi, d_max, order)
    else:
        p0 = encode_free(R_l, s_l, R_h, s_h, d_max); nhead = n + 2
        f = lambda p: residuals_free(p, dd, Gs, N_lo, N_hi, n_lo, n_hi, d_max, W, lam, lam_width)
        unpack = lambda p: unpack_free(p, N_lo, N_hi, d_max)
    head = p0[:nhead].copy()
    s1 = least_squares(lambda q: f(np.concatenate([head, q])), p0[nhead:], method="lm", max_nfev=nfev[0])
    s2 = least_squares(f, np.concatenate([head, s1.x]), method="lm", max_nfev=nfev[1])
    return unpack(s2.x), float(2 * s2.cost)


def order_string(s_lo, s_hi, tol=0.002):
    """Engagement order, L = leading (low-fall) array, R = the other; a tie within tol is written in brackets."""
    items = sorted([(s, "L") for s in s_lo] + [(s, "R") for s in s_hi], key=lambda x: (x[0], x[1] == "R"))
    out, i = [], 0
    while i < len(items):
        j = i
        while j + 1 < len(items) and items[j + 1][0] - items[i][0] < tol: j += 1
        grp = "".join(t[1] for t in items[i:j + 1]); out.append(grp if j == i else "[" + grp + "]"); i = j + 1
    return "".join(out)


def jitter(rng, R_l, s_l, R_h, s_h, d_max, sig=0.3):
    """Perturb a seed: log-normal widths and engagement gaps per array, offsets rescaled to stay inside the stroke."""
    def one(Rx, sx):
        o = np.argsort(sx); Rx = np.asarray(Rx)[o]; sx = np.asarray(sx)[o]
        gaps = np.diff(np.concatenate([[0.0], sx])) * np.exp(rng.normal(0, sig, len(sx)))
        s_new = np.cumsum(gaps); s_new = s_new * min(1.0, 0.98 * d_max / max(s_new[-1], 1e-9))
        return np.maximum(0.05, Rx * np.exp(rng.normal(0, sig, len(Rx)))), s_new
    (a, b), (c, d_) = one(R_l, s_l), one(R_h, s_h)
    return a, b, c, d_
