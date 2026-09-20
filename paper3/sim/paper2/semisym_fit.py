"""
Semi-symmetric RopeComb fitter.

Two banks either side of a central guide, each on its own rope, each rope dead-ended
at the (stationary) central stand. The two ropes drive separate elements of the
fixed-ratio stage, so their contributions ADD rather than being kinematically coupled:

    G_net(d) = n_lo * G_lo(d)  +  n_hi * G_hi(d)          n_lo = k//2 , n_hi = (k+1)//2

with, per bank,   G_bank(d) = SUM_i 2 D_i / sqrt(R_i^2 + D_i^2),  D_i = max(0, d - s_i).

Because block L carries n_hi falls and block R carries n_lo falls, the rope tensions are
in the ratio n_hi : n_lo. The force each bank applies to the carriage is
(fall count) x T x G_bank, so the lateral load on the central guide is driven by

    imbalance(d) = | n_lo * G_lo(d)  -  n_hi * G_hi(d) |

which is zero only if the banks are UNEQUAL, in the ratio  G_lo : G_hi = n_hi : n_lo.
At odd k a fully symmetric machine therefore carries a permanent imbalance of
|n_hi - n_lo| / k  (20% at k=5, 11% at k=9). Semi-symmetry is the condition for balance,
not merely an optimisation.

INTERLEAVING. Engagement order is imposed as a hard constraint by construction:

    s(lo_1) < s(hi_1) < s(lo_2) < s(hi_2) < ...

The bank with FEWER falls leads. G is continuous at engagement and it is the slope that
steps, so the imbalance drifts toward whichever bank engaged last at a rate set by that
bank's fall count. Leading with the low-fall bank makes the excursion both smaller and
faster to close.

Usage
-----
    from semisym_fit import fit_semisym, ideal_target
    dd, Gstar, d_max = ideal_target(M=1000., m=1., v0=10., F=3169.6, v_stop=4.0)
    out = fit_semisym(dd, Gstar, N_lo=12, N_hi=8, k=5, d_max=d_max, L_bank=2.42,
                      lam_balance=1.0, n_starts=24)
"""

import numpy as np
from scipy.optimize import least_squares

R_FLOOR = 0.05          # 10 cm minimum span, per paper section 5.8 (half-width)


# ----------------------------------------------------------------------------- target
def ideal_target(M, m, v0, F, v_stop, n=400, code_dir="."):
    """Ideal uniform-payload-force ratio profile, from the paper's own simulator."""
    import sys
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    from catabult_sim import simulate
    h = v0 ** 2 / (2 * 9.8)
    I = simulate(M, m, h, get_gear_fn=F, min_heavy_speed=v_stop, max_time=.6,
                 max_target_acc=1e9, time_unit=1e-5)
    d_max = I["summary"]["heavy_travel"]
    dd = np.linspace(0.0, d_max, n)
    return dd, np.interp(dd, I["heavy_dist"], I["gear"]), d_max


# ------------------------------------------------------------------------- kinematics
def bank_ratio(d, R, s):
    """G_bank(d). d may be an array; R, s are per-member."""
    d = np.atleast_1d(d)[:, None]
    D = np.maximum(0.0, d - s[None, :])
    return np.sum(2.0 * D / np.sqrt(R[None, :] ** 2 + D ** 2), axis=1)


def falls(k):
    """(low, high) fall counts. They sum to k; equal only when k is even."""
    return k // 2, (k + 1) // 2


def interleaved_slots(N_lo, N_hi):
    """
    Merged engagement order with the low-fall bank leading:
        lo, hi, lo, hi, ... then whichever bank still has members.
    Returns an array of 0 (low bank) / 1 (high bank), length N_lo + N_hi.
    """
    order, i, j = [], 0, 0
    while i < N_lo or j < N_hi:
        if i < N_lo:
            order.append(0); i += 1
        if j < N_hi:
            order.append(1); j += 1
    return np.array(order[:N_lo + N_hi])


# --------------------------------------------------------------------- parametrisation
def _softplus(x):
    return np.logaddexp(0.0, x)


MIN_GAP = 0.008          # 8 mm minimum separation between successive engagements


def _unpack(p, N_lo, N_hi, d_max, order):
    """
    Free vector -> (R_lo, s_lo, R_hi, s_hi).

    Offsets: the MERGED sequence is built from strictly positive gaps, so
    s_1 < s_2 < ... holds by construction; dealing them out by `order` then
    enforces the interleaving constraint exactly. A final sigmoid keeps the
    last engagement strictly inside the stroke.
    """
    n = N_lo + N_hi
    gaps = _softplus(p[:n]) + 1e-9
    cum = np.cumsum(gaps)
    span = 1.0 / (1.0 + np.exp(-np.clip(p[n], -60, 60)))              # fraction of the stroke used
    s_all = d_max * span * cum / cum[-1]
    R_all = R_FLOOR + _softplus(p[n + 1:])
    lo, hi = order == 0, order == 1
    return R_all[lo], s_all[lo], R_all[hi], s_all[hi]


def _residuals(p, dd, Gstar, N_lo, N_hi, n_lo, n_hi, d_max, L_bank,
               lam_balance, lam_width, order):
    R_lo, s_lo, R_hi, s_hi = _unpack(p, N_lo, N_hi, d_max, order)
    g_lo = bank_ratio(dd, R_lo, s_lo)
    g_hi = bank_ratio(dd, R_hi, s_hi)
    scale = max(Gstar.max(), 1e-9)
    r_fit = (n_lo * g_lo + n_hi * g_hi - Gstar) / scale
    r_bal = np.sqrt(lam_balance) * (n_lo * g_lo - n_hi * g_hi) / scale
    w_lo, w_hi = 2.0 * R_lo.sum(), 2.0 * R_hi.sum()
    r_w = np.sqrt(lam_width) * np.array([max(0.0, w_lo - L_bank),
                                         max(0.0, w_hi - L_bank)]) / max(L_bank, 1e-9)
    return np.concatenate([r_fit, r_bal, r_w])


# ---------------------------------------------------------------------------- the fit
def fit_semisym(dd, Gstar, N_lo, N_hi, k, d_max, L_bank,
                lam_balance=1.0, lam_width=50.0, n_starts=24, seed=0, verbose=False):
    """
    Fit both banks jointly. Returns the best of `n_starts` random restarts.

    N_lo : members on the bank serving k//2 falls      (leads engagement)
    N_hi : members on the bank serving (k+1)//2 falls
    L_bank : maximum span of EITHER bank, i.e. the machine length
    lam_balance : weight on the lateral-imbalance term. 0 = fit only,
                  large = force n_lo*G_lo == n_hi*G_hi at every d.
    """
    n_lo, n_hi = falls(k)
    order = interleaved_slots(N_lo, N_hi)
    n = N_lo + N_hi
    rng = np.random.default_rng(seed)
    best = None

    for _ in range(n_starts):
        p0 = np.concatenate([
            rng.normal(-0.5, 0.8, n),                        # gaps
            [rng.normal(1.5, 0.8)],                          # stroke fraction
            rng.normal(np.log(np.expm1(max(L_bank / max(n, 1), 0.02))), 0.7, n),
        ])
        try:
            sol = least_squares(
                _residuals, p0, method="lm", max_nfev=4000,
                args=(dd, Gstar, N_lo, N_hi, n_lo, n_hi, d_max, L_bank,
                      lam_balance, lam_width, order))
        except Exception:
            continue
        R_lo, s_lo, R_hi, s_hi = _unpack(sol.x, N_lo, N_hi, d_max, order)
        g_lo = bank_ratio(dd, R_lo, s_lo)
        g_hi = bank_ratio(dd, R_hi, s_hi)
        net = n_lo * g_lo + n_hi * g_hi
        rms = float(np.sqrt(np.mean((net - Gstar) ** 2)))
        w_lo, w_hi = 2.0 * R_lo.sum(), 2.0 * R_hi.sum()
        if max(w_lo, w_hi) > L_bank * 1.02:                  # reject infeasible
            continue
        imb = float(np.max(np.abs(n_lo * g_lo - n_hi * g_hi)) / max(net.max(), 1e-9))
        key = rms
        if best is None or key < best["rms"]:
            best = dict(rms=rms, imbalance=imb, R_lo=R_lo, s_lo=s_lo,
                        R_hi=R_hi, s_hi=s_hi, width_lo=w_lo, width_hi=w_hi,
                        n_lo=n_lo, n_hi=n_hi, k=k, net=net, g_lo=g_lo, g_hi=g_hi,
                        order=order, terminal=float(net[-1]))
    if best is not None and verbose:
        report(best, Gstar)
    return best


def report(out, Gstar):
    n_lo, n_hi = out["n_lo"], out["n_hi"]
    print("k=%d  falls %d (leading bank) / %d" % (out["k"], n_lo, n_hi))
    print("  net terminal ratio %.2f   target %.2f   rms %.4f"
          % (out["terminal"], Gstar[-1], out["rms"]))
    print("  peak weighted imbalance %.1f%%" % (100 * out["imbalance"]))
    print("  bank lengths  %.3f m (%d members, %d falls) / %.3f m (%d members, %d falls)"
          % (out["width_lo"], len(out["R_lo"]), n_lo,
             out["width_hi"], len(out["R_hi"]), n_hi))
    seq = "".join("L" if o == 0 else "H" for o in out["order"])
    print("  engagement order: %s" % " ".join(seq))
    for nm, R, s, f in (("LEADING bank", out["R_lo"], out["s_lo"], n_lo),
                        ("second  bank", out["R_hi"], out["s_hi"], n_hi)):
        print("\n  %s -- %d falls" % (nm, f))
        print("    %-4s %-14s %-14s" % ("#", "span 2R (cm)", "offset s (cm)"))
        for j, i in enumerate(np.argsort(s), 1):
            print("    %-4d %-14.1f %-14.1f" % (j, 200 * R[i], 100 * s[i]))


def verify_interleaving(out):
    """Hard check that the imposed ordering actually holds in the solution."""
    ev = sorted([(s, 0) for s in out["s_lo"]] + [(s, 1) for s in out["s_hi"]])
    got = [t for _, t in ev]
    want = list(out["order"])
    return got == want, "".join("L" if t == 0 else "H" for t in got)


# ------------------------------------------------------- two-stage structured seeding
def _softplus_inv(y):
    y = np.maximum(y, 1e-9)
    return np.where(y > 30, y, np.log(np.expm1(y)))


def encode(R_lo, s_lo, R_hi, s_hi, d_max, order):
    """(R,s) per bank -> free vector p, the inverse of _unpack."""
    n = len(R_lo) + len(R_hi)
    s_all = np.empty(n); R_all = np.empty(n)
    il = ih = 0
    for j, o in enumerate(order):
        if o == 0: s_all[j], R_all[j] = s_lo[il], R_lo[il]; il += 1
        else:      s_all[j], R_all[j] = s_hi[ih], R_hi[ih]; ih += 1
    idx = np.argsort(s_all); s_all = s_all[idx]; R_all = R_all[idx]
    span = min(0.999, max(1e-3, s_all[-1] / d_max))
    cum = s_all / s_all[-1]
    gaps = np.diff(np.concatenate([[0.0], cum]))
    p = np.concatenate([_softplus_inv(gaps),
                        [np.log(span / (1 - span))],
                        _softplus_inv(R_all - R_FLOOR + 1e-6)])
    return p


def seed_two_stage(R_star, s_star, d_max):
    """
    Stage-2 seed from a single-comb solution (R*, s*):
      leading bank  = the stage-1 solution as-is
      second bank   = same spans, offsets at the MIDPOINTS of the leading bank's,
                      so the two interleave exactly.
    Net ratio is already exact because n_lo + n_hi = k.
    """
    o = np.argsort(s_star)
    R = np.asarray(R_star)[o]; s = np.asarray(s_star)[o]
    mid = np.empty_like(s)
    mid[:-1] = 0.5 * (s[:-1] + s[1:])
    mid[-1] = s[-1] + 0.5 * (d_max - s[-1])
    return (R.copy(), s.copy(), R.copy(), mid)


def fit_semisym_seeded(dd, Gstar, R_star, s_star, k, d_max, L_bank,
                       lam_balance=1.0, lam_width=50.0, jitter=0.0, seed=0):
    """One fit started from the two-stage structured seed."""
    from scipy.optimize import least_squares
    n_lo, n_hi = falls(k)
    N = len(R_star)
    order = interleaved_slots(N, N)
    R_l, s_l, R_h, s_h = seed_two_stage(R_star, s_star, d_max)
    p0 = encode(R_l, s_l, R_h, s_h, d_max, order)
    if jitter:
        p0 = p0 + np.random.default_rng(seed).normal(0.0, jitter, p0.shape)
    sol = least_squares(_residuals, p0, method="lm", max_nfev=4000,
                        args=(dd, Gstar, N, N, n_lo, n_hi, d_max, L_bank,
                              lam_balance, lam_width, order))
    R_lo, s_lo, R_hi, s_hi = _unpack(sol.x, N, N, d_max, order)
    g_lo = bank_ratio(dd, R_lo, s_lo); g_hi = bank_ratio(dd, R_hi, s_hi)
    net = n_lo * g_lo + n_hi * g_hi
    return dict(rms=float(np.sqrt(np.mean((net - Gstar) ** 2))),
                imbalance=float(np.max(np.abs(n_lo*g_lo - n_hi*g_hi))/max(net.max(),1e-9)),
                R_lo=R_lo, s_lo=s_lo, R_hi=R_hi, s_hi=s_hi,
                width_lo=2*R_lo.sum(), width_hi=2*R_hi.sum(),
                n_lo=n_lo, n_hi=n_hi, k=k, order=order, terminal=float(net[-1]), net=net)


def fit_semisym_staged(dd, Gstar, R_star, s_star, k, d_max, L_bank,
                       lam_balance=1.0, lam_width=50.0, jitter=0.0, seed=0):
    """
    Three-phase solve, per the non-convexity of the joint problem:
      phase 0  seed offsets from the single-comb solution (second bank at midpoints)
      phase 1  solve SPAN WIDTHS ONLY with the offsets frozen
      phase 2  solve everything jointly from there
    Freezing the offsets first keeps the engagement pattern from collapsing
    (two offsets migrating onto each other) before the widths have adapted.
    """
    from scipy.optimize import least_squares
    n_lo, n_hi = falls(k)
    N = len(R_star); n = 2 * N
    order = interleaved_slots(N, N)
    R_l, s_l, R_h, s_h = seed_two_stage(R_star, s_star, d_max)
    p0 = encode(R_l, s_l, R_h, s_h, d_max, order)
    if jitter:
        p0 = p0 + np.random.default_rng(seed).normal(0.0, jitter, p0.shape)
    args = (dd, Gstar, N, N, n_lo, n_hi, d_max, L_bank, lam_balance, lam_width, order)

    # ---- phase 1: widths only, offsets frozen -------------------------------
    head = p0[:n + 1].copy()

    def res_widths(q):
        return _residuals(np.concatenate([head, q]), *args)

    s1 = least_squares(res_widths, p0[n + 1:], method="lm", max_nfev=3000)
    p1 = np.concatenate([head, s1.x])

    # ---- phase 2: joint ------------------------------------------------------
    s2 = least_squares(_residuals, p1, method="lm", max_nfev=4000, args=args)

    R_lo, s_lo, R_hi, s_hi = _unpack(s2.x, N, N, d_max, order)
    g_lo = bank_ratio(dd, R_lo, s_lo); g_hi = bank_ratio(dd, R_hi, s_hi)
    net = n_lo * g_lo + n_hi * g_hi
    alls = np.sort(np.concatenate([s_lo, s_hi]))
    return dict(rms=float(np.sqrt(np.mean((net - Gstar) ** 2))),
                imbalance=float(np.max(np.abs(n_lo*g_lo - n_hi*g_hi))/max(net.max(),1e-9)),
                R_lo=R_lo, s_lo=s_lo, R_hi=R_hi, s_hi=s_hi,
                width_lo=2*R_lo.sum(), width_hi=2*R_hi.sum(),
                n_lo=n_lo, n_hi=n_hi, k=k, order=order, terminal=float(net[-1]),
                min_gap=float(np.min(np.diff(alls))), net=net)
