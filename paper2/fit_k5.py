"""Fit and freeze a k = 5 dual-array design, and measure what the engagement
ordering is worth.

Three arrangements at the same member count, width budget and target:

  1. mirror-symmetric      two identical arrays; the lateral load is the
                           structural |n1-n2|/k and no fit can alter it
  2. unequal, order free   the two arrays differ, but their engagement offsets
                           are chosen independently with no interleaving
  3. unequal, interleaved  offsets drawn from one merged increasing sequence,
                           dealt alternately, low-fall array leading

Writes k5_design.json with the full geometry of (3).

    python3 fit_k5.py
"""

import os
import sys
import json
import math
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from ropecomb import target_profile, fit_array, simulate_rigid, simulate_compliant
from ropecomb.geometry import array_ratio
from dualarray_ropecomb.weighting import falls
from dualarray_ropecomb.ordering import interleaved_slots
from dualarray_ropecomb.fit import (_softplus, _residuals, _unpack, R_FLOOR)
from dualarray_ropecomb.geometry import net_ratio_fn

M, m, v0, F, VSTOP, KROPE, ETA = 1000.0, 1.0, 10.0, 3169.6, 4.0, 12000, 0.98
K = 5

spec = target_profile(M, m, v0, VSTOP, F=F, n=400)
dd, Gs, d_max = spec.d, spec.G, spec.d_max
Gt = float(Gs[-1])
n_lo, n_hi = falls(K)                      # 2 and 3
N_LO = math.ceil(Gt / (2 * n_lo) / 2)      # array serving 2 falls (leads) -> 10
N_HI = math.ceil(Gt / (2 * n_hi) / 2)      # array serving 3 falls -> 7
WIDTH = 2.11
print("k = %d, falls %d/%d, terminal target %.2f" % (K, n_lo, n_hi, Gt))
print("member floors: %d on the %d-fall array, %d on the %d-fall array\n"
      % (N_LO, n_lo, N_HI, n_hi))


def _unpack_free(p, N_lo, N_hi, d_max):
    """Each array's offsets from its own increasing sequence: no interleaving."""
    a = 0
    g_lo = _softplus(p[a:a + N_lo]) + 1e-9
    a += N_lo
    sp_lo = 1.0 / (1.0 + np.exp(-np.clip(p[a], -60, 60)))
    a += 1
    g_hi = _softplus(p[a:a + N_hi]) + 1e-9
    a += N_hi
    sp_hi = 1.0 / (1.0 + np.exp(-np.clip(p[a], -60, 60)))
    a += 1
    R_lo = R_FLOOR + _softplus(p[a:a + N_lo])
    a += N_lo
    R_hi = R_FLOOR + _softplus(p[a:a + N_hi])
    s_lo = d_max * sp_lo * np.cumsum(g_lo) / np.cumsum(g_lo)[-1]
    s_hi = d_max * sp_hi * np.cumsum(g_hi) / np.cumsum(g_hi)[-1]
    return R_lo, s_lo, R_hi, s_hi


def _resid_free(p, dd, Gstar, N_lo, N_hi, n_lo, n_hi, d_max, L, lam_b, lam_w):
    R_lo, s_lo, R_hi, s_hi = _unpack_free(p, N_lo, N_hi, d_max)
    g_lo, g_hi = array_ratio(dd, R_lo, s_lo), array_ratio(dd, R_hi, s_hi)
    net = n_lo * g_lo + n_hi * g_hi
    r = [net - Gstar, np.sqrt(lam_b) * (n_lo * g_lo - n_hi * g_hi)]
    for R in (R_lo, R_hi):
        r.append([np.sqrt(lam_w) * max(0.0, 2.0 * R.sum() - L)])
    return np.concatenate([np.atleast_1d(x).ravel() for x in r])


def metrics(R_lo, s_lo, R_hi, s_hi):
    g_lo, g_hi = array_ratio(dd, R_lo, s_lo), array_ratio(dd, R_hi, s_hi)
    net = n_lo * g_lo + n_hi * g_hi
    return (float(np.sqrt(np.mean((net - Gs) ** 2))),
            float(np.max(np.abs(n_lo * g_lo - n_hi * g_hi)) / max(net.max(), 1e-9)),
            float(net[-1]))


def evaluate(pieces):
    g = net_ratio_fn(pieces[0][0], pieces[0][1], pieces[0][2],
                     pieces[1][0], pieces[1][1], pieces[1][2])
    rr = simulate_rigid(M, m, v0, g, max_heavy_dist=d_max, max_time=0.5,
                        max_target_acc=3e4, dt=1e-5)
    ee = simulate_compliant(M, m, v0, g, KROPE, pretension=F,
                            max_heavy_dist=d_max, dt=2e-5)
    cut = lambda a: a[:max(1, int(len(a) * 0.995))]
    Fr = cut(np.asarray(rr["force_target"], float))
    Fc = cut(np.asarray(ee["force_target"], float))
    return dict(vC=float(ee["summary"]["target_final_speed"]),
                pR=float(Fr.max() / Fr.mean()),
                pC=float(Fc.max() / F))


def main():
    out = {}

    # 1. mirror-symmetric
    b = None
    for sd in range(10):
        fit_res = fit_array(spec, N_HI, K, seed=sd, max_width=WIDTH)
        if b is None or fit_res.rms < b[2]:
            b = (np.asarray(fit_res.R), np.asarray(fit_res.s), fit_res.rms)
    R, s, rms = b
    out["mirror"] = dict(rms=float(rms), imbalance=abs(n_hi - n_lo) / K,
                         N_lo=N_HI, N_hi=N_HI, **evaluate([(R, s, n_lo), (R, s, n_hi)]))
    print("1. mirror-symmetric      rms %.3f  imbalance %.1f%%  p2m_C %.2f"
          % (out["mirror"]["rms"], 100 * out["mirror"]["imbalance"], out["mirror"]["pC"]))

    # 2. unequal, order free
    best = None
    rng = np.random.default_rng(3)
    for sd in range(40):
        p0 = np.concatenate([rng.normal(-0.5, 0.8, N_LO), [2.0],
                             rng.normal(-0.5, 0.8, N_HI), [2.0],
                             rng.normal(-1.0, 0.6, N_LO + N_HI)])
        try:
            sol = least_squares(_resid_free, p0, method="lm", max_nfev=4000,
                                args=(dd, Gs, N_LO, N_HI, n_lo, n_hi, d_max, WIDTH, 0.3, 50.0))
        except Exception:
            continue
        g = _unpack_free(sol.x, N_LO, N_HI, d_max)
        r = metrics(*g)
        if best is None or r[0] < best[0]:
            best = (r[0], r[1], r[2], g)
    rms_f, imb_f, term_f, gf = best
    out["free"] = dict(rms=rms_f, imbalance=imb_f, N_lo=N_LO, N_hi=N_HI,
                       R_lo=list(map(float, gf[0])), s_lo=list(map(float, gf[1])),
                       R_hi=list(map(float, gf[2])), s_hi=list(map(float, gf[3])),
                       **evaluate([(gf[0], gf[1], n_lo), (gf[2], gf[3], n_hi)]))
    print("2. unequal, order free   rms %.3f  imbalance %.1f%%  p2m_C %.2f"
          % (rms_f, 100 * imb_f, out["free"]["pC"]))

    # 3. unequal, interleaved
    order = interleaved_slots(N_LO, N_HI)
    best = None
    rng = np.random.default_rng(11)
    for sd in range(40):
        n = N_LO + N_HI
        p0 = np.concatenate([rng.normal(-0.5, 0.8, n), [2.0], rng.normal(-1.0, 0.6, n)])
        try:
            sol = least_squares(_residuals, p0, method="lm", max_nfev=4000,
                                args=(dd, Gs, N_LO, N_HI, n_lo, n_hi, d_max, WIDTH,
                                      0.3, 50.0, order))
        except Exception:
            continue
        g = _unpack(sol.x, N_LO, N_HI, d_max, order)
        r = metrics(*g)
        if best is None or r[0] < best[0]:
            best = (r[0], r[1], r[2], g)
    rms_i, imb_i, term_i, gi = best
    out["interleaved"] = dict(rms=rms_i, imbalance=imb_i, N_lo=N_LO, N_hi=N_HI,
                              terminal=term_i, pct_target=100.0 * term_i / Gt,
                              R_lo=list(map(float, gi[0])), s_lo=list(map(float, gi[1])),
                              R_hi=list(map(float, gi[2])), s_hi=list(map(float, gi[3])),
                              width_lo=float(2 * gi[0].sum()), width_hi=float(2 * gi[2].sum()),
                              order="".join("L" if o == 0 else "H" for o in order),
                              **evaluate([(gi[0], gi[1], n_lo), (gi[2], gi[3], n_hi)]))
    print("3. unequal, interleaved  rms %.3f  imbalance %.1f%%  p2m_C %.2f"
          % (rms_i, 100 * imb_i, out["interleaved"]["pC"]))

    out["_meta"] = dict(k=K, n_lo=n_lo, n_hi=n_hi, N_lo=N_LO, N_hi=N_HI, d_max=d_max,
                        target_terminal=Gt, width_budget=WIDTH, lam_balance=0.3,
                        M=M, m=m, v0=v0, F=F, v_stop=VSTOP, k_rope=KROPE)
    json.dump(out, open(os.path.join(HERE, "k5_design.json"), "w"), indent=1)
    print("\nwrote k5_design.json")


if __name__ == "__main__":
    main()
