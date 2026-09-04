"""Semi-symmetric RopeComb fitter (Paper 2 reference implementation).

Two arrays either side of a central guide, each on its own rope, each rope dead-ended
at the (stationary) central stand. The two ropes drive separate elements of the
fixed-ratio stage, so their contributions ADD rather than being kinematically coupled:

    G_net(d) = n_lo * G_lo(d) + n_hi * G_hi(d)        n_lo = k//2 , n_hi = (k+1)//2
"""

import sys
import os
import numpy as np
from scipy.optimize import least_squares

# Support running directly from paper2/ or repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ropecomb import target_profile
from ropecomb.geometry import array_ratio
from dualarray_ropecomb.weighting import falls, guide_imbalance
from dualarray_ropecomb.ordering import interleaved_slots, verify_interleaving
from dualarray_ropecomb.fit import (R_FLOOR, _softplus, _softplus_inv, _unpack,
                                    encode, seed_two_stage, _residuals,
                                    fit_dual_staged, fit_dual)

bank_ratio = array_ratio


def ideal_target(M=1000.0, m=1.0, v0=10.0, F=3169.6, v_stop=4.0, n=400, **kwargs):
    """Ideal uniform-payload-force ratio profile, using ropecomb.target_profile."""
    spec = target_profile(M, m, v0, v_stop, F=F, n=n)
    return spec.d, spec.G, spec.d_max


def fit_semisym_staged(dd, Gstar, R_star, s_star, k, d_max, L_bank,
                       lam_balance=1.0, lam_width=50.0, jitter=0.0, seed=0):
    """Three-phase staged solve from single-array seed."""
    res = fit_dual_staged(dd, Gstar, R_star, s_star, k, d_max, L_bank,
                          lam_balance=lam_balance, lam_width=lam_width,
                          jitter=jitter, seed=seed)
    return res.raw


def fit_semisym(dd, Gstar, N_lo, N_hi, k, d_max, L_bank,
                lam_balance=1.0, lam_width=50.0, n_starts=20, seed=0, verbose=False):
    """Multi-start free optimization."""
    res = fit_dual(dd, Gstar, N_lo, N_hi, k, d_max, L_bank,
                   lam_balance=lam_balance, lam_width=lam_width,
                   n_starts=n_starts, seed=seed)
    if res is not None and verbose:
        report(res.raw, Gstar)
    return res.raw if res is not None else None


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
