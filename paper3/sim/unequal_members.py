"""Unequal member counts: does giving the low-fall (leading) array more members than the high-fall
array remove the end-of-stroke lateral load?

Reasoning under test. Late in the stroke the deep members saturate at their asymptote of 2, so
G_1 -> 2 N_1 and G_2 -> 2 N_2 and the lateral load tends to |n_1 N_1 - n_2 N_2| / (n_1 N_1 + n_2 N_2),
which with N_1 = N_2 is the mirror residual |n_1 - n_2| / k. Exact balance at saturation needs
N_1 : N_2 = n_2 : n_1, the discrete counterpart of G_1 : G_2 = n_2 : n_1. For k = 7 that is 4:3,
for k = 9 it is 5:4; the member floors N_j >= G*(d_max) / (4 n_j) already ask for the same order.

Setup is that of balanced_study.py (1,000:1, 16,471 N/m, width budgets 2.11 m and 1.96 m). Each
<N_1, N_2> pair is fitted by the staged solve from a single-array seed of N_1 members (the second
array seeded on the first N_2 midpoints), at several balance weights, and simulated rigid and
compliant. Writes unequal_members.json.
"""
import os, sys, json, time, numpy as np
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
sys.path.insert(0, os.path.join(_HERE, "code")); sys.path.insert(0, os.path.join(_HERE, "paper2"))
from friction_sim import make_config, evaluate, lateral_load, simulate_compliant
from ropecomb_fit4 import fit_array
import semisym_fit as SF
from semisym_fit import ideal_target, falls, bank_ratio, interleaved_slots, encode, _residuals, _unpack

M, m, v0, F, VSTOP, KR = 1000., 1., 10., 3169.6, 4.0, 16471
h = v0 ** 2 / (2 * 9.8)
dd, Gs, dmax = ideal_target(M, m, v0, F, VSTOP, n=400, code_dir="code")
Gt = float(Gs[-1])


def seed_unequal(R_star, s_star, N_hi):
    """Leading array = the single-array solution; second array = its first N_hi members, offset to the
    midpoints between successive leading engagements."""
    o = np.argsort(s_star); R = np.asarray(R_star)[o]; s = np.asarray(s_star)[o]
    mid = np.empty_like(s); mid[:-1] = 0.5 * (s[:-1] + s[1:]); mid[-1] = s[-1] + 0.5 * (dmax - s[-1])
    return R.copy(), s.copy(), R[:N_hi].copy(), mid[:N_hi].copy()


def fit_staged_unequal(R_star, s_star, N_hi, k, L_bank, lam_balance, lam_width=50.0):
    n_lo, n_hi = falls(k); N_lo = len(R_star); n = N_lo + N_hi
    order = interleaved_slots(N_lo, N_hi)
    R_l, s_l, R_h, s_h = seed_unequal(R_star, s_star, N_hi)
    p0 = encode(R_l, s_l, R_h, s_h, dmax, order)
    args = (dd, Gs, N_lo, N_hi, n_lo, n_hi, dmax, L_bank, lam_balance, lam_width, order)
    head = p0[:n + 1].copy()
    s1 = least_squares(lambda q: _residuals(np.concatenate([head, q]), *args), p0[n + 1:], method="lm", max_nfev=3000)
    s2 = least_squares(_residuals, np.concatenate([head, s1.x]), method="lm", max_nfev=4000, args=args)
    R_lo, s_lo, R_hi, s_hi = _unpack(s2.x, N_lo, N_hi, dmax, order)
    return R_lo, s_lo, R_hi, s_hi, order


def assess(R_lo, s_lo, R_hi, s_hi, k, sim=True):
    n_lo, n_hi = falls(k)
    g1 = bank_ratio(dd, R_lo, s_lo); g2 = bank_ratio(dd, R_hi, s_hi); net = n_lo * g1 + n_hi * g2
    imb = np.abs(n_lo * g1 - n_hi * g2) / net.max()
    rec = dict(rms=float(np.sqrt(np.mean((net - Gs) ** 2))), rms_pct=100 * float(np.sqrt(np.mean((net - Gs) ** 2)) / Gt),
               terminal_pct=100 * float(net[-1] / Gt), lateral=float(imb.max()), lateral_end=float(imb[-1]),
               lateral_at=float(dd[int(np.argmax(imb))] / dmax),
               lateral_curve=[float(x) for x in np.interp(np.linspace(0, dmax, 41), dd, imb)],
               G1_over_G2_end=float((g1[-1] / n_lo) / (g2[-1] / n_hi)) if g2[-1] > 0 else None,
               width_lo=float(2 * R_lo.sum()), width_hi=float(2 * R_hi.sum()),
               first=("lo" if s_lo.min() <= s_hi.min() else "hi"), last=("lo" if s_lo.max() >= s_hi.max() else "hi"),
               R_lo=list(map(float, R_lo)), s_lo=list(map(float, s_lo)), R_hi=list(map(float, R_hi)), s_hi=list(map(float, s_hi)))
    if sim:
        gear, react, rhos, nn, taus = make_config([(R_lo, s_lo), (R_hi, s_hi)], k, 1.0, True)
        e = evaluate(gear, react, M, m, h, F, dmax, KR)
        ll, rmax = lateral_load(rhos, dmax)
        ee = simulate_compliant(M, m, h, gear, react, KR, F, max_heavy_dist=dmax, time_unit=2e-5)
        Fc = ee["F"][:int(0.995 * len(ee["F"]))]
        rec.update(lateral_sim=ll, lateral_kN=ll * rmax * F / 1e3, pR=e["pR"], pC=e["pC"], vC=e["vC"], vR=e["vR"], minF=float(Fc.min() / F))
    return rec


def main():
    out = {"_meta": dict(M=M, m=m, v0=v0, F=F, v_stop=VSTOP, k_rope=KR, d_max=dmax, G_terminal=Gt,
                         method="staged seeded solve (phase 1 widths, phase 2 joint) from an N_1-member single-array fit; "
                                "second array seeded on the first N_2 midpoints; lam_width=50; interleaving lo,hi,lo,hi,...,lo")}
    LAMS = (0.0, 0.01, 0.03, 0.1, 0.3)
    CASES = {7: dict(W=2.11, pairs=[(7, 7), (8, 6), (7, 6), (8, 7), (7, 5)]),
             9: dict(W=1.96, pairs=[(5, 5), (5, 4), (6, 5), (6, 4)])}
    seeds = {}
    for k, C in CASES.items():
        n_lo, n_hi = falls(k)
        for N_lo, N_hi in C["pairs"]:
            tag = f"k{k}_N{N_lo}-{N_hi}"; out[tag] = dict(k=k, n_lo=n_lo, n_hi=n_hi, N_lo=N_lo, N_hi=N_hi, W=C["W"],
                                                        saturation_lateral=abs(n_lo * N_lo - n_hi * N_hi) / (n_lo * N_lo + n_hi * N_hi))
            if (k, N_lo) not in seeds:
                t0 = time.time(); best = None
                for sd in range(6):
                    R_, s_, _, rms_ = fit_array(dd, Gs, N_lo, k, dmax, seed=sd, width_budget=C["W"], R_bounds=(0.05, 8.0))
                    if best is None or rms_ < best[2]: best = (R_, s_, rms_)
                seeds[(k, N_lo)] = (np.array(best[0]), np.array(best[1]))
                print(f"seed k={k} N={N_lo}: rms {best[2]:.3f} width {2*np.sum(best[0]):.2f} ({time.time()-t0:.0f} s)", flush=True)
            Rst, sst = seeds[(k, N_lo)]
            for lam in LAMS:
                t0 = time.time()
                R_lo, s_lo, R_hi, s_hi, order = fit_staged_unequal(Rst, sst, N_hi, k, C["W"], lam)
                rec = assess(R_lo, s_lo, R_hi, s_hi, k); rec["lam"] = lam; out[tag][str(lam)] = rec
                print(f"{tag:10} lam={lam:<5} rms {rec['rms_pct']:.2f}%  lateral {100*rec['lateral']:5.2f}% (end {100*rec['lateral_end']:4.1f}%, peak at {rec['lateral_at']:.2f})  "
                      f"term {rec['terminal_pct']:.1f}%  G1/G2 {rec['G1_over_G2_end']:.3f}  pR {rec['pR']:.2f} pC {rec['pC']:.3f} minF {rec['minF']:.2f}  vC {rec['vC']:.1f}  "
                      f"w {rec['width_lo']:.2f}/{rec['width_hi']:.2f}  first {rec['first']} last {rec['last']}  ({time.time()-t0:.0f} s)", flush=True)
            json.dump(out, open("unequal_members.json", "w"), indent=1)
    print("wrote unequal_members.json")


if __name__ == "__main__":
    main()
