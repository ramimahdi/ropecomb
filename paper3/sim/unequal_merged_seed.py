"""Author's merged seed: fit ONE array of N_1 + N_2 members at stage ratio k/2 (so that its ratio is 2 G*/k),
deal its members in engagement order to the two arrays as lo, hi, lo, hi, ..., lo (the leading array takes
the first and any extra members), and run the staged balanced solve from there. 30 random restarts of the
single-array fit, each followed by the staged solve; the best joint objective is kept and the spread reported.
Writes unequal_merged_seed.json."""
import os, sys, json, time, numpy as np
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
sys.path.insert(0, os.path.join(_HERE, "code")); sys.path.insert(0, os.path.join(_HERE, "paper2"))
import unequal_members as U
from semisym_fit import falls, bank_ratio, interleaved_slots, encode, _residuals, _unpack
dd, Gs, dmax = U.dd, U.Gs, U.dmax

def deal(R, s, N_lo, N_hi):
    o = np.argsort(s); R = np.asarray(R)[o]; s = np.asarray(s)[o]
    order = interleaved_slots(N_lo, N_hi)
    lo, hi = order == 0, order == 1
    return R[lo], s[lo], R[hi], s[hi], order

def staged(R_l, s_l, R_h, s_h, order, k, W, lam):
    n_lo, n_hi = falls(k); N_lo, N_hi = len(R_l), len(R_h); n = N_lo + N_hi
    p0 = encode(R_l, s_l, R_h, s_h, dmax, order)
    args = (dd, Gs, N_lo, N_hi, n_lo, n_hi, dmax, W, lam, 50.0, order)
    head = p0[:n + 1].copy()
    s1 = least_squares(lambda q: _residuals(np.concatenate([head, q]), *args), p0[n + 1:], method="lm", max_nfev=3000)
    s2 = least_squares(_residuals, np.concatenate([head, s1.x]), method="lm", max_nfev=4000, args=args)
    return _unpack(s2.x, N_lo, N_hi, dmax, order), float(2 * s2.cost)

def main():
    out = {"_meta": dict(method="single array of N1+N2 members fitted at k/2 (30 restarts, width budget 2W), dealt lo,hi,...,lo; staged solve; best joint objective kept")}
    for label, k, W, N_lo, N_hi in (("<5,5,9>", 9, 1.96, 5, 5), ("<5,6,9>", 9, 1.96, 6, 5), ("<4,5,9>", 9, 1.96, 5, 4),
                                    ("<7,7,7>", 7, 2.11, 7, 7), ("<6,8,7>", 7, 2.11, 8, 6)):
        N = N_lo + N_hi; out[label] = {}
        t0 = time.time(); singles = []
        for sd in range(30):
            R, s, _, rms = U.fit_array(dd, Gs, N, k / 2, dmax, seed=sd, width_budget=2 * W, R_bounds=(0.05, 8.0))
            singles.append((R, s, rms))
        print(f"{label} single N={N} at k/2={k/2}: rms {min(x[2] for x in singles):.3f}–{max(x[2] for x in singles):.3f} over 30 starts ({time.time()-t0:.0f} s)", flush=True)
        for lam in (0.01, 0.02, 0.03):
            t0 = time.time(); res = []
            for R, s, rms in singles:
                (R_lo, s_lo, R_hi, s_hi), cost = staged(*deal(R, s, N_lo, N_hi), k, W, lam)
                res.append((cost, R_lo, s_lo, R_hi, s_hi))
            res.sort(key=lambda x: x[0])
            # dynamics for the best, geometry-only spread for all
            spread = []
            for cost, R_lo, s_lo, R_hi, s_hi in res:
                a = U.assess(R_lo, s_lo, R_hi, s_hi, k, sim=False); spread.append((a["lateral"], a["rms_pct"], cost))
            cost, R_lo, s_lo, R_hi, s_hi = res[0]
            a = U.assess(R_lo, s_lo, R_hi, s_hi, k)
            n_lo, n_hi = falls(k); g1 = bank_ratio(dd, R_lo, s_lo); g2 = bank_ratio(dd, R_hi, s_hi); net = n_lo * g1 + n_hi * g2
            a["lateral_mean"] = float((np.abs(n_lo * g1 - n_hi * g2) / net.max()).mean()); a["lam"] = lam; a["cost"] = cost
            lat = np.array([x[0] for x in spread]); a["spread_lateral"] = [float(lat.min()), float(np.median(lat)), float(lat.max())]
            a["n_within_5pct_cost"] = int(sum(1 for x in spread if x[2] <= 1.05 * cost))
            out[label][str(lam)] = a
            print(f"{label} lam={lam:<5} best: max lat {100*a['lateral']:5.2f}%  mean {100*a['lateral_mean']:4.2f}%  end {100*a['lateral_end']:4.1f}%  pC {a['pC']:.3f}  pR {a['pR']:.2f}  vC {a['vC']:.1f}  minF {a['minF']:.2f}  rms {a['rms_pct']:.2f}%  w {a['width_lo']:.2f}/{a['width_hi']:.2f}  "
                  f"| lateral over 30 starts {100*lat.min():.2f}–{100*lat.max():.2f}% (median {100*np.median(lat):.2f}%), {a['n_within_5pct_cost']} within 5% of best cost ({time.time()-t0:.0f} s)", flush=True)
        json.dump(out, open("unequal_merged_seed.json", "w"), indent=1)
    print("wrote unequal_merged_seed.json")


if __name__ == "__main__":
    main()
