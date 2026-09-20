"""Follow-ups to unequal_members.py: (a) where the extra member(s) of the leading array should sit in the
engagement sequence (first/last/middle), for <8,6> at k=7 and <6,5> at k=9; (b) two more k=9 pairs;
(c) a finer lambda ladder for the best pairs. Writes unequal_orders.json."""
import os, sys, json, numpy as np
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
sys.path.insert(0, os.path.join(_HERE, "code")); sys.path.insert(0, os.path.join(_HERE, "paper2"))
import unequal_members as U          # reuses its target, seeds and assess(); running it re-executes the sweep, so guard
from semisym_fit import falls, encode, _residuals, _unpack
dd, Gs, dmax, Gt = U.dd, U.Gs, U.dmax, U.Gt

def seed_for(k, N_lo, W):
    best = None
    for sd in range(6):
        R_, s_, _, r_ = U.fit_array(dd, Gs, N_lo, k, dmax, seed=sd, width_budget=W, R_bounds=(0.05, 8.0))
        if best is None or r_ < best[2]: best = (R_, s_, r_)
    return np.array(best[0]), np.array(best[1])

def fit_with_order(Rst, sst, N_hi, k, W, lam, order):
    n_lo, n_hi = falls(k); N_lo = len(Rst); n = N_lo + N_hi
    o = np.argsort(sst); R = Rst[o]; s = sst[o]
    mid = np.empty_like(s); mid[:-1] = 0.5 * (s[:-1] + s[1:]); mid[-1] = s[-1] + 0.5 * (dmax - s[-1])
    # seed offsets: merge by the requested order, leading members take the single-array offsets in turn,
    # second-array members take midpoints in turn
    p0 = encode(R, s, R[:N_hi], mid[:N_hi], dmax, order)
    args = (dd, Gs, N_lo, N_hi, n_lo, n_hi, dmax, W, lam, 50.0, order)
    head = p0[:n + 1].copy()
    s1 = least_squares(lambda q: _residuals(np.concatenate([head, q]), *args), p0[n + 1:], method="lm", max_nfev=3000)
    s2 = least_squares(_residuals, np.concatenate([head, s1.x]), method="lm", max_nfev=4000, args=args)
    return _unpack(s2.x, N_lo, N_hi, dmax, order)

def orders(N_lo, N_hi):
    base = []
    for i in range(N_hi): base += [0, 1]
    extra = N_lo - N_hi
    return {"extra lo last": np.array(base + [0] * extra),
            "extra lo first": np.array([0] * extra + base),
            "extra lo middle": np.array(base[:N_hi] + [0] * extra + base[N_hi:])}

out = {}
for k, W, N_lo, N_hi in ((7, 2.11, 8, 6), (9, 1.96, 6, 5)):
    Rst, sst = seed_for(k, N_lo, W)
    for name, od in orders(N_lo, N_hi).items():
        for lam in (0.01, 0.03, 0.1):
            R_lo, s_lo, R_hi, s_hi = fit_with_order(Rst, sst, N_hi, k, W, lam, od)
            rec = U.assess(R_lo, s_lo, R_hi, s_hi, k); rec.update(lam=lam, order=list(map(int, od)))
            out[f"k{k}_N{N_lo}-{N_hi}|{name}|{lam}"] = rec
            print(f"k{k} <{N_lo},{N_hi}> {name:15} lam={lam:<5} rms {rec['rms_pct']:.2f}%  lateral {100*rec['lateral']:5.2f}% (end {100*rec['lateral_end']:4.1f}%, peak at {rec['lateral_at']:.2f})  pC {rec['pC']:.3f} pR {rec['pR']:.2f} minF {rec['minF']:.2f}  term {rec['terminal_pct']:.1f}%  w {rec['width_lo']:.2f}/{rec['width_hi']:.2f}", flush=True)
# two more k=9 pairs, default order, and a finer ladder for <8,6> and <6,5>
for k, W, N_lo, N_hi, lams in ((9, 1.96, 7, 5, (0.03,)), (9, 1.96, 7, 6, (0.03,)), (7, 2.11, 8, 6, (0.02, 0.05)), (9, 1.96, 6, 5, (0.02, 0.05))):
    Rst, sst = seed_for(k, N_lo, W)
    for lam in lams:
        R_lo, s_lo, R_hi, s_hi, od = U.fit_staged_unequal(Rst, sst, N_hi, k, W, lam)
        rec = U.assess(R_lo, s_lo, R_hi, s_hi, k); rec.update(lam=lam, order=list(map(int, od)))
        out[f"k{k}_N{N_lo}-{N_hi}|default|{lam}"] = rec
        print(f"k{k} <{N_lo},{N_hi}> default         lam={lam:<5} rms {rec['rms_pct']:.2f}%  lateral {100*rec['lateral']:5.2f}% (end {100*rec['lateral_end']:4.1f}%, peak at {rec['lateral_at']:.2f})  pC {rec['pC']:.3f} pR {rec['pR']:.2f} minF {rec['minF']:.2f}  term {rec['terminal_pct']:.1f}%  w {rec['width_lo']:.2f}/{rec['width_hi']:.2f}", flush=True)
json.dump(out, open("unequal_orders.json", "w"), indent=1); print("wrote unequal_orders.json")
