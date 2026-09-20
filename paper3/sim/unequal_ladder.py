"""Lambda ladder for the five configurations of unequal_table.py, all from the paper's frozen single-array seeds:
equal counts through fit_semisym_staged (Algorithm 1 as published), unequal counts through the author's seed.
Writes unequal_ladder.json."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
sys.path.insert(0, os.path.join(_HERE, "code")); sys.path.insert(0, os.path.join(_HERE, "paper2"))
import unequal_members as U
from unequal_seed_compare import seed_author, solve
from semisym_fit import falls, bank_ratio, fit_semisym_staged
dd, Gs, dmax = U.dd, U.Gs, U.dmax
LAMS = (0.01, 0.02, 0.03, 0.05)
out = {}
for label, tag, k, W, N_lo, N_hi in (("<5,5,9>", "k9", 9, 1.96, 5, 5), ("<5,6,9>", "k9", 9, 1.96, 6, 5), ("<4,5,9>", "k9", 9, 1.96, 5, 4),
                                     ("<7,7,7>", "k7", 7, 2.11, 7, 7), ("<6,8,7>", "k7", 7, 2.11, 8, 6)):
    S = json.load(open(f"single_{tag}.json")); Rst, sst = np.array(S["R"]), np.array(S["s"])
    out[label] = {}
    for lam in LAMS:
        if N_lo == N_hi:
            o = fit_semisym_staged(dd, Gs, Rst, sst, k, dmax, W, lam_balance=lam, lam_width=50.)
            R_lo, s_lo, R_hi, s_hi = o["R_lo"], o["s_lo"], o["R_hi"], o["s_hi"]
        elif N_lo + N_hi - 2 * len(Rst) in (0, 1):
            R_lo, s_lo, R_hi, s_hi = solve(*seed_author(Rst, sst, N_lo, N_hi), k, W, lam)
        else:                                                     # <5,4>: refit seed (one member fewer than the frozen seed)
            best = None
            for sd in range(6):
                R_, s_, _, r_ = U.fit_array(dd, Gs, N_lo, k, dmax, seed=sd, width_budget=W, R_bounds=(0.05, 8.0))
                if best is None or r_ < best[2]: best = (R_, s_, r_)
            R_lo, s_lo, R_hi, s_hi, _ = U.fit_staged_unequal(np.array(best[0]), np.array(best[1]), N_hi, k, W, lam)
        a = U.assess(R_lo, s_lo, R_hi, s_hi, k)
        n_lo, n_hi = falls(k); g1 = bank_ratio(dd, R_lo, s_lo); g2 = bank_ratio(dd, R_hi, s_hi); net = n_lo * g1 + n_hi * g2
        a["lateral_mean"] = float((np.abs(n_lo * g1 - n_hi * g2) / net.max()).mean()); a["lam"] = lam
        out[label][str(lam)] = a
        print(f"{label} lam={lam:<5} max lat {100*a['lateral']:5.2f}%  mean {100*a['lateral_mean']:4.2f}%  end {100*a['lateral_end']:4.1f}%  pC {a['pC']:.3f}  pR {a['pR']:.2f}  vC {a['vC']:.1f}  minF {a['minF']:.2f}  rms {a['rms_pct']:.2f}%  w {a['width_lo']:.2f}/{a['width_hi']:.2f}", flush=True)
json.dump(out, open("unequal_ladder.json", "w"), indent=1); print("wrote unequal_ladder.json")
