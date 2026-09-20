"""Two ways to seed the staged solve for unequal member counts, compared on the converged designs.

  author : start from the paper's own two-stage seed (single-array solution of N members on array 1, the same
           spans at the midpoints on array 2); then, for N_1 - N_2 = 1 append one member to array 1 with the width
           of array 2's last span and an offset half-way from the last engagement to d_max; for N_1 - N_2 = 2 move
           array 2's last member to the end of array 1.  (Algorithm 1 changes by one clause in step 4.)
  refit  : fit a single array of N_1 members, use it as array 1, and seed array 2 with its first N_2 members at
           the midpoints (what unequal_members.py did).
Writes unequal_seed_compare.json."""
import os, sys, json, numpy as np
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
sys.path.insert(0, os.path.join(_HERE, "code")); sys.path.insert(0, os.path.join(_HERE, "paper2"))
import unequal_members as U
from semisym_fit import falls, interleaved_slots, encode, _residuals, _unpack, seed_two_stage
dd, Gs, dmax = U.dd, U.Gs, U.dmax

def seed_author(R_star, s_star, N_lo, N_hi):
    R_l, s_l, R_h, s_h = seed_two_stage(np.asarray(R_star), np.asarray(s_star), dmax)
    N = len(R_star); moved = N - N_hi; added = N_lo + N_hi - 2 * N
    assert moved >= 0 and added >= 0 and N_lo == N + moved + added
    if moved:                                             # array 2's last member(s) go to the end of array 1
        R_l = np.concatenate([R_l, R_h[-moved:]]); s_l = np.concatenate([s_l, s_h[-moved:]])
        R_h, s_h = R_h[:-moved], s_h[:-moved]
    for _ in range(added):                                # one more member after the last engagement
        s_last = max(s_l.max(), s_h.max())
        R_l = np.concatenate([R_l, [R_h[-1]]]); s_l = np.concatenate([s_l, [s_last + 0.5 * (dmax - s_last)]])
    return R_l, s_l, R_h, s_h

def solve(R_l, s_l, R_h, s_h, k, W, lam):
    n_lo, n_hi = falls(k); N_lo, N_hi = len(R_l), len(R_h); n = N_lo + N_hi
    order = interleaved_slots(N_lo, N_hi)
    p0 = encode(R_l, s_l, R_h, s_h, dmax, order)
    args = (dd, Gs, N_lo, N_hi, n_lo, n_hi, dmax, W, lam, 50.0, order)
    head = p0[:n + 1].copy()
    s1 = least_squares(lambda q: _residuals(np.concatenate([head, q]), *args), p0[n + 1:], method="lm", max_nfev=3000)
    s2 = least_squares(_residuals, np.concatenate([head, s1.x]), method="lm", max_nfev=4000, args=args)
    return _unpack(s2.x, N_lo, N_hi, dmax, order)

out = {}
for k, W, tag, N_lo, N_hi in ((7, 2.11, "k7", 8, 6), (9, 1.96, "k9", 6, 5), (9, 1.96, "k9", 5, 4), (7, 2.11, "k7", 8, 7)):
    S = json.load(open(f"single_{tag}.json")); Rst, sst = np.array(S["R"]), np.array(S["s"])
    for lam in (0.01, 0.03, 0.05, 0.1):
        for name in ("author", "refit"):
            if name == "author":
                if N_lo + N_hi - 2 * len(Rst) not in (0, 1): continue
                R_l, s_l, R_h, s_h = seed_author(Rst, sst, N_lo, N_hi)
                R_lo, s_lo, R_hi, s_hi = solve(R_l, s_l, R_h, s_h, k, W, lam)
            else:
                best = None
                for sd in range(6):
                    R_, s_, _, r_ = U.fit_array(dd, Gs, N_lo, k, dmax, seed=sd, width_budget=W, R_bounds=(0.05, 8.0))
                    if best is None or r_ < best[2]: best = (R_, s_, r_)
                R_lo, s_lo, R_hi, s_hi, _ = U.fit_staged_unequal(np.array(best[0]), np.array(best[1]), N_hi, k, W, lam)
            rec = U.assess(R_lo, s_lo, R_hi, s_hi, k); rec.update(lam=lam, seed=name)
            out[f"k{k}_N{N_lo}-{N_hi}|{name}|{lam}"] = rec
            print(f"k{k} <{N_lo},{N_hi}> {name:6} lam={lam:<5} rms {rec['rms_pct']:.2f}%  lateral {100*rec['lateral']:5.2f}% (end {100*rec['lateral_end']:4.1f}%)  pC {rec['pC']:.3f} pR {rec['pR']:.2f} minF {rec['minF']:.2f}  term {rec['terminal_pct']:.1f}%  w {rec['width_lo']:.2f}/{rec['width_hi']:.2f}", flush=True)
json.dump(out, open("unequal_seed_compare.json", "w"), indent=1); print("wrote unequal_seed_compare.json")
