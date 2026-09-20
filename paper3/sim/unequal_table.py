"""Comparison table: frozen equal-count designs against the unequal-count designs (author seed, lambda = 0.03),
all metrics recomputed from the design coordinates by the same code path. Also a jitter robustness check on the
staged solve (8 jittered seeds per unequal design). Writes unequal_table.json."""
import os, sys, json, numpy as np
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
sys.path.insert(0, os.path.join(_HERE, "code")); sys.path.insert(0, os.path.join(_HERE, "paper2"))
import unequal_members as U
import unequal_seed_compare as C
from semisym_fit import falls, interleaved_slots, encode, _residuals, _unpack, bank_ratio
dd, Gs, dmax = U.dd, U.Gs, U.dmax
DZ = json.load(open("paper2/designs_16471.json")); SC = json.load(open("unequal_seed_compare.json")); UM = json.load(open("unequal_members.json"))

def metrics(rec, k):
    R_lo, s_lo, R_hi, s_hi = [np.array(rec[x]) for x in ("R_lo", "s_lo", "R_hi", "s_hi")]
    a = U.assess(R_lo, s_lo, R_hi, s_hi, k)
    n_lo, n_hi = falls(k); g1 = bank_ratio(dd, R_lo, s_lo); g2 = bank_ratio(dd, R_hi, s_hi); net = n_lo * g1 + n_hi * g2
    imb = np.abs(n_lo * g1 - n_hi * g2) / net.max()
    a["lateral_mean"] = float(imb.mean()); a["N_lo"] = len(R_lo); a["N_hi"] = len(R_hi)
    return a

rows = []
for label, rec, k in (("<5,5,9>  frozen", DZ["k9"]["dual"], 9), ("<6,5,9>  author seed", SC["k9_N6-5|author|0.03"], 9),
                      ("<5,4,9>  refit seed", UM["k9_N5-4"]["0.03"], 9),
                      ("<7,7,7>  frozen", DZ["k7"]["dual"], 7), ("<8,6,7>  author seed", SC["k7_N8-6|author|0.03"], 7)):
    a = metrics(rec, k); a["label"] = label; rows.append(a)
    print(f"{label:22} N1={a['N_lo']} N2={a['N_hi']}  max lat {100*a['lateral']:5.2f}%  mean lat {100*a['lateral_mean']:5.2f}%  end {100*a['lateral_end']:4.1f}%  "
          f"pC {a['pC']:.3f}  pR {a['pR']:.2f}  vC {a['vC']:.1f}  vR {a['vR']:.1f}  rms {a['rms_pct']:.2f}%  term {a['terminal_pct']:.1f}%  w {a['width_lo']:.2f}/{a['width_hi']:.2f}", flush=True)

# jitter robustness of the staged solve from the author's seed
def solve_jit(R_l, s_l, R_h, s_h, k, W, lam, jitter, seed):
    n_lo, n_hi = falls(k); N_lo, N_hi = len(R_l), len(R_h); n = N_lo + N_hi
    order = interleaved_slots(N_lo, N_hi)
    p0 = encode(R_l, s_l, R_h, s_h, dmax, order) + np.random.default_rng(seed).normal(0.0, jitter, n + 1 + n)
    args = (dd, Gs, N_lo, N_hi, n_lo, n_hi, dmax, W, lam, 50.0, order)
    head = p0[:n + 1].copy()
    s1 = least_squares(lambda q: _residuals(np.concatenate([head, q]), *args), p0[n + 1:], method="lm", max_nfev=3000)
    s2 = least_squares(_residuals, np.concatenate([head, s1.x]), method="lm", max_nfev=4000, args=args)
    return _unpack(s2.x, N_lo, N_hi, dmax, order)

jit = {}
for k, W, tag, N_lo, N_hi in ((7, 2.11, "k7", 8, 6), (9, 1.96, "k9", 6, 5)):
    S = json.load(open(f"single_{tag}.json")); Rst, sst = np.array(S["R"]), np.array(S["s"])
    R_l, s_l, R_h, s_h = C.seed_author(Rst, sst, N_lo, N_hi)
    res = []
    for sd in range(8):
        R_lo, s_lo, R_hi, s_hi = solve_jit(R_l, s_l, R_h, s_h, k, W, 0.03, 0.3, sd)
        a = U.assess(R_lo, s_lo, R_hi, s_hi, k, sim=False)
        gear, react, rhos, nn, taus = U.make_config([(R_lo, s_lo), (R_hi, s_hi)], k, 1.0, True)
        e = U.evaluate(gear, react, U.M, U.m, U.h, U.F, dmax, U.KR)
        res.append(dict(lateral=a["lateral"], rms_pct=a["rms_pct"], pC=e["pC"], vC=e["vC"], width_lo=a["width_lo"], width_hi=a["width_hi"]))
    lat = np.array([r["lateral"] for r in res]); pc = np.array([r["pC"] for r in res]); rm = np.array([r["rms_pct"] for r in res])
    jit[f"k{k}_N{N_lo}-{N_hi}"] = dict(runs=res, lateral_min=float(lat.min()), lateral_median=float(np.median(lat)), lateral_max=float(lat.max()),
                                      pC_min=float(pc.min()), pC_median=float(np.median(pc)), pC_max=float(pc.max()), rms_min=float(rm.min()), rms_max=float(rm.max()))
    print(f"jitter <{N_lo},{N_hi},{k}> 8 starts (sigma 0.3): lateral {100*lat.min():.2f}–{100*lat.max():.2f}% (median {100*np.median(lat):.2f}%)  pC {pc.min():.3f}–{pc.max():.3f} (median {np.median(pc):.3f})  rms {rm.min():.2f}–{rm.max():.2f}%", flush=True)
json.dump(dict(rows=rows, jitter=jit), open("unequal_table.json", "w"), indent=1); print("wrote unequal_table.json")
