"""Author's hypothesis: the merged single array at k/2 needs ~2x the width of one array, so its random initial
widths and its width cap should be scaled up. Test: init_span_range x1, x2, x3 and cap 2W, 3W, on <4,5,9>, <5,6,9>,
<6,8,7> at lambda 0.03, 30 restarts each. Also reports how close each merged single sits to its member ceiling
2(N_1+N_2) against the required 2G*/k. Writes unequal_width_init.json."""
import os, sys, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
sys.path.insert(0, os.path.join(_HERE, "code")); sys.path.insert(0, os.path.join(_HERE, "paper2"))
import unequal_members as U
from unequal_merged_seed import deal, staged
dd, Gs, dmax = U.dd, U.Gs, U.dmax; Gt = float(Gs[-1])
out = {}
for label, k, W, N_lo, N_hi in (("<4,5,9>", 9, 1.96, 5, 4), ("<5,6,9>", 9, 1.96, 6, 5), ("<6,8,7>", 7, 2.11, 8, 6)):
    N = N_lo + N_hi; need = 2 * Gt / k
    print(f"{label}: merged single needs terminal {need:.1f} against a ceiling of {2*N} ({100*need/(2*N):.0f}% of ceiling)", flush=True)
    for init, capx in (((0.2, 0.8), 2), ((0.4, 1.6), 2), ((0.6, 2.4), 2), ((0.4, 1.6), 3)):
        t0 = time.time()
        singles = [U.fit_array(dd, Gs, N, k / 2, dmax, seed=sd, width_budget=capx * W, R_bounds=(0.05, 8.0), init_span_range=init) for sd in range(30)]
        srms = np.array([x[3] for x in singles]); sw = np.array([2 * x[0].sum() for x in singles])
        res = []
        for R, s, _, rms in singles:
            (R_lo, s_lo, R_hi, s_hi), cost = staged(*deal(R, s, N_lo, N_hi), k, W, 0.03)
            res.append((cost, R_lo, s_lo, R_hi, s_hi))
        res.sort(key=lambda x: x[0]); cost, R_lo, s_lo, R_hi, s_hi = res[0]
        a = U.assess(R_lo, s_lo, R_hi, s_hi, k)
        lat = [U.assess(r[1], r[2], r[3], r[4], k, sim=False)["lateral"] for r in res]
        a.update(init=list(init), cap=capx * W, single_rms=[float(srms.min()), float(srms.max())], single_width=[float(sw.min()), float(sw.median() if hasattr(sw,'median') else np.median(sw)), float(sw.max())], spread=[float(min(lat)), float(np.median(lat)), float(max(lat))])
        out[f"{label}|init{init}|cap{capx}W"] = a
        print(f"{label} init {init} cap {capx}W: single rms {srms.min():.3f}–{srms.max():.3f}, width {sw.min():.2f}–{sw.max():.2f} m (median {np.median(sw):.2f}) | best: max lat {100*a['lateral']:5.2f}%  end {100*a['lateral_end']:4.1f}%  pC {a['pC']:.3f}  rms {a['rms_pct']:.2f}%  w {a['width_lo']:.2f}/{a['width_hi']:.2f}  | spread {100*min(lat):.2f}–{100*max(lat):.2f}% ({time.time()-t0:.0f} s)", flush=True)
    json.dump(out, open("unequal_width_init.json", "w"), indent=1)
print("wrote unequal_width_init.json")
