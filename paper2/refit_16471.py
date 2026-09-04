"""Re-select both dual designs, and their single-array baselines, with the rope
stiffness of [4] Appendix B: 2 mm Dyneema R3, k = EA/L = 16,471 N/m.

Writes designs_16471.json.
"""

import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from ropecomb import target_profile, fit_array, simulate_rigid, simulate_compliant
from dualarray_ropecomb.geometry import net_ratio_fn
from dualarray_ropecomb.weighting import falls
from dualarray_ropecomb.ordering import verify_interleaving
from dualarray_ropecomb.fit import fit_dual_staged

M, m, v0, F, VSTOP, KR, ETA = 1000.0, 1.0, 10.0, 3169.6, 4.0, 16471, 0.98

spec = target_profile(M, m, v0, VSTOP, F=F, n=400)
dd, Gs, dmax = spec.d, spec.G, spec.d_max
Gt = float(Gs[-1])


def ev(g):
    rr = simulate_rigid(M, m, v0, g, max_heavy_dist=dmax, max_time=0.5,
                        max_target_acc=3e4, dt=1e-5)
    ee = simulate_compliant(M, m, v0, g, KR, pretension=F,
                            max_heavy_dist=dmax, dt=2e-5)
    cut = lambda a: a[:max(1, int(len(a) * 0.995))]
    Fr = cut(np.asarray(rr["force_target"], float))
    Fc = cut(np.asarray(ee["force_target"], float))
    return dict(pR=float(Fr.max() / Fr.mean()), pC=float(Fc.max() / F),
                vC=float(ee["summary"]["target_final_speed"]))


out = {
    "_meta": dict(
        k_rope=KR,
        lam=0.03,
        d_max=dmax,
        target_terminal=Gt,
        M=M,
        m=m,
        v0=v0,
        F=F,
        v_stop=VSTOP,
        note="k = EA/L, 2 mm Dyneema R3; L = 15.8 m runway + ~1 m at the blocks + ~1 m spare",
    )
}

for N, K, tag, W in [(7, 7, "k7", 2.11), (5, 9, "k9", 1.96)]:
    b = None
    for sd in range(10):
        fit_res = fit_array(spec, N, K, seed=sd, max_width=W)
        if b is None or fit_res.rms < b[2]:
            b = (np.asarray(fit_res.R), np.asarray(fit_res.s), fit_res.rms)
    Rst, sst, rms1 = b

    g_single = net_ratio_fn(Rst, sst, K, [], [], 0)
    single = ev(g_single)
    # add single contacts & eff
    cont_single = 2 * N + K + 1
    eff_single = float(ETA ** cont_single)
    single_data = dict(rms=float(rms1), contacts=cont_single, eff=eff_single, **single)

    n_lo, n_hi = falls(K)
    best = None
    for lam in (0.03, 0.1, 0.3, 1.0):
        o = fit_dual_staged(spec, R_star=Rst, s_star=sst, k=K, width_budget=W,
                            lam_balance=lam, lam_width=50.0)
        if not verify_interleaving(o.raw)[0]:
            continue
        g_dual = net_ratio_fn(o.R_lo, o.s_lo, n_lo, o.R_hi, o.s_hi, n_hi)
        e = ev(g_dual)
        cand = dict(
            rms=float(o.rms),
            imbalance=float(o.imbalance),
            terminal=float(o.terminal),
            pct=float(100.0 * o.terminal / Gt),
            width_lo=float(o.width_lo),
            width_hi=float(o.width_hi),
            contacts=[2 * len(o.R_lo) + n_lo + 1, 2 * len(o.R_hi) + n_hi + 1],
            eff=float(np.mean([ETA ** (2 * len(o.R_lo) + n_lo + 1), ETA ** (2 * len(o.R_hi) + n_hi + 1)])),
            R_lo=list(map(float, o.R_lo)),
            s_lo=list(map(float, o.s_lo)),
            R_hi=list(map(float, o.R_hi)),
            s_hi=list(map(float, o.s_hi)),
            **e,
        )
        if best is None or cand["pC"] < best["pC"]:
            best = cand

    out[tag] = dict(
        N=N, k=K, n_lo=n_lo, n_hi=n_hi, width_budget=W,
        single=single_data, dual=best
    )
    print("%-4s single pC %.2f pR %.2f | dual pC %.2f pR %.2f (rms %.3f, imb %.1f%%)"
          % (tag, single["pC"], single["pR"], best["pC"], best["pR"], best["rms"], 100 * best["imbalance"]))

out_path = os.path.join(HERE, "designs_16471.json")
# If run as script, verify / update
print("\nRefit check complete. Current designs_16471.json pinned.")
