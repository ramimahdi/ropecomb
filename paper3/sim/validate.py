"""Reproduce the frozen designs_16471.json metrics with the friction integrators at eta = 1."""
import os, sys, json, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
sys.path.insert(0, os.path.join(_HERE, "code")); sys.path.insert(0, os.path.join(_HERE, "paper2"))
from friction_sim import make_config, evaluate, lateral_load
from semisym_fit import ideal_target
from ropecomb_fit4 import fit_array
M, m, v0, F, VSTOP, KR = 1000., 1., 10., 3169.6, 4.0, 16471
h = v0**2/(2*9.8)
dd, Gs, dmax = ideal_target(M, m, v0, F, VSTOP, n=400, code_dir="code")
print("d_max", dmax, "terminal", Gs[-1])
J = json.load(open("paper2/designs_16471.json"))
for tag in ("k7", "k9"):
    c = J[tag]; x = c["dual"]; K = c["k"]
    gear, react, rhos, n, taus = make_config([(x["R_lo"], x["s_lo"]), (x["R_hi"], x["s_hi"])], K, 1.0, True)
    e = evaluate(gear, react, M, m, h, F, dmax, KR)
    ll, _ = lateral_load(rhos, dmax)
    print(f"{tag} dual eta=1: pR {e['pR']:.3f} (json {x['pR']:.3f})  pC {e['pC']:.3f} (json {x['pC']:.3f})  vC {e['vC']:.1f} (json {x['vC']:.1f})  lateral {100*ll:.2f}% (json {100*x['imbalance']:.2f}%)  stops {e['stopR']}/{e['stopC']}")
    # single-array refit at matched N,k, seeds 0..9, as refit_16471.py
    N = c["N"]; W = c["width_budget"]; b = None
    for sd in range(10):
        R, s, _, rms = fit_array(dd, Gs, N, K, dmax, seed=sd, R_bounds=(0.05, 8.), width_budget=W, overshoot_weight=0.0)
        if b is None or rms < b[2]: b = (np.asarray(R), np.asarray(s), rms)
    Rst, sst, rms1 = b
    gear1, react1, _, _, _ = make_config([(Rst, sst)], K, 1.0, False)
    e1 = evaluate(gear1, react1, M, m, h, F, dmax, KR)
    print(f"{tag} single refit: rms {rms1:.4f} (json {c['single']['rms']:.4f})  pR {e1['pR']:.3f} (json {c['single']['pR']:.3f})  pC {e1['pC']:.3f} (json {c['single']['pC']:.3f})  vC {e1['vC']:.1f} (json {c['single']['vC']:.1f})")
    json.dump(dict(R=list(map(float,Rst)), s=list(map(float,sst)), rms=float(rms1)), open(f"single_{tag}.json","w"))
