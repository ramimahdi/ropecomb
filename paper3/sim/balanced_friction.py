"""Lateral load of the near-balanced designs under per-contact friction (path-loss asymmetry alone)."""
import os
import sys, json, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
sys.path.insert(0, os.path.join(_HERE, "code")); sys.path.insert(0, os.path.join(_HERE, "paper2"))
from friction_sim import make_config, evaluate, lateral_load
B3 = json.load(open("balanced_study3.json")); J = json.load(open("paper2/designs_16471.json"))
M, m, v0, F, KR = 1000., 1., 10., 3169.6, 16471; h = v0**2/19.6; dmax = B3["_meta"]["d_max"]
cases = {"<9,7> 3.3 m lam=10": B3["N9_W3.3"]["10.0"], "<9,7> 3.3 m lam=3": B3["N9_W3.3"]["3.0"], "<7,7> 2.11 m lam=0.03 (frozen)": J["k7"]["dual"]}
res = {}
for name, d in cases.items():
    res[name] = {}
    for eta in (1.0, 0.99, 0.98):
        gear, react, rhos, n, taus = make_config([(d["R_lo"], d["s_lo"]), (d["R_hi"], d["s_hi"])], 7, eta, True)
        ll, rmax = lateral_load(rhos, dmax); e = evaluate(gear, react, M, m, h, F, dmax, KR)
        res[name][str(eta)] = dict(lateral=ll, lateral_kN=ll*rmax*F/1e3, pC=e["pC"], vC=e["vC"], taus=list(taus))
        print(f"{name:32s} eta={eta:<5} lateral {100*ll:5.2f}% ({ll*rmax*F/1e3:4.1f} kN)  tau1/tau2 {taus[0]/taus[1]:.4f}  pC {e['pC']:.3f}  vC {e['vC']:.1f}")
json.dump(res, open("balanced_friction.json","w"), indent=1)
