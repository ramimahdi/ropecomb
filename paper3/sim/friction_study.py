"""Per-contact friction study on the frozen 1,000:1 designs. Writes friction_study.json."""
import os, sys, json, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
sys.path.insert(0, os.path.join(_HERE, "code")); sys.path.insert(0, os.path.join(_HERE, "paper2"))
from friction_sim import make_config, evaluate, lateral_load, stage_factors
from semisym_fit import ideal_target
M, m, v0, F, VSTOP, KR = 1000., 1., 10., 3169.6, 4.0, 16471
h = v0**2/(2*9.8)
dd, Gs, dmax = ideal_target(M, m, v0, F, VSTOP, n=400, code_dir="code")
J = json.load(open("paper2/designs_16471.json")); L = json.load(open("locked_4ms.json"))["1000to1"]
S7 = json.load(open("single_k7.json")); S9 = json.load(open("single_k9.json"))
configs = {
 "<9,5> one-line single":  dict(k=5, two=False, lines=[(L["R"], L["s"])]),
 "<9,5> two-line single":  dict(k=5, two=True,  lines=[(L["R"], L["s"]), (L["R"], L["s"])]),
 "<7,7> one-line single":  dict(k=7, two=False, lines=[(S7["R"], S7["s"])]),
 "<7,7> two-line single":  dict(k=7, two=True,  lines=[(S7["R"], S7["s"]), (S7["R"], S7["s"])]),
 "<7,7> unequal pair":     dict(k=7, two=True,  lines=[(J["k7"]["dual"]["R_lo"], J["k7"]["dual"]["s_lo"]), (J["k7"]["dual"]["R_hi"], J["k7"]["dual"]["s_hi"])]),
 "<5,9> one-line single":  dict(k=9, two=False, lines=[(S9["R"], S9["s"])]),
 "<5,9> two-line single":  dict(k=9, two=True,  lines=[(S9["R"], S9["s"]), (S9["R"], S9["s"])]),
 "<5,9> unequal pair":     dict(k=9, two=True,  lines=[(J["k9"]["dual"]["R_lo"], J["k9"]["dual"]["s_lo"]), (J["k9"]["dual"]["R_hi"], J["k9"]["dual"]["s_hi"])]),
}
etas = [1.0, 0.995, 0.99, 0.98]
out = {"_meta": dict(M=M, m=m, v0=v0, F=F, v_stop=VSTOP, k_rope=KR, d_max=dmax, etas=etas,
        model="per-contact efficiency eta; see friction_sim.py docstring")}
for name, c in configs.items():
    out[name] = {}
    for eta in etas:
        gear, react, rhos, n, taus = make_config(c["lines"], c["k"], eta, c["two"])
        e = evaluate(gear, react, M, m, h, F, dmax, KR)
        rec = dict(eta=eta, taus=list(taus), **{k: (float(v) if isinstance(v,(int,float,np.floating)) else v) for k, v in e.items()})
        if c["two"]:
            ll, rmax = lateral_load(rhos, dmax); rec["lateral"] = ll; rec["lateral_kN"] = ll * rmax * F / 1e3; rec["peak_reaction_kN"] = rmax * F / 1e3
        else:
            d = np.linspace(0, dmax, 4001); rec["peak_reaction_kN"] = max(rhos[0](x) for x in d) * F / 1e3
        out[name][str(eta)] = rec
        print(f"{name:24s} eta={eta:<6} vC {e['vC']:6.1f}  E/E1 {e['E_payload']/out[name]['1.0']['E_payload']:.3f}  pC {e['pC']:.3f}  pR {e['pR']:.3f}  "
              f"lat {100*rec.get('lateral',float('nan')):5.2f}%  Rpk {rec['peak_reaction_kN']:5.0f} kN  vh_end {e['vhC']:.2f}  stop {e['stopC']}/{e['stopR']}")
json.dump(out, open("friction_study.json", "w"), indent=1)
