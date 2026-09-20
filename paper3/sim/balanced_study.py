"""Near-balanced <7,7> and <5,9> designs at large lambda, with their dynamics. Writes balanced_study.json."""
import os, sys, json, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
sys.path.insert(0, os.path.join(_HERE, "code")); sys.path.insert(0, os.path.join(_HERE, "paper2"))
from friction_sim import make_config, evaluate, lateral_load
from semisym_fit import ideal_target, fit_semisym_staged, verify_interleaving, falls
M, m, v0, F, VSTOP, KR = 1000., 1., 10., 3169.6, 4.0, 16471
h = v0**2/(2*9.8)
dd, Gs, dmax = ideal_target(M, m, v0, F, VSTOP, n=400, code_dir="code")
out = {"_meta": dict(M=M, m=m, v0=v0, F=F, v_stop=VSTOP, k_rope=KR, d_max=dmax, method="fit_semisym_staged from the single-array seed (refit_16471.py), lam_width=50")}
for tag, K, W in (("k7", 7, 2.11), ("k9", 9, 1.96)):
    S = json.load(open(f"single_{tag}.json")); Rst, sst = np.array(S["R"]), np.array(S["s"])
    n_lo, n_hi = falls(K); out[tag] = {}
    for lam in (0.03, 0.3, 1.0, 3.0, 10.0, 30.0):
        o = fit_semisym_staged(dd, Gs, Rst, sst, K, dmax, W, lam_balance=lam, lam_width=50.)
        ok, _ = verify_interleaving(o) if isinstance(verify_interleaving(o), tuple) else (verify_interleaving(o), None)
        gear, react, rhos, n, taus = make_config([(o["R_lo"], o["s_lo"]), (o["R_hi"], o["s_hi"])], K, 1.0, True)
        e = evaluate(gear, react, M, m, h, F, dmax, KR)
        ll, rmax = lateral_load(rhos, dmax)
        g1 = np.array([rhos[0](x) for x in np.linspace(0, dmax, 4001)]); g2 = np.array([rhos[1](x) for x in np.linspace(0, dmax, 4001)])
        rec = dict(lam=lam, rms=float(o["rms"]), lateral=ll, lateral_kN=ll*rmax*F/1e3, terminal=float(o["terminal"]), pct=100*float(o["terminal"])/float(Gs[-1]),
                   width_lo=float(o["width_lo"]), width_hi=float(o["width_hi"]), G1_over_G2_end=float(g1[-1]/n_lo/(g2[-1]/n_hi)),
                   interleaved=bool(ok), pR=e["pR"], pC=e["pC"], vC=e["vC"],
                   R_lo=list(map(float,o["R_lo"])), s_lo=list(map(float,o["s_lo"])), R_hi=list(map(float,o["R_hi"])), s_hi=list(map(float,o["s_hi"])))
        out[tag][str(lam)] = rec
        print(f"{tag} lam={lam:<5} rms {rec['rms']:.3f}  lateral {100*ll:5.2f}% ({rec['lateral_kN']:4.1f} kN)  term {rec['pct']:.1f}%  G1/G2 {rec['G1_over_G2_end']:.3f}  pR {e['pR']:.2f}  pC {e['pC']:.3f}  vC {e['vC']:.1f}  interleaved {ok}  widths {rec['width_lo']:.2f}/{rec['width_hi']:.2f}")
json.dump(out, open("balanced_study.json", "w"), indent=1)
