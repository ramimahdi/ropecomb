"""Near-balanced k=7 designs ABOVE the member floor (N=9 and N=11 per array). Writes balanced_study2.json."""
import os, sys, json, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
sys.path.insert(0, os.path.join(_HERE, "code")); sys.path.insert(0, os.path.join(_HERE, "paper2"))
from friction_sim import make_config, evaluate, lateral_load
from semisym_fit import ideal_target, fit_semisym_staged, verify_interleaving, falls
from ropecomb_fit4 import fit_array
M, m, v0, F, VSTOP, KR = 1000., 1., 10., 3169.6, 4.0, 16471
h = v0**2/(2*9.8)
dd, Gs, dmax = ideal_target(M, m, v0, F, VSTOP, n=400, code_dir="code")
out = {"_meta": dict(M=M, m=m, v0=v0, F=F, v_stop=VSTOP, k_rope=KR, d_max=dmax, width_budget=2.11)}
K, W = 7, 2.11; n_lo, n_hi = falls(K)
for N in (9, 11):
    b = None
    for sd in range(10):
        R, s, _, rms = fit_array(dd, Gs, N, K, dmax, seed=sd, R_bounds=(0.05, 8.), width_budget=W, overshoot_weight=0.0)
        if b is None or rms < b[2]: b = (np.asarray(R), np.asarray(s), rms)
    Rst, sst, rms1 = b
    gear1, react1, _, _, _ = make_config([(Rst, sst)], K, 1.0, False); e1 = evaluate(gear1, react1, M, m, h, F, dmax, KR)
    print(f"N={N} single one-line: rms {rms1:.3f} pR {e1['pR']:.2f} pC {e1['pC']:.3f} vC {e1['vC']:.1f}")
    out[f"N{N}"] = {"single": dict(rms=float(rms1), pR=e1["pR"], pC=e1["pC"], vC=e1["vC"], R=list(map(float,Rst)), s=list(map(float,sst)))}
    for lam in (0.03, 1.0, 3.0, 10.0, 30.0):
        o = fit_semisym_staged(dd, Gs, Rst, sst, K, dmax, W, lam_balance=lam, lam_width=50.)
        ok = verify_interleaving(o); ok = ok[0] if isinstance(ok, tuple) else ok
        gear, react, rhos, n, taus = make_config([(o["R_lo"], o["s_lo"]), (o["R_hi"], o["s_hi"])], K, 1.0, True)
        e = evaluate(gear, react, M, m, h, F, dmax, KR); ll, rmax = lateral_load(rhos, dmax)
        rec = dict(lam=lam, rms=float(o["rms"]), lateral=ll, lateral_kN=ll*rmax*F/1e3, terminal=float(o["terminal"]), pct=100*float(o["terminal"])/float(Gs[-1]),
                   width_lo=float(o["width_lo"]), width_hi=float(o["width_hi"]), interleaved=bool(ok), pR=e["pR"], pC=e["pC"], vC=e["vC"],
                   R_lo=list(map(float,o["R_lo"])), s_lo=list(map(float,o["s_lo"])), R_hi=list(map(float,o["R_hi"])), s_hi=list(map(float,o["s_hi"])))
        out[f"N{N}"][str(lam)] = rec
        print(f"N={N} lam={lam:<5} rms {rec['rms']:.3f}  lateral {100*ll:5.2f}% ({rec['lateral_kN']:4.1f} kN)  term {rec['pct']:.1f}%  pR {e['pR']:.2f}  pC {e['pC']:.3f}  vC {e['vC']:.1f}  widths {rec['width_lo']:.2f}/{rec['width_hi']:.2f}  il {ok}")
json.dump(out, open("balanced_study2.json", "w"), indent=1)
