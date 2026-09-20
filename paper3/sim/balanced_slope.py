"""Balance sweep with a slope-regularised residual: r = [(net-G*)/maxG*, sqrt(mu)(net'-G*')/maxG*', sqrt(lam)(n1G1-n2G2)/maxG*, width].
Tests whether the tension dips of the balanced 2.11 m designs are an artefact of the value-only objective of Algorithm 1.
Writes balanced_slope.json."""
import os
import sys, json, time, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
sys.path.insert(0, os.path.join(_HERE, "code")); sys.path.insert(0, os.path.join(_HERE, "paper2"))
from scipy.optimize import least_squares
from friction_sim import make_config, evaluate, lateral_load, simulate_compliant
from semisym_fit import ideal_target, falls, interleaved_slots, seed_two_stage, encode, _unpack, bank_ratio, verify_interleaving
M, m, v0, F, VSTOP, KR = 1000., 1., 10., 3169.6, 4.0, 16471
h = v0**2/19.6
dd, Gs, dmax = ideal_target(M, m, v0, F, VSTOP, n=400, code_dir="code")
dGs = np.gradient(Gs, dd); SG = Gs.max(); SD = dGs.max()

def residuals(p, N, n_lo, n_hi, L_bank, lam, mu, order, lam_width=50.):
    R_lo, s_lo, R_hi, s_hi = _unpack(p, N, N, dmax, order)
    g_lo = bank_ratio(dd, R_lo, s_lo); g_hi = bank_ratio(dd, R_hi, s_hi)
    net = n_lo*g_lo + n_hi*g_hi
    r_fit = (net - Gs)/SG
    r_slope = np.sqrt(mu) * (np.gradient(net, dd) - dGs)/SD
    r_bal = np.sqrt(lam) * (n_lo*g_lo - n_hi*g_hi)/SG
    r_w = np.sqrt(lam_width) * np.array([max(0., 2*R_lo.sum()-L_bank), max(0., 2*R_hi.sum()-L_bank)])/L_bank
    return np.concatenate([r_fit, r_slope, r_bal, r_w])

def fit(Rst, sst, k, W, lam, mu):
    n_lo, n_hi = falls(k); N = len(Rst); n = 2*N; order = interleaved_slots(N, N)
    R_l, s_l, R_h, s_h = seed_two_stage(Rst, sst, dmax); p0 = encode(R_l, s_l, R_h, s_h, dmax, order)
    args = (N, n_lo, n_hi, W, lam, mu, order); head = p0[:n+1].copy()
    s1 = least_squares(lambda q: residuals(np.concatenate([head, q]), *args), p0[n+1:], method="lm", max_nfev=3000)
    s2 = least_squares(residuals, np.concatenate([head, s1.x]), method="lm", max_nfev=4000, args=args)
    R_lo, s_lo, R_hi, s_hi = _unpack(s2.x, N, N, dmax, order)
    return dict(R_lo=R_lo, s_lo=s_lo, R_hi=R_hi, s_hi=s_hi, n_lo=n_lo, n_hi=n_hi, k=k, order=order, width_lo=2*R_lo.sum(), width_hi=2*R_hi.sum())

out = {"_meta": dict(M=M, m=m, v0=v0, F=F, v_stop=VSTOP, k_rope=KR, d_max=dmax, note="value + slope residual, mu = slope weight")}
S = json.load(open("single_k7.json")); Rst, sst = np.array(S["R"]), np.array(S["s"])
for W in (2.11, 3.3):
  for mu in (0.0, 1.0, 10.0):
    for lam in (0.03, 1.0, 3.0, 10.0):
        t0 = time.time(); o = fit(Rst, sst, 7, W, lam, mu)
        gear, react, rhos, n, taus = make_config([(o["R_lo"], o["s_lo"]), (o["R_hi"], o["s_hi"])], 7, 1.0, True)
        e = evaluate(gear, react, M, m, h, F, dmax, KR); ll, rmax = lateral_load(rhos, dmax)
        ee = simulate_compliant(M, m, h, gear, react, KR, F, max_heavy_dist=dmax, time_unit=2e-5)
        Fc = ee["F"][:int(0.995*len(ee["F"]))]
        net = o["n_lo"]*bank_ratio(dd, o["R_lo"], o["s_lo"]) + o["n_hi"]*bank_ratio(dd, o["R_hi"], o["s_hi"])
        rec = dict(W=W, mu=mu, lam=lam, lateral=ll, lateral_kN=ll*rmax*F/1e3, rms=float(np.sqrt(np.mean((net-Gs)**2))), pct=100*float(net[-1]/Gs[-1]),
                   pR=e["pR"], pC=e["pC"], vC=e["vC"], minF=float(Fc.min()/F), width_lo=float(o["width_lo"]), width_hi=float(o["width_hi"]),
                   interleaved=bool(verify_interleaving(o)[0] if isinstance(verify_interleaving(o), tuple) else verify_interleaving(o)),
                   R_lo=list(map(float, o["R_lo"])), s_lo=list(map(float, o["s_lo"])), R_hi=list(map(float, o["R_hi"])), s_hi=list(map(float, o["s_hi"])))
        out[f"W{W}_mu{mu}_lam{lam}"] = rec
        print(f"W={W} mu={mu:<4} lam={lam:<5} lat {100*ll:5.2f}% rms {rec['rms']:.3f} term {rec['pct']:.0f}%  pC {e['pC']:.2f} pR {e['pR']:.2f} minF {rec['minF']:.2f} vC {e['vC']:.1f} W {rec['width_lo']:.2f}/{rec['width_hi']:.2f} il {rec['interleaved']} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(out, open("balanced_slope.json", "w"), indent=1)
