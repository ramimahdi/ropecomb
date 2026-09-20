"""Balance sweep with a RELATIVE fit residual: r_fit = (net - G*)/(G* + c), c = 0.1 max(G*).
Same staged solver, seed and parametrisation as semisym_fit.fit_semisym_staged; only the residual weighting differs.
Writes balanced_relative.json."""
import os, sys, json, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
sys.path.insert(0, os.path.join(_HERE, "code")); sys.path.insert(0, os.path.join(_HERE, "paper2"))
import semisym_fit as SF
from scipy.optimize import least_squares
from friction_sim import make_config, evaluate, lateral_load
from semisym_fit import ideal_target, falls, interleaved_slots, seed_two_stage, encode, _unpack, bank_ratio, verify_interleaving
M, m, v0, F, VSTOP, KR = 1000., 1., 10., 3169.6, 4.0, 16471
h = v0**2/(2*9.8)
dd, Gs, dmax = ideal_target(M, m, v0, F, VSTOP, n=400, code_dir="code")
CREL = 0.1 * Gs.max()

def residuals_rel(p, dd, Gstar, N_lo, N_hi, n_lo, n_hi, d_max, L_bank, lam_balance, lam_width, order):
    R_lo, s_lo, R_hi, s_hi = _unpack(p, N_lo, N_hi, d_max, order)
    g_lo = bank_ratio(dd, R_lo, s_lo); g_hi = bank_ratio(dd, R_hi, s_hi)
    net = n_lo*g_lo + n_hi*g_hi
    r_fit = (net - Gstar) / (Gstar + CREL)
    r_bal = np.sqrt(lam_balance) * (n_lo*g_lo - n_hi*g_hi) / (Gstar + CREL)
    w_lo, w_hi = 2.0*R_lo.sum(), 2.0*R_hi.sum()
    r_w = np.sqrt(lam_width) * np.array([max(0.0, w_lo - L_bank), max(0.0, w_hi - L_bank)]) / max(L_bank, 1e-9)
    return np.concatenate([r_fit, r_bal, r_w])

def fit_staged_rel(dd, Gstar, R_star, s_star, k, d_max, L_bank, lam_balance, lam_width=50.):
    n_lo, n_hi = falls(k); N = len(R_star); n = 2*N
    order = interleaved_slots(N, N)
    R_l, s_l, R_h, s_h = seed_two_stage(R_star, s_star, d_max)
    p0 = encode(R_l, s_l, R_h, s_h, d_max, order)
    args = (dd, Gstar, N, N, n_lo, n_hi, d_max, L_bank, lam_balance, lam_width, order)
    head = p0[:n+1].copy()
    s1 = least_squares(lambda q: residuals_rel(np.concatenate([head, q]), *args), p0[n+1:], method="lm", max_nfev=3000)
    p1 = np.concatenate([head, s1.x])
    s2 = least_squares(residuals_rel, p1, method="lm", max_nfev=4000, args=args)
    R_lo, s_lo, R_hi, s_hi = _unpack(s2.x, N, N, d_max, order)
    g_lo = bank_ratio(dd, R_lo, s_lo); g_hi = bank_ratio(dd, R_hi, s_hi); net = n_lo*g_lo + n_hi*g_hi
    alls = np.sort(np.concatenate([s_lo, s_hi]))
    return dict(rms=float(np.sqrt(np.mean((net-Gstar)**2))), imbalance=float(np.max(np.abs(n_lo*g_lo-n_hi*g_hi))/max(net.max(),1e-9)),
                R_lo=R_lo, s_lo=s_lo, R_hi=R_hi, s_hi=s_hi, width_lo=2*R_lo.sum(), width_hi=2*R_hi.sum(), n_lo=n_lo, n_hi=n_hi, k=k, order=order,
                terminal=float(net[-1]), min_gap=float(np.min(np.diff(alls))), net=net)

out = {"_meta": dict(M=M, m=m, v0=v0, F=F, v_stop=VSTOP, k_rope=KR, d_max=dmax, c_rel=float(CREL), note="relative residual (net-G*)/(G*+c)")}
for tag, K, W, N in (("k7_W2.11", 7, 2.11, 7), ("k7_W2.7", 7, 2.7, 7), ("k9_W1.96", 9, 1.96, 5)):
    S = json.load(open(f"single_{'k7' if K==7 else 'k9'}.json")); Rst, sst = np.array(S["R"]), np.array(S["s"])
    out[tag] = {}
    for lam in (0.03, 0.3, 1.0, 3.0, 10.0, 30.0):
        o = fit_staged_rel(dd, Gs, Rst, sst, K, dmax, W, lam)
        ok = verify_interleaving(o); ok = ok[0] if isinstance(ok, tuple) else ok
        gear, react, rhos, n, taus = make_config([(o["R_lo"], o["s_lo"]), (o["R_hi"], o["s_hi"])], K, 1.0, True)
        e = evaluate(gear, react, M, m, h, F, dmax, KR); ll, rmax = lateral_load(rhos, dmax)
        g = np.array([gear(x) for x in dd]); track10 = float(np.interp(0.1*dmax, dd, g/np.maximum(Gs,1e-9)))
        rec = dict(lam=lam, rms=float(o["rms"]), lateral=ll, lateral_kN=ll*rmax*F/1e3, terminal=float(o["terminal"]), pct=100*float(o["terminal"])/float(Gs[-1]),
                   width_lo=float(o["width_lo"]), width_hi=float(o["width_hi"]), interleaved=bool(ok), pR=e["pR"], pC=e["pC"], vC=e["vC"], track10=track10,
                   R_lo=list(map(float,o["R_lo"])), s_lo=list(map(float,o["s_lo"])), R_hi=list(map(float,o["R_hi"])), s_hi=list(map(float,o["s_hi"])))
        out[tag][str(lam)] = rec
        print(f"{tag} lam={lam:<5} rms {rec['rms']:.3f}  lateral {100*ll:5.2f}% ({rec['lateral_kN']:4.1f} kN)  term {rec['pct']:.1f}%  G/G* at 10% {track10:.2f}  pR {e['pR']:.2f}  pC {e['pC']:.3f}  vC {e['vC']:.1f}  widths {rec['width_lo']:.2f}/{rec['width_hi']:.2f}  il {ok}")
json.dump(out, open("balanced_relative.json", "w"), indent=1)
