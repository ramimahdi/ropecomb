"""Arrest mode (Supplementary Section S17): the transmission run in reverse against a hold-down mass.
Target and fit as in target.py / fit.py (a direct least-squares fit in the relative residual sampled along the
body's travel, 40 random starts; NOT Algorithm 1 of the main text), integrator sim.py (semi-implicit Euler, output
member Kelvin-Voigt with damping ratio zeta on the reflected body mass; NOT the undamped model of S12).
Runs the worked case (50 kg, 100 -> 10 m/s in 30 m, 4 t hold-down through a 2:1 tackle, R_b = 2.5 m, <8:6,7>),
the same hold-down at a fixed ratio chosen to end the 30 m at 10 m/s, the scaled family 20/50/100 kg, and the
stiffness x damping grid. Writes arrest_results.json and the traces main_rec.npy / fixed_rec.npy for the figure."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np
from sim import Comb, Fixed, simulate, metrics
from target import build
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file

d = json.load(open("fit_k7_8_6_L2.5.json")); cb = Comb(d)
T = build(100., Rp=2.5, j=2, n=30001)
i1 = int(np.searchsorted(T["x"], T["x1"]))
out = {"design": {k: d[k] for k in ("k", "n1", "n2", "N1", "N2", "R1", "s1", "R2", "s2", "rel_rms", "imb", "w1", "w2", "lam", "Lw", "S")},
       "target": dict(C0=100.0, Rp=2.5, j=2, M=4000.0, m=50.0, v0=100.0, v1=10.0, X=30.0, Fu=float(T["Fu"]), x_ramp_end=float(T["x1"]),
                      stroke=float(T["u"][-1]), C_after_ramp=float(T["C"][i1]), C_end=float(T["C"][-1]))}
# fitted net ratio over the lift, for the record
S = d["S"]; tt = np.linspace(0, S, 200); Cfit = np.array([cb.G(t) for t in tt])
out["design"]["C_fit_start"] = float(Cfit[-1]); out["design"]["C_fit_end"] = float(Cfit[0])

def run(ratio, **kw):
    rec = simulate(ratio, **kw); m = metrics(rec, Fu=T["Fu"], m=kw.get("m", 50.)); return rec, {k: float(v) for k, v in m.items()}

rec, m = run(cb, zeta=0.3); out["main"] = m; np.save("main_rec.npy", rec)
t, x, vx, u, vu, Tn, F, C = rec.T
out["main"]["T_after_20ms_max"] = float(Tn[t > 0.02].max()); out["main"]["T_mean_uniform"] = float(Tn[x > T["x1"]].mean())
out["main"]["rope_speed_max"] = float(np.abs(np.gradient(np.array([cb.payout(ui) for ui in u]), t)).max())
out["main"]["E_mass_end_kJ"] = float(0.5 * 4000 * (2 * vu[-1]) ** 2 / 1e3 + 4000 * 9.8 * 2 * u[-1] / 1e3)
out["main"]["Fc_mean_kN"] = float((Tn * C).mean() / 1e3); out["main"]["Fc_peak_kN"] = float((Tn * C).max() / 1e3)
print("main:", {k: round(v, 3) for k, v in out["main"].items()})

# fixed ratio: bisect so that the 30 m stroke ends at 10 m/s with the same hold-down, line and damping
lo, hi = 10.0, 60.0
for _ in range(20):
    mid = 0.5 * (lo + hi); rb, mb = run(Fixed(mid), zeta=0.3)
    if mb["v_end"] > 10.0: hi = mid        # too little braking: the mass barely moves at a large ratio, so lower it
    else: lo = mid
Cf = 0.5 * (lo + hi); recf, mf = run(Fixed(Cf), zeta=0.3); mf["ratio"] = Cf; out["fixed"] = mf; np.save("fixed_rec.npy", recf)
print("fixed ratio %.1f:" % Cf, {k: round(v, 3) for k, v in mf.items()})

# scaled family: hold-down mass, stiffness and pre-tension in proportion to the body mass, same comb
out["family"] = {}
for mb_ in (20., 50., 100.):
    sc = mb_ / 50.; r_, m_ = run(cb, m=mb_, M=4000. * sc, K=29000. * sc, T0=300. * sc, zeta=0.3)
    out["family"][str(int(mb_))] = m_; print("family %3.0f kg:" % mb_, "p2m %.3f g_peak %.1f v_end %.2f Tpeak %.0f Fmean %.0f" % (m_["p2m_all"], m_["g_peak"], m_["v_end"], m_["Tpeak"], m_["Fmean_all"]))

# stiffness x damping grid
out["grid"] = []
for K in (10000., 29000., 580000.):
    for z in (0.0, 0.1, 0.2, 0.3, 0.4, 0.6):
        r_, m_ = run(cb, K=K, zeta=z); out["grid"].append(dict(K=K, zeta=z, p2m=m_["p2m_all"], v_end=m_["v_end"], slack=m_["slack_frac"]))
        print("grid K %6.0f zeta %.1f: p2m %.2f v_end %.2f slack %.2f" % (K, z, m_["p2m_all"], m_["v_end"], m_["slack_frac"]))
json.dump(out, open("arrest_results.json", "w"), indent=1); print("wrote arrest_results.json")
