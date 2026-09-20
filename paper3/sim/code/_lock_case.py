"""Lock one case to the highest-scoring restart of a chosen (N,k) cell, regenerate
its four figures and update locked_4ms.json.

    python3 _lock_case.py <tag> <N,k>        e.g.  _lock_case.py 1000 9,5
"""
import sys, os, json
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
os.chdir(os.path.join(_HERE, ".."))   # locked_4ms.json and the grid caches live in sim/
import numpy as np
from catabult_sim import simulate
from catabult_sim_elastic import simulate_elastic
from gear_fn_bridge import make_gear_fn
from ropecomb_fit4 import array_ratio
from plot32 import plot_3x2
from comb_figure import comb_figure
from instability_figure import instability_figure

CASE = {"100": (100., 999.0, "100:1", "100to1"),
        "1000": (1000., 3169.6, "1,000:1", "1000to1"),
        "10000": (10000., 10033.8, "10,000:1", "10000to1")}
tag, cell = sys.argv[1], sys.argv[2]
M, F, lab, key = CASE[tag]
h = 10. ** 2 / (2 * 9.8)

g = json.load(open("_grid4_%s.json" % tag))
b = max(g[cell]["cands"], key=lambda c: c["score3"])
R = np.array(b["R"]); s = np.array(b["s"]); k = b["k"]; N = b["N_eff"]; W = float(2 * R.sum())

I = simulate(M, 1., h, get_gear_fn=F, min_heavy_speed=4., max_time=.6,
             max_target_acc=1e9, time_unit=1e-5)
dmax = I["summary"]["heavy_travel"]; ideal = I["summary"]["target_final_speed"]
dd = np.linspace(0, dmax, 400); Gs = np.interp(dd, I["heavy_dist"], I["gear"])
gf = make_gear_fn(R, s, k)
rr = simulate(M, 1., h, get_gear_fn=gf, max_heavy_dist=dmax, max_time=.5,
              max_target_acc=3e4, time_unit=1e-5)
ee = simulate_elastic(M, 1., h, gf, k_rope=16471, max_heavy_dist=dmax,
                      pretension=F, time_unit=2e-5)
pk = float(np.asarray(ee["force_target"]).max() / F)

plot_3x2(rr, "FIG_rigid_%s.png" % key, f_ref=F, f_ymin=0.0, e_ymin=0.0,
         title="%s  |  %d members, k=%g, comb %.2f m — RIGID (inextensible)" % (lab, N, k, W),
         ideal=I)
plot_3x2(ee, "FIG_dynamics_%s.png" % key, f_ref=F, f_ymin=0.0, e_ymin=0.0,
         title="%s  |  %d members, k=%g, comb %.2f m — pre-tensioned elastic, peak %.2fx"
               % (lab, N, k, W, pk), ideal=I)
comb_figure(R, s, k, dmax, "FIG_comb_%s.png" % key,
            title="%s — %d members, k=%g" % (lab, N, k), ideal=(dd, Gs))
instability_figure(rr, "FIG_instability_%s.png" % key,
                   title="%s — %d members, k=%g" % (lab, N, k))

res = 100 * float(np.sqrt(np.mean((k * array_ratio(dd, R, s) - Gs) ** 2))) / np.ptp(Gs)
L = json.load(open("locked_4ms.json"))
L[key].update(N=N, k=k, built=N, R=[float(x) for x in R], s=[float(x) for x in s],
              width=W, comb_width=W, rigid=b["exit_rigid"], exit=b["exit_elastic"],
              vh_end=float(ee["summary"]["heavy_final_speed"]), peak=pk,
              p2m_rigid_995=b["pm_rigid"], p2m_comp_995=b["pm_elastic"],
              residual_pct=res, rms=b["rms"], score3=b["score3"],
              spans=[round(2 * x * 100) for _, x in sorted(zip(s, R))],
              provenance=("grid search 90 cells x 50 restarts, top 5 per cell simulated; "
                          "admissible set (p2m_c<=1.3, exit>=99% ideal) ranked by "
                          "width-penalised score"))
json.dump(L, open("locked_4ms.json", "w"), indent=2)

Fv = np.asarray(rr["force_target"]); Ev = np.asarray(ee["force_target"])
print("%s LOCKED  N=%d k=%g  comb %.2f m  residual %.2f%%" % (lab, N, k, W, res))
print("  spans (cm):", [round(2 * x * 100) for _, x in sorted(zip(s, R))])
print("  exit_r %.1f | exit_c %.1f (%.1f%% of ideal %.1f) | p2m_r %.3f | p2m_c %.3f | peak/F %.3f"
      % (b["exit_rigid"], b["exit_elastic"], 100 * b["exit_elastic"] / ideal, ideal,
         b["pm_rigid"], b["pm_elastic"], pk))
print("  rigid terminal peak %.1f kN | rigid full p2m %.1f | compliant peak %.2fx design"
      % (Fv.max() / 1e3, Fv.max() / Fv.mean(), Ev.max() / F))
