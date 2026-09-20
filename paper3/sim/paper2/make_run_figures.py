"""
Six-panel run charts for both configured designs against the ideal, in the layout
of [4] Figures 12 and 13. One figure for the rigid integrator, one for the compliant.

    python3 make_run_figures.py     ->  fig_runs_rigid.png, fig_runs_compliant.png
"""
import os, sys, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code")); sys.path.insert(0, HERE)
from catabult_sim import simulate
from catabult_sim_elastic import simulate_elastic

D = json.load(open(os.path.join(HERE, "designs_16471.json")))
M_ = D["_meta"]; M, m, v0, F, KR = M_["M"], M_["m"], M_["v0"], M_["F"], M_["k_rope"]
h = v0**2/(2*9.8); dmax = M_["d_max"]

def gear(P):
    P = [(list(R), list(s), float(mu)) for R, s, mu in P]
    def f(d):
        t = 0.0
        for R, s, mu in P:
            b = 0.0
            for Ri, si in zip(R, s):
                Di = d - si
                if Di > 0: b += 2.0*Di/math.sqrt(Ri*Ri + Di*Di)
            t += mu*b
        return t
    return f

def runs(mode):
    """ideal plus both designs, under one integrator."""
    kw = dict(max_heavy_dist=dmax, max_time=.5, max_target_acc=3e4, time_unit=1e-5)
    if mode == "rigid":
        ideal = simulate(M, m, h, get_gear_fn=F, min_heavy_speed=M_["v_stop"],
                         max_time=.6, max_target_acc=1e9, time_unit=1e-5)
        sim = lambda g: simulate(M, m, h, get_gear_fn=g, **kw)
    else:
        ideal = simulate(M, m, h, get_gear_fn=F, min_heavy_speed=M_["v_stop"],
                         max_time=.6, max_target_acc=1e9, time_unit=1e-5)
        sim = lambda g: simulate_elastic(M, m, h, g, k_rope=KR, max_heavy_dist=dmax,
                                         pretension=F, time_unit=2e-5)
    out = {"ideal": ideal}
    for tag, lab in (("k7", r"$\langle 7,7\rangle$, $k=7$"), ("k9", r"$\langle 5,9\rangle$, $k=9$")):
        d = D[tag]["dual"]
        out[lab] = sim(gear([(d["R_lo"], d["s_lo"], D[tag]["n_lo"]),
                             (d["R_hi"], d["s_hi"], D[tag]["n_hi"])]))
    return out

PANELS = [("heavy_dist", "a) source braking distance", "m"),
          ("gear",       "b) net displacement ratio", "ratio (y:1)"),
          ("heavy_speed","c) source speed", "m/s"),
          ("target_speed","d) payload speed", "m/s"),
          ("force_heavy","e) force on the source", "N"),
          ("force_target","f) force on the payload", "N")]
# colour and line style both carry the distinction, so the figures survive
# greyscale printing as well as colour
IDEAL = dict(color="k", lw=1.4, ls=(0, (6, 2)))
STYLE = {r"$\langle 7,7\rangle$, $k=7$": dict(color="crimson", lw=1.4, ls="-"),
         r"$\langle 5,9\rangle$, $k=9$": dict(color="royalblue", lw=1.4, ls=(0, (5, 1.8)))}

def figure(mode, path, title):
    R = runs(mode); ideal = R.pop("ideal")
    fig, axes = plt.subplots(3, 2, figsize=(12.2, 10.2))
    for ax, (key, lab, unit) in zip(axes.ravel(), PANELS):
        ax.plot(ideal["t"]*1e3, ideal[key], label="ideal", **IDEAL)
        for name, r in R.items():
            ax.plot(r["t"]*1e3, r[key], label=name, **STYLE[name])
        ax.set_title(lab, fontsize=13.5); ax.set_ylabel(unit, fontsize=12)
        ax.grid(alpha=0.25); ax.tick_params(labelsize=10.5)
        if key in ("force_heavy", "force_target"):
            v = np.concatenate([np.asarray(r[key], float) for r in R.values()])
            v = v[np.isfinite(v)]
            lo, hi = np.percentile(v, [0.5, 99.0])
            ax.set_ylim(min(0, lo), max(hi*1.25, 2.2*F if key == "force_target" else hi*1.25))
        if key == "force_target":
            ax.axhline(F, color="0.35", ls="-.", lw=0.9)
            ax.text(0.015, F, " design force", transform=ax.get_yaxis_transform(),
                    fontsize=9, color="0.35", va="bottom")
            for name, r in R.items():
                # same measure as the tables: peak over the design force, first 99.5%
                Fc = np.asarray(r[key], float); Fc = Fc[:max(1, int(len(Fc)*0.995))]
                SHORT = {r"$\langle 7,7\rangle$, $k=7$": r"$\langle 7,7\rangle$",
                         r"$\langle 5,9\rangle$, $k=9$": r"$\langle 5,9\rangle$"}
                ax.plot([], [], " ", label="%s peak/design %.2f" % (SHORT[name], Fc.max()/F))
    for ax in axes[-1]: ax.set_xlabel("time (ms)", fontsize=12)
    axes[0][0].legend(fontsize=10, loc="upper left")
    axes[-1][-1].legend(fontsize=9.5, loc="upper left")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(HERE, path), dpi=140, bbox_inches="tight")
    plt.close(fig); print("wrote", path)

figure("rigid", "fig_runs_rigid.png",
       "Rigid-body dynamics, both configured designs against the ideal profile")
figure("compliant", "fig_runs_compliant.png",
       "The same designs with a compliant, pre-tensioned output member (16,471 N/m)")
