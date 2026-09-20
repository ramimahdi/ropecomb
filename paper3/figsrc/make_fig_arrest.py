"""Supplementary figure for Section S17 (arrest mode): the transmission run in reverse, from sim/arrest/
(run_arrest.py -> arrest_results.json, main_rec.npy, fixed_rec.npy). Writes figs/fig_si_arrest.*"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); AR = os.path.join(HERE, "..", "sim", "arrest")
sys.path.insert(0, HERE); sys.path.insert(0, AR)
import figstyle as S; S.use()
import matplotlib.pyplot as plt
os.chdir(AR)
from sim import Comb
from target import build, gp
R = json.load(open("arrest_results.json")); d = json.load(open("fit_k7_8_6_L2.5.json")); cb = Comb(d); Sd = d["S"]
T = build(R["target"]["C0"], Rp=R["target"]["Rp"], j=R["target"]["j"], n=30001)
x, u, C, F, vb, vc = T["x"], T["u"], np.minimum(T["C"], R["target"]["C0"]), T["F"], T["vb"], T["vc"]
sub = slice(0, len(x), 30); xs, us = x[sub], u[sub]
Cfit = np.array([cb.G(Sd - ui) for ui in us])
main = np.load("main_rec.npy"); fixed = np.load("fixed_rec.npy")
tm, xm, vxm, um, vum, Tm, Fm, Cm = main.T; tf, xf, vxf, uf, vuf, Tf, Ff, Cf = fixed.T
g = 9.8; m = R["target"]["m"]

fig, ax = plt.subplots(2, 2, figsize=(S.W_FULL, 4.8))
a = ax[0, 0]
a.semilogy(x, C, label="required $C^*$", **S.IDEAL); a.semilogy(xs, Cfit, label=r"fitted $n_1 G_1 + n_2 G_2$, $\langle 8{:}6, 7\rangle$", **S.DESIGN)
a.set_xlim(0, 30); a.set_ylim(3, 200); a.set_ylabel("net ratio of the arrays"); a.legend(loc="lower left")
S.panel(a, "a", "required and fitted net ratio against body travel")
a = ax[0, 1]
a.plot(x, F / 1e3, label="target force on the body (kN)", **S.IDEAL); a.set_ylim(0, 10); a.set_ylabel("force (kN)")
b = S.twin_ok(a.twinx()); b.plot(x, vb, color=S.GREY, lw=1.0, label="body speed (m/s)"); b.plot(x, R["target"]["j"] * vc, color=S.GREY, lw=1.0, ls=(0, (1, 1.2)), label="hold-down mass speed (m/s)")
b.set_ylim(0, 110); b.set_ylabel("speed (m/s)"); a.set_xlim(0, 30)
h1, l1 = a.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels(); a.legend(h1 + h2, l1 + l2, loc="center right")
S.panel(a, "b", "target: a short ramp, then uniform force")
a = ax[1, 0]
a.plot(xf, Ff / m / g, color=S.GREY, lw=0.9, label="fixed ratio %.1f, peak-to-mean %.1f" % (R["fixed"]["ratio"], R["fixed"]["p2m_all"]))
a.plot(xm, Fm / m / g, label="arrays, peak-to-mean %.2f" % R["main"]["p2m_all"], **S.DESIGN)
a.axhline(R["target"]["Fu"] / m / g, label="target, uniform phase", **S.IDEAL)
a.set_xlim(0, 30); a.set_ylim(0, 175); a.set_ylabel("deceleration of the body (g)"); a.set_xlabel("body travel $x$ (m)"); a.legend(loc="upper right")
S.panel(a, "c", "simulated deceleration, damped output member")
a = ax[1, 1]
a.plot(tm * 1e3, Tm / 1e3, label="output-member tension (kN)", **S.DESIGN); a.set_ylim(0, 18); a.set_ylabel("tension (kN)")
b = S.twin_ok(a.twinx()); b.plot(tm * 1e3, um, color=S.GREY, lw=1.0, label="carriage lift (m)"); b.plot(tm * 1e3, R["target"]["j"] * vum, color=S.GREY, lw=1.0, ls=(0, (1, 1.2)), label="hold-down mass speed (m/s)")
b.set_ylim(0, 8); b.set_ylabel("lift (m), speed (m/s)"); a.set_xlim(0, tm[-1] * 1e3); a.set_xlabel("time (ms)")
h1, l1 = a.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels(); a.legend(h1 + h2, l1 + l2, loc="center right")
S.panel(a, "d", "tension, lift and hold-down mass speed")
for a in ax[0]: a.set_xlabel("body travel $x$ (m)")
fig.tight_layout(h_pad=1.0, w_pad=1.6)
S.save(fig, "fig_si_arrest")
