"""Figure for Section 6.4 (balance by member count): reaction imbalance through the stroke and compliant payload force,
the equal-count designs <7,7> and <5,9> against the count-balanced <8:6,7> and <5:4,9> (and <6:5,9> thin), from
sim/designs_counts_16471.json. Writes figs/fig_counts.*"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "sim", "paper2"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import figstyle as S; S.use()
from semisym_fit import bank_ratio
D = json.load(open(os.path.join(HERE, "..", "sim", "designs_counts_16471.json")))
F = D["_meta"]["F"]; dmax = D["_meta"]["d_max"]; dd = np.linspace(0, dmax, 800)


def lateral(g, nlo, nhi):
    gl = bank_ratio(dd, np.array(g["R_lo"]), np.array(g["s_lo"])); gh = bank_ratio(dd, np.array(g["R_hi"]), np.array(g["s_hi"]))
    net = nlo * gl + nhi * gh
    return 100 * np.abs(nlo * gl - nhi * gh) / net.max()


def design(tag, lam=None):
    r = D[tag]
    if lam is None: return r["geometry"], r["dynamics"]["1.0"], r["k"], r["N_lo"], r["N_hi"]
    e = r["by_lambda"][str(lam)]; return e["geometry"], e["dynamics"]["1.0"], r["k"], r["N_lo"], r["N_hi"]


ROWS = ((7, 3, 4, [("k7_7-7", None, r"$\langle 7,7\rangle$", dict(color=S.GREY, lw=1.0)),
                   ("k7_8-6", 0.03, r"$\langle 8{:}6,7\rangle$", dict(color=S.BLACK, lw=1.2))]),
        (9, 4, 5, [("k9_5-5", None, r"$\langle 5,9\rangle$", dict(color=S.GREY, lw=1.0)),
                   ("k9_5-4", 0.02, r"$\langle 5{:}4,9\rangle$", dict(color=S.BLACK, lw=1.2)),
                   ("k9_6-5", 0.03, r"$\langle 6{:}5,9\rangle$", dict(color=S.BLUE, lw=0.9, ls=(0, (1, 1.2))))]))
fig, axes = plt.subplots(2, 2, figsize=(S.W_FULL, 4.6))
for i, (k, nlo, nhi, items) in enumerate(ROWS):
    aL, aF = axes[i]; mirror = 100 * abs(nhi - nlo) / k
    aL.axhline(mirror, color=S.GREY, ls=(0, (4, 2)), lw=0.9, label="mirror-symmetric, same $k$")
    aF.axhline(F / 1e3, label="design force", **S.IDEAL)
    for tag, lam, name, st in items:
        g, dy, _, N_lo, N_hi = design(tag, lam)
        imb = lateral(g, nlo, nhi)
        aL.plot(dd, imb, label=f"{name}  {imb.max():.1f}%", **st)
        aF.plot(dy["trace_d"], np.array(dy["trace_F"]) / 1e3, label=f"{name}  {dy['pC']:.2f}", **st)
    aL.set_ylim(0, mirror * 1.45); aL.yaxis.set_major_locator(MaxNLocator(5, integer=True)); aL.set_xlim(0, dmax); aF.set_xlim(0, dmax); aF.set_ylim(0, 5.2)
    aL.set_ylabel("reaction imbalance (% of peak reaction)"); aF.set_ylabel("payload force, compliant (kN)")
    aL.legend(loc="upper left", fontsize=6.0); aF.legend(loc="upper left", fontsize=6.0)
    S.panel(aL, "ac"[i], f"reaction imbalance through the stroke, $k = {k}$"); S.panel(aF, "bd"[i], f"payload force, $k = {k}$")
for a in axes[1]: a.set_xlabel("carriage displacement $d$ (m)")
fig.tight_layout(h_pad=0.9, w_pad=1.4)
S.save(fig, "fig_counts")
