"""Graphical abstract on the count-balanced <8:6,7> design (sim/designs_counts_16471.json, lambda = 0.03), in the
layout of fig_graphical_abstract. Writes figs/graphical_abstract.pdf/.png/.tiff (13 x 5 cm)."""
import os, json, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0]]
import make_figures as MF
D = json.load(open(os.path.join(HERE, "..", "sim", "designs_counts_16471.json")))
r = D["k7_8-6"]; g = r["by_lambda"]["0.03"]["geometry"]
MF.DUAL["k7c"] = dict(dual=dict(R_lo=g["R_lo"], s_lo=g["s_lo"], R_hi=g["R_hi"], s_hi=g["s_hi"]), n_lo=r["n_lo"], n_hi=r["n_hi"], k=r["k"])
lat = 100 * g["lateral"]
MF.fig_graphical_abstract("k7c", title="the dual-array RopeComb ($\\langle 8{:}6\\rangle$, $k$ = 7)",
                          guide_text="reaction imbalance %.1f%% of peak:\n%.1f%% of a single array's cancelled" % (lat, 100 - lat))
