"""Design figures (the layout of Figures 10 and 11) for the count-balanced designs of sim/designs_counts_16471.json:
<8:6,7> at lambda = 0.03 and <5:4,9> at lambda = 0.02 (Supplementary Section S16).
Writes figs/fig_design_k7c.* and fig_design_k9c.*"""
import os, json, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0]]
import make_figures as MF
D = json.load(open(os.path.join(HERE, "..", "sim", "designs_counts_16471.json")))
for tag, key, lam in (("k7c", "k7_8-6", "0.03"), ("k9c", "k9_5-4", "0.02")):
    r = D[key]; g = r["by_lambda"][lam]["geometry"]
    MF.DUAL[tag] = dict(dual=dict(R_lo=g["R_lo"], s_lo=g["s_lo"], R_hi=g["R_hi"], s_hi=g["s_hi"]), n_lo=r["n_lo"], n_hi=r["n_hi"], k=r["k"])
    MF.fig_design(tag)
