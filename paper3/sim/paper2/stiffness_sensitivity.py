"""
Is the compliant peak-to-mean sensitive to the assumed rope stiffness?

Paper 1 Appendix B derives k = EA/L = (E/sigma) F_break / L and notes that E/sigma
lies between about 34 and 66 across UHMWPE, steel wire rope, carbon fibre and PBO,
so working strain is of order 1% whatever the material is. It reports 1.6e4 N/m for
its worked case and states the result is insensitive to stiffness over that range.
This re-tests that on the two dual-array designs of Section 9.

    python3 stiffness_sensitivity.py
"""
import os, sys, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code")); sys.path.insert(0, HERE)
from catabult_sim_elastic import simulate_elastic

D = json.load(open(os.path.join(HERE, "designs_16471.json")))
M, m, v0, F = 1000., 1., 10., 3169.6
h = v0**2/(2*9.8); dmax = D["_meta"]["d_max"]

def gear(pieces):
    P=[(list(R),list(s),float(mu)) for R,s,mu in pieces]
    def f(d):
        t=0.0
        for R,s,mu in P:
            b=0.0
            for Ri,si in zip(R,s):
                Di=d-si
                if Di>0: b+=2.0*Di/math.sqrt(Ri*Ri+Di*Di)
            t+=mu*b
        return t
    return f

print("compliant peak-to-mean against assumed rope stiffness")
print("(paper 1 Appendix B: k = (E/sigma) F_break / L, E/sigma in [34, 66])\n")
print("  %-22s %s" % ("stiffness (N/m)", "  ".join("%8s" % s for s in ("k=7 design","k=9 design"))))
print("  " + "-"*50)
rows=[]
for kr in (8000, 10000, 12000, 15730, 16471, 19663, 20000, 30000):
    vals=[]
    for key in ("k7","k9"):
        du=D[key]["dual"]
        g=gear([(du["R_lo"],du["s_lo"],D[key]["n_lo"]),
                (du["R_hi"],du["s_hi"],D[key]["n_hi"])])
        ee=simulate_elastic(M,m,h,g,k_rope=kr,max_heavy_dist=dmax,pretension=F,time_unit=2e-5)
        Fc=np.asarray(ee["force_target"],float)
        vals.append(Fc[:max(1,int(len(Fc)*0.995))].max()/F)
    rows.append((kr,vals))
    mark = "   <- physical band" if kr in (15730,19663) else ("   <- used here" if kr==16471 else "")
    print("  %-22s %8.2f  %8.2f%s" % ("%,d"%kr if False else format(kr,","), vals[0], vals[1], mark))
a=[r[1][0] for r in rows]; b=[r[1][1] for r in rows]
print("\n  spread over the full range tested: k=7 %.2f to %.2f, k=9 %.2f to %.2f"
      % (min(a),max(a),min(b),max(b)))
