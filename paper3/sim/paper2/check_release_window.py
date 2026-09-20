"""Where does the rigid peak-to-mean come from? Full curve vs truncated release."""
import os, sys, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import numpy as np
from catabult_sim import simulate

D = json.load(open(os.path.join(HERE, "designs_16471.json")))
M, m, v0, F, VSTOP = 1000., 1., 10., 3169.6, 4.0
h = v0**2/(2*9.8)
dmax = D["_meta"]["d_max"]

def gear(pieces):
    P = [(list(R), list(s), float(mu)) for R, s, mu in pieces]
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

d7 = D["k7"]["dual"]
g = gear([(d7["R_lo"], d7["s_lo"], D["k7"]["n_lo"]),
          (d7["R_hi"], d7["s_hi"], D["k7"]["n_hi"])])
rr = simulate(M, m, h, get_gear_fn=g, max_heavy_dist=dmax, max_time=.5,
              max_target_acc=3e4, time_unit=1e-5)
Fp = np.asarray(rr["force_target"], float)
n = len(Fp)
print("frozen designs_16471.json rigid p2m  : %.2f" % d7["pR"])
print("samples in the force history          : %d\n" % n)
print("  %-34s %8s" % ("window", "peak/mean"))
print("  " + "-"*46)
for label, frac in [("full curve, no truncation", 1.000),
                    ("first 99.9%", 0.999), ("first 99.5%", 0.995),
                    ("first 99.0%", 0.990), ("first 98%", 0.980), ("first 95%", 0.950)]:
    c = Fp[:max(1, int(n*frac))]
    print("  %-34s %8.2f" % (label, c.max()/c.mean()))
print("\n  last 10 samples of the force history (N):")
print("   ", np.round(Fp[-10:], 0))
print("  peak occurs at sample %d of %d (%.2f%% through)" % (Fp.argmax(), n, 100*Fp.argmax()/n))
