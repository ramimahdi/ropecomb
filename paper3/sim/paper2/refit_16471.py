"""
Re-select both dual designs, and their single-array baselines, with the rope
stiffness of [4] Appendix B: 2 mm Dyneema R3, k = EA/L = 16,471 N/m.

Writes designs_16471.json.
"""
import os, sys, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code")); sys.path.insert(0, HERE)
from catabult_sim import simulate
from catabult_sim_elastic import simulate_elastic
from ropecomb_fit4 import fit_array
from semisym_fit import ideal_target, falls, fit_semisym_staged, verify_interleaving

M, m, v0, F, VSTOP, KR, ETA = 1000., 1., 10., 3169.6, 4.0, 16471, 0.98
h = v0**2/(2*9.8)
dd, Gs, dmax = ideal_target(M, m, v0, F, VSTOP, n=400, code_dir=os.path.join(HERE,"..","code"))
Gt = float(Gs[-1])

def gear(P):
    P=[(list(R),list(s),float(mu)) for R,s,mu in P]
    def f(d):
        t=0.
        for R,s,mu in P:
            b=0.
            for Ri,si in zip(R,s):
                Di=d-si
                if Di>0: b+=2.*Di/math.sqrt(Ri*Ri+Di*Di)
            t+=mu*b
        return t
    return f

def ev(g):
    kw=dict(max_heavy_dist=dmax,max_time=.5,max_target_acc=3e4,time_unit=1e-5)
    rr=simulate(M,m,h,get_gear_fn=g,**kw)
    ee=simulate_elastic(M,m,h,g,k_rope=KR,max_heavy_dist=dmax,pretension=F,time_unit=2e-5)
    cut=lambda a:a[:max(1,int(len(a)*0.995))]
    Fr=cut(np.asarray(rr["force_target"],float)); Fc=cut(np.asarray(ee["force_target"],float))
    return dict(pR=float(Fr.max()/Fr.mean()), pC=float(Fc.max()/F),
                vC=float(ee["summary"]["target_final_speed"]))

out={"_meta":dict(k_rope=KR, note="EA/L, 2 mm Dyneema R3, L~17 m, per [4] App. B",
                  M=M,m=m,v0=v0,F=F,v_stop=VSTOP,d_max=dmax,target_terminal=Gt)}
for N,K,tag,W in [(7,7,"k7",2.11),(5,9,"k9",1.96)]:
    b=None
    for sd in range(10):
        R,s,_,rms=fit_array(dd,Gs,N,K,dmax,seed=sd,R_bounds=(0.05,8.),width_budget=W,overshoot_weight=0.0)
        if b is None or rms<b[2]: b=(np.asarray(R),np.asarray(s),rms)
    Rst,sst,rms1=b
    single=ev(gear([(Rst,sst,K)]))
    n_lo,n_hi=falls(K); best=None
    for lam in (0.03,0.1,0.3,1.0):
        o=fit_semisym_staged(dd,Gs,Rst,sst,K,dmax,W,lam_balance=lam,lam_width=50.)
        if not verify_interleaving(o)[0]: continue
        e=ev(gear([(o["R_lo"],o["s_lo"],n_lo),(o["R_hi"],o["s_hi"],n_hi)]))
        cand=dict(lam=lam,rms=float(o["rms"]),imbalance=float(o["imbalance"]),
                  terminal=float(o["terminal"]),
                  R_lo=list(map(float,o["R_lo"])),s_lo=list(map(float,o["s_lo"])),
                  R_hi=list(map(float,o["R_hi"])),s_hi=list(map(float,o["s_hi"])),**e)
        if best is None or cand["pC"]<best["pC"]: best=cand
    out[tag]=dict(N=N,k=K,n_lo=n_lo,n_hi=n_hi,width_budget=W,
                  single=dict(rms=float(rms1),**single), dual=best)
    print("%-4s single pC %.2f pR %.2f | dual pC %.2f pR %.2f (lam %g, rms %.3f, imb %.1f%%)"
          % (tag,single["pC"],single["pR"],best["pC"],best["pR"],best["lam"],best["rms"],100*best["imbalance"]))
json.dump(out,open(os.path.join(HERE,"designs_16471.json"),"w"),indent=1)
print("\nwrote designs_16471.json")
for tag in ("k7","k9"):
    d=out[tag]["dual"]
    print("\n%s spans (mm)  lead: %s" % (tag,[round(1000*x) for x in d["R_lo"]]))
    print("   %s      second: %s" % (" "*len(tag),[round(1000*x) for x in d["R_hi"]]))
    print("   offsets(mm) lead: %s" % [round(1000*x) for x in d["s_lo"]])
    print("   %s      second: %s" % (" "*len(tag),[round(1000*x) for x in d["s_hi"]]))
