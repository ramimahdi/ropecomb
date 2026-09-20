"""Compare the two lead assignments as TRADE CURVES, not single points."""
import os,sys,json
import numpy as np
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,"..","code")); sys.path.insert(0,HERE)
from ropecomb_fit4 import fit_array
from semisym_fit import (ideal_target,falls,bank_ratio,interleaved_slots,
                         seed_two_stage,encode,_residuals,_unpack)
M,m,v0,F,VSTOP,W=1000.,1.,10.,3169.6,4.0,2.11
dd,Gs,dmax=ideal_target(M,m,v0,F,VSTOP,n=400,code_dir=os.path.join(HERE,"..","code"))
cache={}
def stage1(N,k,sd):
    if (N,k,sd) not in cache:
        R,s,_,_=fit_array(dd,Gs,N,k,dmax,seed=sd,R_bounds=(0.05,8.),width_budget=W,overshoot_weight=0.0)
        cache[(N,k,sd)]=(np.asarray(R),np.asarray(s))
    return cache[(N,k,sd)]
def fit(k,N,lam,order,sd,jit):
    n_lo,n_hi=falls(k); Rst,sst=stage1(N,k,sd%8)
    p0=encode(*seed_two_stage(Rst,sst,dmax),dmax,order)
    if jit: p0=p0+np.random.default_rng(sd).normal(0,jit,p0.shape)
    args=(dd,Gs,N,N,n_lo,n_hi,dmax,W,lam,50.0,order); n=2*N; head=p0[:n+1].copy()
    s1=least_squares(lambda q:_residuals(np.concatenate([head,q]),*args),p0[n+1:],method="lm",max_nfev=2500)
    s2=least_squares(_residuals,np.concatenate([head,s1.x]),method="lm",max_nfev=3500,args=args)
    Rlo,slo,Rhi,shi=_unpack(s2.x,N,N,dmax,order)
    gl,gh=bank_ratio(dd,Rlo,slo),bank_ratio(dd,Rhi,shi); net=n_lo*gl+n_hi*gh
    return (float(np.sqrt(np.mean((net-Gs)**2))),
            float(np.max(np.abs(n_lo*gl-n_hi*gh))/max(net.max(),1e-9)),float(s2.cost))
for k,N in [(9,5),(7,7)]:
    print("\n=== k = %d, %d members per array: trade curves ===" % (k,N))
    print("  %-6s | %-22s | %-22s" % ("lambda","low-fall leads","high-fall leads"))
    print("  %-6s | %10s %11s | %10s %11s" % ("","rms","imbalance","rms","imbalance"))
    print("  "+"-"*58)
    for lam in (0.1,0.3,1.0,3.0,10.0,30.0):
        row=[]
        for lab in ("low","high"):
            order=interleaved_slots(N,N)
            if lab=="high": order=1-order
            runs=[]
            for sd in range(14):
                try: runs.append(fit(k,N,lam,order,sd,0.0 if sd==0 else 0.25))
                except Exception: pass
            runs.sort(key=lambda r:r[2]); row.append(runs[0] if runs else (float('nan'),)*3)
        print("  %-6g | %10.4f %10.1f%% | %10.4f %10.1f%%"
              % (lam,row[0][0],100*row[0][1],row[1][0],100*row[1][1]))
