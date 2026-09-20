import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, json, sys
from scipy.optimize import least_squares
from target import build, gp
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
C0=100.; Rp=2.5; j=2
T=build(C0,Rp=Rp,j=j,n=30001)
x,u,C=T['x'],T['u'],np.minimum(T['C'],C0)
S=u[-1]
# target as function of depth coordinate t = S - u  (t=0 at end of arrest, t=S at start)
xs=np.linspace(0.02,x[-1],500)          # sample along the drone's travel: force matters per metre of arrest
ug=np.interp(xs,x,u)
Cg=np.interp(xs,x,C)
t=S-ug

def arr(R,s,tt):
    D=np.maximum(0,tt[:,None]-s[None,:]); return (2*D/np.sqrt(R[None,:]**2+D**2)).sum(1)

def unpack(p,N1,N2):
    R1=0.05+np.logaddexp(0,p[:N1]); R2=0.05+np.logaddexp(0,p[N1:N1+N2])
    s1=p[N1+N2:2*N1+N2]; s2=p[2*N1+N2:]
    return R1,s1,R2,s2

def resid(p,k,n1,n2,N1,N2,lam,Lw,Gt):
    R1,s1,R2,s2=unpack(p,N1,N2)
    G1=arr(R1,s1,t); G2=arr(R2,s2,t)
    Gn=n1*G1+n2*G2
    scale=Gt+0.05*Gt.max()
    r1=(Gn-Gt)/scale
    r2=np.sqrt(lam)*(n1*G1-n2*G2)/scale
    w=[10*max(0,2*R1.sum()-Lw),10*max(0,2*R2.sum()-Lw)]
    return np.concatenate([r1,r2,w])

def fit(k,N1,N2,lam=0.03,Lw=6.0,starts=40,seed=0):
    n1,n2=k//2,k-k//2
    Gt=Cg   # net ratio n1*G1+n2*G2; the falls already carry k
    rng=np.random.default_rng(seed); best=None
    for st in range(starts):
        R10=rng.uniform(0.2,1.5,N1); R20=rng.uniform(0.2,1.5,N2)
        q1=np.sort(rng.uniform(0,1,N1)); q2=np.sort(rng.uniform(0,1,N2))
        s10=S-(S+1.0)*(1-q1)**rng.uniform(1.5,3); s20=S-(S+1.0)*(1-q2)**rng.uniform(1.5,3)
        s10=np.minimum(s10,S-0.01); s20=np.minimum(s20,S-0.01)
        R10=np.sort(R10)[::-1]*rng.uniform(0.3,1.5); R20=np.sort(R20)[::-1]*rng.uniform(0.3,1.5)
        p0=np.concatenate([np.log(np.expm1(np.maximum(R10-0.05,1e-3))),np.log(np.expm1(np.maximum(R20-0.05,1e-3))),s10,s20])
        lb=np.concatenate([-20*np.ones(N1+N2),-4*np.ones(N1+N2)]); ub=np.concatenate([5*np.ones(N1+N2),S*np.ones(N1+N2)])
        try:
            r=least_squares(resid,p0,bounds=(lb,ub),args=(k,n1,n2,N1,N2,lam,Lw,Gt),max_nfev=4000)
        except Exception as e: continue
        rr=resid(r.x,k,n1,n2,N1,N2,0,Lw,Gt)[:len(t)]
        rms=np.sqrt(np.mean(rr**2))
        if best is None or r.cost<best[0]: best=(r.cost,r.x,rms)
    R1,s1,R2,s2=unpack(best[1],N1,N2)
    G1=arr(R1,s1,t);G2=arr(R2,s2,t)
    imb=np.abs(n1*G1-n2*G2).max()/(n1*G1+n2*G2).max()
    return dict(k=k,n1=n1,n2=n2,N1=N1,N2=N2,R1=R1.tolist(),s1=s1.tolist(),R2=R2.tolist(),s2=s2.tolist(),
                rel_rms=best[2],imb=imb,w1=2*R1.sum(),w2=2*R2.sum(),lam=lam,Lw=Lw,S=S)
if __name__=="__main__":
    k,N1,N2,Lw=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),float(sys.argv[4])
    d=fit(k,N1,N2,Lw=Lw,starts=int(sys.argv[5]) if len(sys.argv)>5 else 30,lam=float(sys.argv[6]) if len(sys.argv)>6 else 0.03)
    print(f"k={k} {N1}:{N2} Lw={Lw} relrms={100*d['rel_rms']:.2f}% imb={100*d['imb']:.1f}% widths {d['w1']:.2f}/{d['w2']:.2f}")
    print(' R1',np.round(d['R1'],3),'\n s1',np.round(d['s1'],3),'\n R2',np.round(d['R2'],3),'\n s2',np.round(d['s2'],3))
    json.dump(d,open(f"fit_k{k}_{N1}_{N2}_L{Lw}.json","w"),indent=1)
