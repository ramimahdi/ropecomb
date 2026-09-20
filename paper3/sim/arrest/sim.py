import numpy as np, json
from target import gp, Yp
g=9.8
def Yarr(R,s,tt):
    R=np.asarray(R); s=np.asarray(s); D=np.maximum(0,tt-s); return (2*(np.sqrt(R**2+D**2)-R)).sum()
def Garr(R,s,tt):
    R=np.asarray(R); s=np.asarray(s); D=np.maximum(0,tt-s); return (2*D/np.sqrt(R**2+D**2)).sum()

class Comb:
    def __init__(s,d):
        s.d=d; s.S=d['S']; s.n1=d['n1']; s.n2=d['n2']
        s.Y0=s.Y(s.S)
    def Y(s,tt): d=s.d; return s.n1*Yarr(d['R1'],d['s1'],tt)+s.n2*Yarr(d['R2'],d['s2'],tt)
    def G(s,tt): d=s.d; return s.n1*Garr(d['R1'],d['s1'],tt)+s.n2*Garr(d['R2'],d['s2'],tt)
    def G12(s,tt): d=s.d; return s.n1*Garr(d['R1'],d['s1'],tt), s.n2*Garr(d['R2'],d['s2'],tt)
    def payout(s,u): return s.Y0-s.Y(s.S-u)

class Fixed:
    def __init__(s,C): s.C=C; s.S=1e9
    def G(s,tt): return s.C
    def payout(s,u): return s.C*u

def simulate(ratio, m=50., v0=100., X=30., M=4000., j=2, Rp=2.5, K=29000., T0=300., dt=2e-5, zeta=0.0, tmax=2.0, cu=0.0):
    Me, We = j*j*M, j*M*g
    e0=T0/K
    x=0.; vx=v0; u=0.; vu=0.; t=0.
    c = 0.0
    rec=[]
    while x<X and t<tmax and vx>0:
        e=Yp(x,Rp)-ratio.payout(u)+e0
        gpx=gp(x,Rp); C=ratio.G(ratio.S-u) if isinstance(ratio,Comb) else ratio.C
        edot=gpx*vx-C*vu
        if zeta>0:
            meff=m/max(gpx**2,1e-3); c=2*zeta*np.sqrt(K*meff)
        T=max(0.,K*e+c*edot) if e>0 else 0.
        ax=-T*gpx/m
        au=(T*C-We-cu*vu)/Me
        if u<=0 and au<0: au=0.; vu=max(vu,0.)
        # semi-implicit Euler
        vx+=ax*dt; x+=vx*dt
        vu+=au*dt; u=max(0.,u+vu*dt)
        t+=dt
        rec.append((t,x,vx,u,vu,T,T*gpx,C))
    return np.array(rec)

def metrics(rec, Fu=None, m=50., ramp_frac=0.9):
    t,x,vx,u,vu,T,F,C=rec.T
    Fm=F[x<=x[-1]]
    i0=np.argmax(F>=ramp_frac*(Fu if Fu else F.max()))
    Fr=F[i0:]
    return dict(t_end=t[-1],x_end=x[-1],v_end=vx[-1],KE_absorbed=1-(vx[-1]/100)**2,
                stroke=u.max(),vmass_end=2*vu[-1],Tpeak=T.max(),Fpeak=F.max(),
                Fmean_uniform=Fr.mean(),p2m_uniform=Fr.max()/Fr.mean(),
                Fmean_all=np.trapezoid(F,x)/(x[-1]-x[0]),p2m_all=F.max()/(np.trapezoid(F,x)/(x[-1]-x[0])),
                g_peak=F.max()/m/9.8,x_ramp=x[i0],slack_frac=float(np.mean(T[i0:]<=0)),Fmin_uniform=Fr.min())
