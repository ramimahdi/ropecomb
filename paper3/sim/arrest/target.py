import numpy as np
g=9.8
def gp(x,Rp): return 2*x/np.sqrt(Rp**2+x**2)
def Yp(x,Rp): return 2*(np.sqrt(Rp**2+x**2)-Rp)

def build(C0, m=50., v0=100., v1=10., X=30., M=4000., Rp=1.0, n=60001, j=1):
    """Target: phase A comb ratio fixed at C0 (=k*Gnet at full depth); phase B uniform force F_u.
    Returns dict with x, vb, vc, u, F, C (=k Gnet required), x1, F_u."""
    E0=0.5*m*v0**2
    Mm, M = M, j*j*M   # effective mass seen at the carriage through a j:1 hold-down tackle
    Wg = j*Mm*g        # effective weight
    x=np.linspace(0,X,n); h=x[1]-x[0]
    # phase A: integrate u with u'=gp/C0
    uA=np.concatenate([[0],np.cumsum(0.5*(gp(x[1:],Rp)+gp(x[:-1],Rp))*h)])/C0
    upA=gp(x,Rp)/C0
    vbA=np.sqrt(2*(E0-Wg*uA)/(m+M*upA**2))
    FA=-m*vbA*np.gradient(vbA,x)
    # find x1 self-consistently
    Fu=None
    for i in range(1,n):
        Fu_i=(0.5*m*vbA[i]**2-0.5*m*v1**2)/(X-x[i]) if X-x[i]>1e-9 else np.inf
        if FA[i]>=Fu_i: i1=i; Fu=Fu_i; break
    if Fu is None: return None
    x1=x[i1]
    vb=vbA.copy(); u=uA.copy()
    vb[i1:]=np.sqrt(np.maximum(vbA[i1]**2-2*Fu*(x[i1:]-x1)/m,1e-12))
    # integrate u in phase B
    for i in range(i1,n-1):
        e=E0-0.5*m*vb[i]**2-Wg*u[i]
        vc=np.sqrt(max(2*e/M,0)); up=vc/vb[i]
        u[i+1]=u[i]+up*h
    vc=np.sqrt(np.maximum(2*(E0-0.5*m*vb**2-Wg*u)/M,0))
    up=np.where(x<x1,upA,vc/vb)
    with np.errstate(divide='ignore',invalid='ignore'):
        C=np.where(x<x1,C0,gp(x,Rp)/up)
    F=np.where(x<x1,FA,Fu)
    return dict(x=x,vb=vb,vc=vc,u=u,F=F,C=C,x1=x1,Fu=Fu,M=M,Wg=Wg,j=j,m=m,Rp=Rp,up=up,T=F/np.maximum(gp(x,Rp),1e-9))

def uniform_peak(m=50.,v0=100.,v1=10.,X=30.,M=4000.,Rp=1.0,j=1,n=300001):
    Me,We=j*j*M,j*M*g
    x=np.linspace(1e-6,X,n);h=x[1]-x[0]
    F=0.5*m*(v0**2-v1**2)/X
    vb=np.sqrt(v0**2-2*F*x/m)
    u=0;us=[]
    for xi,v in zip(x,vb):
        us.append(u); e=F*xi-We*u; u+=np.sqrt(max(2*e/Me,0))/v*h
    u=np.array(us); vc=np.sqrt(np.maximum(2*(F*x-We*u)/Me,0)); C=gp(x,Rp)/(vc/vb)
    i=np.argmax(C); return C[i],x[i]
if __name__=="__main__":
    import sys
    Rp=float(sys.argv[1]); j=int(sys.argv[2])
    Cp,xp=uniform_peak(Rp=Rp,j=j); print("uniform peak",Cp,xp)
    for C0 in np.arange(round(Cp)-5,round(Cp)+25,2):
        r=build(C0,Rp=Rp,j=j,n=30001)
        if r is None: print(C0,'none'); continue
        C=r['C'];x=r['x'];i=np.searchsorted(x,r['x1'])
        print(f"C0={C0} x1={r['x1']:.2f} Fu={r['Fu']:.0f} stroke={r['u'][-1]:.3f} Cend={C[-1]:.2f} maxC_after={C[i:].max():.1f} Tpeak={r['T'][5:].max():.0f} Fmax={r['F'].max():.0f}")
