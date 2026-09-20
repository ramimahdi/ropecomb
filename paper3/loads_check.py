"""Component loads for Table 14 of main.tex, derived from frozen designs_16471.json and locked_4ms.json.
T_j = n_j F; anchor = T_j; per-array reaction = n_j F G_j(d); block speed = v_exit/k; sheave rev/min at 100 mm; MBL at SF 3."""
import json, os, numpy as np
_S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim")
d=json.load(open(os.path.join(_S,'paper2','designs_16471.json'))); l=json.load(open(os.path.join(_S,'locked_4ms.json')))['1000to1']
F=3169.6
def Garr(R,s,x):
    D=np.maximum(0,x-np.array(s)); return float(np.sum(2*D/np.sqrt(np.array(R)**2+D**2)))
for key in ['k7','k9']:
    c=d[key]; x=c['dual']; dmax=d['_meta']['d_max']; n1,n2,k=c['n_lo'],c['n_hi'],c['k']
    xs=np.linspace(0,dmax,4001)
    g1=np.array([Garr(x['R_lo'],x['s_lo'],t) for t in xs]); g2=np.array([Garr(x['R_hi'],x['s_hi'],t) for t in xs])
    net=n1*g1+n2*g2; imb=np.abs(n1*g1-n2*g2); v=x['vC']
    print(f"{key}: T1={n1*F/1e3:.1f} kN T2={n2*F/1e3:.1f} kN | MBL SF3 {3*n1*F/1e3:.1f}/{3*n2*F/1e3:.1f} kN | peak reaction {F*net.max()/1e3:.0f} kN, per array {F*n1*g1.max()/1e3:.0f}/{F*n2*g2.max()/1e3:.0f} kN | lateral {imb.max()/net.max()*100:.2f}% = {F*imb.max()/1e3:.1f} kN | block {v/k:.1f} m/s | sheave {v/(np.pi*0.1)*60:,.0f} rev/min")
xs=np.linspace(0,l['d_max'],4001); g=np.array([Garr(l['R'],l['s'],t) for t in xs]); k=l['k']; v=l['exit']
print(f"single <9,5>: T=kF={k*F/1e3:.1f} kN | MBL SF3 {3*k*F/1e3:.1f} kN | peak reaction {F*k*g.max()/1e3:.0f} kN | block {v/k:.1f} m/s | sheave {v/(np.pi*0.1)*60:,.0f} rev/min")
