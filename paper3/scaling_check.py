"""Verify the dimensionless form of the target equation against the frozen Table 3 numbers.
Dimensional: dy/dd = sqrt(2Fy/m)/sqrt(v0^2 + 2 g d - 2Fy/M).
Dimensionless: yh = 2Fy/(M v0^2), delta = d/dmax, gamma = 2 g dmax/v0^2, Phi = 2 F dmax/(sqrt(Mm) v0^2)
  d yh/d delta = Phi sqrt(yh/(1+gamma delta - yh)),  G* = sqrt(M/m) sqrt(yh/(1+gamma delta - yh))
"""
import numpy as np
from scipy.integrate import solve_ivp
g=9.8; v0=10.0; nu=0.4
cases={'100:1':(100.,1.,999.0,0.8541213821783579,25.0,99.9),
       '1,000:1':(1000.,1.,3169.6,0.8541739050539013,79.3,317.0),
       '10,000:1':(10000.,1.,10033.8,0.8541707785332826,251.0,1003.5)}
print("case      gamma   Phi     yh(1)   1+g-nu^2  G*(1)  tab   vexit  tab    T(ms)  ymax(m)")
for name,(M,m,F,dmax,Gt,vx) in cases.items():
    gamma=2*g*dmax/v0**2; Phi=2*F*dmax/(np.sqrt(M*m)*v0**2)
    def rhs(delta,yh): 
        return [Phi*np.sqrt(max(yh[0],0)/(1+gamma*delta-yh[0]))]
    # seed from small-d expansion: y ~ (2F/m) d^2/(4 v0^2) -> yh ~ Phi^2 delta^2/4 ... check: yh=2Fy/(Mv0^2)
    d0=1e-6; yh0=2*F*((2*F/m)*(d0*dmax)**2/(4*v0**2))/(M*v0**2)
    sol=solve_ivp(rhs,[d0,1.0],[yh0],rtol=1e-10,atol=1e-14,dense_output=True,max_step=1e-3)
    yh1=sol.y[0,-1]
    G1=np.sqrt(M/m)*np.sqrt(yh1/(1+gamma-yh1))
    vexit=v0*np.sqrt(M/m)*np.sqrt(yh1)
    # stroke time T = (dmax/v0) * int d delta / sqrt(1+gamma delta - yh)
    dd=np.linspace(d0,1,20001); yy=sol.sol(dd)[0]
    T=(dmax/v0)*np.trapezoid(1/np.sqrt(1+gamma*dd-yy),dd)
    ymax=yh1*M*v0**2/(2*F)
    print(f"{name:9s} {gamma:.4f}  {Phi:.4f}  {yh1:.4f}  {1+gamma-nu**2:.4f}    {G1:6.1f} {Gt:5.1f}  {vexit:6.1f} {vx:6.1f}  {T*1000:5.1f}  {ymax:6.2f}")
print()
# closed form for gamma = 0 (ram/spring source)
th=np.arccos(nu); Phi0=th+nu*np.sqrt(1-nu**2)
print("gamma=0 closed form: Phi =",round(Phi0,4)," terminal G*/sqrt(M/m) = tan(theta_end) =",round(np.tan(th),4),
      " tau = T v0/dmax = 2 sqrt(1-nu^2)/Phi =",round(2*np.sqrt(1-nu**2)/Phi0,4), " eta =",1-nu**2)
# numeric check of closed form at gamma=0
def rhs0(delta,yh): return [Phi0*np.sqrt(max(yh[0],0)/(1-yh[0]))]
s0=solve_ivp(rhs0,[1e-6,1.0],[Phi0**2*1e-12/4],rtol=1e-10,atol=1e-14,dense_output=True,max_step=1e-3)
print("numeric yh(1) at gamma=0:",round(s0.y[0,-1],5)," expected 1-nu^2 =",1-nu**2)
dd=np.linspace(1e-6,1,20001); yy=s0.sol(dd)[0]
print("numeric tau:",round(np.trapezoid(1/np.sqrt(1-yy),dd),4))
# gravity-case numbers for the paper: Phi(gamma) and tau(gamma) at nu=0.4 for the frozen gamma
gamma=2*g*0.8541739/v0**2
print("gravity case (frozen 1,000:1): gamma=%.4f  Phi=%.4f  tau=T v0/dmax=%.4f  G*(1)/sqrt(M/m)=%.4f"%(gamma,2*3169.6*0.8541739/(np.sqrt(1000)*100),0.1*10/0.8541739,np.sqrt(1+gamma-0.16)/0.4))
