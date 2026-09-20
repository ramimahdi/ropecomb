import sys,json,os,time
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,_HERE); os.chdir(os.path.join(_HERE, ".."))
import numpy as np
from catabult_sim import simulate
from catabult_sim_elastic import simulate_elastic
from ropecomb_fit4 import fit_array
from gear_fn_bridge import make_gear_fn
M,F,vh_stop=1000.,3031.3,5.0; h=10.**2/(2*9.8)
OUT="_p5_1000.json"; res=json.load(open(OUT)) if os.path.exists(OUT) else {}
I=simulate(M,1.,h,get_gear_fn=F,min_heavy_speed=vh_stop,max_time=.6,max_target_acc=1e9,time_unit=1e-5)
dmax=I["summary"]["heavy_travel"]; dd=np.linspace(0,dmax,400)
Gs=np.interp(dd,I["heavy_dist"],I["gear"])
tms=np.interp(dd,I["heavy_dist"],I["t"])*1e3
kw=dict(max_heavy_dist=dmax,max_time=.5,max_target_acc=3e4,time_unit=1e-5)
for arg in sys.argv[1:]:
    N,k,a,ow=arg.split(","); N=int(N);k=float(k);a=float(a);ow=float(ow)
    if arg in res: continue
    t0=time.time()
    w=1.0/(1.0+tms**a) if a>0 else np.ones_like(tms); w=w/w.mean()
    b=None
    for sd in range(6):
        R,s,_,rms=fit_array(dd,Gs,N,k,dmax,seed=sd,R_bounds=(0.05,8.),width_budget=12.,
                            resid_weight=np.sqrt(w),overshoot_weight=ow)
        if b is None or rms<b[2]: b=(np.asarray(R),np.asarray(s),rms)
    R,s,rms=b
    g0=make_gear_fn(R,s,k)
    rt=simulate(M,1.,h,get_gear_fn=g0,**kw)["summary"]["heavy_travel"]
    et=simulate_elastic(M,1.,h,g0,k_rope=16471,max_heavy_dist=dmax,pretension=F,time_unit=2e-5)["summary"]["heavy_travel"]
    keep=s<max(rt,et); R2,s2=R[keep],s[keep]; gg=make_gear_fn(R2,s2,k)
    rr=simulate(M,1.,h,get_gear_fn=gg,**kw)
    ee=simulate_elastic(M,1.,h,gg,k_rope=16471,max_heavy_dist=dmax,pretension=F,time_unit=2e-5)
    res[arg]=dict(N=N,k=k,alpha=a,ow=ow,built=int(len(R2)),width=float(2*R2.sum()),rms=float(rms),
       rigid=float(rr["summary"]["target_final_speed"]),
       exit=float(ee["summary"]["target_final_speed"]),
       peak=float(ee["force_target"].max()/F),
       rigid_peak=float(rr["force_target"].max()/rr["summary"]["mean_force_target"]),
       vh_end=float(ee["summary"]["heavy_final_speed"]),
       spans=[round(2*x*100) for _,x in sorted(zip(s2,R2))],R=R2.tolist(),s=s2.tolist())
    json.dump(res,open(OUT,"w"),indent=1)
    v=res[arg]
    print("N%d k%g a=%-5g ow%-6g | built %d | w %5.2f | rigid %5.1f | exit %5.1f (%3.0f%%) | peak %5.2fx | rigidP/M %5.1fx (%.0fs)"
      %(N,k,a,ow,v["built"],v["width"],v["rigid"],v["exit"],100*v["exit"]/I["summary"]["target_final_speed"],
        v["peak"],v["rigid_peak"],time.time()-t0),flush=True)
print("ideal exit %.1f  d_max %.4f  final ratio %.1f"%(I["summary"]["target_final_speed"],dmax,I["summary"]["final_gear"]))
