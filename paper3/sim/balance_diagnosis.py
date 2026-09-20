"""Why the compliant peak-to-mean force rises along the balance trade (Section 6.7 / SI S15).

Two diagnostics on every design of the balance sweeps (balanced_study*.json, balanced_relative.json):
  * the minimum tension in the compliant trace, min F / F_design, the fraction of the stroke spent slack
    and the position of the first slack event;
  * the local slope ratio  G_net'(d) / G*'(d)  averaged over 3 cm windows, and its minimum over
    15..97 % of the stroke ("flat").
And a control: the ideal profile G* perturbed by smooth 1 % errors of several shapes, simulated with the
same compliant integrator, to show that smooth tracking error alone does not produce the peaks.
Writes balance_diagnosis.json.
"""
import os
import json, sys, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
sys.path.insert(0, os.path.join(_HERE, "code")); sys.path.insert(0, os.path.join(_HERE, "paper2"))
from semisym_fit import ideal_target, falls, bank_ratio
from friction_sim import make_config, simulate_compliant, evaluate

M, m, v0, F, VSTOP, KR = 1000., 1., 10., 3169.6, 4.0, 16471
h = v0**2 / 19.6
dd, Gs, dmax = ideal_target(M, m, v0, F, VSTOP, n=2000, code_dir="code")
dGs = np.gradient(Gs, dd)

def smooth(x, w):
    return np.convolve(x, np.ones(w) / w, mode="same")

def diagnose(Rl, sl, Rh, sh, k):
    nlo, nhi = falls(k)
    net = nlo * bank_ratio(dd, Rl, sl) + nhi * bank_ratio(dd, Rh, sh)
    w = int(0.03 / (dd[1] - dd[0]))
    ratio = smooth(np.gradient(net, dd), w) / np.maximum(smooth(dGs, w), 1e-9)
    sel = (dd > 0.15 * dmax) & (dd < 0.97 * dmax)
    i = int(np.argmin(np.where(sel, ratio, 9.0)))
    gear, react, rhos, n, taus = make_config([(Rl, sl), (Rh, sh)], k, 1.0, True)
    ee = simulate_compliant(M, m, h, gear, react, KR, F, max_heavy_dist=dmax, time_unit=2e-5)
    n995 = int(0.995 * len(ee["F"])); Fc = ee["F"][:n995]; dh = ee["d"][:n995]
    slack = Fc <= 0.0
    return dict(minF=float(Fc.min() / F), dip_at=float(dh[int(np.argmin(Fc))] / dmax),
                slack_frac=float(slack.mean()), first_slack_at=(float(dh[int(np.argmax(slack))] / dmax) if slack.any() else None),
                min_slope_ratio=float(ratio[i]), flat_at=float(dd[i] / dmax), terminal_pct=100 * float(net[-1] / Gs[-1]))

out = {"_meta": dict(M=M, m=m, v0=v0, F=F, v_stop=VSTOP, k_rope=KR, d_max=dmax,
                     note="min tension, slack and local slope deficit of every balance-sweep design; smooth-error control on G*")}
rows = []
for fn, K in (("balanced_study.json", None), ("balanced_relative.json", None), ("balanced_study2.json", 7), ("balanced_study3.json", 7)):
    d = json.load(open(fn))
    for tag in [t for t in d if not t.startswith("_")]:
        k = K or int(tag[1])
        for lam, rec in d[tag].items():
            if "R_lo" not in rec: continue
            Rl, sl, Rh, sh = [np.array(rec[x]) for x in ("R_lo", "s_lo", "R_hi", "s_hi")]
            r = diagnose(Rl, sl, Rh, sh, k)
            r.update(file=fn, family=tag, lam=float(lam), k=k, N=len(Rl), lateral=rec["lateral"], pC=rec["pC"], pR=rec["pR"])
            rows.append(r)
            print(f"{fn[:14]:14} {tag:9} lam={float(lam):<5} lat {100*rec['lateral']:5.2f}%  pC {rec['pC']:.2f}  minF/F {r['minF']:.2f}  slack {100*r['slack_frac']:4.1f}%  min slope ratio {r['min_slope_ratio']:.2f} at {r['flat_at']:.2f}  terminal {r['terminal_pct']:.0f}%")
out["designs"] = rows
a = np.array([(r["pC"], r["minF"], r["min_slope_ratio"]) for r in rows])
out["correlations"] = dict(pC_minF=float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]),
                           pC_min_slope=float(np.corrcoef(a[:, 0], a[:, 2])[0, 1]),
                           minF_min_slope=float(np.corrcoef(a[:, 1], a[:, 2])[0, 1]))
print("correlations:", out["correlations"])

# ---- control: smooth errors on the ideal profile
shapes = {"ideal": lambda x: 0 * x, "-1% uniform": lambda x: -np.ones_like(x), "+1% uniform": lambda x: np.ones_like(x),
          "1% half-sine bump": lambda x: np.sin(np.pi * x), "1% full sine": lambda x: np.sin(2 * np.pi * x),
          "1% late ramp": lambda x: -np.clip((x - 0.75) / 0.25, 0, 1), "2% full sine": lambda x: 2 * np.sin(2 * np.pi * x)}
ctrl = {}
for name, sh in shapes.items():
    g = lambda d, sh=sh: float(np.interp(d, dd, Gs * (1 + 0.01 * sh(d / dmax))))
    e = evaluate(g, g, M, m, h, F, dmax, KR)
    ee = simulate_compliant(M, m, h, g, g, KR, F, max_heavy_dist=dmax, time_unit=2e-5)
    ctrl[name] = dict(pC=e["pC"], pR=e["pR"], minF=float(ee["F"].min() / F), vC=e["vC"])
    print(f"control {name:18} pC {e['pC']:.3f}  pR {e['pR']:.3f}  min F/F {ctrl[name]['minF']:.2f}")
out["smooth_error_control"] = ctrl
out["target_slope"] = {f"{f:.2f}": float(np.interp(f * dmax, dd, dGs)) for f in (0.5, 0.75, 0.9, 1.0)}
json.dump(out, open("balance_diagnosis.json", "w"), indent=1)
