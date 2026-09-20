"""<4,5,9> (N_1 = 5 on the leading four-fall array, N_2 = 4 on the five-fall array, k = 9): four solver strategies
compared in charts at lambda = 0.03 and 0.01, against the frozen <5,5,9> design.

  refit      single array of N_1 = 5 members at k (best of 6 starts by rms), array 2 = its first 4 members at the
             midpoints, one staged solve                                            (unequal_members.py)
  merged k/2 single array of 9 members at k/2 (30 starts), dealt lo,hi,...,lo, staged solve from each, best by
             the joint least-squares objective                                       (unequal_merged_seed.py)
  merged k   the same at the full ratio k                                            (unequal_merged_k.py)
  unseeded   the joint LM fit from 30 random starts, no seed, best by fit rms       (semisym_fit.fit_semisym)

Selection inside every strategy is geometric: the joint objective (fit + lambda x balance + width penalty) or, for
the unseeded fit, the fit rms. No dynamic quantity enters the selection. Writes figs/unequal_min9_compare.png and
unequal_min9_charts.json."""
import os, sys, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
sys.path.insert(0, os.path.join(_HERE, "code")); sys.path.insert(0, os.path.join(_HERE, "paper2")); sys.path.insert(0, os.path.join(HERE, "..", "figsrc"))
import figstyle as S; S.use()
import unequal_members as U
from unequal_merged_seed import deal, staged
from semisym_fit import falls, bank_ratio, fit_semisym
from friction_sim import make_config, simulate_compliant
dd, Gs, dmax = U.dd, U.Gs, U.dmax
k, W, N_lo, N_hi = 9, 1.96, 5, 4
n_lo, n_hi = falls(k)
DZ = json.load(open("paper2/designs_16471.json"))["k9"]["dual"]

def curves(R_lo, s_lo, R_hi, s_hi):
    g1 = bank_ratio(dd, R_lo, s_lo); g2 = bank_ratio(dd, R_hi, s_hi); net = n_lo * g1 + n_hi * g2
    imb = 100 * np.abs(n_lo * g1 - n_hi * g2) / net.max()
    gear, react, rhos, nn, taus = make_config([(R_lo, s_lo), (R_hi, s_hi)], k, 1.0, True)
    ee = simulate_compliant(U.M, U.m, U.h, gear, react, U.KR, U.F, max_heavy_dist=dmax, time_unit=2e-5)
    n995 = int(0.995 * len(ee["F"]))
    return imb, ee["d"][:n995], ee["F"][:n995] / 1e3, float(ee["F"][:n995].max() / U.F)

designs = {}
for lam in (0.03, 0.01):
    # refit
    best = None
    for sd in range(6):
        R_, s_, _, r_ = U.fit_array(dd, Gs, N_lo, k, dmax, seed=sd, width_budget=W, R_bounds=(0.05, 8.0))
        if best is None or r_ < best[2]: best = (R_, s_, r_)
    R_lo, s_lo, R_hi, s_hi, _ = U.fit_staged_unequal(np.array(best[0]), np.array(best[1]), N_hi, k, W, lam)
    designs[("refit", lam)] = (R_lo, s_lo, R_hi, s_hi, None)
    # merged at k/2 and at k, 30 starts each, best by joint objective; keep the cloud of lateral curves
    for name, kk in (("merged k/2", k / 2), ("merged k", k)):
        res = []
        for sd in range(30):
            R, s, _, rms = U.fit_array(dd, Gs, N_lo + N_hi, kk, dmax, seed=sd, width_budget=3 * W, R_bounds=(0.05, 8.0), init_span_range=(0.4, 1.6))
            (a, b, c, d_), cost = staged(*deal(R, s, N_lo, N_hi), k, W, lam)
            res.append((cost, a, b, c, d_))
        res.sort(key=lambda x: x[0])
        cloud = [curves(*r[1:])[0] for r in res] if name == "merged k/2" else None
        designs[(name, lam)] = (*res[0][1:], cloud)
    # unseeded joint fit, 30 random starts, best by rms
    o = fit_semisym(dd, Gs, N_lo, N_hi, k, dmax, W, lam_balance=lam, lam_width=50.0, n_starts=30, seed=0)
    designs[("unseeded", lam)] = (o["R_lo"], o["s_lo"], o["R_hi"], o["s_hi"], None)
    print(f"lambda {lam}: designs solved", flush=True)

styles = {"refit": dict(color=S.BLUE, lw=1.3), "merged k/2": dict(color=S.RED, lw=1.3), "merged k": dict(color=S.ORANGE, lw=1.1, ls=(0, (4, 1.5))),
          "unseeded": dict(color=S.GREEN, lw=1.1, ls=(0, (1, 1.2)))}
ref_imb, ref_d, ref_F, ref_p = curves(*[np.array(DZ[x]) for x in ("R_lo", "s_lo", "R_hi", "s_hi")])
fig, axes = plt.subplots(2, 2, figsize=(S.W_FULL, 5.4), sharex=True)
summary = {}
for i, lam in enumerate((0.03, 0.01)):
    aF, aI = axes[i]
    aF.plot(ref_d, ref_F, color=S.GREY, lw=1.0, label=f"<5,5,9> frozen, peak/mean {ref_p:.2f}")
    aI.plot(dd, ref_imb, color=S.GREY, lw=1.0)
    for name in ("refit", "merged k/2", "merged k", "unseeded"):
        R_lo, s_lo, R_hi, s_hi, cloud = designs[(name, lam)]
        imb, d_, F_, p = curves(R_lo, s_lo, R_hi, s_hi)
        if cloud is not None:
            for c in cloud: aI.plot(dd, c, color=S.RED, lw=0.4, alpha=0.18)
        aF.plot(d_, F_, label=f"{name}, peak/mean {p:.2f}", **styles[name]); aI.plot(dd, imb, **styles[name])
        summary[f"{name}|{lam}"] = dict(pC=p, lateral_max=float(imb.max()), lateral_mean=float(imb.mean()), lateral_end=float(imb[-1]),
                                       rms_pct=100 * float(np.sqrt(np.mean((n_lo * bank_ratio(dd, R_lo, s_lo) + n_hi * bank_ratio(dd, R_hi, s_hi) - Gs) ** 2)) / Gs[-1]),
                                       width_lo=float(2 * np.sum(R_lo)), width_hi=float(2 * np.sum(R_hi)),
                                       R_lo=list(map(float, R_lo)), s_lo=list(map(float, s_lo)), R_hi=list(map(float, R_hi)), s_hi=list(map(float, s_hi)))
        print(f"lam={lam} {name:11} pC {p:.3f}  max lat {imb.max():5.2f}%  mean {imb.mean():4.2f}%  end {imb[-1]:4.1f}%  rms {summary[f'{name}|{lam}']['rms_pct']:.2f}%  w {2*np.sum(R_lo):.2f}/{2*np.sum(R_hi):.2f}", flush=True)
    aF.axhline(U.F / 1e3, color=S.BLACK, lw=0.8, ls=(0, (5, 2.5)))
    aF.set_ylabel("payload force, compliant (kN)"); aF.set_ylim(0, 7); aF.legend(loc="upper left", fontsize=6.2)
    aI.set_ylabel("lateral load (% of peak reaction)"); aI.set_ylim(0, 16)
    aI.axhline(100 * abs(n_lo - n_hi) / k, color=S.GREY, lw=0.8, ls=(0, (4, 2)))
    S.panel(aF, "ab"[i], f"payload force, $\\lambda = {lam}$"); S.panel(aI, "cd"[i], f"lateral load, $\\lambda = {lam}$")
for a in axes[1]: a.set_xlabel("carriage displacement $d$ (m)")
fig.suptitle("<4,5,9>: five members on the leading four-fall array, four on the five-fall array; 30 starts of the merged k/2 strategy shown faint", fontsize=7.5, y=0.995)
fig.tight_layout(h_pad=0.6, w_pad=1.2)
S.save(fig, "unequal_min9_compare")
json.dump(summary, open("unequal_min9_charts.json", "w"), indent=1); print("wrote unequal_min9_charts.json")
