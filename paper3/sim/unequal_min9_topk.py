"""<4,5,9>, merged k/2 strategy: does simulating more than the objective-best restart pay?

30 restarts of the merged k/2 seed (one array of 9 members fitted at k/2, width cap 3W, init_span_range (0.4, 1.6)),
dealt lo,hi,...,lo and solved by the staged solve at each lambda. The 30 solutions are ranked by the joint objective
(fit + lambda x balance + width penalty). The top five are simulated rigid and compliant and scored by

    J = pC * mean(lateral^2) / sqrt(vC)

with pC the compliant peak/mean, lateral the geometric lateral load as a fraction of the peak reaction (mean of its
square over the stroke) and vC the compliant exit speed. The cumulative table then reports, for j = 1..5, the best
of the first j simulated solutions by J, so row j shows what simulating j restarts instead of one buys.
Writes unequal_min9_topk.json.
"""
import os, sys, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
sys.path.insert(0, os.path.join(_HERE, "code")); sys.path.insert(0, os.path.join(_HERE, "paper2"))
import unequal_members as U
from unequal_merged_seed import deal, staged
from semisym_fit import falls, bank_ratio
from friction_sim import make_config, simulate_compliant
import types
_src = open("friction_sim.py").read().replace("return dict(stop=stop, t=np.array(t_hist), F=np.array(F_hist), slack=np.array(slack_hist),",
                                             "return dict(stop=stop, t=np.array(t_hist), F=np.array(F_hist), slack=np.array(slack_hist), hs=np.array(hs_hist),")
_mod = types.ModuleType("friction_sim_hs"); _mod.__file__ = os.path.abspath("friction_sim.py"); exec(compile(_src, "friction_sim.py", "exec"), _mod.__dict__)
simulate_rigid = _mod.simulate_rigid
dd, Gs, dmax = U.dd, U.Gs, U.dmax
k, W, N_lo, N_hi = 9, 1.96, 5, 4
n_lo, n_hi = falls(k)
TOP, NSTART = 5, 30


def metrics(R_lo, s_lo, R_hi, s_hi):
    g1 = bank_ratio(dd, R_lo, s_lo); g2 = bank_ratio(dd, R_hi, s_hi); net = n_lo * g1 + n_hi * g2
    imb = np.abs(n_lo * g1 - n_hi * g2) / net.max()
    rms = 100 * float(np.sqrt(np.mean((net - Gs) ** 2)) / Gs[-1])
    gear, react, rhos, nn, taus = make_config([(R_lo, s_lo), (R_hi, s_hi)], k, 1.0, True)
    ee = simulate_compliant(U.M, U.m, U.h, gear, react, U.KR, U.F, max_heavy_dist=dmax, time_unit=2e-5)
    rr = simulate_rigid(U.M, U.m, U.h, gear, react, max_heavy_dist=dmax, max_time=0.5, max_target_acc=3e4, time_unit=1e-5)
    Fc = ee["F"][:int(0.995 * len(ee["F"]))]; Fr = rr["F"][:int(0.995 * len(rr["F"]))]
    pC = float(Fc.max() / U.F); pR = float(Fr.max() / Fr.mean()); vC = float(ee["v_exit"]); vR = float(rr["v_exit"])
    return dict(rms_pct=rms, lat_max=100 * float(imb.max()), lat_mean=100 * float(imb.mean()), lat_msq=float(np.mean(imb ** 2)),
                lat_end=100 * float(imb[-1]), pC=pC, pR=pR, vC=vC, vR=vR, minF=float(Fc.min() / U.F),
                J=pC * float(np.mean(imb ** 2)) / np.sqrt(vC), width_lo=float(2 * np.sum(R_lo)), width_hi=float(2 * np.sum(R_hi)))


def main():
    out = {"_meta": dict(design="<4,5,9>: N_lo=5 on the four-fall (leading) array, N_hi=4 on the five-fall array", k=k, W=W, restarts=NSTART, top=TOP,
                         J="pC * mean(lateral_fraction^2) / sqrt(vC)")}
    for lam in (0.03, 0.02, 0.01):
        t0 = time.time(); res = []
        for sd in range(NSTART):
            R, s, _, rms = U.fit_array(dd, Gs, N_lo + N_hi, k / 2, dmax, seed=sd, width_budget=3 * W, R_bounds=(0.05, 8.0), init_span_range=(0.4, 1.6))
            (a, b, c, d_), cost = staged(*deal(R, s, N_lo, N_hi), k, W, lam)
            res.append(dict(seed=sd, cost=float(cost), R_lo=list(map(float, a)), s_lo=list(map(float, b)), R_hi=list(map(float, c)), s_hi=list(map(float, d_))))
        res.sort(key=lambda r: r["cost"]); best_cost = res[0]["cost"]
        print(f"\nlambda = {lam}: {NSTART} restarts solved in {time.time()-t0:.0f} s; objective of rank 1..{TOP}: "
              + ", ".join(f"{r['cost']:.4f}" for r in res[:TOP]) + f"; rank 6..10: " + ", ".join(f"{r['cost']:.4f}" for r in res[TOP:10]), flush=True)
        top = res[:TOP]
        print(f"{'rank':>4} {'seed':>4} {'objective':>9} {'gap%':>5} {'rms%':>5} {'lat max%':>8} {'lat mean%':>9} {'lat end%':>8} {'pC':>6} {'pR':>5} {'minF':>5} {'vC':>6} {'vR':>6} {'w lo/hi':>10} {'J x1e5':>7}")
        for i, r in enumerate(top):
            r.update(metrics(np.array(r["R_lo"]), np.array(r["s_lo"]), np.array(r["R_hi"]), np.array(r["s_hi"]))); r["rank"] = i + 1
            r["gap_pct"] = 100 * (r["cost"] / best_cost - 1)
            print(f"{i+1:>4} {r['seed']:>4} {r['cost']:>9.4f} {r['gap_pct']:>5.1f} {r['rms_pct']:>5.2f} {r['lat_max']:>8.2f} {r['lat_mean']:>9.2f} {r['lat_end']:>8.1f} "
                  f"{r['pC']:>6.3f} {r['pR']:>5.2f} {r['minF']:>5.2f} {r['vC']:>6.1f} {r['vR']:>6.1f} {r['width_lo']:>4.2f}/{r['width_hi']:<4.2f} {1e5*r['J']:>7.3f}", flush=True)
        cum = []
        print(f"\ncumulative: best of the first j simulated by J")
        print(f"{'j':>2} {'pick':>4} {'seed':>4} {'objective':>9} {'lat max%':>8} {'lat mean%':>9} {'pC':>6} {'pR':>5} {'vC':>6} {'J x1e5':>7}")
        for j in range(1, TOP + 1):
            b = min(top[:j], key=lambda r: r["J"])
            cum.append(dict(j=j, pick=b["rank"], seed=b["seed"], cost=b["cost"], lat_max=b["lat_max"], lat_mean=b["lat_mean"], pC=b["pC"], pR=b["pR"], vC=b["vC"], J=b["J"]))
            print(f"{j:>2} {b['rank']:>4} {b['seed']:>4} {b['cost']:>9.4f} {b['lat_max']:>8.2f} {b['lat_mean']:>9.2f} {b['pC']:>6.3f} {b['pR']:>5.2f} {b['vC']:>6.1f} {1e5*b['J']:>7.3f}", flush=True)
        # the objective-ranked top five are usually one basin; group the 30 solutions into distinct designs (any coordinate
        # differing by more than 1 cm) and simulate the objective-best member of each distinct basin instead
        clusters = []
        for r in res:
            v = np.concatenate([r["R_lo"], r["s_lo"], r["R_hi"], r["s_hi"]])
            for c in clusters:
                if np.max(np.abs(v - c["v"])) < 0.01: c["n"] += 1; break
            else:
                clusters.append(dict(v=v, n=1, rep=r))
        print(f"\ndistinct basins among the {NSTART} restarts: {len(clusters)}  (sizes {[c['n'] for c in clusters]})")
        print(f"{'basin':>5} {'size':>4} {'seed':>4} {'objective':>9} {'gap%':>5} {'rms%':>5} {'lat max%':>8} {'lat mean%':>9} {'lat end%':>8} {'pC':>6} {'pR':>5} {'minF':>5} {'vC':>6} {'vR':>6} {'w lo/hi':>10} {'J x1e5':>7}")
        reps = []
        for i, c in enumerate(clusters[:TOP]):
            r = c["rep"]
            if "J" not in r: r.update(metrics(np.array(r["R_lo"]), np.array(r["s_lo"]), np.array(r["R_hi"]), np.array(r["s_hi"])))
            r["gap_pct"] = 100 * (r["cost"] / best_cost - 1); r["basin"] = i + 1; r["basin_size"] = c["n"]; reps.append(r)
            print(f"{i+1:>5} {c['n']:>4} {r['seed']:>4} {r['cost']:>9.4f} {r['gap_pct']:>5.2f} {r['rms_pct']:>5.2f} {r['lat_max']:>8.2f} {r['lat_mean']:>9.2f} {r['lat_end']:>8.1f} "
                  f"{r['pC']:>6.3f} {r['pR']:>5.2f} {r['minF']:>5.2f} {r['vC']:>6.1f} {r['vR']:>6.1f} {r['width_lo']:>4.2f}/{r['width_hi']:<4.2f} {1e5*r['J']:>7.3f}", flush=True)
        cum_b = []
        print(f"cumulative over basins: best of the first j basins by J")
        for j in range(1, len(reps) + 1):
            b = min(reps[:j], key=lambda r: r["J"])
            cum_b.append(dict(j=j, pick=b["basin"], seed=b["seed"], cost=b["cost"], lat_max=b["lat_max"], lat_mean=b["lat_mean"], pC=b["pC"], pR=b["pR"], vC=b["vC"], J=b["J"]))
            print(f"{j:>2} basin {b['basin']} (seed {b['seed']}) objective {b['cost']:.4f}  lat max {b['lat_max']:.2f}%  mean {b['lat_mean']:.2f}%  pC {b['pC']:.3f}  pR {b['pR']:.2f}  vC {b['vC']:.1f}  J {1e5*b['J']:.3f}", flush=True)
        out[str(lam)] = dict(top=top, cumulative=cum, basins=[{k_: v_ for k_, v_ in r.items()} for r in reps], cumulative_basins=cum_b, costs_all=[r["cost"] for r in res])
        json.dump(out, open("unequal_min9_topk.json", "w"), indent=1)
    print("wrote unequal_min9_topk.json")


if __name__ == "__main__":
    main()
