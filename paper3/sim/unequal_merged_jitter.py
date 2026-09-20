"""Merged k/2 with real restarts. The 30 single-array fits at k/2 converge to one design (3 mm spread), so the
earlier 30 'restarts' were one start. Here the dealt seed is perturbed before each staged solve (log-normal widths and
offset gaps, sigma given), the best solution is kept by the joint objective and the solutions are clustered into basins
(any coordinate more than 1 cm apart). The objective-best member of each basin is simulated. <4,5,9>, W = 1.96 m.
Writes unequal_merged_jitter.json."""
import os, sys, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
sys.path.insert(0, os.path.join(_HERE, "code")); sys.path.insert(0, os.path.join(_HERE, "paper2"))
import unequal_members as U
from unequal_merged_seed import deal, staged
from semisym_fit import falls, bank_ratio, interleaved_slots
from unequal_min9_topk import metrics, k, W, N_lo, N_hi, n_lo, n_hi
dd, Gs, dmax = U.dd, U.Gs, U.dmax
SIG, NJ = 0.3, 30
order = interleaved_slots(N_lo, N_hi)

R, s, _, _ = U.fit_array(dd, Gs, N_lo + N_hi, k / 2, dmax, seed=0, width_budget=3 * W, R_bounds=(0.05, 8.0), init_span_range=(0.4, 1.6))
seed0 = deal(R, s, N_lo, N_hi)[:4]


def jitter(rng, R_l, s_l, R_h, s_h):
    def one(Rx, sx):
        o = np.argsort(sx); Rx = np.asarray(Rx)[o]; sx = np.asarray(sx)[o]
        gaps = np.diff(np.concatenate([[0.0], sx])); gaps = gaps * np.exp(rng.normal(0, SIG, len(gaps)))
        s_new = np.cumsum(gaps); s_new = s_new * min(1.0, 0.98 * dmax / max(s_new[-1], 1e-9))
        return np.maximum(0.05, Rx * np.exp(rng.normal(0, SIG, len(Rx)))), s_new
    (a, b), (c, d_) = one(R_l, s_l), one(R_h, s_h)
    return a, b, c, d_


out = {"_meta": dict(design="<4,5,9>", sigma=SIG, n_jitter=NJ, note="start 0 is the unperturbed dealt seed")}
for lam in (0.03, 0.02, 0.01):
    rng = np.random.default_rng(1); t0 = time.time(); res = []
    for j in range(NJ + 1):
        st = seed0 if j == 0 else jitter(rng, *seed0)
        try:
            (a, b, c, d_), cost = staged(*st, order, k, W, lam)
        except Exception as e:
            print("start", j, "failed:", e); continue
        res.append(dict(start=j, cost=float(cost), R_lo=list(map(float, a)), s_lo=list(map(float, b)), R_hi=list(map(float, c)), s_hi=list(map(float, d_))))
    res.sort(key=lambda r: r["cost"]); best = res[0]["cost"]
    clusters = []
    for r in res:
        v = np.concatenate([r["R_lo"], r["s_lo"], r["R_hi"], r["s_hi"]])
        for c in clusters:
            if np.max(np.abs(v - c["v"])) < 0.01: c["n"] += 1; break
        else: clusters.append(dict(v=v, n=1, rep=r))
    print(f"\nlambda = {lam}: {len(res)} starts in {time.time()-t0:.0f} s, {len(clusters)} basins (sizes {[c['n'] for c in clusters]}); unperturbed start ranks "
          f"{[i for i, r in enumerate(res) if r['start'] == 0][0] + 1} with objective {[r['cost'] for r in res if r['start'] == 0][0]:.5f}", flush=True)
    print(f"{'basin':>5} {'size':>4} {'objective':>9} {'gap%':>5} {'rms%':>5} {'lat max%':>8} {'lat mean%':>9} {'lat end%':>8} {'pC':>6} {'pR':>5} {'minF':>5} {'vC':>6} {'w lo/hi':>10} {'J x1e5':>7}")
    reps = []
    for i, c in enumerate(clusters[:6]):
        r = c["rep"]; r.update(metrics(np.array(r["R_lo"]), np.array(r["s_lo"]), np.array(r["R_hi"]), np.array(r["s_hi"])))
        r["gap_pct"] = 100 * (r["cost"] / best - 1); r["basin"] = i + 1; r["basin_size"] = c["n"]; reps.append(r)
        print(f"{i+1:>5} {c['n']:>4} {r['cost']:>9.5f} {r['gap_pct']:>5.2f} {r['rms_pct']:>5.2f} {r['lat_max']:>8.2f} {r['lat_mean']:>9.2f} {r['lat_end']:>8.1f} "
              f"{r['pC']:>6.3f} {r['pR']:>5.2f} {r['minF']:>5.2f} {r['vC']:>6.1f} {r['width_lo']:>4.2f}/{r['width_hi']:<4.2f} {1e5*r['J']:>7.3f}", flush=True)
    out[str(lam)] = dict(basins=reps, costs_all=[r["cost"] for r in res], starts_all=[r["start"] for r in res])
    json.dump(out, open("unequal_merged_jitter.json", "w"), indent=1)
print("wrote unequal_merged_jitter.json")
