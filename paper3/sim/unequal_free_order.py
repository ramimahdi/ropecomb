"""<4,5,9>, merged k/2 strategy with the engagement order enforced (interleave=True) or free (interleave=False).

Seed in both cases: one array of 9 members fitted at k/2, dealt lo, hi, ..., lo (L first and last). Starts: the plain
dealt seed, 30 jittered copies of it (sigma 0.3, rng 1) and, for the free solve, the constrained optimum itself (does
the free solver leave it?). Best by objective, basins clustered (any coordinate > 1 cm apart), each basin's engagement
order written L/R with ties in brackets, best-of-basin simulated. Writes unequal_free_order.json."""
import os, sys, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
sys.path.insert(0, os.path.join(_HERE, "code")); sys.path.insert(0, os.path.join(_HERE, "paper2"))
import unequal_members as U
from unequal_merged_seed import deal
from dual_fit_free import staged_solve, order_string, jitter
from unequal_min9_topk import metrics, k, W, N_lo, N_hi
dd, Gs, dmax = U.dd, U.Gs, U.dmax
NJ = 30

R, s, _, _ = U.fit_array(dd, Gs, N_lo + N_hi, k / 2, dmax, seed=0, width_budget=3 * W, R_bounds=(0.05, 8.0), init_span_range=(0.4, 1.6))
seed0 = deal(R, s, N_lo, N_hi)[:4]
print("seed order:", order_string(seed0[1], seed0[3]))


def run(lam, interleave, extra_start=None):
    rng = np.random.default_rng(1); res = []
    starts = [("plain", seed0)] + [(f"jit{j}", jitter(rng, *seed0, dmax)) for j in range(1, NJ + 1)]
    if extra_start is not None: starts.append(("from constrained", extra_start))
    for name, st in starts:
        try:
            (a, b, c, d_), cost = staged_solve(*st, k, W, lam, dd, Gs, dmax, interleave=interleave)
        except Exception as e:
            print("  start", name, "failed:", e); continue
        res.append(dict(start=name, cost=float(cost), order=order_string(b, d_), R_lo=list(map(float, a)), s_lo=list(map(float, b)), R_hi=list(map(float, c)), s_hi=list(map(float, d_))))
    res.sort(key=lambda r: r["cost"])
    clusters = []
    for r in res:
        v = np.concatenate([r["R_lo"], r["s_lo"], r["R_hi"], r["s_hi"]])
        for c in clusters:
            if np.max(np.abs(v - c["v"])) < 0.01: c["n"] += 1; break
        else: clusters.append(dict(v=v, n=1, rep=r))
    return res, clusters


out = {"_meta": dict(design="<4,5,9>", W=W, starts="plain dealt seed + 30 jittered (sigma 0.3) [+ constrained optimum for the free solve]")}
for lam in (0.03, 0.02, 0.01):
    t0 = time.time()
    resC, clC = run(lam, True)
    bestC = resC[0]; extra = tuple(np.array(bestC[x]) for x in ("R_lo", "s_lo", "R_hi", "s_hi"))
    resF, clF = run(lam, False, extra_start=extra)
    print(f"\nlambda = {lam}  ({time.time()-t0:.0f} s)")
    rec = {}
    for tag, res, cl in (("interleave=True", resC, clC), ("interleave=False", resF, clF)):
        print(f"  {tag}: best objective {res[0]['cost']:.5f} from start '{res[0]['start']}', order {res[0]['order']}; {len(cl)} basins (sizes {[c['n'] for c in cl][:8]})")
        if tag.endswith("False"):
            fc = [r for r in res if r["start"] == "from constrained"]
            if fc: print(f"    free solve started at the constrained optimum: objective {fc[0]['cost']:.5f} (constrained {resC[0]['cost']:.5f}), order {fc[0]['order']}")
        print(f"  {'basin':>5} {'size':>4} {'objective':>9} {'gap%':>5} {'order':<14} {'rms%':>5} {'lat max%':>8} {'lat mean%':>9} {'lat end%':>8} {'pC':>6} {'pR':>5} {'minF':>5} {'vC':>6} {'w lo/hi':>10}")
        reps = []
        for i, c in enumerate(cl[:5]):
            r = dict(c["rep"]); r.update(metrics(np.array(r["R_lo"]), np.array(r["s_lo"]), np.array(r["R_hi"]), np.array(r["s_hi"])))
            r["gap_pct"] = 100 * (r["cost"] / res[0]["cost"] - 1); r["basin"] = i + 1; r["basin_size"] = c["n"]; reps.append(r)
            print(f"  {i+1:>5} {c['n']:>4} {r['cost']:>9.5f} {r['gap_pct']:>5.2f} {r['order']:<14} {r['rms_pct']:>5.2f} {r['lat_max']:>8.2f} {r['lat_mean']:>9.2f} {r['lat_end']:>8.1f} "
                  f"{r['pC']:>6.3f} {r['pR']:>5.2f} {r['minF']:>5.2f} {r['vC']:>6.1f} {r['width_lo']:>4.2f}/{r['width_hi']:<4.2f}", flush=True)
        rec[tag] = dict(best=res[0], basins=reps, costs_all=[r["cost"] for r in res], orders_all=[r["order"] for r in res])
    out[str(lam)] = rec
    json.dump(out, open("unequal_free_order.json", "w"), indent=1)
print("wrote unequal_free_order.json")
