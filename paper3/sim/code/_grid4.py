"""Grid search, wider version: 50 restarts per (N,k) cell, top 5 kept for simulation.

Identical to _grid3.py in physics and in the fit call, so the seed 0..29 fits already
cached in _fits_<tag>.json are reused verbatim; only seeds 30..49 are new.

Scoring follows the paper (§4.7):
    admissible : p2m_compliant <= 1.3  AND  v_compliant >= 0.99 * v_ideal
    score3     = (55 - N_eff - k - W)/50 * v_rigid * v_compliant / sqrt(p2m_r + p2m_c)
score and score2 (the earlier 52-based forms) are retained for comparison only.
"""
import sys, os, json, time, math
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE); os.chdir(os.path.join(_HERE, ".."))
import numpy as np
from catabult_sim import simulate
from catabult_sim_elastic import simulate_elastic
from ropecomb_fit4 import fit_array
from gear_fn_bridge import make_gear_fn

CASE = {"1000": (1000., 3169.6, 4.0), "100": (100., 999.0, 4.0), "10000": (10000., 10033.8, 4.0)}
tag = sys.argv[1]
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 32.0
M, F, STOP = CASE[tag]
h = 10. ** 2 / (2 * 9.8)

NS = list(range(2, 17)); KS = [1., 2., 3., 5., 7., 9.]
NSEED = 50; TOP = 5

FITS = "_fits50b_%s.json" % tag
OUT = "_grid4_%s.json" % tag
if not os.path.exists(FITS) and os.path.exists("_fits_%s.json" % tag):
    json.dump(json.load(open("_fits_%s.json" % tag)), open(FITS + ".tmp", "w")); os.replace(FITS + ".tmp", FITS)   # seed the cache
fits = json.load(open(FITS)) if os.path.exists(FITS) else {}
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

I = simulate(M, 1., h, get_gear_fn=F, min_heavy_speed=STOP, max_time=.6,
             max_target_acc=1e9, time_unit=1e-5)
dmax = I["summary"]["heavy_travel"]; ideal = I["summary"]["target_final_speed"]
dd = np.linspace(0, dmax, 400); Gs = np.interp(dd, I["heavy_dist"], I["gear"])
kw = dict(max_heavy_dist=dmax, max_time=.5, max_target_acc=3e4, time_unit=1e-5)


def pm995(Fv):
    Fv = np.asarray(Fv); c = max(2, int(0.995 * len(Fv)))
    return float(Fv[:c].max() / Fv[:c].mean())


t0 = time.time(); nf = 0


def _atomic(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh); fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)


def flush():
    _atomic(fits, FITS); _atomic(res, OUT)


for N in NS:
    for k in KS:
        cell = "%d,%g" % (N, k)
        if cell in res:
            continue
        for sd in range(NSEED):
            fk = "%s,%d" % (cell, sd)
            if fk in fits:
                continue
            if time.time() - t0 > BUDGET:
                flush(); print("budget; %d fits this run; cells %d/90" % (nf, len(res))); sys.exit(0)
            try:
                R, s, _, rms = fit_array(dd, Gs, N, k, dmax, seed=sd,
                                         R_bounds=(0.05, 8.0), width_budget=20.0)
                fits[fk] = [float(rms), [round(float(x), 6) for x in R],
                            [round(float(x), 6) for x in s]]
            except Exception:
                fits[fk] = None
            nf += 1
            if nf % 6 == 0:
                flush()

        got = [fits["%s,%d" % (cell, sd)] for sd in range(NSEED)]
        got = [g for g in got if g]
        if not got:
            continue
        got.sort(key=lambda g: g[0]); cands = []
        for rms, Rl, sl in got[:TOP]:
            R = np.array(Rl); s = np.array(sl); g0 = make_gear_fn(R, s, k)
            rt = simulate(M, 1., h, get_gear_fn=g0, **kw)["summary"]["heavy_travel"]
            et = simulate_elastic(M, 1., h, g0, k_rope=16471, max_heavy_dist=dmax,
                                  pretension=F, time_unit=2e-5)["summary"]["heavy_travel"]
            keep = s < max(rt, et); Rp, sp = R[keep], s[keep]
            if len(Rp) < 2:
                continue
            gp = make_gear_fn(Rp, sp, k)
            rr = simulate(M, 1., h, get_gear_fn=gp, **kw)
            ee = simulate_elastic(M, 1., h, gp, k_rope=16471, max_heavy_dist=dmax,
                                  pretension=F, time_unit=2e-5)
            ne = int(len(Rp)); W = float(2 * Rp.sum())
            exr = float(rr["summary"]["target_final_speed"])
            exe = float(ee["summary"]["target_final_speed"])
            pmr = pm995(rr["force_target"]); pme = pm995(ee["force_target"])
            cands.append(dict(
                N_fit=N, N_eff=ne, k=k, pruned=int(len(R) - ne), rms=rms, width=W,
                exit_rigid=exr, exit_elastic=exe, pm_rigid=pmr, pm_elastic=pme,
                peakF_elastic=float(np.asarray(ee["force_target"]).max() / F),
                admissible=bool(pme <= 1.3 and exe >= 0.99 * ideal),
                score=((52.0 - ne - k) / 50.0) * exr / math.sqrt(pmr),
                score2=((52.0 - ne - k) / 50.0) * (exr * exe) / math.sqrt(pmr + pme),
                score3=((55.0 - ne - k - W) / 50.0) * (exr * exe) / math.sqrt(pmr + pme),
                R=[round(x, 6) for x in Rp], s=[round(x, 6) for x in sp]))
        if cands:
            res[cell] = dict(N=N, k=k, ideal=ideal, cands=cands); flush()
            print("  %-6s Neff=%s (%.0fs)" % (cell, [c["N_eff"] for c in cands],
                                              time.time() - t0), flush=True)
flush(); print("COMPLETE - cells %d/90" % len(res))
