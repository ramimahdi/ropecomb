"""
Which array should engage first: the one driving the lower fall count, or the higher?

For each (k, N, lambda) and each of the two assignments, run many restarts of the
staged solve and keep the distribution, not one fit. Restarts vary both the stage-1
single-array seed and a jitter on the structured starting point.

    python3 lead_order_study.py [restarts] [budget_seconds]

Checkpoints to lead_order_study.json after every cell, so it can be re-run to resume.
"""
import os, sys, json, time
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code")); sys.path.insert(0, HERE)
from ropecomb_fit4 import fit_array
from semisym_fit import (ideal_target, falls, bank_ratio, interleaved_slots,
                         seed_two_stage, encode, _residuals, _unpack)

RESTARTS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
BUDGET   = float(sys.argv[2]) if len(sys.argv) > 2 else 520.0
OUT = os.path.join(HERE, "lead_order_study.json")
M, m, v0, F, VSTOP, WIDTH = 1000., 1., 10., 3169.6, 4.0, 2.11

dd, Gs, dmax = ideal_target(M, m, v0, F, VSTOP, n=400,
                            code_dir=os.path.join(HERE, "..", "code"))
Gt = float(Gs[-1])
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
_seed_cache = {}


def stage1(N, k, sd):
    key = (N, k, sd)
    if key not in _seed_cache:
        R, s, _, rms = fit_array(dd, Gs, N, k, dmax, seed=sd, R_bounds=(0.05, 8.),
                                 width_budget=WIDTH, overshoot_weight=0.0)
        _seed_cache[key] = (np.asarray(R), np.asarray(s))
    return _seed_cache[key]


def one_fit(k, N, lam, order, sd, jitter):
    n_lo, n_hi = falls(k)
    Rst, sst = stage1(N, k, sd % 8)
    R_l, s_l, R_h, s_h = seed_two_stage(Rst, sst, dmax)
    p0 = encode(R_l, s_l, R_h, s_h, dmax, order)
    if jitter:
        p0 = p0 + np.random.default_rng(sd).normal(0.0, jitter, p0.shape)
    args = (dd, Gs, N, N, n_lo, n_hi, dmax, WIDTH, lam, 50.0, order)
    n = 2 * N; head = p0[:n + 1].copy()
    s1 = least_squares(lambda q: _residuals(np.concatenate([head, q]), *args),
                       p0[n + 1:], method="lm", max_nfev=2500)
    s2 = least_squares(_residuals, np.concatenate([head, s1.x]), method="lm",
                       max_nfev=3500, args=args)
    Rlo, slo, Rhi, shi = _unpack(s2.x, N, N, dmax, order)
    gl, gh = bank_ratio(dd, Rlo, slo), bank_ratio(dd, Rhi, shi)
    net = n_lo * gl + n_hi * gh
    return dict(cost=float(s2.cost),
                rms=float(np.sqrt(np.mean((net - Gs) ** 2))),
                imb=float(np.max(np.abs(n_lo*gl - n_hi*gh)) / max(net.max(), 1e-9)),
                terminal=float(net[-1]))


cells = [(k, N, lam, lab) for k, N in [(5, 10), (7, 7), (9, 5)]
                          for lam in (0.3, 1.0)
                          for lab in ("low_leads", "high_leads")]
t0 = time.time()
for k, N, lam, lab in cells:
    key = "%d,%d,%g,%s" % (k, N, lam, lab)
    if key in res: continue
    if time.time() - t0 > BUDGET:
        print("budget reached, %d/%d cells done" % (len(res), len(cells))); break
    order = interleaved_slots(N, N)
    if lab == "high_leads": order = 1 - order
    runs = []
    for sd in range(RESTARTS):
        try:
            runs.append(one_fit(k, N, lam, order, sd, 0.0 if sd == 0 else 0.25))
        except Exception:
            continue
    if not runs: continue
    runs.sort(key=lambda r: r["cost"])
    best = runs[0]
    res[key] = dict(k=k, N=N, lam=lam, order=lab, n=len(runs), best=best,
                    rms_med=float(np.median([r["rms"] for r in runs])),
                    imb_med=float(np.median([r["imb"] for r in runs])),
                    imb_min=float(min(r["imb"] for r in runs)),
                    imb_p25=float(np.percentile([r["imb"] for r in runs], 25)))
    json.dump(res, open(OUT, "w"), indent=1)
    print("  %-22s n=%2d  best cost -> rms %.4f imb %5.1f%% | median imb %5.1f%% | min imb %5.1f%%  (%.0fs)"
          % (key, len(runs), best["rms"], 100*best["imb"], 100*res[key]["imb_med"],
             100*res[key]["imb_min"], time.time()-t0), flush=True)
print("\ncells complete: %d/%d" % (len(res), len(cells)))
