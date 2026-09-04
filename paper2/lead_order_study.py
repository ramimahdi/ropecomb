"""Which array should engage first: the one driving the lower fall count, or the higher?

For each (k, N, lambda) and each of the two assignments, run many restarts of the
staged solve and keep the distribution, not one fit. Restarts vary both the stage-1
single-array seed and a jitter on the structured starting point.

    python3 lead_order_study.py [restarts] [budget_seconds]

Checkpoints to lead_order_study.json after every cell, so it can be re-run to resume.
"""

import os
import sys
import json
import time
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from ropecomb import target_profile, fit_array
from ropecomb.geometry import array_ratio
from dualarray_ropecomb.weighting import falls
from dualarray_ropecomb.ordering import interleaved_slots
from dualarray_ropecomb.fit import (encode, seed_two_stage, _residuals, _unpack)

RESTARTS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 520.0
OUT = os.path.join(HERE, "lead_order_study.json")
M, m, v0, F, VSTOP, WIDTH = 1000.0, 1.0, 10.0, 3169.6, 4.0, 2.11

spec = target_profile(M, m, v0, VSTOP, F=F, n=400)
dd, Gs, dmax = spec.d, spec.G, spec.d_max
Gt = float(Gs[-1])
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
_seed_cache = {}


def stage1(N, k, sd):
    key = (N, k, sd)
    if key not in _seed_cache:
        fit_res = fit_array(spec, N, k, seed=sd, max_width=WIDTH)
        _seed_cache[key] = (np.asarray(fit_res.R), np.asarray(fit_res.s))
    return _seed_cache[key]


def one_fit(k, N, lam, order, sd, jitter):
    n_lo, n_hi = falls(k)
    Rst, sst = stage1(N, k, sd % 8)
    R_l, s_l, R_h, s_h = seed_two_stage(Rst, sst, dmax)
    p0 = encode(R_l, s_l, R_h, s_h, dmax, order)
    if jitter:
        p0 = p0 + np.random.default_rng(sd).normal(0.0, jitter, p0.shape)
    args = (dd, Gs, N, N, n_lo, n_hi, dmax, WIDTH, lam, 50.0, order)
    n = 2 * N
    head = p0[:n + 1].copy()
    s1 = least_squares(lambda q: _residuals(np.concatenate([head, q]), *args),
                       p0[n + 1:], method="lm", max_nfev=2500)
    s2 = least_squares(_residuals, np.concatenate([head, s1.x]), method="lm",
                       max_nfev=3500, args=args)
    Rlo, slo, Rhi, shi = _unpack(s2.x, N, N, dmax, order)
    gl, gh = array_ratio(dd, Rlo, slo), array_ratio(dd, Rhi, shi)
    net = n_lo * gl + n_hi * gh
    return dict(cost=float(s2.cost),
                rms=float(np.sqrt(np.mean((net - Gs) ** 2))),
                imb=float(np.max(np.abs(n_lo * gl - n_hi * gh)) / max(net.max(), 1e-9)),
                terminal=float(net[-1]))


def main():
    cells = [(k, N, lam, lab) for k, N in [(5, 10), (7, 7), (9, 5)]
                              for lam in (0.3, 1.0)
                              for lab in ("low_leads", "high_leads")]
    t0 = time.time()
    for k, N, lam, lab in cells:
        key = "%d,%d,%g,%s" % (k, N, lam, lab)
        if key in res:
            continue
        if time.time() - t0 > BUDGET:
            print("budget reached, %d/%d cells done" % (len(res), len(cells)))
            break
        order = np.asarray(interleaved_slots(N, N, low_leads=(lab == "low_leads")))
        runs = []
        for sd in range(RESTARTS):
            try:
                runs.append(one_fit(k, N, lam, order, sd, 0.0 if sd == 0 else 0.25))
            except Exception:
                continue
        if not runs:
            continue
        runs.sort(key=lambda r: r["cost"])
        best = runs[0]
        res[key] = dict(k=k, N=N, lam=lam, order=lab, n=len(runs), best=best,
                        rms_med=float(np.median([r["rms"] for r in runs])),
                        imb_med=float(np.median([r["imb"] for r in runs])),
                        imb_min=float(min(r["imb"] for r in runs)),
                        imb_p25=float(np.percentile([r["imb"] for r in runs], 25)))
        json.dump(res, open(OUT, "w"), indent=1)
        print("  %-22s n=%2d  best cost -> rms %.4f imb %5.1f%% | median imb %5.1f%% | min imb %5.1f%%  (%.0fs)"
              % (key, len(runs), best["rms"], 100 * best["imb"], 100 * res[key]["imb_med"],
                 100 * res[key]["imb_min"], time.time() - t0), flush=True)
    print("\ncells complete: %d/%d" % (len(res), len(cells)))


if __name__ == "__main__":
    main()
