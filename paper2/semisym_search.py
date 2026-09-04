"""Grid search over semi-symmetric RopeComb designs.

    python3 semisym_search.py <k> <budget_seconds>

Sweeps member counts on both banks and the balance weight, with many restarts per
cell. Checkpoints atomically after every cell, so it can be run repeatedly and will
resume where it stopped.

Cells are keyed "N_lo,N_hi,lam". For each cell the best few fits by residual are kept
so they can be simulated afterwards -- fit residual does NOT predict force quality
(paper section 4.5), so ranking must be done on simulated peak-to-mean, not on rms.
"""

import sys
import os
import json
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from semisym_fit import ideal_target, fit_semisym, verify_interleaving, falls

K = int(sys.argv[1]) if len(sys.argv) > 1 else 7
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
OUT = os.path.join(HERE, "_semisym_search_k%d.json" % K)

N_RESTARTS = 20
TOP_KEEP = 3
LAMS = (0.1, 0.3, 1.0)

M, F, STOP = 1000.0, 1.0, 4.0
F_DESIGN = 3169.6


def atomic(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def main():
    dd, Gstar, d_max = ideal_target(M=M, m=F, v0=10.0, F=F_DESIGN, v_stop=STOP)
    Gt = float(Gstar[-1])
    n_lo, n_hi = falls(K)

    L_bank = 2.11 if K == 7 else 1.96

    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    t0 = time.time()
    print("k=%d  n_lo=%d  n_hi=%d  budget=%.0fs  out=%s" % (K, n_lo, n_hi, BUDGET, OUT))

    done = 0
    for lam in LAMS:
        for N_lo in range(max(1, K // 2), max(1, K // 2) + 6):
            for N_hi in range(max(1, (K + 1) // 2), max(1, (K + 1) // 2) + 6):
                key = "%d,%d,%g" % (N_lo, N_hi, lam)
                if key in res:
                    continue
                if time.time() - t0 > BUDGET:
                    print("budget exhausted after %d cells" % done)
                    return

                fits = []
                for s in range(N_RESTARTS):
                    o = fit_semisym(dd, Gstar, N_lo, N_hi, K, d_max, L_bank,
                                    lam_balance=lam, n_starts=1, seed=s + 100)
                    if o is None:
                        continue
                    ok, _ = verify_interleaving(o)
                    if not ok:
                        continue
                    fits.append(dict(
                        rms=float(o["rms"]),
                        imbalance=float(o["imbalance"]),
                        width_lo=float(o["width_lo"]),
                        width_hi=float(o["width_hi"]),
                        R_lo=list(map(float, o["R_lo"])),
                        s_lo=list(map(float, o["s_lo"])),
                        R_hi=list(map(float, o["R_hi"])),
                        s_hi=list(map(float, o["s_hi"])),
                    ))

                fits.sort(key=lambda x: x["rms"])
                res[key] = fits[:TOP_KEEP]
                atomic(res, OUT)
                done += 1
                b = fits[0] if fits else None
                if b:
                    print("%-14s  best rms %.4f  imb %.1f%%  w %.2f/%.2fm"
                          % (key, b["rms"], 100 * b["imbalance"], b["width_lo"], b["width_hi"]))
                else:
                    print("%-14s  no valid fits" % key)


if __name__ == "__main__":
    main()
