"""Does moving to the nearest EVEN k remove the need for unequal banks?

This answers the strongest obviousness objection available to an examiner:
"at odd k the banks are unbalanced, so just use k+1, which splits evenly and is
balanced under mirror symmetry for free."

Two experiments, both at fixed members-per-bank and fixed width budget so that
only k and the balance weight vary.

    python3 k_parity_study.py            # both experiments
"""

import sys
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from ropecomb import target_profile, fit_array, simulate_rigid, simulate_compliant
from ropecomb.geometry import array_ratio
from dualarray_ropecomb.weighting import falls
from dualarray_ropecomb.ordering import verify_interleaving
from dualarray_ropecomb.geometry import net_ratio_fn
from semisym_fit import fit_semisym

M, m, v0, F, VSTOP, KROPE, ETA = 1000.0, 1.0, 10.0, 3169.6, 4.0, 12000, 0.98
N_BANK, WIDTH = 7, 2.11

spec = target_profile(M, m, v0, VSTOP, F=F, n=400)
D_MAX = spec.d_max
DD = spec.d
GS = spec.G
V_IDEAL = spec.ideal_exit
SIMKW = dict(max_heavy_dist=D_MAX, max_time=0.5, max_target_acc=3e4, dt=1e-5)


def evaluate(pieces):
    """Drop members that never engage, then simulate rigid and compliant."""
    g0 = net_ratio_fn(pieces[0][0], pieces[0][1], pieces[0][2],
                      pieces[1][0], pieces[1][1], pieces[1][2])
    cut = max(simulate_rigid(M, m, v0, g0, **SIMKW)["summary"]["heavy_travel"],
              simulate_compliant(M, m, v0, g0, KROPE, max_heavy_dist=D_MAX,
                                 pretension=F, dt=2e-5)["summary"]["heavy_travel"])
    kept, built = [], []
    for R, s, mu in pieces:
        R, s = np.asarray(R), np.asarray(s)
        kp = s < cut
        kept.append((R[kp], s[kp], mu))
        built.append(int(kp.sum()))

    g = net_ratio_fn(kept[0][0], kept[0][1], kept[0][2],
                     kept[1][0], kept[1][1], kept[1][2])
    rr = simulate_rigid(M, m, v0, g, **SIMKW)
    ee = simulate_compliant(M, m, v0, g, KROPE, max_heavy_dist=D_MAX,
                            pretension=F, dt=2e-5)
    cont = [2 * b + int(mu) + 1 for b, (_, _, mu) in zip(built, kept)]
    return dict(built=built, contacts=cont,
                eff=float(np.mean([ETA ** c for c in cont])),
                vR=float(rr["summary"]["target_final_speed"]),
                vC=float(ee["summary"]["target_final_speed"]),
                pR=float(rr["force_target"].max() / rr["summary"]["mean_force_target"]),
                pC=float(ee["force_target"].max() / F))


def best_semisym(k, lam, restarts=20, seed0=131):
    best = None
    for sd in range(restarts):
        o = fit_semisym(DD, GS, N_BANK, N_BANK, k, D_MAX, WIDTH,
                        lam_balance=lam, n_starts=1, seed=seed0 * sd + 5)
        if o is None:
            continue
        ok, _ = verify_interleaving(o)
        if ok and (best is None or o["rms"] < best["rms"]):
            best = o
    return best


def experiment_1():
    """k=7 unequal banks vs k=8 mirror-symmetric, simulated."""
    print("\n=== 1. Is even k a free escape from the imbalance? ===")
    print("ideal exit %.1f m/s, d_max %.4f m, %d members per bank, width <= %.2f m\n"
          % (V_IDEAL, D_MAX, N_BANK, WIDTH))
    rows = []

    o = best_semisym(7, 0.3)
    nl, nh = falls(7)
    r = evaluate([(o["R_lo"], o["s_lo"], nl), (o["R_hi"], o["s_hi"], nh)])
    r.update(label="k=7 unequal banks", imb=o["imbalance"], rms=o["rms"])
    rows.append(r)

    b = None
    for sd in range(12):
        fit_res = fit_array(spec, N_BANK, 8, seed=sd, max_width=WIDTH)
        if b is None or fit_res.rms < b[2]:
            b = (np.asarray(fit_res.R), np.asarray(fit_res.s), fit_res.rms)
    R, s, rms = b
    r = evaluate([(R, s, 4), (R, s, 4)])
    r.update(label="k=8 mirror-symmetric", imb=0.0, rms=rms)
    rows.append(r)

    hdr = "%-22s %7s %7s %7s %8s %9s %6s %8s" % (
        "design", "rms", "exit", "%ideal", "p2m_C", "contacts", "eff", "imbalance")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("%-22s %7.3f %7.1f %6.1f%% %8.2f %9s %6.3f %7.1f%%"
              % (r["label"], r["rms"], r["vC"], 100 * r["vC"] / V_IDEAL, r["pC"],
                 "/".join(map(str, r["contacts"])), r["eff"], 100 * r["imb"]))
    print("\nRigid peak-to-mean is omitted: it is dominated by the last members to\n"
          "engage and is not reproducible across refits. Use compliant p2m.")


def experiment_2():
    """How the balance/accuracy trade differs between odd and even k."""
    print("\n=== 2. What the balance constraint costs at each parity ===\n")
    hdr = "%-4s %-6s %9s %11s %14s" % ("k", "lambda", "rms", "imbalance", "bank spans differ")
    print(hdr)
    print("-" * len(hdr))
    for k in (7, 8):
        for lam in (0.0, 0.3, 1.0):
            o = best_semisym(k, lam)
            if o is None:
                print("%-4d %-6g   no feasible fit" % (k, lam))
                continue
            dif = float(np.mean(np.abs(np.sort(o["R_lo"]) - np.sort(o["R_hi"]))))
            print("%-4d %-6g %9.4f %10.1f%% %13.4f m"
                  % (k, lam, o["rms"], 100 * o["imbalance"], dif))
        print()


if __name__ == "__main__":
    experiment_1()
    experiment_2()
