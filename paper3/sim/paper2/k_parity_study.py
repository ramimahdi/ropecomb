"""
Does moving to the nearest EVEN k remove the need for unequal banks?

This answers the strongest obviousness objection available to an examiner:
"at odd k the banks are unbalanced, so just use k+1, which splits evenly and is
balanced under mirror symmetry for free."

Two experiments, both at fixed members-per-bank and fixed width budget so that
only k and the balance weight vary.

    python3 k_parity_study.py            # both experiments

Requires the parent filing's code/ directory on the path (../code).
"""
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
sys.path.insert(0, HERE)

from catabult_sim import simulate
from catabult_sim_elastic import simulate_elastic
from ropecomb_fit4 import fit_array
from semisym_fit import fit_semisym, falls, verify_interleaving

M, m, v0, F, VSTOP, KROPE, ETA = 1000., 1., 10., 3169.6, 4.0, 12000, 0.98
N_BANK, WIDTH = 7, 2.11
h = v0 ** 2 / (2 * 9.8)

I = simulate(M, m, h, get_gear_fn=F, min_heavy_speed=VSTOP, max_time=.6,
             max_target_acc=1e9, time_unit=1e-5)
D_MAX = I["summary"]["heavy_travel"]
DD = np.linspace(0, D_MAX, 400)
GS = np.interp(DD, I["heavy_dist"], I["gear"])
V_IDEAL = I["summary"]["target_final_speed"]
SIMKW = dict(max_heavy_dist=D_MAX, max_time=.5, max_target_acc=3e4, time_unit=1e-5)


def gear_fn(pieces):
    """pieces = [(R, s, falls), ...]; net ratio = sum falls_i * G_bank_i(d)."""
    P = [(np.asarray(R, float), np.asarray(s, float), float(mu)) for R, s, mu in pieces]

    def f(d):
        t = 0.0
        for R, s, mu in P:
            D = d - s
            e = D > 0
            if e.any():
                t += mu * np.sum(2.0 * D[e] / np.sqrt(R[e] ** 2 + D[e] ** 2))
        return float(t)
    return f


def evaluate(pieces):
    """Drop members that never engage, then simulate rigid and compliant."""
    g0 = gear_fn(pieces)
    cut = max(simulate(M, m, h, get_gear_fn=g0, **SIMKW)["summary"]["heavy_travel"],
              simulate_elastic(M, m, h, g0, k_rope=KROPE, max_heavy_dist=D_MAX,
                               pretension=F, time_unit=2e-5)["summary"]["heavy_travel"])
    kept, built = [], []
    for R, s, mu in pieces:
        R, s = np.asarray(R), np.asarray(s)
        kp = s < cut
        kept.append((R[kp], s[kp], mu)); built.append(int(kp.sum()))
    g = gear_fn(kept)
    rr = simulate(M, m, h, get_gear_fn=g, **SIMKW)
    ee = simulate_elastic(M, m, h, g, k_rope=KROPE, max_heavy_dist=D_MAX,
                          pretension=F, time_unit=2e-5)
    # one contact per engagement member, one per fixed support, one per fall, plus
    # the stage entry: verified against the frozen designs as 2*built + falls + 1
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
    r.update(label="k=7 unequal banks", imb=o["imbalance"], rms=o["rms"]); rows.append(r)

    b = None
    for sd in range(12):
        R, s, _, rms = fit_array(DD, GS, N_BANK, 8, D_MAX, seed=sd,
                                 R_bounds=(0.05, 8.), width_budget=WIDTH,
                                 overshoot_weight=0.0)
        if b is None or rms < b[2]:
            b = (np.asarray(R), np.asarray(s), rms)
    R, s, rms = b
    r = evaluate([(R, s, 4), (R, s, 4)])
    r.update(label="k=8 mirror-symmetric", imb=0.0, rms=rms); rows.append(r)

    hdr = "%-22s %7s %7s %7s %8s %9s %6s %8s" % (
        "design", "rms", "exit", "%ideal", "p2m_C", "contacts", "eff", "imbalance")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print("%-22s %7.3f %7.1f %6.1f%% %8.2f %9s %6.3f %7.1f%%"
              % (r["label"], r["rms"], r["vC"], 100 * r["vC"] / V_IDEAL, r["pC"],
                 "/".join(map(str, r["contacts"])), r["eff"], 100 * r["imb"]))
    print("\nRigid peak-to-mean is omitted here only because this comparison turns on\n"
          "the compliant figure. It is reproducible when measured over the release\n"
          "window of Section 9 (first 99.5% of the trace); taking max() over the full\n"
          "history instead picks up the terminal traction-loss spike, which the payload\n"
          "never sees because release precedes it.")


def experiment_2():
    """How the balance/accuracy trade differs between odd and even k."""
    print("\n=== 2. What the balance constraint costs at each parity ===\n")
    hdr = "%-4s %-6s %9s %11s %14s" % ("k", "lambda", "rms", "imbalance", "bank spans differ")
    print(hdr); print("-" * len(hdr))
    for k in (7, 8):
        for lam in (0.0, 0.3, 1.0):
            o = best_semisym(k, lam)
            if o is None:
                print("%-4d %-6g   no feasible fit" % (k, lam)); continue
            dif = float(np.mean(np.abs(np.sort(o["R_lo"]) - np.sort(o["R_hi"]))))
            print("%-4d %-6g %9.4f %10.1f%% %13.4f m"
                  % (k, lam, o["rms"], 100 * o["imbalance"], dif))
        print()


if __name__ == "__main__":
    experiment_1()
    experiment_2()
