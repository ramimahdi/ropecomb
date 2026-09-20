"""Merged seed at the full ratio: one array of N_1 + N_2 members fitted at k (not k/2), dealt lo,hi,...,lo, then the
staged balanced solve; 30 restarts. Also reports the seed BEFORE optimisation (net ratio, fit and lateral load of the
dealt arrays) for both the k/2 and the k variants, and the engagement-gap statistics that govern the compliant
ripple. Writes unequal_merged_k.json."""
import os, sys, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
sys.path.insert(0, os.path.join(_HERE, "code")); sys.path.insert(0, os.path.join(_HERE, "paper2"))
import unequal_members as U
from unequal_merged_seed import deal, staged
from semisym_fit import falls, bank_ratio
dd, Gs, dmax = U.dd, U.Gs, U.dmax; Gt = float(Gs[-1])

def seed_metrics(R_l, s_l, R_h, s_h, k):
    n_lo, n_hi = falls(k); g1 = bank_ratio(dd, R_l, s_l); g2 = bank_ratio(dd, R_h, s_h); net = n_lo * g1 + n_hi * g2
    return dict(terminal_pct=100 * float(net[-1] / Gt), rms_pct=100 * float(np.sqrt(np.mean((net - Gs) ** 2)) / Gt),
                lateral=float((np.abs(n_lo * g1 - n_hi * g2) / max(net.max(), 1e-9)).max()),
                lateral_end=float(abs(n_lo * g1[-1] - n_hi * g2[-1]) / max(net.max(), 1e-9)))

def gaps(s_lo, s_hi):
    s = np.sort(np.concatenate([s_lo, s_hi])); g = np.diff(np.concatenate([s, [dmax]]))
    return dict(max_gap=float(g.max()), max_gap_at=float(s[int(np.argmax(g))] / dmax), mean_gap=float(g.mean()))

out = {}
for label, k, W, N_lo, N_hi in (("<5,5,9>", 9, 1.96, 5, 5), ("<5,6,9>", 9, 1.96, 6, 5), ("<4,5,9>", 9, 1.96, 5, 4),
                                ("<7,7,7>", 7, 2.11, 7, 7), ("<6,8,7>", 7, 2.11, 8, 6)):
    N = N_lo + N_hi; out[label] = {}
    for variant, kk, budget in (("k/2", k / 2, 2 * W), ("k", k, 2 * W)):
        t0 = time.time(); singles = [U.fit_array(dd, Gs, N, kk, dmax, seed=sd, width_budget=budget, R_bounds=(0.05, 8.0)) for sd in range(30)]
        R, s, _, rms = min(singles, key=lambda x: x[3])
        R_l, s_l, R_h, s_h, order = deal(R, s, N_lo, N_hi)
        sm = seed_metrics(R_l, s_l, R_h, s_h, k)
        print(f"{label} seed@{variant:3}: single rms {rms:.3f}, width {2*R.sum():.2f} m | dealt seed before solve: net terminal {sm['terminal_pct']:.0f}% of target, rms {sm['rms_pct']:.1f}%, lateral {100*sm['lateral']:.1f}% (end {100*sm['lateral_end']:.1f}%)", flush=True)
        out[label][f"seed@{variant}"] = dict(single_rms=rms, single_width=float(2 * R.sum()), **sm)
        if variant == "k":
            for lam in (0.02, 0.03):
                res = []
                for R, s, _, rms in singles:
                    (R_lo, s_lo, R_hi, s_hi), cost = staged(*deal(R, s, N_lo, N_hi), k, W, lam)
                    res.append((cost, R_lo, s_lo, R_hi, s_hi))
                res.sort(key=lambda x: x[0]); cost, R_lo, s_lo, R_hi, s_hi = res[0]
                a = U.assess(R_lo, s_lo, R_hi, s_hi, k); a.update(lam=lam, cost=cost, **gaps(s_lo, s_hi))
                lat = [U.assess(r[1], r[2], r[3], r[4], k, sim=False)["lateral"] for r in res]
                a["spread_lateral"] = [float(min(lat)), float(np.median(lat)), float(max(lat))]
                out[label][f"solved@k|{lam}"] = a
                print(f"{label} solved from seed@k lam={lam:<4}: max lat {100*a['lateral']:5.2f}%  end {100*a['lateral_end']:4.1f}%  pC {a['pC']:.3f}  pR {a['pR']:.2f}  vC {a['vC']:.1f}  rms {a['rms_pct']:.2f}%  w {a['width_lo']:.2f}/{a['width_hi']:.2f}  max gap {100*a['max_gap']/dmax:.1f}% of stroke at {a['max_gap_at']:.2f}  | spread {100*min(lat):.2f}–{100*max(lat):.2f}%", flush=True)
    json.dump(out, open("unequal_merged_k.json", "w"), indent=1)
# engagement gaps of the reference designs (k/2 seed results, lambda 0.03)
M = json.load(open("unequal_merged_seed.json"))
for label in ("<5,5,9>", "<5,6,9>", "<4,5,9>", "<7,7,7>", "<6,8,7>"):
    r = M[label]["0.03"]; g = gaps(np.array(r["s_lo"]), np.array(r["s_hi"]))
    print(f"{label} lam=0.03 (k/2 seed): pC {r['pC']:.3f}  members {len(r['s_lo'])+len(r['s_hi'])}  max gap {100*g['max_gap']/dmax:.1f}% of stroke at {g['max_gap_at']:.2f}  mean gap {100*g['mean_gap']/dmax:.1f}%")
print("wrote unequal_merged_k.json")
