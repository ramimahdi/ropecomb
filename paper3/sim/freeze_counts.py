"""Freeze run for the count-balanced designs (PLAN_COUNTS_UPDATE.md, Step 0).

Same target and setup as paper2/designs_16471.json: 1,000:1, M = 1000 kg, m = 1 kg, v0 = 10 m/s, F = 3,169.6 N,
v_stop = 4 m/s, K_rope = 16,471 N/m, width budgets 2.11 m (k = 7) and 1.96 m (k = 9), span floor 0.05 m, lam_width 50.

Designs   <8:6,7> at lambda 0.02 and 0.03; <5:4,9> at 0.02 and 0.03; <6:5,9> at 0.03; a lambda ladder
          (0.01, 0.02, 0.03, 0.05) for the first two.
Solver    merged k/2 seed (one array of N_1 + N_2 members fitted at k/2, width cap 3 L_w, init_span_range (0.4, 1.6),
          seed 0), dealt L, R, ..., L; plain seed + 30 jittered starts (sigma 0.3, rng 1); staged solve with the order
          enforced; best by the joint objective. Checks logged: the free-order solve started from the result, the
          refit and unseeded strategies, and the placement of the extra leading members (first / middle / last).
Metrics   geometry (rms absolute and % of terminal, terminal ratio, lateral load max / mean / end / location, kN,
          widths, order), dynamics by friction_sim.evaluate (rigid dt 1e-5, compliant dt 2e-5, p2m over the first
          99.5 %), minimum compliant tension, friction at eta_c 1 / 0.995 / 0.99 / 0.98 (friction_study.py model),
          loads (per-array peak reaction, lateral kN, line tensions).
Baselines the frozen <7,7> and <5,9> designs re-evaluated through the same functions (a check of the pipeline against
          designs_16471.json), and single arrays of 9, 10 and 11 members at k = 9 fitted as in refit_16471.py
          (10 seeds, best rms) for the matched-count comparison; the 10-member single checks against the frozen one.
Writes designs_counts_16471.json.
"""
import os, sys, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
sys.path.insert(0, os.path.join(_HERE, "code")); sys.path.insert(0, os.path.join(_HERE, "paper2"))
import unequal_members as U
from unequal_merged_seed import deal
from dual_fit_free import staged_solve, order_string, jitter
from semisym_fit import falls, bank_ratio, interleaved_slots, encode, _residuals, fit_semisym
from friction_sim import make_config, evaluate, lateral_load, simulate_compliant
M, m, h, F, KR = U.M, U.m, U.h, U.F, U.KR
dd, Gs, dmax, Gt = U.dd, U.Gs, U.dmax, U.Gt
ETAS = (1.0, 0.995, 0.99, 0.98)
NJ = 30
LOG = open("designs_counts_16471.log", "w")


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def geometry(R_lo, s_lo, R_hi, s_hi, k):
    n_lo, n_hi = falls(k)
    R_lo, s_lo, R_hi, s_hi = map(lambda a: np.asarray(a, float), (R_lo, s_lo, R_hi, s_hi))
    g1 = bank_ratio(dd, R_lo, s_lo); g2 = bank_ratio(dd, R_hi, s_hi); net = n_lo * g1 + n_hi * g2
    imb = np.abs(n_lo * g1 - n_hi * g2) / net.max()
    rms = float(np.sqrt(np.mean((net - Gs) ** 2)))
    return dict(rms=rms, rms_pct=100 * rms / Gt, terminal=float(net[-1]), terminal_pct=100 * float(net[-1] / Gt),
                lateral=float(imb.max()), lateral_mean=float(imb.mean()), lateral_end=float(imb[-1]),
                lateral_at_m=float(dd[int(np.argmax(imb))]), peak_reaction_kN=float(net.max() * F / 1e3),
                lateral_kN=float(imb.max() * net.max() * F / 1e3),
                peak_reaction_array_kN=[float(n_lo * g1.max() * F / 1e3), float(n_hi * g2.max() * F / 1e3)],
                line_tension_kN=[n_lo * F / 1e3, n_hi * F / 1e3],
                G1_over_G2_end=float((g1[-1] / n_lo) / (g2[-1] / n_hi)),
                width_lo=float(2 * R_lo.sum()), width_hi=float(2 * R_hi.sum()), order=order_string(s_lo, s_hi),
                lateral_curve=[float(x) for x in np.interp(np.linspace(0, dmax, 81), dd, imb)],
                R_lo=list(map(float, R_lo)), s_lo=list(map(float, s_lo)), R_hi=list(map(float, R_hi)), s_hi=list(map(float, s_hi)))


def dynamics(R_lo, s_lo, R_hi, s_hi, k):
    lines = [(np.asarray(R_lo), np.asarray(s_lo)), (np.asarray(R_hi), np.asarray(s_hi))]
    out = {}
    for eta in ETAS:
        gear, react, rhos, nn, taus = make_config(lines, k, eta, True)
        e = evaluate(gear, react, M, m, h, F, dmax, KR)
        ll, rmax = lateral_load(rhos, dmax)
        rec = dict(pR=e["pR"], pC=e["pC"], vC=e["vC"], vR=e["vR"], E_payload=e["E_payload"], stopC=e["stopC"], stopR=e["stopR"],
                   lateral=ll, lateral_kN=ll * rmax * F / 1e3, peak_reaction_kN=rmax * F / 1e3)
        if eta == 1.0:
            ee = simulate_compliant(M, m, h, gear, react, KR, F, max_heavy_dist=dmax, time_unit=2e-5)
            Fc = ee["F"][:int(0.995 * len(ee["F"]))]; rec["minF"] = float(Fc.min() / F)
            rec["trace_d"] = [float(x) for x in ee["d"][:len(Fc):20]]; rec["trace_F"] = [float(x) for x in Fc[::20]]
        out[str(eta)] = rec
    E1 = out["1.0"]["E_payload"]
    for eta in ETAS: out[str(eta)]["E_retained"] = out[str(eta)]["E_payload"] / E1
    return out


def objective(R_lo, s_lo, R_hi, s_hi, k, W, lam, order):
    n_lo, n_hi = falls(k)
    p = encode(np.asarray(R_lo), np.asarray(s_lo), np.asarray(R_hi), np.asarray(s_hi), dmax, order)
    r = _residuals(p, dd, Gs, len(R_lo), len(R_hi), n_lo, n_hi, dmax, W, lam, 50.0, order)
    return float(r @ r)


def solve_design(k, W, N_lo, N_hi, lam, seed0, with_checks):
    order = interleaved_slots(N_lo, N_hi); rng = np.random.default_rng(1); res = []
    for j in range(NJ + 1):
        st = seed0 if j == 0 else jitter(rng, *seed0, dmax)
        try:
            des, cost = staged_solve(*st, k, W, lam, dd, Gs, dmax, interleave=True); res.append((cost, des, j))
        except Exception as ex:
            say("   start", j, "failed:", ex)
    res.sort(key=lambda x: x[0]); cost, des, j0 = res[0]
    plain = [r[0] for r in res if r[2] == 0][0]
    vecs = [np.concatenate(r[1]) for r in res]; basins = []
    for v in vecs:
        if not any(np.max(np.abs(v - b)) < 0.01 for b in basins): basins.append(v)
    rec = dict(objective=cost, from_start=("plain" if j0 == 0 else f"jitter {j0}"), plain_objective=plain,
               plain_rank=1 + sum(r[0] < plain for r in res), n_starts=len(res), n_basins=len(basins),
               within_1pct=int(sum(r[0] < 1.01 * cost for r in res)))
    if with_checks:
        (a, b, c, d_), cf = staged_solve(*des, k, W, lam, dd, Gs, dmax, interleave=False)
        rec["free_check"] = dict(objective=cf, order=order_string(b, d_), max_move_m=float(np.max(np.abs(np.concatenate([a, b, c, d_]) - np.concatenate(des)))))
        # refit strategy: N_lo single at k (best of 6), second array on the first N_hi midpoints
        best = None
        for sd in range(6):
            R_, s_, _, r_ = U.fit_array(dd, Gs, N_lo, k, dmax, seed=sd, width_budget=W, R_bounds=(0.05, 8.0))
            if best is None or r_ < best[2]: best = (R_, s_, r_)
        rf = U.fit_staged_unequal(np.array(best[0]), np.array(best[1]), N_hi, k, W, lam)[:4]
        rec["refit_objective"] = objective(*rf, k, W, lam, order)
        o = fit_semisym(dd, Gs, N_lo, N_hi, k, dmax, W, lam_balance=lam, lam_width=50.0, n_starts=30, seed=0)
        rec["unseeded_objective"] = objective(o["R_lo"], o["s_lo"], o["R_hi"], o["s_hi"], k, W, lam, order)
    return des, rec


def extra_placement(k, W, N_lo, N_hi, lam, seed0):
    """Where the extra leading members go: last (the default), first, middle. Seeds re-dealt by encode() to each order."""
    base = []
    for i in range(N_hi): base += [0, 1]
    extra = N_lo - N_hi
    orders = {"last": base + [0] * extra, "first": [0] * extra + base, "middle": base[:N_hi] + [0] * extra + base[N_hi:]}
    out = {}
    for name, od in orders.items():
        rng = np.random.default_rng(1); res = []
        for j in range(NJ + 1):
            st = seed0 if j == 0 else jitter(rng, *seed0, dmax)
            try:
                des, cost = staged_solve(*st, k, W, lam, dd, Gs, dmax, interleave=True, order=np.array(od)); res.append((cost, des))
            except Exception:
                pass
        res.sort(key=lambda x: x[0]); g = geometry(*res[0][1], k)
        out[name] = dict(objective=res[0][0], lateral=g["lateral"], lateral_mean=g["lateral_mean"], lateral_end=g["lateral_end"], rms=g["rms"], order=g["order"])
        say(f"   extra members {name:6}: objective {res[0][0]:.5f}  lateral {100*g['lateral']:.2f}% mean {100*g['lateral_mean']:.2f}% end {100*g['lateral_end']:.1f}%  order {g['order']}")
    return out


def merged_seed(k, W, N_lo, N_hi):
    R, s, _, rms = U.fit_array(dd, Gs, N_lo + N_hi, k / 2, dmax, seed=0, width_budget=3 * W, R_bounds=(0.05, 8.0), init_span_range=(0.4, 1.6))
    return deal(R, s, N_lo, N_hi)[:4], float(rms)


out = {"_meta": dict(k_rope=KR, M=M, m=m, v0=U.v0, F=F, v_stop=U.VSTOP, d_max=dmax, target_terminal=Gt, lam_width=50.0, R_floor=0.05,
                     solver="merged k/2 seed dealt L,R,...,L; plain + 30 jittered starts (sigma 0.3, rng 1); staged solve, order enforced; best by objective",
                     etas=list(ETAS), date="2026-09-14")}

# ---------------------------------------------------------------- frozen baselines through the same pipeline
J = json.load(open("paper2/designs_16471.json"))
for tag, key in (("k7_7-7", "k7"), ("k9_5-5", "k9")):
    d = J[key]["dual"]; k = J[key]["k"]
    g = geometry(d["R_lo"], d["s_lo"], d["R_hi"], d["s_hi"], k); dy = dynamics(d["R_lo"], d["s_lo"], d["R_hi"], d["s_hi"], k)
    out[tag] = dict(k=k, W=J[key]["width_budget"], N_lo=len(d["R_lo"]), N_hi=len(d["R_hi"]), lam=d.get("lam", J["_meta"].get("lam", 0.03)), source="paper2/designs_16471.json", geometry=g, dynamics=dy)
    say(f"{tag}: frozen pC {d['pC']:.3f} pR {d['pR']:.3f} vC {d['vC']:.1f} rms {d['rms']:.3f} lat {100*d['imbalance']:.2f}%  |  re-evaluated pC {dy['1.0']['pC']:.3f} pR {dy['1.0']['pR']:.3f} "
        f"vC {dy['1.0']['vC']:.1f} rms {g['rms']:.3f} lat {100*g['lateral']:.2f}% mean {100*g['lateral_mean']:.2f}% end {100*g['lateral_end']:.1f}%  order {g['order']}  minF {dy['1.0']['minF']:.2f}")
json.dump(out, open("designs_counts_16471.json", "w"), indent=1)

# ---------------------------------------------------------------- matched single arrays at k = 9 (refit_16471.py procedure)
out["singles_k9"] = {}
for N in (9, 10, 11):
    b = None
    for sd in range(10):
        R, s, _, rms = U.fit_array(dd, Gs, N, 9, dmax, seed=sd, R_bounds=(0.05, 8.0), width_budget=1.96, overshoot_weight=0.0)
        if b is None or rms < b[2]: b = (np.asarray(R), np.asarray(s), rms)
    R, s, rms = b
    gear, react, rhos, nn, taus = make_config([(R, s)], 9, 1.0, False)
    e = evaluate(gear, react, M, m, h, F, dmax, KR)
    out["singles_k9"][str(N)] = dict(N=N, k=9, rms=float(rms), rms_pct=100 * float(rms) / Gt, width=float(2 * R.sum()), pR=e["pR"], pC=e["pC"], vC=e["vC"], vR=e["vR"],
                                     R=list(map(float, R)), s=list(map(float, s)), contacts_one_line=None)
    say(f"single N={N} k=9: rms {rms:.3f}  width {2*R.sum():.2f}  pR {e['pR']:.2f}  pC {e['pC']:.3f}  vC {e['vC']:.1f}" + ("   (frozen <5,9> single: 2.881 / 2.191 / 328.1)" if N == 10 else ""))
json.dump(out, open("designs_counts_16471.json", "w"), indent=1)

# ---------------------------------------------------------------- the count-balanced designs
DESIGNS = [dict(tag="k7_8-6", k=7, W=2.11, N_lo=8, N_hi=6, lams=(0.02, 0.03), ladder=(0.01, 0.02, 0.03, 0.05), checks=True),
           dict(tag="k9_5-4", k=9, W=1.96, N_lo=5, N_hi=4, lams=(0.02, 0.03), ladder=(0.01, 0.02, 0.03, 0.05), checks=True),
           dict(tag="k9_6-5", k=9, W=1.96, N_lo=6, N_hi=5, lams=(0.03,), ladder=(0.03,), checks=False)]
for D in DESIGNS:
    k, W, N_lo, N_hi = D["k"], D["W"], D["N_lo"], D["N_hi"]; n_lo, n_hi = falls(k)
    t0 = time.time(); seed0, rms_single = merged_seed(k, W, N_lo, N_hi)
    gseed = geometry(*seed0, k)
    rec = dict(k=k, W=W, N_lo=N_lo, N_hi=N_hi, n_lo=n_lo, n_hi=n_hi, saturation_lateral=abs(n_lo * N_lo - n_hi * N_hi) / (n_lo * N_lo + n_hi * N_hi),
               seed=dict(single_rms_at_k_over_2=rms_single, geometry=gseed), by_lambda={})
    say(f"\n{D['tag']}: seed (k/2 single rms {rms_single:.3f}) as dealt: rms {gseed['rms']:.3f} terminal {gseed['terminal_pct']:.1f}% lat {100*gseed['lateral']:.2f}% order {gseed['order']}")
    for lam in D["ladder"]:
        t1 = time.time(); full = lam in D["lams"]
        des, srec = solve_design(k, W, N_lo, N_hi, lam, seed0, with_checks=(full and D["checks"]))
        g = geometry(*des, k); entry = dict(lam=lam, solve=srec, geometry=g)
        if full:
            entry["dynamics"] = dynamics(*des, k)
            if D["checks"]: entry["extra_placement"] = extra_placement(k, W, N_lo, N_hi, lam, seed0)
        else:
            gear, react, rhos, nn, taus = make_config([(np.asarray(des[0]), np.asarray(des[1])), (np.asarray(des[2]), np.asarray(des[3]))], k, 1.0, True)
            e = evaluate(gear, react, M, m, h, F, dmax, KR); entry["dynamics"] = {"1.0": dict(pR=e["pR"], pC=e["pC"], vC=e["vC"], vR=e["vR"])}
        rec["by_lambda"][str(lam)] = entry
        dy = entry["dynamics"]["1.0"]
        say(f"  lam {lam}: objective {srec['objective']:.5f} (plain rank {srec['plain_rank']}, {srec['n_basins']} basins, {srec['within_1pct']} within 1%)"
            + (f"; free {srec['free_check']['objective']:.5f} moved {1000*srec['free_check']['max_move_m']:.1f} mm; refit {srec['refit_objective']:.5f}; unseeded {srec['unseeded_objective']:.5f}" if "free_check" in srec else ""))
        say(f"         rms {g['rms']:.3f} ({g['rms_pct']:.2f}%)  terminal {g['terminal_pct']:.1f}%  lateral {100*g['lateral']:.2f}% mean {100*g['lateral_mean']:.2f}% end {100*g['lateral_end']:.1f}% at {g['lateral_at_m']:.2f} m  "
            f"{g['lateral_kN']:.1f} kN of {g['peak_reaction_kN']:.0f}  pC {dy['pC']:.3f} pR {dy['pR']:.2f}" + (f" minF {dy['minF']:.2f}" if "minF" in dy else "") + f"  vC {dy['vC']:.1f}  w {g['width_lo']:.2f}/{g['width_hi']:.2f}  order {g['order']}  ({time.time()-t1:.0f} s)")
        if full:
            for eta in ETAS[1:]:
                f_ = entry["dynamics"][str(eta)]
                say(f"         eta {eta}: E retained {100*f_['E_retained']:.1f}%  vC {f_['vC']:.0f}  pC {f_['pC']:.2f}  lateral {100*f_['lateral']:.1f}%  peak reaction {f_['peak_reaction_kN']:.0f} kN")
        out[D["tag"]] = rec; json.dump(out, open("designs_counts_16471.json", "w"), indent=1)
    say(f"{D['tag']} done in {time.time()-t0:.0f} s")
say("wrote designs_counts_16471.json")
