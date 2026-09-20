"""<4,5,9>: the merged k/2 seed as dealt (no optimisation) against the staged solve from it with the engagement
order enforced and free, at lambda 0.03, 0.02, 0.01; compliant force, rigid force and lateral load, frozen <5,5,9>
in grey. Each solve: plain dealt seed + 30 jittered starts (sigma 0.3, rng 1), best by objective (the free solve also
starts once from the constrained optimum). The seed does not depend on lambda, so its curves repeat in every row.
Writes figs/unequal_seed_vs_final.* and unequal_seed_vs_final.json."""
import os, sys, json, time, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
sys.path.insert(0, os.path.join(_HERE, "code")); sys.path.insert(0, os.path.join(_HERE, "paper2")); sys.path.insert(0, os.path.join(HERE, "..", "figsrc"))
import figstyle as S; S.use()
import unequal_members as U
from unequal_merged_seed import deal
from dual_fit_free import staged_solve, order_string, jitter, residuals_free, encode_free
from semisym_fit import falls, bank_ratio
from friction_sim import make_config, simulate_compliant
import types
_src = open("friction_sim.py").read().replace("return dict(stop=stop, t=np.array(t_hist), F=np.array(F_hist), slack=np.array(slack_hist),",
                                             "return dict(stop=stop, t=np.array(t_hist), F=np.array(F_hist), slack=np.array(slack_hist), hs=np.array(hs_hist),")
_mod = types.ModuleType("friction_sim_hs"); _mod.__file__ = os.path.abspath("friction_sim.py"); exec(compile(_src, "friction_sim.py", "exec"), _mod.__dict__)
simulate_rigid = _mod.simulate_rigid
dd, Gs, dmax = U.dd, U.Gs, U.dmax
k, W, N_lo, N_hi = 9, 1.96, 5, 4
n_lo, n_hi = falls(k)
NJ = 30
DZ = json.load(open("paper2/designs_16471.json"))["k9"]["dual"]


def curves(R_lo, s_lo, R_hi, s_hi):
    R_lo, s_lo, R_hi, s_hi = map(np.asarray, (R_lo, s_lo, R_hi, s_hi))
    g1 = bank_ratio(dd, R_lo, s_lo); g2 = bank_ratio(dd, R_hi, s_hi); net = n_lo * g1 + n_hi * g2
    imb = 100 * np.abs(n_lo * g1 - n_hi * g2) / net.max()
    gear, react, rhos, nn, taus = make_config([(R_lo, s_lo), (R_hi, s_hi)], k, 1.0, True)
    ee = simulate_compliant(U.M, U.m, U.h, gear, react, U.KR, U.F, max_heavy_dist=dmax, time_unit=2e-5)
    rr = simulate_rigid(U.M, U.m, U.h, gear, react, max_heavy_dist=dmax, max_time=0.5, max_target_acc=3e4, time_unit=1e-5)
    nc = int(0.995 * len(ee["F"])); nr = int(0.995 * len(rr["F"]))
    dr = np.concatenate([[0.0], np.cumsum(rr["hs"][1:] * np.diff(rr["t"]))])
    Fc = ee["F"][:nc]; Fr = rr["F"][:nr]
    return dict(imb=imb, dc=ee["d"][:nc], Fc=Fc / 1e3, dr=dr[:nr], Fr=Fr / 1e3, pC=float(Fc.max() / U.F), pR=float(Fr.max() / Fr.mean()),
                vC=float(ee["v_exit"]), vR=float(rr["v_exit"]), minF=float(Fc.min() / U.F), net=net,
                rms_pct=100 * float(np.sqrt(np.mean((net - Gs) ** 2)) / Gs[-1]), terminal_pct=100 * float(net[-1] / Gs[-1]),
                order=order_string(s_lo, s_hi), width_lo=float(2 * R_lo.sum()), width_hi=float(2 * R_hi.sum()),
                R_lo=list(map(float, R_lo)), s_lo=list(map(float, s_lo)), R_hi=list(map(float, R_hi)), s_hi=list(map(float, s_hi)))


def objective(des, lam):
    p = encode_free(*des, dmax)
    r = residuals_free(p, dd, Gs, N_lo, N_hi, n_lo, n_hi, dmax, W, lam, 50.0)
    return float(r @ r)


def best_of(lam, interleave, extra=None):
    rng = np.random.default_rng(1); res = []
    starts = [seed0] + [jitter(rng, *seed0, dmax) for _ in range(NJ)] + ([extra] if extra is not None else [])
    for st in starts:
        try:
            des, cost = staged_solve(*st, k, W, lam, dd, Gs, dmax, interleave=interleave); res.append((cost, des))
        except Exception:
            pass
    res.sort(key=lambda x: x[0]); return res[0]


# the seed: one array of nine members fitted at k/2, dealt L, R, L, R, ..., L
R, s, _, rms_single = U.fit_array(dd, Gs, N_lo + N_hi, k / 2, dmax, seed=0, width_budget=3 * W, R_bounds=(0.05, 8.0), init_span_range=(0.4, 1.6))
seed0 = deal(R, s, N_lo, N_hi)[:4]
seed_c = curves(*seed0)
print(f"seed: single-array rms {rms_single:.3f}, dealt order {seed_c['order']}, net rms {seed_c['rms_pct']:.2f}%  terminal {seed_c['terminal_pct']:.1f}%  "
      f"lateral max {seed_c['imb'].max():.2f}% mean {seed_c['imb'].mean():.2f}% end {seed_c['imb'][-1]:.1f}%  pC {seed_c['pC']:.3f} pR {seed_c['pR']:.2f} vC {seed_c['vC']:.1f}  "
      f"w {seed_c['width_lo']:.2f}/{seed_c['width_hi']:.2f}", flush=True)
print("  seed R_lo", np.round(seed0[0], 3), "s_lo", np.round(seed0[1], 3)); print("  seed R_hi", np.round(seed0[2], 3), "s_hi", np.round(seed0[3], 3))

ref = curves(*[np.array(DZ[x]) for x in ("R_lo", "s_lo", "R_hi", "s_hi")])
LAMS = (0.03, 0.02, 0.01)
styles = {"seed": dict(color=S.ORANGE, lw=1.1, ls=(0, (1, 1.2))), "enforced": dict(color=S.BLUE, lw=1.2), "free": dict(color=S.RED, lw=1.2, ls=(0, (4, 1.5)))}
labels = {"seed": "seed as dealt", "enforced": "solved, order enforced", "free": "solved, order free"}
fig, axes = plt.subplots(3, 3, figsize=(S.W_FULL, 7.0), sharex=True)
summary = {"seed": {kk: v for kk, v in seed_c.items() if not isinstance(v, np.ndarray)}}
for i, lam in enumerate(LAMS):
    t0 = time.time()
    costC, desC = best_of(lam, True); costF, desF = best_of(lam, False, extra=desC)
    cs = {"seed": seed_c, "enforced": curves(*desC), "free": curves(*desF)}
    costs = {"seed": objective(seed0, lam), "enforced": costC, "free": costF}
    aC, aR, aI = axes[i]
    aC.plot(ref["dc"], ref["Fc"], color=S.GREY, lw=0.9, label=f"<5,5,9> frozen  {ref['pC']:.2f}")
    aR.plot(ref["dr"], ref["Fr"], color=S.GREY, lw=0.7, label=f"<5,5,9> frozen  {ref['pR']:.2f}")
    aI.plot(dd, ref["imb"], color=S.GREY, lw=0.9, label=f"<5,5,9> frozen  {ref['imb'].max():.1f}%")
    print(f"\nlambda = {lam}  ({time.time()-t0:.0f} s)")
    for name in ("seed", "enforced", "free"):
        c = cs[name]
        aC.plot(c["dc"], c["Fc"], label=f"{labels[name]}  {c['pC']:.2f}", **styles[name])
        aR.plot(c["dr"], c["Fr"], label=f"{labels[name]}  {c['pR']:.2f}", **{**styles[name], "lw": 0.8})
        aI.plot(dd, c["imb"], label=f"{labels[name]}  {c['imb'].max():.1f}%", **styles[name])
        rec = {kk: v for kk, v in c.items() if not isinstance(v, np.ndarray)}; rec.update(objective=costs[name], lateral_max=float(c["imb"].max()), lateral_mean=float(c["imb"].mean()), lateral_end=float(c["imb"][-1]))
        summary[f"{name}|{lam}"] = rec
        print(f"  {name:9} objective {costs[name]:.5f}  order {c['order']:<13} rms {c['rms_pct']:.2f}%  lat max {c['imb'].max():5.2f}% mean {c['imb'].mean():4.2f}% end {c['imb'][-1]:3.1f}%  "
              f"pC {c['pC']:.3f} pR {c['pR']:.2f} minF {c['minF']:.2f} vC {c['vC']:.1f}  w {c['width_lo']:.2f}/{c['width_hi']:.2f}", flush=True)
    aC.axhline(U.F / 1e3, color=S.BLACK, lw=0.8, ls=(0, (5, 2.5))); aR.axhline(U.F / 1e3, color=S.BLACK, lw=0.8, ls=(0, (5, 2.5)))
    aC.set_ylim(0, 7); aR.set_ylim(0, 12); aI.set_ylim(0, 16); aI.axhline(100 * abs(n_lo - n_hi) / k, color=S.GREY, lw=0.8, ls=(0, (4, 2)))
    for a_ in (aC, aR, aI): a_.legend(loc="upper left", fontsize=5.6, handlelength=1.8)
    aC.set_ylabel("payload force (kN)")
    S.panel(aC, "adg"[i], f"compliant, $\\lambda = {lam}$"); S.panel(aR, "beh"[i], f"rigid, $\\lambda = {lam}$"); S.panel(aI, "cfi"[i], f"lateral load (% of peak), $\\lambda = {lam}$")
for a_ in axes[2]: a_.set_xlabel("carriage displacement $d$ (m)")
fig.suptitle("<4,5,9>, merged k/2: the dealt seed (no optimisation) and the staged solve from it with the engagement order enforced or free; legend numbers are peak/mean (force) and peak lateral load", fontsize=6.8, y=0.995)
fig.tight_layout(h_pad=0.6, w_pad=1.0)
S.save(fig, "unequal_seed_vs_final")
json.dump(summary, open("unequal_seed_vs_final.json", "w"), indent=1); print("wrote unequal_seed_vs_final.json")
