"""<4,5,9> at lambda 0.03, 0.02, 0.01: compliant force, rigid force and lateral load for three solver strategies
(refit seed; merged k/2 with the dealt seed and 30 perturbations of it; unseeded joint LM with 30 starts), frozen <5,5,9> in grey.
Writes figs/unequal_min9_lam.* and unequal_min9_lam.json."""
import os, sys, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
sys.path.insert(0, os.path.join(_HERE, "code")); sys.path.insert(0, os.path.join(_HERE, "paper2")); sys.path.insert(0, os.path.join(HERE, "..", "figsrc"))
import figstyle as S; S.use()
import unequal_members as U
from unequal_merged_seed import deal, staged
from semisym_fit import falls, bank_ratio, fit_semisym, interleaved_slots
from friction_sim import make_config, simulate_compliant


def jitter(rng, R_l, s_l, R_h, s_h, sig=0.3):
    """Perturb a dealt seed: log-normal widths, log-normal engagement gaps (offsets rescaled to stay inside the stroke)."""
    def one(Rx, sx):
        o = np.argsort(sx); Rx = np.asarray(Rx)[o]; sx = np.asarray(sx)[o]
        gaps = np.diff(np.concatenate([[0.0], sx])) * np.exp(rng.normal(0, sig, len(sx)))
        s_new = np.cumsum(gaps); s_new = s_new * min(1.0, 0.98 * U.dmax / max(s_new[-1], 1e-9))
        return np.maximum(0.05, Rx * np.exp(rng.normal(0, sig, len(Rx)))), s_new
    (a, b), (c, d_) = one(R_l, s_l), one(R_h, s_h)
    return a, b, c, d_
# the rigid integrator keeps the source-speed history but does not return it; load a copy that does
import types
_src = open("friction_sim.py").read().replace("return dict(stop=stop, t=np.array(t_hist), F=np.array(F_hist), slack=np.array(slack_hist),",
                                             "return dict(stop=stop, t=np.array(t_hist), F=np.array(F_hist), slack=np.array(slack_hist), hs=np.array(hs_hist),")
_mod = types.ModuleType("friction_sim_hs"); _mod.__file__ = os.path.abspath("friction_sim.py"); exec(compile(_src, "friction_sim.py", "exec"), _mod.__dict__)
simulate_rigid = _mod.simulate_rigid
dd, Gs, dmax = U.dd, U.Gs, U.dmax
k, W, N_lo, N_hi = 9, 1.96, 5, 4
n_lo, n_hi = falls(k)
DZ = json.load(open("paper2/designs_16471.json"))["k9"]["dual"]

def curves(R_lo, s_lo, R_hi, s_hi):
    g1 = bank_ratio(dd, R_lo, s_lo); g2 = bank_ratio(dd, R_hi, s_hi); net = n_lo * g1 + n_hi * g2
    imb = 100 * np.abs(n_lo * g1 - n_hi * g2) / net.max()
    gear, react, rhos, nn, taus = make_config([(R_lo, s_lo), (R_hi, s_hi)], k, 1.0, True)
    ee = simulate_compliant(U.M, U.m, U.h, gear, react, U.KR, U.F, max_heavy_dist=dmax, time_unit=2e-5)
    rr = simulate_rigid(U.M, U.m, U.h, gear, react, max_heavy_dist=dmax, max_time=0.5, max_target_acc=3e4, time_unit=1e-5)
    nc = int(0.995 * len(ee["F"])); nr = int(0.995 * len(rr["F"]))
    dr = np.concatenate([[0.0], np.cumsum(rr["hs"][1:] * np.diff(rr["t"]))])
    Fc = ee["F"][:nc]; Fr = rr["F"][:nr]
    return dict(imb=imb, dc=ee["d"][:nc], Fc=Fc / 1e3, dr=dr[:nr], Fr=Fr / 1e3, pC=float(Fc.max() / U.F), pR=float(Fr.max() / Fr.mean()),
                vC=float(ee["v_exit"]), vR=float(rr["v_exit"]), minF=float(Fc.min() / U.F))

def solve_all(lam):
    out = {}
    best = None
    for sd in range(6):
        R_, s_, _, r_ = U.fit_array(dd, Gs, N_lo, k, dmax, seed=sd, width_budget=W, R_bounds=(0.05, 8.0))
        if best is None or r_ < best[2]: best = (R_, s_, r_)
    out["refit"] = U.fit_staged_unequal(np.array(best[0]), np.array(best[1]), N_hi, k, W, lam)[:4]
    # merged k/2 with real restarts: the k/2 single-array fit converges to one design whatever its seed, so the
    # restarts must perturb the dealt seed (log-normal widths and gaps, sigma 0.3); 30 perturbations + the plain seed
    R, s, _, _ = U.fit_array(dd, Gs, N_lo + N_hi, k / 2, dmax, seed=0, width_budget=3 * W, R_bounds=(0.05, 8.0), init_span_range=(0.4, 1.6))
    seed0 = deal(R, s, N_lo, N_hi)[:4]; order = interleaved_slots(N_lo, N_hi); rng = np.random.default_rng(1); res = []
    for j in range(31):
        st = seed0 if j == 0 else jitter(rng, *seed0)
        try:
            (a, b, c, d_), cost = staged(*st, order, k, W, lam); res.append((cost, a, b, c, d_))
        except Exception:
            pass
    plain = res[0][0]; res.sort(key=lambda x: x[0]); out["merged k/2"] = res[0][1:]
    print(f"lam={lam} merged k/2: {len(res)} starts, best objective {res[0][0]:.5f} (plain dealt seed {plain:.5f}, rank {1 + sum(r[0] < plain for r in res)})", flush=True)
    o = fit_semisym(dd, Gs, N_lo, N_hi, k, dmax, W, lam_balance=lam, lam_width=50.0, n_starts=30, seed=0)
    out["unseeded"] = (o["R_lo"], o["s_lo"], o["R_hi"], o["s_hi"])
    return out

styles = {"refit": dict(color=S.BLUE, lw=1.2), "merged k/2": dict(color=S.RED, lw=1.2, ls=(0, (4, 1.5))), "unseeded": dict(color=S.GREEN, lw=1.1, ls=(0, (1, 1.2)))}
ref = curves(*[np.array(DZ[x]) for x in ("R_lo", "s_lo", "R_hi", "s_hi")])
LAMS = (0.03, 0.02, 0.01)
fig, axes = plt.subplots(3, 3, figsize=(S.W_FULL, 7.0), sharex=True)
summary = {}
for i, lam in enumerate(LAMS):
    des = solve_all(lam); aC, aR, aI = axes[i]
    aC.plot(ref["dc"], ref["Fc"], color=S.GREY, lw=0.9, label=f"<5,5,9> frozen  {ref['pC']:.2f}")
    aR.plot(ref["dr"], ref["Fr"], color=S.GREY, lw=0.7, label=f"<5,5,9> frozen  {ref['pR']:.2f}")
    aI.plot(dd, ref["imb"], color=S.GREY, lw=0.9, label=f"<5,5,9> frozen  {ref['imb'].max():.1f}%")
    for name, (R_lo, s_lo, R_hi, s_hi) in des.items():
        c = curves(R_lo, s_lo, R_hi, s_hi)
        aC.plot(c["dc"], c["Fc"], label=f"{name}  {c['pC']:.2f}", **styles[name])
        aR.plot(c["dr"], c["Fr"], label=f"{name}  {c['pR']:.2f}", **{**styles[name], "lw": 0.8})
        aI.plot(dd, c["imb"], label=f"{name}  {c['imb'].max():.1f}%", **styles[name])
        rms = 100 * float(np.sqrt(np.mean((n_lo * bank_ratio(dd, R_lo, s_lo) + n_hi * bank_ratio(dd, R_hi, s_hi) - Gs) ** 2)) / Gs[-1])
        summary[f"{name}|{lam}"] = dict(pC=c["pC"], pR=c["pR"], vC=c["vC"], vR=c["vR"], minF=c["minF"], lateral_max=float(c["imb"].max()), lateral_mean=float(c["imb"].mean()),
                                       lateral_end=float(c["imb"][-1]), rms_pct=rms, width_lo=float(2 * np.sum(R_lo)), width_hi=float(2 * np.sum(R_hi)),
                                       R_lo=list(map(float, R_lo)), s_lo=list(map(float, s_lo)), R_hi=list(map(float, R_hi)), s_hi=list(map(float, s_hi)))
        print(f"lam={lam} {name:11} pC {c['pC']:.3f}  pR {c['pR']:.2f}  vC {c['vC']:.1f}  vR {c['vR']:.1f}  minF {c['minF']:.2f}  max lat {c['imb'].max():5.2f}%  mean {c['imb'].mean():4.2f}%  end {c['imb'][-1]:4.1f}%  rms {rms:.2f}%  w {2*np.sum(R_lo):.2f}/{2*np.sum(R_hi):.2f}", flush=True)
    aC.axhline(U.F / 1e3, color=S.BLACK, lw=0.8, ls=(0, (5, 2.5))); aR.axhline(U.F / 1e3, color=S.BLACK, lw=0.8, ls=(0, (5, 2.5)))
    aC.set_ylim(0, 7); aR.set_ylim(0, 12); aI.set_ylim(0, 16); aI.axhline(100 * abs(n_lo - n_hi) / k, color=S.GREY, lw=0.8, ls=(0, (4, 2)))
    for a_ in (aC, aR, aI): a_.legend(loc="upper left", fontsize=5.6, handlelength=1.8)
    aC.set_ylabel("payload force (kN)")
    S.panel(aC, "adg"[i], f"compliant, $\\lambda = {lam}$"); S.panel(aR, "beh"[i], f"rigid, $\\lambda = {lam}$"); S.panel(aI, "cfi"[i], f"lateral load (% of peak), $\\lambda = {lam}$")
for a_ in axes[2]: a_.set_xlabel("carriage displacement $d$ (m)")
fig.suptitle("<4,5,9>: five members on the leading four-fall array, four on the five-fall array, k = 9; legend numbers are peak/mean (force) and peak lateral load", fontsize=7.2, y=0.995)
fig.tight_layout(h_pad=0.6, w_pad=1.0)
S.save(fig, "unequal_min9_lam")
json.dump(summary, open("unequal_min9_lam.json", "w"), indent=1); print("wrote unequal_min9_lam.json")
