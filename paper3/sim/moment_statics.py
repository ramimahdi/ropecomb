"""Pitch-moment statics of the frozen designs: no refit, no simulation, geometry only.

Each fold is symmetric about its member, so the member's force n_j F g_i(d), g_i = 2 D_i / sqrt(R_i^2 + D_i^2), is axial
(along the guide). It acts at the member's position x_i along the array, so the guide sees, besides the axial reaction
F_c = F_1 + F_2, the pitch moment M = M_2 - M_1 with M_j = sum_i x_i f_i, reacted by two bearings at spacing h as a
couple M / h. The paper's balance quantity is the reaction imbalance dF = |F_1 - F_2|; the moment equals e-bar dF
only if the two arrays' centres of pressure coincide. A span's position along the array is free (the ratio is a sum),
so placements are compared:
  drawing : spans in engagement order outward from the guide (Figures 10 and 11 of the main text)
  best    : the placement minimising max_d |M_2 - M_1| found by exhaustive search where N_1! N_2! <= 2e5, else by
            pairwise-swap descent from the drawing order and 30 random orders
For the single-array reference <9,5> the centre of pressure x-bar(d) and its force-weighted mean are given, with the
net moment for several offsets of the driving mass. Guide friction: E_guide = (2 mu / h) int |M| dd for both single
and dual arrays (S9), so the reduction factor int M_single dd / int |M_2 - M_1| dd is a design property.
Writes moment_statics.json and prints a LaTeX-ready summary."""
import os
import json, sys, itertools, math, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
sys.path.insert(0, os.path.join(_HERE, "code")); sys.path.insert(0, os.path.join(_HERE, "paper2"))
import unequal_members as U
from semisym_fit import falls
D = json.load(open("designs_counts_16471.json")); J = json.load(open("paper2/designs_16471.json")); L = json.load(open("locked_4ms.json"))["1000to1"]
dd, F, dmax = U.dd, U.F, U.dmax
w = np.gradient(dd)                                                   # trapezoid-like weights for integrals over d


def terms(R, s):
    R = np.asarray(R, float); s = np.asarray(s, float)
    Dm = np.maximum(0.0, dd[:, None] - s[None, :]); return 2 * Dm / np.sqrt(R[None, :] ** 2 + Dm ** 2)


def positions(R, perm):
    """x_i (m from the guide axis) of member i when the spans are laid out in the order `perm` outward from the guide."""
    R = np.asarray(R, float)[perm]; x = np.concatenate([[0.0], np.cumsum(2 * R)])[:-1] + R
    out = np.empty(len(R)); out[np.asarray(perm)] = x; return out


class Design:
    def __init__(self, d, k):
        self.k = k; self.n1, self.n2 = falls(k); self.R1, self.R2 = np.asarray(d["R_lo"], float), np.asarray(d["R_hi"], float)
        self.g1, self.g2 = terms(d["R_lo"], d["s_lo"]), terms(d["R_hi"], d["s_hi"])
        self.F1, self.F2 = self.n1 * F * self.g1.sum(1), self.n2 * F * self.g2.sum(1)
        self.o1, self.o2 = np.argsort(d["s_lo"]), np.argsort(d["s_hi"])

    def moments(self, p1, p2):
        return self.n1 * F * (self.g1 * positions(self.R1, p1)[None, :]).sum(1), self.n2 * F * (self.g2 * positions(self.R2, p2)[None, :]).sum(1)

    def score(self, p1, p2):
        M1, M2 = self.moments(p1, p2); return np.abs(M2 - M1).max()

    def report(self, p1, p2):
        M1, M2 = self.moments(p1, p2); Fc = self.F1 + self.F2; dM = np.abs(M2 - M1); Mone = M1 + M2
        with np.errstate(invalid="ignore", divide="ignore"): xb1, xb2 = M1 / self.F1, M2 / self.F2
        return dict(eps_F=float(np.abs(self.F1 - self.F2).max() / Fc.max()), dF_kN=float(np.abs(self.F1 - self.F2).max() / 1e3),
                    eps_M=float(dM.max() / Mone.max()), dM_kNm=float(dM.max() / 1e3), Mone_kNm=float(Mone.max() / 1e3), Fc_kN=float(Fc.max() / 1e3),
                    d_at_dM=float(dd[int(np.argmax(dM))]), int_dM_kJm=float((dM * w).sum() / 1e3), int_Mone_kJm=float((Mone * w).sum() / 1e3),
                    xbar_end=[float(xb1[-1]), float(xb2[-1])], perm1=list(map(int, p1)), perm2=list(map(int, p2)))


def best_placement(des, rng):
    n1, n2 = len(des.o1), len(des.o2)
    if math.factorial(n1) * math.factorial(n2) <= 200_000:
        best = (np.inf, None)
        for p1 in itertools.permutations(range(n1)):
            p1 = np.array(p1)
            for p2 in itertools.permutations(range(n2)):
                sc = des.score(p1, np.array(p2))
                if sc < best[0]: best = (sc, (p1, np.array(p2)))
        return best[1], "exhaustive"
    best = (np.inf, None)
    starts = [(des.o1.copy(), des.o2.copy())] + [(rng.permutation(n1), rng.permutation(n2)) for _ in range(30)]
    for p1, p2 in starts:
        p1, p2 = p1.copy(), p2.copy(); cur = des.score(p1, p2)
        while True:
            improved = False
            for which in (0, 1):
                p = p1 if which == 0 else p2
                for i, j in itertools.combinations(range(len(p)), 2):
                    q = p.copy(); q[i], q[j] = q[j], q[i]
                    sc = des.score(q if which == 0 else p1, q if which == 1 else p2)
                    if sc < cur - 1e-9: cur = sc; p[:] = q; improved = True
            if not improved: break
        if cur < best[0]: best = (cur, (p1, p2))
    return best[1], "swap descent, 31 starts"


# ---------------------------------------------------------------- single-array reference <9,5>
gs = terms(L["R"], L["s"]); o = np.argsort(L["s"]); xs = positions(L["R"], o)
Fs = 5 * F * gs.sum(1); Ms = 5 * F * (gs * xs[None, :]).sum(1)
with np.errstate(invalid="ignore", divide="ignore"): xbs = Ms / Fs
live = Fs > 0.01 * Fs.max()
single = dict(width=float(2 * sum(L["R"])), Fc_kN=float(Fs.max() / 1e3), M_kNm=float(Ms.max() / 1e3), xbar_first=float(xbs[live][0]), d_first=float(dd[live][0]),
              xbar_end=float(xbs[-1]), xbar_min=float(np.nanmin(xbs[live])), xbar_max=float(np.nanmax(xbs[live])),
              xbar_eff=float((Ms * w).sum() / (Fs * w).sum()), int_M_kJm=float((Ms * w).sum() / 1e3), int_F_kJ=float((Fs * w).sum() / 1e3),
              cg_test={str(c): [float(np.nanmin((Fs * (xbs - c))[live]) / 1e3), float(np.nanmax((Fs * (xbs - c))[live]) / 1e3)] for c in (0.5, 0.8, 1.0, 1.2, 1.58)})
print(f"single <9,5>: width {single['width']:.2f} m, F_c peak {single['Fc_kN']:.0f} kN, moment peak {single['M_kNm']:.0f} kN m, x-bar {single['xbar_first']:.2f} m at d = {single['d_first']:.2f} "
      f"-> {single['xbar_end']:.2f} m at d_max, force-weighted mean {single['xbar_eff']:.2f} m, int M dd {single['int_M_kJm']:.1f} kJ m, int F dd {single['int_F_kJ']:.1f} kJ")
for c, (lo, hi) in single["cg_test"].items(): print(f"   driving mass {c} m from the guide: net moment {lo:7.1f} to {hi:6.1f} kN m")

# ---------------------------------------------------------------- dual designs
rng = np.random.default_rng(0); out = {"_meta": {"single_reference": single}}
designs = (("<7,7>", J["k7"]["dual"], 7), ("<5,9>", J["k9"]["dual"], 9), ("<8:6,7>", D["k7_8-6"]["by_lambda"]["0.03"]["geometry"], 7),
           ("<5:4,9>", D["k9_5-4"]["by_lambda"]["0.02"]["geometry"], 9), ("<6:5,9>", D["k9_6-5"]["by_lambda"]["0.03"]["geometry"], 9))
print(f"\n{'design':8} {'eps_F':>6} {'dF kN':>6} | drawing: {'eps_M':>6} {'dM kNm':>7} {'x1/x2 end':>10} {'red.':>5} | best: {'eps_M':>6} {'dM kNm':>7} {'x1/x2 end':>10} {'red.':>5}  method")
for name, d, k in designs:
    des = Design(d, k); draw = des.report(des.o1, des.o2)
    (p1, p2), method = best_placement(des, rng); best = des.report(p1, p2)
    for r in (draw, best): r["reduction_vs_single"] = single["int_M_kJm"] / r["int_dM_kJm"]; r["bearing_kN_per_m"] = r["dM_kNm"]
    out[name] = dict(drawing=draw, best=best, method=method)
    print(f"{name:8} {100*draw['eps_F']:5.2f}% {draw['dF_kN']:6.1f} | {100*draw['eps_M']:6.2f}% {draw['dM_kNm']:7.1f} {draw['xbar_end'][0]:4.2f}/{draw['xbar_end'][1]:<4.2f} {draw['reduction_vs_single']:5.1f} |"
          f" {100*best['eps_M']:6.2f}% {best['dM_kNm']:7.1f} {best['xbar_end'][0]:4.2f}/{best['xbar_end'][1]:<4.2f} {best['reduction_vs_single']:5.1f}  {method}; order1 {best['perm1']} order2 {best['perm2']}")
json.dump(out, open("moment_statics.json", "w"), indent=1); print("wrote moment_statics.json")
