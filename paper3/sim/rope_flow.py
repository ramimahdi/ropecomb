"""Rope passed over each contact x wrap angle (a friction/wear proxy), for span orders along an array.
Rope enters at the outboard end (from the stage) and flows toward the dead-end anchor at the guide. Rope sliding over
member i = half its own take-up + the take-up of every fold anchor-side of it; wrap at member i = 2 atan(D_i/R_i);
each stationary support between two members turns the rope by the sum of the two adjacent half-angles and passes
the take-up of every fold anchor-side of it. Metric: integral over the stroke of sum_contacts wrap x rope passed
(radian-metres), plus the worst single contact. Writes rope_flow.json (Supplementary Section S9)."""
import json, sys, os, numpy as np
from itertools import permutations
_HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)   # run from anywhere; outputs land beside this file
meta = json.load(open("designs_counts_16471.json"))["_meta"]; dmax = meta["d_max"]; dd = np.linspace(0, dmax, 400)
D = json.load(open("designs_counts_16471.json")); J = json.load(open("paper2/designs_16471.json")); MS = json.load(open("moment_statics.json"))
L = json.load(open("locked_4ms.json"))["1000to1"]

def flow(R, s, order):
    """order: member indices from the ENTRY (outboard) to the ANCHOR (guide)."""
    R = np.asarray(R)[order]; s = np.asarray(s)[order]; Dm = np.maximum(0, dd[:, None] - s[None, :])
    Y = 2 * (np.sqrt(R**2 + Dm**2) - R)                       # take-up per fold, 400 x N
    dY = np.diff(Y, axis=0)                                    # increments
    half = np.arctan2(Dm, R)                                   # half-angle per member
    theta_m = 2 * half                                         # wrap at member
    behind = np.cumsum(dY[:, ::-1], axis=1)[:, ::-1] - dY      # take-up increments of folds anchor-side of i
    slide_m = 0.5 * dY + behind
    # supports: between member i and i+1 (anchor-side), passes take-up of folds i+1.. ; and one before the first member
    theta_s = half[:, :-1] + half[:, 1:]; slide_s = behind[:, :-1]
    theta_s0 = half[:, :1]; slide_s0 = (behind + dY)[:, :1]     # entry support before the first member
    Wm = (theta_m[1:] * slide_m).sum(); Ws = (theta_s[1:] * slide_s).sum() + (theta_s0[1:] * slide_s0).sum()
    worst = max((theta_m[1:] * slide_m).sum(0).max(), (theta_s[1:] * slide_s).sum(0).max())
    return Wm + Ws, worst

OUT = {}
def report(name, R, s, best_perm=None):
    N = len(R); eng = list(np.argsort(s))                      # engagement order (first-engaging first)
    orders = {"drawing (deepest at anchor)": eng[::-1], "reversed (deepest at entry)": eng}
    if best_perm is not None:
        # moment_statics perm: slot order outward from the guide (anchor) -> entry order is its reverse
        orders["searched placement"] = list(best_perm)[::-1]
    out = {k: flow(R, s, v) for k, v in orders.items()}
    lo = min(f for f, _ in out.values())
    OUT[CUR + " " + name] = {k: dict(total_rad_m=float(f), worst_rad_m=float(w)) for k, (f, w) in out.items()}
    for k, (f, w) in out.items(): print(f"  {name:10} {k:28} total {f:7.2f} rad m  ({f/lo:4.2f}x)   worst contact {w:6.2f} rad m")
    if N <= 8:
        allp = [(flow(R, s, list(p))[0], p) for p in permutations(range(N))]
        f, p = min(allp); OUT[CUR + " " + name]["best of all orders"] = dict(total_rad_m=float(f), entry_to_anchor=list(map(int, p)))
        print(f"  {name:10} {'best of all orders':28} total {f:7.2f} rad m  entry->anchor {list(p)}, engagement ranks {[eng.index(i)+1 for i in p]}")

CUR = "<9,5>"; print("<9,5> single"); report("single", L["R"], L["s"])
for tag, key, lam, nm in (("k7_7-7", None, None, "<7,7>"), ("k7_8-6", None, "0.03", "<8:6,7>")):
    r = D[tag]; g = r["by_lambda"][lam]["geometry"] if lam else r["geometry"]
    ms = MS[nm]["best"]
    CUR = nm; print(nm); report("array 1", g["R_lo"], g["s_lo"], ms["perm1"]); report("array 2", g["R_hi"], g["s_hi"], ms["perm2"])
json.dump(OUT, open("rope_flow.json", "w"), indent=1); print("wrote rope_flow.json")
