"""Regenerate every data figure of the manuscript in one style (figstyle.py).

Inputs are the frozen designs and the author's own integrators, copied unmodified:
    ../sim/locked_4ms.json              single-array references at 100:1, 1,000:1, 10,000:1
    ../sim/paper2/designs_16471.json    the <7,7> and <5,9> dual designs
    code/_decel4_*.json                 the deceleration-target sweep (SI S10)
    code/_searchfit_data.json           search-vs-fit curve (SI S11)
    ../sim/code/catabult_sim.py, catabult_sim_elastic.py, gear_fn_bridge.py (the author's integrators)

    python3 make_figures.py [names...]      (no names: everything)

Every figure is written to ../figs/<name>.pdf (+ .png at 300 dpi).
"""
import os, sys, json, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim", "code"))       # the author's integrators, unmodified
import figstyle as S
from catabult_sim import simulate
from catabult_sim_elastic import simulate_elastic
from gear_fn_bridge import make_gear_fn
S.use()

G = 9.8
CASES = {"100to1":   dict(M=100.,   F=999.0,   label="100:1"),
         "1000to1":  dict(M=1000.,  F=3169.6,  label="1,000:1"),
         "10000to1": dict(M=10000., F=10033.8, label="10,000:1")}
V0, VSTOP, KR = 10.0, 4.0, 16471.0
H0 = V0**2 / (2 * G)
LOCKED = json.load(open(os.path.join(HERE, "..", "sim", "locked_4ms.json")))
DUAL = json.load(open(os.path.join(HERE, "..", "sim", "paper2", "designs_16471.json")))
_cache = {}


# ------------------------------------------------------------------ runs
def ideal_run(M, F, v_stop=VSTOP):
    key = ("ideal", M, F, v_stop)
    if key not in _cache:
        _cache[key] = simulate(M, 1., H0, get_gear_fn=F, min_heavy_speed=v_stop, max_time=.6,
                               max_target_acc=1e9, time_unit=1e-5)
    return _cache[key]


def rigid_run(M, gf, dmax):
    return simulate(M, 1., H0, get_gear_fn=gf, max_heavy_dist=dmax, max_time=.5, max_target_acc=3e4, time_unit=1e-5)


def compliant_run(M, F, gf, dmax):
    return simulate_elastic(M, 1., H0, gf, k_rope=KR, max_heavy_dist=dmax, pretension=F, time_unit=2e-5)


def single_design(key):
    L = LOCKED[key]
    R, s, k = np.array(L["R"]), np.array(L["s"]), float(L["k"])
    return R, s, k


def dual_gear(tag):
    d = DUAL[tag]["dual"]; nlo, nhi = DUAL[tag]["n_lo"], DUAL[tag]["n_hi"]
    P = [(np.array(d["R_lo"]), np.array(d["s_lo"]), nlo), (np.array(d["R_hi"]), np.array(d["s_hi"]), nhi)]
    def f(dd):
        t = 0.0
        for R, s, n in P:
            D = np.maximum(0.0, dd - s)
            t += n * float(np.sum(2 * D / np.sqrt(R * R + D * D)))
        return t
    return f


def array_ratio(dd, R, s):
    D = np.maximum(dd[:, None] - s[None, :], 0.0)
    return np.sum(2 * D / np.sqrt(R[None, :]**2 + D**2), axis=1)


# ------------------------------------------------------------------ helpers
def robust_ylim(ax, series, lo_fix=None, floor_hi=None, pad=0.06, headroom=0.0):
    v = np.concatenate([np.asarray(x, float).ravel() for x in series if x is not None])
    v = v[np.isfinite(v)]
    lo, hi = np.percentile(v, [0.5, 99.0])
    span = hi - lo
    lo, hi = lo - pad * span, hi + (pad + headroom) * span
    if floor_hi is not None: hi = max(hi, floor_hi)
    if lo_fix is not None: lo = lo_fix
    ax.set_ylim(lo, hi)
    if v.max() > hi:
        ax.text(0.98, 0.95, "max %s (off scale)" % format(v.max(), ",.0f"), transform=ax.transAxes,
                ha="right", va="top", fontsize=6.5, color=S.RED)


def six_panel(runs, path, f_ref, styles, labels, from_zero=True, p2m_note=True, force_only=()):
    """runs: list of run dicts, the first the ideal. 3x2 layout of the manuscript; forces in kN.
    force_only: extra (run, style, label) triples drawn on the two force panels only, beneath the main traces."""
    PAN = [("heavy_dist", "source braking distance", "m"), ("gear", "net displacement ratio", "ratio"),
           ("heavy_speed", "source speed", "m/s"), ("target_speed", "payload speed", "m/s"),
           ("force_heavy", "force on the source", "kN"), ("force_target", "force on the payload", "kN")]
    fig, axes = plt.subplots(3, 2, figsize=(S.W_FULL, 6.4))
    fk = f_ref / 1e3
    for ax, (key, ttl, unit), letter in zip(axes.ravel(), PAN, "abcdef"):
        scale = 1e-3 if key.startswith("force") else 1.0
        for r, st, lab in zip(runs, styles, labels):
            ax.plot(r["t"] * 1e3, np.asarray(r[key]) * scale, label=lab, **st)
        S.panel(ax, letter, ttl); ax.set_ylabel(unit)
        if key in ("force_heavy", "force_target"):
            for r, st, lab in force_only:
                ax.plot(r["t"] * 1e3, np.asarray(r[key]) * scale, label=lab, zorder=1.5, **st)
            vv = [np.asarray(r[key]) * scale for r in runs[1:]] or [np.asarray(runs[0][key]) * scale]
            peak = max(float(np.nanmax(v)) for v in vv)
            if key == "force_target":
                mult = 3.0 if peak > 2.0 * fk else 2.0
                robust_ylim(ax, vv, lo_fix=0.0 if from_zero else None, floor_hi=1.08 * mult * fk)
                top = ax.get_ylim()[1]
                for m_ in (2.0, 3.0):
                    if m_ * fk <= top:
                        ax.axhline(m_ * fk, color=S.LIGHT, ls=(0, (4, 3)), lw=0.7)
                        ax.text(0.01, m_ * fk, " %.0fx design force" % m_, transform=ax.get_yaxis_transform(),
                                fontsize=6.2, color=S.GREY, va="bottom")
                ax.axhline(fk, color=S.GREY, ls="-.", lw=0.7)
                ax.text(0.01, fk, " design force", transform=ax.get_yaxis_transform(), fontsize=6.2, color=S.GREY, va="bottom")
            else:
                robust_ylim(ax, vv, lo_fix=0.0 if from_zero else None, headroom=0.3)
        ax.axvline(runs[1]["t"][-1] * 1e3 if len(runs) > 1 else runs[0]["t"][-1] * 1e3, color=S.ORANGE, ls=":", lw=0.9)
    for ax in axes[-1]: ax.set_xlabel("time (ms)")
    axes[0][0].legend(loc="upper left")
    if force_only:
        axes[2][0].legend(loc="upper left")
    if p2m_note and len(runs) > 1:
        notes = []
        for r, lab in zip(runs[1:], labels[1:]):
            Fc = np.asarray(r["force_target"], float); Fc = Fc[:max(1, int(len(Fc) * 0.995))]
            notes.append("%s: peak/design %.2f" % (lab, Fc.max() / f_ref))
        for r, st, lab in force_only:
            Fr = np.asarray(r["force_target"], float); Fr = Fr[:max(1, int(len(Fr) * 0.995))]
            notes.append("%s: peak/mean %.2f" % (lab, Fr.max() / Fr.mean()))
        axes[-1][-1].text(0.02, 0.04, "\n".join(notes), transform=axes[-1][-1].transAxes, fontsize=6.5, va="bottom",
                          bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    fig.tight_layout(h_pad=1.0, w_pad=1.2)
    S.save(fig, path)


# ------------------------------------------------------------------ figures
def fig_targets():
    """Ratio profiles for uniform payload force, three mass ratios, two durations (Fig. 3)."""
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(3, 3, figsize=(S.W_FULL, 3.7), sharex="col")
    cols = [("heavy_speed", "source velocity (m/s)"), ("gear", "required ratio"), ("target_speed", "payload velocity (m/s)")]
    for i, key in enumerate(("100to1", "1000to1", "10000to1")):
        c = CASES[key]
        r1 = ideal_run(c["M"], c["F"])
        # the force that stretches the same deceleration over 0.2 s, found by bisection on the stroke duration
        lo, hi = 0.2 * c["F"], c["F"]
        for _ in range(30):
            mid = 0.5 * (lo + hi); T = ideal_run(c["M"], mid)["t"][-1]
            lo, hi = (mid, hi) if T > 0.2 else (lo, mid)
        r2 = ideal_run(c["M"], 0.5 * (lo + hi))
        for j, (k_, lab) in enumerate(cols):
            ax = axes[i][j]
            ax.plot(r1["t"] * 1e3, r1[k_], color=S.BLACK, lw=1.2, label="0.1 s transfer")
            ax.plot(r2["t"] * 1e3, r2[k_], color=S.BLUE, lw=1.2, ls=(0, (4, 1.5)), label="0.2 s transfer")
            if i == 0: ax.set_title(lab, loc="center")
            if j == 0: ax.set_ylabel(c["label"] + "\n" + "m/s")
            elif j == 1: ax.set_ylabel("ratio")
            else: ax.set_ylabel("m/s")
            if i == 2: ax.set_xlabel("time (ms)")
            ax.set_xlim(0, 210)
            ax.yaxis.set_major_locator(MaxNLocator(4))
    axes[0][2].legend(loc="lower right")
    fig.tight_layout(h_pad=0.4, w_pad=1.0)
    S.save(fig, "fig_targets")

def _draw_single_array(ax, R, s, dmax):
    """One array drawn to scale: supports, members at their deepest deflection, the tension member at five points
    of the stroke, and the span widths dimensioned beneath. Returns the array width. The dual-array counterpart is
    _draw_dual; the two share their conventions."""
    order = np.argsort(s); R = np.asarray(R)[order]; s = np.asarray(s)[order]
    sup = np.concatenate([[0.0], np.cumsum(2 * R)]); pin = sup[:-1] + R; W = sup[-1]
    deep = np.max(dmax - s); yc = 0.0; yb = -(deep + 0.10 * dmax); yd = yb - 0.10 * dmax
    for y in (yc, yb): ax.plot([0, W], [y, y], color="0.15", lw=2.2, solid_capstyle="butt", zorder=4)
    for x in sup:
        ax.plot([x, x], [yb, yc], color="0.55", lw=1.1, zorder=2)
        ax.plot(x, yc, "o", ms=2.6, mfc="white", mec="0.2", mew=0.6, zorder=6)
    for fr, g_, ls in zip(S.STROKE_FRACS, S.STROKE_GREYS, S.STROKE_LS):
        d = fr * dmax; pts = [(sup[0], 0.0)]
        for i in range(len(R)):
            Dp = max(0.0, d - s[i])
            if Dp > 0: pts.append((pin[i], -Dp))
            pts.append((sup[i + 1], 0.0))
        ax.plot([q[0] for q in pts], [q[1] for q in pts], color=g_, ls=ls, lw=1.0, zorder=3, label="%d%%" % (100 * fr))
    for i in range(len(R)):
        Dp = max(0.0, dmax - s[i])
        ax.plot(pin[i], -Dp, "o", ms=3.4, mfc="white", mec="black", mew=0.7, zorder=7)
        tight = (i + 1 < len(R) and abs(pin[i + 1] - pin[i]) < 0.11 * W) or (i > 0 and abs(pin[i] - pin[i - 1]) < 0.11 * W)
        dx, dy = (0, -9 if i % 2 else -17) if tight else (0, -9)   # stagger downward: the carriage bar is above
        ax.annotate("%d" % (i + 1), (pin[i], -Dp), textcoords="offset points", xytext=(dx, dy), ha="center", va="center", fontsize=5.5, zorder=12,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.9) if tight else None)
    widths = 2 * R; roomy = widths >= 0.11 * W; i = 0
    while i < len(R):
        if roomy[i]:
            ax.annotate("", (sup[i], yd), (sup[i + 1], yd), arrowprops=dict(arrowstyle="<->", lw=0.5, color="0.35"))
            ax.text(pin[i], yd - 0.05 * dmax, "%.2f" % widths[i], ha="center", va="top", fontsize=5.5, color="0.3"); i += 1
        else:
            j = i
            while j < len(R) and not roomy[j]: j += 1
            tick = 0.022 * dmax
            ax.plot([sup[i], sup[j]], [yd, yd], color="0.35", lw=0.5, zorder=3)
            for x in (sup[i], sup[j]): ax.plot([x, x], [yd - tick, yd + tick], color="0.35", lw=0.5, zorder=3)
            lo_w, hi_w = widths[i:j].min(), widths[i:j].max()
            txt = "%d spans of %.2f" % (j - i, lo_w) if hi_w - lo_w < 0.005 else "%d spans, %.2f\u2013%.2f" % (j - i, lo_w, hi_w)
            ax.text(0.5 * (sup[i] + sup[j]), yd - 0.05 * dmax, txt, ha="center", va="top", fontsize=5.3, color="0.3"); i = j
    ax.text(0.5 * W, yd - 0.17 * dmax, "span widths (m); total array %.2f m" % W, ha="center", va="top", fontsize=6)
    ax.set_xlim(-0.04 * W, 1.04 * W); ax.set_ylim(yd - 0.30 * dmax, yc + 0.10 * dmax)
    ax.set_aspect("equal"); ax.axis("off")
    return W


def fig_comb(key):
    """Single-array reference drawn to scale, with its design variables and ratio profile."""
    c = CASES[key]; R, s, k = single_design(key)
    I = ideal_run(c["M"], c["F"]); dmax = I["summary"]["heavy_travel"]
    dd = np.linspace(0, dmax, 400); Gs = np.interp(dd, I["heavy_dist"], I["gear"])
    N = len(R)
    fig = plt.figure(figsize=(S.W_FULL, 4.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], hspace=0.55, wspace=0.35, left=0.02, right=0.98, top=0.92, bottom=0.1)
    ax = fig.add_subplot(gs[:, 0]); W = _draw_single_array(ax, R, s, dmax)
    ax.legend(title="tension member at % of stroke", ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=0.9)
    ax.set_title("%s: %d members, $k$ = %g, stroke %.2f m" % (c["label"], N, k, dmax), loc="left", pad=26, fontsize=8.5)
    axL = fig.add_subplot(gs[0, 1]); order = np.argsort(s)
    axL.bar(np.arange(1, N + 1), 2 * R[order], 0.62, facecolor="white", edgecolor="black", lw=0.7, hatch=S.HATCH_A, label="span width $2R_i$")
    axL.set_xlabel("engagement order"); axL.set_ylabel("span width $2R_i$ (m)"); S.panel(axL, "a", "design variables in engagement order")
    axL.set_xticks(np.arange(1, N + 1))
    ax2 = S.twin_ok(axL.twinx()); ax2.plot(np.arange(1, N + 1), s[order], color=S.BLUE, marker="o", ms=3, lw=1.0, mfc="white", label="offset $s_i$")
    ax2.set_ylabel("engagement offset $s_i$ (m)", color=S.BLUE); ax2.tick_params(axis="y", colors=S.BLUE)
    h1, l1 = axL.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); axL.legend(h1 + h2, l1 + l2, loc="center right")
    axR = fig.add_subplot(gs[1, 1])
    axR.plot(dd, Gs, label="target profile $G^*$", **S.IDEAL)
    axR.plot(dd, k * array_ratio(dd, R, s), label="this array, $k\\,G_{\\mathrm{array}}$", **S.DESIGN)
    for si in s: axR.axvline(si, color="0.9", lw=0.5, zorder=0)
    axR.set_xlabel("carriage displacement $d$ (m)"); axR.set_ylabel("net ratio"); S.panel(axR, "b", "ratio profile against the target")
    axR.legend(loc="upper left")
    S.save(fig, "fig_comb_" + key)


def fig_runs_single(key, mode):
    c = CASES[key]; R, s, k = single_design(key)
    I = ideal_run(c["M"], c["F"]); dmax = I["summary"]["heavy_travel"]; gf = make_gear_fn(R, s, k)
    if mode == "rigid":
        r = rigid_run(c["M"], gf, dmax)
        six_panel([I, r], "fig_rigid_%s" % key, c["F"], [S.IDEAL, S.DESIGN], ["ideal", "this array, rigid member"])
    else:
        r = compliant_run(c["M"], c["F"], gf, dmax); rr = rigid_run(c["M"], gf, dmax)
        six_panel([I, r], "fig_compliant_%s" % key, c["F"], [S.IDEAL, S.DESIGN], ["ideal", "this array, compliant member"],
                  force_only=[(rr, dict(color=S.GREEN, lw=0.7, ls="-"), "this array, rigid member")])


def fig_instability():
    c = CASES["1000to1"]; R, s, k = single_design("1000to1")
    I = ideal_run(c["M"], c["F"]); dmax = I["summary"]["heavy_travel"]
    r = rigid_run(c["M"], make_gear_fn(R, s, k), dmax)
    t = r["t"] * 1e3; F = r["force_target"]; vt = r["target_speed"]; vrope = r["gear"] * r["heavy_speed"]; slack = r["slack"]
    mean_F = r["summary"]["mean_force_target"]; t_end = t[-1]
    tail = (t_end - 2.0, t_end); mid_c = 0.62 * t_end; mid = (mid_c - 6.0, mid_c + 6.0)
    fig = plt.figure(figsize=(S.W_FULL, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1], hspace=0.55, wspace=0.42)
    axA = fig.add_subplot(gs[0, :]); axB = fig.add_subplot(gs[1, 0]); axC = fig.add_subplot(gs[1, 1]); axD = fig.add_subplot(gs[1, 2])
    axA.plot(t, F, color=S.BLACK, lw=0.7)
    axA.axhline(mean_F, color=S.GREY, ls="-.", lw=0.8, label="mean %s N" % format(mean_F, ",.0f"))
    axA.set_ylim(0, max(np.percentile(F, 99) * 1.3, 2.2 * mean_F))
    axA.text(0.99, 0.95, "peak %s N (off scale)" % format(F.max(), ",.0f"), transform=axA.transAxes, fontsize=6.5, ha="right", va="top", color=S.RED)
    for (lo, hi), col, lbl in ((mid, S.BLUE, "(b)"), (tail, S.RED, "(c), (d)")):
        axA.axvspan(lo, hi, color=col, alpha=0.12, lw=0)
        axA.text(0.5 * (lo + hi), axA.get_ylim()[1] * 0.06, lbl, ha="center", fontsize=7, color=col)
    axA.set_xlabel("time (ms)"); axA.set_ylabel("force on payload (N)"); S.panel(axA, "a", "payload force, whole stroke"); axA.legend(loc="upper left")
    m = (t >= mid[0]) & (t <= mid[1]); w = (t >= tail[0]) & (t <= tail[1])
    axB.plot(t[m], F[m], color=S.BLACK, lw=0.9); axB.axhline(mean_F, color=S.GREY, ls="-.", lw=0.8)
    axB.set_xlabel("time (ms)"); axB.set_ylabel("force on payload (N)"); S.panel(axB, "b", "mid-stroke: bounded")
    axC.plot(t[w], F[w], color=S.BLACK, lw=0.9); axC.axhline(mean_F, color=S.GREY, ls="-.", lw=0.8)
    if slack[w].any():
        axC.axvline(t[w][slack[w]][0], color=S.ORANGE, ls=":", lw=1.0, label="first slack"); axC.legend(loc="upper left")
    axC.set_xlabel("time (ms)"); S.panel(axC, "c", "terminal: pulses diverge"); axC.set_ylabel("force on payload (N)")
    axD.plot(t[w], vrope[w], color=S.BLUE, lw=1.0, label="rope tip $G_{\\mathrm{net}}\\,v_h$")
    axD.plot(t[w], vt[w], color=S.BLACK, lw=1.0, label="payload $v_t$")
    over = vt[w] > vrope[w]
    if over.any(): axD.fill_between(t[w], vrope[w], vt[w], where=over, color=S.RED, alpha=0.3, lw=0, label="payload outrunning rope")
    axD.set_xlabel("time (ms)"); axD.set_ylabel("speed (m/s)"); S.panel(axD, "d", "mechanism"); axD.legend(loc="upper left")
    S.save(fig, "fig_instability")


def fig_target_selection():
    """SI S10: the deceleration-target sweep, from code/_decel4_*.json."""
    fig, axes = plt.subplots(1, 3, figsize=(S.W_FULL, 2.2))
    sty = {"100": dict(color=S.BLUE, marker="o"), "1000": dict(color=S.RED, marker="s"), "10000": dict(color=S.GREEN, marker="^")}
    for tag in ("100", "1000", "10000"):
        d = json.load(open(os.path.join(HERE, "code", "_decel4_%s.json" % tag)))
        rows = sorted(d.values(), key=lambda r: -r["stop"])
        st = [r["stop"] for r in rows]
        ref = [r for r in rows if abs(r["stop"] - 3.0) < 1e-6][0]["exit"]
        kw = dict(ms=3.5, lw=1.0, mfc="white", label=CASES[tag + "to1"]["label"], **sty[tag])
        axes[0].plot(st, [r["peak"] for r in rows], **kw)
        axes[1].plot(st, [100 * r["slack_frac"] for r in rows], **kw)
        axes[2].plot(st, [100 * r["exit"] / ref for r in rows], **kw)
    axes[0].axhline(1.3, color=S.GREY, ls=":", lw=0.8); axes[0].text(6.0, 1.305, "1.3 criterion", fontsize=6.5, color=S.GREY, va="bottom")
    for ax, (letter, ttl, yl) in zip(axes, [("a", "peak force, compliant", "peak / design force"), ("b", "traction loss", "stroke with member unloaded (%)"),
                                            ("c", "exit velocity", "% of the 3 m/s value")]):
        S.panel(ax, letter, ttl); ax.set_ylabel(yl); ax.set_xlabel("specified source final speed (m/s)"); ax.invert_xaxis()
    axes[0].legend(loc="upper left")
    fig.tight_layout(w_pad=1.2)
    S.save(fig, "fig_target_selection")


def fig_search_vs_fit():
    d = json.load(open(os.path.join(HERE, "code", "_searchfit_data.json")))
    fig, ax = plt.subplots(figsize=(S.W_HALF + 0.6, 2.5))
    ax.plot(d["ns"], d["curve"], color=S.BLACK, marker="o", ms=3.5, mfc="white", lw=1.1, label="random search")
    ax.plot([d["gc"]], [d["grid"]], marker="s", ms=6, color=S.RED, ls="none", label="structured grid, 4 parameters (9,600)")
    ax.axhline(d["best"], color=S.BLUE, ls=(0, (4, 1.5)), lw=1.1, label="least squares, ~2,000 evaluations")
    ax.plot([2000], [d["best"]], marker="*", ms=8, color=S.BLUE, ls="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("objective evaluations"); ax.set_ylabel("best rms ratio error (% of full scale)")
    ax.legend(loc="center left", fontsize=6.5)
    fig.tight_layout()
    S.save(fig, "fig_search_vs_fit")


def fig_member():
    """SI: the single-member ratio curve, recomposed under the existing artwork of fig04 panel (a)."""
    from PIL import Image
    x = np.linspace(0, 3.5, 400); y = 2 * x / np.sqrt(1 + x**2)
    fig, ax = plt.subplots(figsize=(S.W_FULL, 1.9))
    ax.plot(x, y, color=S.BLACK, lw=1.3)
    ax.axhline(2, color=S.GREY, ls=":", lw=0.8); ax.text(3.45, 2.02, "asymptote $dY/dD = 2$", ha="right", va="bottom", fontsize=6.5, color=S.GREY)
    ax.set_xlabel("normalised depth $D/R$"); ax.set_ylabel("$dY/dD$"); ax.set_ylim(0, 2.25); ax.set_xlim(0, 3.5)
    S.panel(ax, "b", "the ratio one member contributes, against normalised depth")
    fig.tight_layout()
    pb = S.save(fig, "_member_curve", png=True)
    # artwork band of the original figure: locate by ink profile, take everything above the old chart
    src = Image.open(os.path.join(HERE, "..", "figs", "drawn", "fig04_single_member.png")).convert("RGB")
    a = np.asarray(src); ink = (a < 200).any(axis=2).mean(axis=1)
    rows = np.where(ink > 0.002)[0]
    # the old chart is the last large ink block; cut at the largest blank gap in the lower half
    blank = ink < 0.0005; cut = None; best = 0; i = int(0.45 * len(blank))
    while i < len(blank):
        if blank[i]:
            j = i
            while j < len(blank) and blank[j]: j += 1
            if j - i > best: best, cut = j - i, (i, j)
            i = j
        else: i += 1
    art = src.crop((0, rows[0] - 10, src.width, cut[0] + 5))
    curve = Image.open(os.path.join(HERE, "..", "figs", "_member_curve.png")).convert("RGB")
    curve = curve.resize((art.width, int(curve.height * art.width / curve.width)), Image.LANCZOS)
    out = Image.new("RGB", (art.width, art.height + curve.height + 30), "white")
    out.paste(art, (0, 0)); out.paste(curve, (0, art.height + 30))
    out.save(os.path.join(HERE, "..", "figs", "fig_member.png"))
    for f in ("_member_curve.png", "_member_curve.pdf"):
        os.remove(os.path.join(HERE, "..", "figs", f))
    print("wrote fig_member.png", out.size)


# ------------------------------------------------------------------ dual-array figures
def fig_parity():
    ks = np.array([3, 5, 7, 9, 11]); n1 = ks // 2; n2 = (ks + 1) // 2
    fig, (a, b) = plt.subplots(1, 2, figsize=(S.W_FULL, 2.3))
    a.bar(ks, 100 * (n2 - n1) / ks, 1.1, facecolor="white", edgecolor="black", lw=0.7, hatch=S.HATCH_A)
    for k_, v in zip(ks, 100 * (n2 - n1) / ks): a.text(k_, v + 0.8, "%.0f%%" % v, ha="center", fontsize=6.5)
    a.set_xlabel("fixed-stage ratio $k$"); a.set_ylabel("reaction imbalance, mirror pair (% of reaction)"); a.set_xticks(ks); a.set_ylim(0, 40)
    S.panel(a, "a", "mirror-symmetric arrays are unbalanced at odd $k$")
    b.plot(ks, n2 / n1, color=S.BLACK, marker="o", ms=4, mfc="white", lw=1.0)
    for k_, v, x, y in zip(ks, n2 / n1, n1, n2): b.annotate("%d:%d" % (y, x), (k_, v), textcoords="offset points", xytext=(6, 3), fontsize=6.5)
    b.axhline(1.0, color=S.GREY, ls=":", lw=0.8); b.text(11, 1.02, "equal arrays", ha="right", fontsize=6.5, color=S.GREY)
    b.set_xlabel("fixed-stage ratio $k$"); b.set_ylabel("required $G_1/G_2 = n_2/n_1$"); b.set_xticks(ks); b.set_ylim(0.9, 2.15)
    S.panel(b, "b", "balance requires unequal arrays")
    fig.tight_layout(w_pad=2.0)
    S.save(fig, "fig_parity")


def fig_compare():
    tags = ("k7", "k9"); xl = ["$N$=7, $k$=7", "$N$=5, $k$=9"]
    sg = [DUAL[t]["single"] for t in tags]; du = [DUAL[t]["dual"] for t in tags]
    fig, axes = plt.subplots(1, 3, figsize=(S.W_FULL, 2.3))
    x = np.arange(2); w = 0.36
    def bars(ax, vs, vd):
        ax.bar(x - w / 2, vs, w, facecolor="white", edgecolor="black", lw=0.7, hatch=S.HATCH_B, label="one-line single array")
        ax.bar(x + w / 2, vd, w, facecolor="white", edgecolor="black", lw=0.7, hatch=S.HATCH_A, label="unequal pair (busiest line)")
        for xi, v in zip(x - w / 2, vs): ax.text(xi, v, "%.2f" % v if v < 10 else "%d" % v, ha="center", va="bottom", fontsize=6.2)
        for xi, v in zip(x + w / 2, vd): ax.text(xi, v, "%.2f" % v if v < 10 else "%d" % v, ha="center", va="bottom", fontsize=6.2)
        ax.set_xticks(x); ax.set_xticklabels(xl)
    bars(axes[0], [s["pR"] for s in sg], [d["pR"] for d in du]); axes[0].set_ylabel("peak / mean payload force"); S.panel(axes[0], "a", "rigid member"); axes[0].set_ylim(0, 3.4)
    bars(axes[1], [s["pC"] for s in sg], [d["pC"] for d in du]); axes[1].set_ylabel("peak / mean payload force"); S.panel(axes[1], "b", "compliant member"); axes[1].set_ylim(0, 2.6)
    axes[1].axhline(1.3, color=S.GREY, ls=":", lw=0.8); axes[1].text(0.5, 1.32, "1.3 bound", fontsize=6.2, color=S.GREY, va="bottom", ha="center")
    bars(axes[2], [s["contacts"] for s in sg], [max(d["contacts"]) for d in du]); axes[2].set_ylabel("elements per tension member"); S.panel(axes[2], "c", "sheave count along each path"); axes[2].set_ylim(0, 30)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=1.4, rect=(0, 0.06, 1, 1))
    S.save(fig, "fig_compare")


def fig_trade():
    """SI: (a) the k=7 lambda sweep of the lambda-sweep table; (b) the member floor against k."""
    lam = [0.03, 0.1, 0.3]; imb = [8.1, 7.2, 6.4]; p2m = [1.12, 1.14, 1.23]
    fig, (a, b) = plt.subplots(1, 2, figsize=(S.W_FULL, 2.3))
    a.plot(imb, p2m, color=S.BLACK, marker="o", ms=4, mfc="white", lw=1.0)
    for l, i, p in zip(lam, imb, p2m): a.annotate("$\\lambda$ = %g" % l, (i, p), textcoords="offset points", xytext=(5, -9), fontsize=6.5)
    a.axhline(1.3, color=S.GREY, ls=":", lw=0.8); a.text(8.15, 1.295, "1.3 bound", fontsize=6.5, color=S.GREY, ha="right", va="top")
    a.set_xlabel("peak reaction imbalance (% of reaction)"); a.set_ylabel("peak / mean payload force, compliant"); a.set_ylim(1.08, 1.32)
    S.panel(a, "a", "balance trades against force uniformity")
    ks = np.array([5, 7, 9, 11]); Gt = DUAL["_meta"]["target_terminal"]
    floors = np.array([int(np.ceil(Gt / (2 * (kk // 2)) / 2)) for kk in ks])
    b.plot(ks, floors, color=S.BLACK, marker="s", ms=4, mfc="white", lw=1.0)
    for kk, f in zip(ks, floors): b.annotate("%d" % f, (kk, f), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6.5)
    b.set_xlabel("fixed-stage ratio $k$"); b.set_ylabel("minimum members, leading array"); b.set_xticks(ks); b.set_ylim(3, 11)
    S.panel(b, "b", "raising $k$ lowers the member floor")
    fig.tight_layout(w_pad=2.0)
    S.save(fig, "fig_trade")


def fig_runs_dual(mode):
    c = CASES["1000to1"]; F = c["F"]; dmax = DUAL["_meta"]["d_max"]
    I = ideal_run(c["M"], F)
    runs = [I]
    for tag in ("k7", "k9"):
        gf = dual_gear(tag)
        runs.append(rigid_run(c["M"], gf, dmax) if mode == "rigid" else compliant_run(c["M"], F, gf, dmax))
    six_panel(runs, "fig_runs_" + mode, F, [S.IDEAL, S.DESIGN, S.SECOND], ["ideal", "$\\langle 7,7\\rangle$, $k$ = 7", "$\\langle 5,9\\rangle$, $k$ = 9"])


def _dual_xlim(Wl, Wh):
    return (-Wl * 1.12, Wh * 1.12)


def _dual_ylim(yd, yc, dmax):
    return (yd - 0.42 * dmax, yc + 0.46 * dmax)


def _shift_gs(gs):
    return gs


def _draw_dual(ax, Rl, sl, Rh, sh, dmax, nlo, nhi):
    def one(R, s, sign, label):
        order = np.argsort(s); R = np.asarray(R)[order]; s = np.asarray(s)[order]
        sup = np.concatenate([[0.0], np.cumsum(2 * R)]); pin = sup[:-1] + R; W = sup[-1]
        deep = max(np.max(dmax - sl), np.max(dmax - sh)); yc = 0.20 * dmax; yb = -(deep + 0.10 * dmax); yd = yb - 0.10 * dmax
        for y in (yc, yb): ax.plot([0, sign * W], [y, y], color="0.15", lw=2.2, solid_capstyle="butt", zorder=4)
        for x in sup:
            ax.plot([sign * x, sign * x], [yb, 0], color="0.55", lw=1.1, zorder=2)
            ax.plot(sign * x, 0, "o", ms=2.6, mfc="white", mec="0.2", mew=0.6, zorder=6)
        for f, g_, ls in zip(S.STROKE_FRACS, S.STROKE_GREYS, S.STROKE_LS):
            d = f * dmax; pts = [(sign * sup[0], 0.0)]
            for i in range(len(R)):
                Dp = max(0.0, d - s[i])
                if Dp > 0: pts.append((sign * pin[i], -Dp))
                pts.append((sign * sup[i + 1], 0.0))
            ax.plot([q[0] for q in pts], [q[1] for q in pts], color=g_, ls=ls, lw=1.0, zorder=3, label=("%d%%" % (100 * f)) if sign > 0 else None)
        for i in range(len(R)):
            Dp = max(0.0, dmax - s[i])
            ax.plot([sign * pin[i]] * 2, [yc, -Dp], color="0.45", lw=1.0, zorder=2)
            ax.plot(sign * pin[i], -Dp, "o", ms=3.4, mfc="white", mec="black", mew=0.7, zorder=7)
            tight = (i + 1 < len(R) and abs(pin[i + 1] - pin[i]) < 0.11 * W) or (i > 0 and abs(pin[i] - pin[i - 1]) < 0.11 * W)
            dx, dy = (0, 10 if i % 2 else -10) if tight else (sign * 7, 0)
            ax.annotate("%d" % (i + 1), (sign * pin[i], -Dp), textcoords="offset points", xytext=(dx, dy), ha="center", va="center", fontsize=5.5, zorder=12,
                        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.9) if tight else None)
        widths = 2 * R; roomy = widths >= 0.115 * W; i = 0
        while i < len(R):
            if roomy[i]:
                ax.annotate("", (sign * sup[i], yd), (sign * sup[i + 1], yd), arrowprops=dict(arrowstyle="<->", lw=0.5, color="0.35"))
                ax.text(sign * pin[i], yd - 0.05 * dmax, "%.2f" % widths[i], ha="center", va="top", fontsize=5.5, color="0.3"); i += 1
            else:
                j = i
                while j < len(R) and not roomy[j]: j += 1
                tick = 0.022 * dmax; a_, b_ = sign * sup[i], sign * sup[j]
                ax.plot([a_, b_], [yd, yd], color="0.35", lw=0.5, zorder=3)
                for x in (a_, b_): ax.plot([x, x], [yd - tick, yd + tick], color="0.35", lw=0.5, zorder=3)
                lo_w, hi_w = widths[i:j].min(), widths[i:j].max()
                txt = "%d spans of %.2f" % (j - i, lo_w) if hi_w - lo_w < 0.005 else "%d spans, %.2f–%.2f" % (j - i, lo_w, hi_w)
                ax.text(sign * 0.5 * (sup[i] + sup[j]), yd - 0.125 * dmax, txt, ha="center", va="top", fontsize=5.3, color="0.3"); i = j
        ax.text(sign * W * 0.5, yd - 0.25 * dmax, "%s\n%d members, %.2f m" % (label, len(R), W), ha="center", va="top", fontsize=6)
        return W, yb, yc, yd
    Wl, yb, yc, yd = one(Rl, sl, -1, "array 1, leads (%d falls)" % nlo)
    Wh, _, _, _ = one(Rh, sh, +1, "array 2 (%d falls)" % nhi)
    ax.plot([0, 0], [yb, yc + 0.05 * dmax], color="0.15", lw=2.6, zorder=8)
    ax.plot(0, 0, "s", ms=4.5, mfc="black", mec="black", zorder=9)
    ax.text(0, yc + 0.08 * dmax, "anchors on the\nstationary guide", ha="center", va="bottom", fontsize=5.8, zorder=10)
    ax.set_xlim(*_dual_xlim(Wl, Wh)); ax.set_ylim(*_dual_ylim(yd, yc, dmax)); ax.set_aspect("equal"); ax.axis("off")
    hs, ls = ax.get_legend_handles_labels(); order = [0, 3, 1, 4, 2]     # row-major reading order in three columns
    ax.legend([hs[i] for i in order], [ls[i] for i in order], title="tension member at % of stroke", ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.0), handlelength=1.6, columnspacing=0.9, borderaxespad=0.0)


def fig_design(tag):
    d = DUAL[tag]["dual"]; nlo, nhi = DUAL[tag]["n_lo"], DUAL[tag]["n_hi"]; k = DUAL[tag]["k"]
    Rl, sl, Rh, sh = [np.array(d[x]) for x in ("R_lo", "s_lo", "R_hi", "s_hi")]
    dmax = DUAL["_meta"]["d_max"]; c = CASES["1000to1"]
    I = ideal_run(c["M"], c["F"]); dd = np.linspace(0, dmax, 600); Gs = np.interp(dd, I["heavy_dist"], I["gear"])
    gl = array_ratio(dd, Rl, sl); gh = array_ratio(dd, Rh, sh); net = nlo * gl + nhi * gh
    # top panel sized to the drawing's own aspect, so no white space is left above or below it
    Wl, Wh = 2 * Rl.sum(), 2 * Rh.sum(); deep = max(np.max(dmax - sl), np.max(dmax - sh))
    yc = 0.20 * dmax; yb = -(deep + 0.10 * dmax); yd = yb - 0.10 * dmax
    (x0, x1), (y0, y1) = _dual_xlim(Wl, Wh), _dual_ylim(yd, yc, dmax)
    w_top = 0.98 * S.W_FULL; h_top = w_top * (y1 - y0) / (x1 - x0)
    h_row, gap, top_pad, bot_pad = 1.55, 0.62, 0.05, 0.42       # inches: panel rows, row gap (titles + x labels), margins
    H = top_pad + h_top + 0.32 + 2 * h_row + gap + bot_pad
    fig = plt.figure(figsize=(S.W_FULL, H))
    ax = fig.add_axes([0.01, 1 - (top_pad + h_top) / H, 0.98, h_top / H]); _draw_dual(ax, Rl, sl, Rh, sh, dmax, nlo, nhi)
    gs = fig.add_gridspec(2, 2, hspace=gap / h_row, wspace=0.34, left=0.09, right=0.98, top=(2 * h_row + gap + bot_pad) / H, bottom=bot_pad / H)
    gs = _shift_gs(gs)
    # (a) spans and offsets
    a = fig.add_subplot(gs[0, 0]); ol, oh = np.argsort(sl), np.argsort(sh); N = max(len(Rl), len(Rh))
    a.bar(np.arange(1, len(Rl) + 1) - 0.19, 2 * Rl[ol], 0.38, facecolor="white", edgecolor="black", lw=0.7, hatch=S.HATCH_A, label="array 1 (%d falls)" % nlo)
    a.bar(np.arange(1, len(Rh) + 1) + 0.19, 2 * Rh[oh], 0.38, facecolor="white", edgecolor="black", lw=0.7, hatch=S.HATCH_B, label="array 2 (%d falls)" % nhi)
    a.set_xlabel("engagement order within the array"); a.set_ylabel("span width $2R_i$ (m)"); a.set_xticks(np.arange(1, N + 1)); a.set_ylim(0, 2 * max(Rl.max(), Rh.max()) * 1.4)
    a2 = S.twin_ok(a.twinx()); a2.plot(np.arange(1, len(sl) + 1), sl[ol], color=S.BLUE, marker="o", ms=3, mfc="white", lw=0.9, label="$s_i$, array 1")
    a2.plot(np.arange(1, len(sh) + 1), sh[oh], color=S.RED, marker="s", ms=3, mfc="white", lw=0.9, ls=(0, (4, 1.5)), label="$s_i$, array 2")
    a2.set_ylabel("engagement offset $s_i$ (m)"); h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, loc="center right", fontsize=6.2); S.panel(a, "a", "span widths and engagement offsets")
    # (b) engagement sequence along the stroke
    b = fig.add_subplot(gs[0, 1])
    b.plot(sl, np.ones_like(sl), ls="none", marker="o", ms=5, mfc="white", mec=S.BLUE, mew=1.0, label="array 1 (leads)")
    b.plot(sh, np.zeros_like(sh), ls="none", marker="s", ms=5, mfc=S.RED, mec=S.RED, label="array 2")
    def number(xs, y, dy):
        xs = np.sort(xs); i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] - xs[j] < 0.035 * dmax: j += 1
            txt = "%d" % (i + 1) if j == i else "%d–%d" % (i + 1, j + 1)
            b.annotate(txt, (0.5 * (xs[i] + xs[j]), y), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=5.8); i = j + 1
    number(sl, 1, 7); number(sh, 0, -11)
    b.set_yticks([0, 1]); b.set_yticklabels(["array 2", "array 1"]); b.set_ylim(-0.8, 1.8); b.set_xlim(-0.02, dmax * 1.02)
    b.set_xlabel("engagement offset $s_i$ (m of carriage travel)"); S.panel(b, "b", "engagements alternate, array 1 leading")
    # (c) weighted contributions
    c_ = fig.add_subplot(gs[1, 0])
    c_.plot(dd, Gs, label="target $G^*$", **S.IDEAL); c_.plot(dd, net, label="$n_1G_1 + n_2G_2$", **S.DESIGN)
    c_.plot(dd, nlo * gl, color=S.BLUE, lw=0.8, ls=(0, (1, 1.2)), label="$n_1G_1$"); c_.plot(dd, nhi * gh, color=S.RED, lw=0.8, ls=(0, (1, 1.2)), label="$n_2G_2$")
    c_.set_xlabel("carriage displacement $d$ (m)"); c_.set_ylabel("ratio"); c_.legend(loc="upper left"); S.panel(c_, "c", "weighted contributions and their sum")
    # (d) reaction imbalance through the stroke
    e = fig.add_subplot(gs[1, 1]); mirror = 100 * abs(nhi - nlo) / k
    with np.errstate(divide="ignore", invalid="ignore"):
        imb = 100 * np.abs(nlo * gl - nhi * gh) / net.max()
    e.plot(dd, imb, color=S.BLACK, lw=1.1, label="this design")
    e.axhline(mirror, color=S.GREY, ls=(0, (4, 2)), lw=0.9, label="mirror-symmetric, same $k$")
    e.set_xlabel("carriage displacement $d$ (m)"); e.set_ylabel("reaction imbalance (% of peak reaction)"); e.set_ylim(0, mirror * 1.6); e.legend(loc="upper left")
    S.panel(e, "d", "reaction imbalance through the stroke")
    S.save(fig, "fig_design_" + tag)


def fig_configurations():
    """Right-hand plot of the comb-shaping figure, recomposed beside the existing C1..C4 drawing."""
    from PIL import Image
    # three-member arrays: C1 equal spans, equal heights; C2 staggered heights; C3 narrowing spans; C4 both
    dmax = 2.5     # in units of R: spans 2R, 2R, 2R (C1, C2) and 2R, 1.5R, 1R (C3, C4), heights staggered by 0.6R in C2 and C4
    cfgs = {"C1": ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]), "C2": ([1.0, 1.0, 1.0], [0.0, 0.6, 1.2]),
            "C3": ([1.0, 0.75, 0.5], [0.0, 0.0, 0.0]), "C4": ([1.0, 0.75, 0.5], [0.0, 0.6, 1.2])}
    dd = np.linspace(0, dmax, 400)
    fig, ax = plt.subplots(figsize=(2.9, 2.9))
    sty = (dict(color=S.BLACK, ls="-"), dict(color=S.BLUE, ls=(0, (4, 1.5))), dict(color=S.RED, ls=(0, (1, 1.2))), dict(color=S.GREEN, ls=(0, (5, 1.5, 1, 1.5))))
    for (name, (R, s)), st in zip(cfgs.items(), sty):
        ax.plot(dd, array_ratio(dd, np.array(R), np.array(s)), lw=1.2, label=name, **st)
    ax.set_xlabel("carriage displacement (units of $R$)"); ax.set_ylabel("array ratio $G_{\\mathrm{array}}$"); ax.set_ylim(0, 6.3); ax.legend(loc="lower right")
    fig.tight_layout()
    S.save(fig, "_configs_plot")
    src = Image.open(os.path.join(HERE, "..", "figs", "drawn", "fig_configurations.png")).convert("RGB")
    a = np.asarray(src); ink = (a < 200).any(axis=2).mean(axis=0)
    # the drawing occupies the left; find the blank column gap before the old plot
    blank = ink < 0.001; best, cut, i = 0, None, int(0.4 * len(blank))
    while i < len(blank):
        if blank[i]:
            j = i
            while j < len(blank) and blank[j]: j += 1
            if j - i > best: best, cut = j - i, (i, j)
            i = j
        else: i += 1
    art = src.crop((0, 0, cut[0] + 10, src.height))
    plot = Image.open(os.path.join(HERE, "..", "figs", "_configs_plot.png")).convert("RGB")
    ph = int(art.height * 0.78); plot = plot.resize((int(plot.width * ph / plot.height), ph), Image.LANCZOS)
    out = Image.new("RGB", (art.width + plot.width + 60, art.height), "white")
    out.paste(art, (0, 0)); out.paste(plot, (art.width + 60, (art.height - plot.height) // 2))
    out.save(os.path.join(HERE, "..", "figs", "fig_configs.png"))
    for f in ("_configs_plot.png", "_configs_plot.pdf"): os.remove(os.path.join(HERE, "..", "figs", f))
    print("wrote fig_configs.png", out.size)


def fig_graphical_abstract(tag="k9", title=None, guide_text=None, out="graphical_abstract"):
    """Elsevier graphical abstract, 13 x 5 cm: the machine DUAL[tag] to scale, its synthesised ratio, and its payload force rigid and compliant.
    Peak-to-mean figures in the annotation are computed from the runs; `guide_text` supplies the reaction-imbalance line."""
    d = DUAL[tag]["dual"]; nlo, nhi = DUAL[tag]["n_lo"], DUAL[tag]["n_hi"]
    Rl, sl, Rh, sh = [np.array(d[x]) for x in ("R_lo", "s_lo", "R_hi", "s_hi")]
    dmax = DUAL["_meta"]["d_max"]; c = CASES["1000to1"]; F = c["F"]
    I = ideal_run(c["M"], F); dd = np.linspace(0, dmax, 600); Gs = np.interp(dd, I["heavy_dist"], I["gear"])
    net = nlo * array_ratio(dd, Rl, sl) + nhi * array_ratio(dd, Rh, sh)
    r = compliant_run(c["M"], F, dual_gear(tag), dmax)
    fig = plt.figure(figsize=(13 / 2.54, 5 / 2.54))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1, 1], wspace=0.3, left=0.01, right=0.985, top=0.86, bottom=0.2)
    # (1) the machine, to scale, tension members at 50 and 100 % of the stroke
    ax = fig.add_subplot(gs[0])
    def one(R, s, sign):
        o = np.argsort(s); R = R[o]; s = s[o]
        sup = np.concatenate([[0.0], np.cumsum(2 * R)]); pin = sup[:-1] + R; W = sup[-1]
        deep = max(np.max(dmax - sl), np.max(dmax - sh)); yc = 0.3 * dmax; yb = -(deep + 0.08 * dmax)
        for y in (yc, yb): ax.plot([0, sign * W], [y, y], color="0.15", lw=1.6, solid_capstyle="butt", zorder=4)
        for x in sup: ax.plot([sign * x] * 2, [yb, 0], color="0.6", lw=0.7, zorder=2)
        for f, col, ls in ((0.5, "0.6", (0, (2, 1))), (1.0, S.BLUE, "-")):
            dd_ = f * dmax; pts = [(sign * sup[0], 0.0)]
            for i in range(len(R)):
                Dp = max(0.0, dd_ - s[i])
                if Dp > 0: pts.append((sign * pin[i], -Dp))
                pts.append((sign * sup[i + 1], 0.0))
            ax.plot([q[0] for q in pts], [q[1] for q in pts], color=col, ls=ls, lw=0.9, zorder=3)
        for i in range(len(R)):
            Dp = max(0.0, dmax - s[i]); ax.plot([sign * pin[i]] * 2, [yc, -Dp], color="0.35", lw=0.7, zorder=2)
            ax.plot(sign * pin[i], -Dp, "o", ms=2.2, mfc="white", mec="black", mew=0.5, zorder=7)
        return W, yb, yc
    Wl, yb, yc = one(Rl, sl, -1); Wh, _, _ = one(Rh, sh, +1)
    ax.plot([0, 0], [yb, yc + 0.2 * dmax], color="0.15", lw=1.8, zorder=8); ax.plot(0, 0, "s", ms=3, color="black", zorder=9)
    ax.text(-Wl / 2, yc + 0.08 * dmax, "array 1: %d falls" % nlo, ha="center", va="bottom", fontsize=5.5)
    ax.text(Wh / 2, yc + 0.08 * dmax, "array 2: %d falls" % nhi, ha="center", va="bottom", fontsize=5.5)
    ax.text(0.5 * (Wh - Wl), yb - 0.1 * dmax, "two combs on one carriage deflect two\ntension members into a row of fixed spans", ha="center", va="top", fontsize=5.5, color="0.25")
    ax.set_xlim(-Wl * 1.04, Wh * 1.04); ax.set_ylim(yb - 0.4 * dmax, yc + 0.5 * dmax); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title or "the dual-array RopeComb ($\\langle 5,9\\rangle$, $k$ = 9)", loc="center", fontsize=7, pad=2)
    # (2) the synthesised ratio
    b = fig.add_subplot(gs[1])
    b.plot(dd, Gs, color="black", lw=0.9, ls=(0, (4, 2)), label="target $G^*$")
    b.plot(dd, net, color=S.BLUE, lw=1.0, label="$n_1G_1 + n_2G_2$")
    b.set_xlabel("carriage braking distance (m)", fontsize=6, labelpad=1); b.set_ylabel("ratio", fontsize=6, labelpad=1); b.tick_params(labelsize=5.5, length=2, pad=1)
    b.legend(fontsize=5.5, loc="upper left", handlelength=1.8); b.set_title("ratio synthesised by least squares", loc="center", fontsize=7, pad=2)
    # (3) the payload force of the same design, rigid and compliant
    rr = rigid_run(c["M"], dual_gear(tag), dmax)
    e = fig.add_subplot(gs[2])
    nR = int(0.995 * len(rr["force_target"])); nC = int(0.995 * len(r["force_target"]))
    e.plot(rr["t"][:nR] * 1e3, np.asarray(rr["force_target"][:nR]) / 1e3, color="0.62", lw=0.6, label="rigid member")
    e.plot(r["t"][:nC] * 1e3, np.asarray(r["force_target"][:nC]) / 1e3, color=S.BLUE, lw=0.9, label="compliant member")
    e.axhline(F / 1e3, color="black", lw=0.9, ls=(0, (4, 2)), label="uniform force (ideal)")
    e.set_ylim(0, 2.65 * F / 1e3); e.set_xlabel("time (ms)", fontsize=6, labelpad=1); e.set_ylabel("payload force (kN)", fontsize=6, labelpad=1); e.tick_params(labelsize=5.5, length=2, pad=1)
    FR = np.asarray(rr["force_target"][:nR]); FC = np.asarray(r["force_target"][:nC])
    pR, pC = FR.max() / FR.mean(), FC.max() / F; vexit = r["summary"]["target_final_speed"]
    e.text(0.97, 0.04, "peak / mean:\nrigid %.2f, compliant %.2f\n%s" % (pR, pC, guide_text or "guide load 7.0% (mirror 11.1%)"), transform=e.transAxes, ha="right", va="bottom", fontsize=5.2, bbox=dict(fc="white", ec="none", alpha=0.45, pad=1.0))
    e.legend(fontsize=5.2, loc="upper left", handlelength=1.6, borderaxespad=0.2); e.set_title("1 kg to %.0f m/s at 1,000:1" % vexit, loc="center", fontsize=7, pad=2)
    print("graphical abstract: pR %.3f pC %.3f exit %.1f" % (pR, pC, vexit))
    out = os.path.join(HERE, "..", "figs", out)
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300, facecolor="white"); fig.savefig(out + ".tiff", dpi=300, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig); print("wrote graphical_abstract (13 x 5 cm)")


ALL = {
    "targets": fig_targets,
    "comb_100to1": lambda: fig_comb("100to1"), "comb_1000to1": lambda: fig_comb("1000to1"), "comb_10000to1": lambda: fig_comb("10000to1"),
    "rigid_100to1": lambda: fig_runs_single("100to1", "rigid"), "rigid_1000to1": lambda: fig_runs_single("1000to1", "rigid"), "rigid_10000to1": lambda: fig_runs_single("10000to1", "rigid"),
    "compliant_100to1": lambda: fig_runs_single("100to1", "compliant"), "compliant_1000to1": lambda: fig_runs_single("1000to1", "compliant"), "compliant_10000to1": lambda: fig_runs_single("10000to1", "compliant"),
    "instability": fig_instability, "target_selection": fig_target_selection, "search_vs_fit": fig_search_vs_fit, "member": fig_member,
    "parity": fig_parity, "compare": fig_compare, "trade": fig_trade,
    "runs_rigid": lambda: fig_runs_dual("rigid"), "runs_compliant": lambda: fig_runs_dual("compliant"),
    "design_k7": lambda: fig_design("k7"), "design_k9": lambda: fig_design("k9"), "configurations": fig_configurations,
    "graphical_abstract": fig_graphical_abstract,
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    for n in names:
        ALL[n]()
