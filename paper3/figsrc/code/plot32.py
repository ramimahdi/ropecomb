"""
Six-panel run chart, matching the layout of the original paper's Figure 12.

    a) heavy braking distance     b) net gear ratio
    c) heavy source speed         d) target speed
    e) force on heavy             f) force on target

The 3x3 layout this replaces carried two redundant pairs: payload acceleration
duplicated force on payload (the payload is 1 kg, so the curves are identical up
to units), and "speed ratio vs commanded gear" duplicated the net gear ratio
whenever the rope is taut, which is everywhere except the terminal transient.

    from plot32 import plot_3x2
    plot_3x2(run, "fig.png", title="...", ideal=ideal_run)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _robust_ylim(ax, series, pad=0.06, floor_hi=None, headroom=0.0, force_lo=None):
    """Clip to the bulk of the data, labelling anything left off-scale.

    floor_hi  - the upper limit is never set below this value, so a force
                panel always shows a fixed multiple of the design force even
                when the trace is well behaved.
    headroom  - extra fraction of the span added above the upper limit.
    """
    v = np.concatenate([np.asarray(s, float).ravel() for s in series if s is not None])
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    lo, hi = np.percentile(v, [0.5, 99.0])
    if hi <= lo:
        lo, hi = v.min(), v.max()
    if hi <= lo:
        return
    span = hi - lo
    lo, hi = lo - pad * span, hi + (pad + headroom) * span
    if floor_hi is not None:
        hi = max(hi, floor_hi)
    if force_lo is not None:
        lo = force_lo
    ax.set_ylim(lo, hi)
    vmax, vmin = v.max(), v.min()
    notes = []
    if vmax > hi:
        notes.append(f"max {vmax:,.0f}")
    if vmin < lo:
        notes.append(f"min {vmin:,.0f}")
    if notes:
        ax.text(0.985, 0.94, "\n".join(notes) + "\n(off scale)", transform=ax.transAxes,
                ha="right", va="top", fontsize=9.5, color="crimson",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))


def plot_3x2(r, path, title="", ideal=None, force_unit="N", f_ref=None,
             f_span=2.0, f_span_shock=3.0, shock_at=2.0, e_headroom=0.35,
             f_ymin=None, e_ymin=None):
    """
    f_ref  - the design (ideal uniform) payload force. Panel (f) is scaled to
             show at least f_span * f_ref, and at least f_span_shock * f_ref
             if the trace exceeds shock_at * f_ref anywhere. Taken from the
             ideal run if not given.
    """
    t = r["t"] * 1e3
    ti = ideal["t"] * 1e3 if ideal is not None else None
    if f_ref is None and ideal is not None:
        f_ref = float(np.median(ideal["force_target"]))
    if f_ref is None:
        f_ref = r["summary"].get("mean_force_target")

    fig, axes = plt.subplots(3, 2, figsize=(12.2, 10.2))
    (aA, aB), (aC, aD), (aE, aF) = axes

    def draw(ax, y, yi, label, ylab, clip=False, floor_hi=None, headroom=0.0, force_lo=None):
        ax.plot(t, y, "k", lw=1.2, label="designed")
        if yi is not None:
            ax.plot(ti, yi, "b--", lw=1.2, label="ideal gear")
        ax.set_title(label, fontsize=13.5)
        ax.set_ylabel(ylab, fontsize=12)
        ax.tick_params(labelsize=10.5)
        ax.grid(alpha=0.25)
        if clip:
            _robust_ylim(ax, [y, yi], floor_hi=floor_hi, headroom=headroom,
                         force_lo=force_lo)

    g = lambda key: (ideal[key] if ideal is not None else None)

    draw(aA, r["heavy_dist"], g("heavy_dist"), "a) heavy braking distance", "m")
    draw(aB, r["gear"], g("gear"), "b) net gear ratio", "ratio (y:1)")
    draw(aC, r["heavy_speed"], g("heavy_speed"), "c) heavy source speed", "m/s")
    draw(aD, r["target_speed"], g("target_speed"), "d) target speed", "m/s")
    # Panel (f): always show at least f_span x the design force, and at least
    # f_span_shock x it when the trace spikes, so a well-behaved run and a
    # shocking one are read on comparable scales.
    peak_F = float(np.nanmax(r["force_target"]))
    f_floor = None
    if f_ref:
        mult = f_span_shock if peak_F > shock_at * f_ref else f_span
        f_floor = 1.08 * mult * f_ref      # headroom so the marker line sits inside

    draw(aE, r["force_heavy"], g("force_heavy"), "e) force on heavy", force_unit,
         clip=True, headroom=e_headroom, force_lo=e_ymin)
    draw(aF, r["force_target"], g("force_target"), "f) force on target", force_unit,
         clip=True, floor_hi=f_floor, force_lo=f_ymin)

    if f_ref:
        top = aF.get_ylim()[1]
        for mlt in (2.0, 3.0):
            y = mlt * f_ref
            if y <= top:
                aF.axhline(y, color="0.55", ls=(0, (4, 4)), lw=0.9)
                aF.text(0.012, y, f" {mlt:.0f}x design force",
                        transform=aF.get_yaxis_transform(),
                        ha="left", va="top", fontsize=9, color="0.45")
    mean_F = r["summary"].get("mean_force_target")
    if mean_F:
        aF.axhline(mean_F, color="g", ls="-.", lw=0.9)
        aF.text(0.02, 0.04, f"peak/mean = {peak_F/mean_F:,.2f}x",
                transform=aF.transAxes, fontsize=10.5, va="bottom",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))

    # mark the end of the designed stroke on every panel
    for ax in axes.ravel():
        ax.axvline(t[-1], color="orange", ls=":", lw=1.1)
    for ax in (aE, aF):
        ax.set_xlabel("time (ms)", fontsize=12)

    aA.legend(fontsize=10.5, loc="upper left")
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
