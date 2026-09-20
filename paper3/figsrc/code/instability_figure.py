"""
Replacement for Figure 13: the traction-loss instability, with the mechanism
shown rather than only the symptom.

Four panels:
  a) force on the payload over the whole stroke, with both zoom windows marked
  b) a mid-stroke engagement pulse -- the normal, bounded transient
  c) the terminal window -- pulses growing without bound
  d) the mechanism: payload speed against rope-tip speed. Slack begins wherever
     the payload curve rises above the rope-tip curve, and each subsequent
     re-contact delivers an impulse.

Rope-tip speed is  gear * heavy_speed, i.e. the speed at which the transmission
is paying out rope, which is what the payload must not exceed.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _window(t, lo, hi):
    return (t >= lo) & (t <= hi)


def instability_figure(r, path, title="", tail_ms=2.0, mid_frac=0.62, mid_ms=12.0):
    t = r["t"] * 1e3
    F = r["force_target"]
    vt = r["target_speed"]
    vrope = r["gear"] * r["heavy_speed"]
    gap = r["rope_end_dist"] - r["target_dist"]
    slack = r["slack"]

    t_end = t[-1]
    tail = (max(t_end - tail_ms, t[0]), t_end)
    mid_c = mid_frac * t_end
    mid = (max(mid_c - mid_ms / 2, t[0]), min(mid_c + mid_ms / 2, t_end))

    fig = plt.figure(figsize=(14.5, 8.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.34, wspace=0.26)
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])
    axD = fig.add_subplot(gs[1, 2])

    # --- a) whole stroke --------------------------------------------------
    axA.plot(t, F, "k", lw=0.9)
    mean_F = r["summary"]["mean_force_target"]
    axA.axhline(mean_F, color="g", ls="-.", lw=0.9, label=f"mean {mean_F:,.0f} N")
    p99 = np.percentile(F, 99.0)
    axA.set_ylim(0, max(p99 * 1.3, mean_F * 2.2))
    axA.text(0.995, 0.93, f"peak {F.max():,.0f} N (off scale)", transform=axA.transAxes,
             fontsize=10.2, ha="right", va="top", color="crimson")
    for (lo, hi), c, lbl in ((mid, "tab:blue", "b) mid-stroke"), (tail, "tab:red", "c,d) terminal")):
        axA.axvspan(lo, hi, color=c, alpha=0.13)
        axA.text((lo + hi) / 2, axA.get_ylim()[1] * 0.06, lbl, ha="center",
                 fontsize=10.9, color=c)
    axA.set_xlabel("time (ms)", fontsize=11.6)
    axA.set_ylabel("force on payload (N)", fontsize=11.6)
    axA.set_title("a) force on payload, whole stroke", fontsize=13.8)
    axA.legend(fontsize=10.2, loc="upper left")
    axA.tick_params(labelsize=10.2)
    axA.grid(alpha=0.25)

    # --- b) a normal engagement pulse -------------------------------------
    m = _window(t, *mid)
    axB.plot(t[m], F[m], "k", lw=1.1)
    axB.axhline(mean_F, color="g", ls="-.", lw=0.9)
    axB.set_xlabel("time (ms)", fontsize=11.6)
    axB.set_ylabel("force on payload (N)", fontsize=11.6)
    axB.set_title("b) mid-stroke: bounded sawtooth", fontsize=13.8)
    axB.tick_params(labelsize=10.2)
    axB.grid(alpha=0.25)

    # --- c) terminal force -------------------------------------------------
    w = _window(t, *tail)
    axC.plot(t[w], F[w], "k", lw=1.1)
    axC.axhline(mean_F, color="g", ls="-.", lw=0.9)
    if slack[w].any():
        axC.axvline(t[w][slack[w]][0], color="orange", ls=":", lw=1.2,
                    label="first slack")
        axC.legend(fontsize=10.2)
    axC.set_xlabel("time (ms)", fontsize=11.6)
    axC.set_ylabel("force on payload (N)", fontsize=11.6)
    axC.set_title("c) terminal: pulses diverge", fontsize=13.8)
    axC.tick_params(labelsize=10.2)
    axC.grid(alpha=0.25)

    # --- d) the mechanism ---------------------------------------------------
    axD.plot(t[w], vrope[w], color="tab:blue", lw=1.2, label="rope tip  $G\\,v_h$")
    axD.plot(t[w], vt[w], "k", lw=1.2, label="payload  $v_t$")
    over = vt[w] > vrope[w]
    if over.any():
        axD.fill_between(t[w], vrope[w], vt[w], where=over, color="crimson",
                         alpha=0.25, label="payload outrunning rope")
    axD.set_xlabel("time (ms)", fontsize=11.6)
    axD.set_ylabel("speed (m/s)", fontsize=11.6)
    axD.set_title("d) mechanism: payload outruns the rope tip", fontsize=13.8)
    axD.legend(fontsize=10.2, loc="lower left")
    axD.tick_params(labelsize=10.2)
    axD.grid(alpha=0.25)

    fig.suptitle(title or "traction-loss instability", fontsize=17.4)
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    n_cross = int(np.sum(np.diff((vt > vrope).astype(int)) > 0))
    return dict(path=path, peak=float(F.max()), mean=float(mean_F),
                first_slack_ms=float(t[slack][0]) if slack.any() else None,
                crossings=n_cross, slack_pct=float(slack.mean() * 100))
