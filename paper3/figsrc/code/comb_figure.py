"""
Drawing of the comb geometry.

    left   - the array drawn to scale, with the tension member shown at five
             points through the stroke, and span widths dimensioned beneath
    top-r  - span widths in engagement order (the design variables)
    bot-r  - the resulting ratio profile against the target it was fitted to

The drawing is at equal aspect. Because carriage travel is usually comparable
to the total comb width, the drawing is TALLER than it is wide, so it gets a
tall cell of its own on the left rather than a wide strip across the top --
otherwise matplotlib shrinks it to a fraction of the panel and the span
dimensions become illegible.

    from comb_figure import comb_figure
    comb_figure(R, s, k, d_max, "fig.png", title="1000:1", ideal=(d, G))
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def layout(R):
    """Support x-positions (N+1 of them) and pin x-positions (N)."""
    sup, pin, x = [0.0], [], 0.0
    for Ri in R:
        pin.append(x + Ri)
        x += 2.0 * Ri
        sup.append(x)
    return np.array(sup), np.array(pin)


def rope_path(R, s, d):
    sup, pin = layout(R)
    xs, ys = [sup[0]], [0.0]
    for i, si in enumerate(s):
        xs.append(pin[i]); ys.append(min(0.0, si - d))
        xs.append(sup[i + 1]); ys.append(0.0)
    return np.array(xs), np.array(ys)


def comb_figure(R, s, k, d_max, path, title="", ideal=None,
                fracs=(0.0, 0.25, 0.5, 0.75, 1.0), bar_clearance=0.10):
    R = np.asarray(R, float)
    s = np.asarray(s, float)
    N = len(R)
    sup, pin = layout(R)
    width = sup[-1]

    cmap = plt.cm.viridis
    # Draw the pin stems long enough that the carriage bar still clears the
    # baseline at the end of the stroke by bar_clearance metres. Stem length is
    # purely presentational -- span widths 2R and engagement offsets s are the
    # design quantities and are untouched.
    carriage_top = d_max + bar_clearance
    ground_y = float(s.min()) - d_max - 0.10 * d_max

    ybar = ground_y - 0.14 * d_max
    tight = (2.0 * R.min()) < 0.10 * width
    rows = (0.07, 0.20) if tight else (0.07, 0.07)
    y_cap = ybar - (0.36 if tight else 0.22) * d_max
    y_lo = y_cap - 0.06 * d_max

    # Choose the layout from the drawing's own aspect. A narrow comb (carriage
    # travel comparable to comb width) is TALLER than it is wide and belongs in
    # a tall cell down the left; a wide comb is short and wide and belongs in a
    # strip across the top. Using one layout for both squeezes one of them.
    data_w = 1.10 * width
    data_h = (carriage_top + 0.06 * d_max) - y_lo
    aspect = data_h / data_w

    if aspect < 0.70:                                   # wide, short drawing
        top_h = float(np.clip(15.0 * aspect, 3.0, 8.0))
        fig = plt.figure(figsize=(15.0, top_h + 4.6))
        gs = fig.add_gridspec(2, 2, height_ratios=[top_h, 4.6],
                              hspace=0.34, wspace=0.22)
        ax = fig.add_subplot(gs[0, :])
        axL = fig.add_subplot(gs[1, 0])
        axR = fig.add_subplot(gs[1, 1])
    else:                                               # tall, narrow drawing
        fig = plt.figure(figsize=(14.5, 8.6))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0],
                              hspace=0.36, wspace=0.20)
        ax = fig.add_subplot(gs[:, 0])
        axL = fig.add_subplot(gs[0, 1])
        axR = fig.add_subplot(gs[1, 1])

    ax.plot([sup[0] - 0.03 * width, sup[-1] + 0.03 * width], [ground_y, ground_y],
            color="0.15", lw=4, solid_capstyle="butt", zorder=1)
    for xs_ in sup:
        ax.plot([xs_, xs_], [ground_y, 0.0], color="0.4", lw=3,
                solid_capstyle="butt", zorder=2)
        ax.plot(xs_, 0.0, "o", ms=5, mfc="white", mec="0.25", mew=1.4, zorder=4)

    for f in fracs:
        xr, yr = rope_path(R, s, f * d_max)
        ax.plot(xr, yr, color=cmap(f * 0.85), lw=1.9, zorder=3, label=f"{int(f*100)}%")

    yc = carriage_top - d_max
    ax.plot([sup[0] - 0.02 * width, sup[-1] + 0.02 * width], [yc, yc],
            color="0.15", lw=5, solid_capstyle="round", zorder=5)
    for i, (xp, si) in enumerate(zip(pin, s)):
        tip = si - d_max
        ax.plot([xp, xp], [yc, tip], color="0.45", lw=2.2, zorder=5)
        ax.plot(xp, tip, "o", ms=7, mfc=cmap(0.85), mec="0.15", mew=1.2, zorder=6)
        ax.annotate(f"{i+1}", (xp, tip), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=9, color="0.2")

    for i, Ri in enumerate(R):
        x0, x1 = sup[i], sup[i + 1]
        xm = 0.5 * (x0 + x1)
        ax.annotate("", (x0, ybar), xytext=(x1, ybar),
                    arrowprops=dict(arrowstyle="<->", color="0.45", lw=0.9))
        ytxt = ybar - (rows[i % 2] if tight else rows[0]) * d_max
        if tight and i % 2 == 1:
            ax.plot([xm, xm], [ybar - 0.015 * d_max, ytxt + 0.012 * d_max],
                    color="0.72", lw=0.7, zorder=2)
        ax.text(xm, ytxt, f"{2*Ri:.2f}", ha="center", va="top", fontsize=9, color="0.25")
    ax.text(width / 2, y_cap, f"span widths (m) — total comb {width:.2f} m",
            ha="center", va="top", fontsize=10, color="0.3")

    ax.axhline(0, color="0.7", lw=0.7, ls="--", zorder=1)
    ax.set_xlim(-0.05 * width, 1.05 * width)
    ax.set_ylim(y_lo, carriage_top + 0.06 * d_max)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.legend(loc="upper center", fontsize=9, ncol=5, frameon=False,
              bbox_to_anchor=(0.5, 1.045), title="tension member at % of stroke",
              title_fontsize=9)
    ax.set_title(f"{title}   —   carriage travel {d_max:.2f} m",
                 fontsize=12.5, pad=34)

    order = np.argsort(s)
    axL.bar(np.arange(1, N + 1), 2 * R[order],
            color=[cmap(0.15 + 0.7 * i / max(N - 1, 1)) for i in range(N)])
    axL.set_xlabel("engagement order", fontsize=9)
    axL.set_ylabel("span width $2R_i$ (m)", fontsize=9)
    axL.set_title("span width narrows as the stroke proceeds", fontsize=10)
    axL.tick_params(labelsize=8)
    axL.grid(alpha=0.25, axis="y")
    ax2 = axL.twinx()
    ax2.plot(np.arange(1, N + 1), s[order], "k.-", lw=1, ms=6)
    ax2.set_ylabel("engagement offset $s_i$ (m)", fontsize=9)
    ax2.tick_params(labelsize=8)

    dd = np.linspace(0, d_max, 400)
    D = np.maximum(dd[:, None] - s[None, :], 0.0)
    G = k * np.sum(2 * D / np.sqrt(R[None, :] ** 2 + D ** 2), axis=1)
    axR.plot(dd, G, "k", lw=1.7, label="this array")
    if ideal is not None:
        axR.plot(ideal[0], ideal[1], "b--", lw=1.4, label="target profile")
    for si in s:
        axR.axvline(si, color="0.85", lw=0.6, zorder=0)
    axR.set_xlabel("carriage displacement (m)", fontsize=9)
    axR.set_ylabel("net ratio", fontsize=9)
    axR.set_title("ratio profile vs the profile fitted to", fontsize=10)
    axR.legend(fontsize=8.5)
    axR.tick_params(labelsize=8)
    axR.grid(alpha=0.25)

    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
