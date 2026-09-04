"""FIG. 27 - line-art architecture schematic, black and white, 37 CFR 1.84 style."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch

BK = "black"
def num(ax, txt, xy, xytext, fs=7):
    ax.annotate(txt, xy=xy, xytext=xytext, fontsize=fs, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=BK, lw=0.7, shrinkA=0, shrinkB=2),
                bbox=dict(boxstyle="square,pad=0.22", fc="white", ec=BK, lw=0.7), zorder=20)

def sheave(ax, x, y, r, lw=1.0):
    ax.add_patch(Circle((x, y), r, fc="white", ec=BK, lw=lw, zorder=10))
    ax.add_patch(Circle((x, y), r*0.30, fc="white", ec=BK, lw=lw*0.8, zorder=11))

fig = plt.figure(figsize=(8.27, 11.69))

# ---------------------------------------------------------------- (a) whole machine
ax = fig.add_axes([0.06, 0.575, 0.88, 0.31]); ax.axis("off")
ax.set_xlim(-6.6, 10.2); ax.set_ylim(-4.6, 3.4); ax.set_aspect("equal")

GY, CY = -2.2, 1.5                      # base line, carriage line
ax.plot([-5.6, 8.6], [GY, GY], color=BK, lw=2.2)                       # base
ax.plot([-5.0, 5.0], [CY, CY], color=BK, lw=2.2)                       # carriage 2012
ax.plot([0, 0], [GY, 3.0], color=BK, lw=2.6)                           # guide 2010

# source mass halves
for sgn in (-1, 1):
    ax.add_patch(Rectangle((sgn*0.7 - (0.0 if sgn>0 else 1.6), CY+0.12), 1.6, 0.75,
                           fc="white", ec=BK, lw=1.1, hatch="////", zorder=8))
num(ax, "source mass", (-1.1, CY+0.5), (-2.6, 3.0))
num(ax, "carriage", (-4.6, CY), (-5.9, 2.2))
num(ax, "guide", (0, 2.4), (1.9, 3.0))

# banks: supports (fixed) and engagement members (on carriage)
def bank(sgn, nmem, tag_sup, tag_mem, tag_rope, tag_anchor):
    xs = np.linspace(0.55, 4.7, nmem+1) * sgn
    for x in xs:
        ax.plot([x, x], [GY, 0], color=BK, lw=1.4)                     # fixed support
        sheave(ax, x, 0, 0.13)
    dep = np.linspace(1.35, 0.30, nmem)
    for i in range(nmem):
        xm = 0.5*(xs[i]+xs[i+1])
        ax.plot([xm, xm], [CY, -dep[i]], color=BK, lw=1.2)             # engagement member
        sheave(ax, xm, -dep[i], 0.13)
    pts = [(xs[0], 0)]
    for i in range(nmem):
        pts.append((0.5*(xs[i]+xs[i+1]), -dep[i])); pts.append((xs[i+1], 0))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=BK, lw=1.6, zorder=9)
    ax.plot([sgn*0.16, xs[0]], [0, 0], color=BK, lw=1.6, zorder=9)      # rope to its anchor
    ax.plot(sgn*0.16, 0, "s", ms=5, mfc="white", mec=BK, mew=1.2, zorder=12)
    num(ax, tag_anchor, (sgn*0.16, 0.06), (sgn*0.62, 0.88))             # 2042 / 2052
    num(ax, tag_sup, (xs[2], GY+0.55), (sgn*2.6, GY-0.80))
    num(ax, tag_mem, (0.5*(xs[1]+xs[2]), 0.62), (sgn*3.3, 2.55))
    num(ax, tag_rope, (0.5*(xs[3]+xs[4]), -dep[3]*0.55), (sgn*5.72, 0.95))
    return xs[-1]

xl = bank(-1, 4, "engagement\nmembers", "array 1", "tension\nmember 1", "anchor 1")
xr = bank(+1, 4, "engagement\nmembers", "array 2", "tension\nmember 2", "anchor 2")

# return runs to the fixed-ratio stage
ax.plot([xl, xl-0.55, xl-0.55, 6.3], [0, -0.45, -2.35, -2.35], color=BK, lw=1.4, ls=(0,(5,2)))
ax.plot([xr, xr+0.55, xr+0.55, 6.3], [0, -0.45, -2.05, -2.05], color=BK, lw=1.4)
ax.text(0.0, -2.70, "return runs to the fixed-ratio stage", fontsize=6.2, ha="center")

# fixed-ratio stage block
ax.add_patch(FancyBboxPatch((6.3, -1.95), 2.2, 3.2, boxstyle="round,pad=0.06",
                            fc="white", ec=BK, lw=1.3, zorder=6))
ax.text(7.4, 0.78, "fixed-ratio\nstage", ha="center", va="center", fontsize=6.6, zorder=7)
ax.text(7.4, -0.35, "detail:\nsee (b)", ha="center", va="center", fontsize=6.2, style="italic", zorder=7)
ax.annotate("", (8.5, -1.5), (9.0, -2.6), arrowprops=dict(arrowstyle="<-", lw=1.4, color=BK))
ax.plot([8.5, 9.0], [-1.5, -1.5], color=BK, lw=1.6)
num(ax, "output\nmember", (8.9, -1.9), (9.5, -3.3))
ax.text(0, -4.35, "(a)  the dual array, two arrays either side of the stationary guide",
        ha="center", fontsize=7.2)

# ---------------------------------------------------------------- (b) stage detail
ax = fig.add_axes([0.16, 0.30, 0.68, 0.235]); ax.axis("off")
ax.set_xlim(-1.4, 8.4); ax.set_ylim(-2.9, 6.3); ax.set_aspect("equal")

nL, nR = 3, 4                                        # k = 7
ax.plot([-0.9, 7.6], [4.15, 4.15], color=BK, lw=2.2)                  # fixed head
ax.text(3.3, 4.34, "fixed head", ha="center", fontsize=6.4)

# two movable elements, one per bank rope
for j, (x0, nf, lbl, tag) in enumerate([(0.35, nL, "driven by\ntension member 1", "movable\nelement 1"),
                                        (4.0, nR, "driven by\ntension member 2", "movable\nelement 2")]):
    yb = 1.35 if j == 0 else 0.95
    w = 0.42*(nf-1) + 0.9
    ax.add_patch(Rectangle((x0-0.28, yb-0.22), w, 0.44, fc="white", ec=BK, lw=1.3, zorder=8))
    for i in range(nf):
        sheave(ax, x0 + 0.42*i, 4.15-0.30, 0.17)
        sheave(ax, x0 + 0.42*i, yb, 0.17)
        ax.plot([x0+0.42*i, x0+0.42*i], [4.15-0.30, yb], color=BK, lw=1.1, zorder=7)
    ax.plot([x0-0.28+w*0.5, x0-0.28+w*0.5], [yb-0.22, 0.15], color=BK, lw=1.6)
    ax.annotate("", (x0-0.28+w*0.5, 0.15), (x0-0.28+w*0.5, -0.35),
                arrowprops=dict(arrowstyle="->", lw=1.4, color=BK))
    ax.text(x0-0.28+w*0.5, -0.72, lbl, ha="center", fontsize=6.0)
    num(ax, tag, (x0-0.28+w*0.5, yb), (x0-0.28+w*0.5 + (-1.15 if j==0 else 1.25), yb+0.9))
    ax.text(x0-0.28+w*0.5, 5.05, r"$n_%d = %d$ falls" % (j+1, nf), ha="center", fontsize=7.6)

# output member threading both blocks in series
ax.plot([0.35-0.55, 0.35], [3.85, 3.85], color=BK, lw=1.6)
ax.plot([0.35+0.42*(nL-1), 4.0], [3.85, 3.85], color=BK, lw=1.6, ls=(0,(4,2)))
ax.plot([4.0+0.42*(nR-1), 7.0], [3.85, 3.85], color=BK, lw=1.6)
ax.annotate("", (7.0, 3.85), (7.45, 3.85), arrowprops=dict(arrowstyle="->", lw=1.6, color=BK))
num(ax, "output\nmember", (7.3, 3.85), (8.0, 2.8))
ax.text(3.3, 5.85, r"$n_1 + n_2 = k$", ha="center", fontsize=9)
ax.text(3.3, -1.62, "the output member is reeved in series through both movable elements;\neach element is driven by its own array's tension member", ha="center", fontsize=6.4)
ax.text(3.3, -2.62, "(b)  the fixed-ratio stage: the k falls are partitioned between the two arrays",
        ha="center", fontsize=7.2)

fig.savefig("fig27.png", dpi=220, bbox_inches="tight", facecolor="white")
print("wrote fig27.png")
