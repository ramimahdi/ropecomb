"""Figure 6 - the dual-array architecture and its fixed-ratio stage, as line art.

Same drawing as the earlier fig27, redrawn at the manuscript's figure width so the
lettering is set at its true size (8 pt, the figstyle base), with the vertical
white space between and around the two panels removed, the sub-captions reduced
to panel tags (the caption in the text carries the description) and the
"output member" label of panel (b) placed to the right of the drawing rather
than beneath it.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import figstyle as S

S.use()
BK = S.BLACK
FS = 8          # label size
FS_S = 7.5      # secondary text


def num(ax, txt, xy, xytext, fs=FS):
    ax.annotate(txt, xy=xy, xytext=xytext, fontsize=fs, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=BK, lw=0.7, shrinkA=0, shrinkB=2),
                bbox=dict(boxstyle="square,pad=0.22", fc="white", ec=BK, lw=0.7), zorder=20)


def sheave(ax, x, y, r, lw=1.0):
    ax.add_patch(Circle((x, y), r, fc="white", ec=BK, lw=lw, zorder=10))
    ax.add_patch(Circle((x, y), r * 0.30, fc="white", ec=BK, lw=lw * 0.8, zorder=11))


# panel extents in data units; heights follow from equal aspect at the figure width
XA, YA = (-6.7, 10.5), (-3.35, 3.45)
XB, YB = (-1.9, 11.4), (-1.3, 3.75)
WB = 0.80                                   # panel (b) as a fraction of the figure width
HA = S.W_FULL * (YA[1] - YA[0]) / (XA[1] - XA[0])
HB = S.W_FULL * WB * (YB[1] - YB[0]) / (XB[1] - XB[0])
GAP = 0.10
H = HA + HB + GAP
fig = plt.figure(figsize=(S.W_FULL, H))

# ---------------------------------------------------------------- (a) whole machine
ax = fig.add_axes([0, (HB + GAP) / H, 1, HA / H]); ax.axis("off")
ax.set_xlim(*XA); ax.set_ylim(*YA); ax.set_aspect("equal")

GY, CY = -2.2, 1.5                      # base line, carriage line
ax.plot([-5.6, 8.6], [GY, GY], color=BK, lw=2.2)                       # base
ax.plot([-5.0, 5.0], [CY, CY], color=BK, lw=2.2)                       # carriage
ax.plot([0, 0], [GY, 3.0], color=BK, lw=2.6)                           # guide

for sgn in (-1, 1):                                                    # source-mass halves
    ax.add_patch(Rectangle((sgn * 0.7 - (0.0 if sgn > 0 else 1.6), CY + 0.12), 1.6, 0.75,
                           fc="white", ec=BK, lw=1.1, hatch="////", zorder=8))
num(ax, "source mass", (-1.1, CY + 0.5), (-2.6, 3.0))
num(ax, "carriage", (-4.6, CY), (-5.9, 2.2))
num(ax, "guide", (0, 2.4), (1.9, 3.0))


def array(sgn, nmem, tag_sup, tag_mem, tag_rope, tag_anchor):
    xs = np.linspace(0.55, 4.7, nmem + 1) * sgn
    for x in xs:
        ax.plot([x, x], [GY, 0], color=BK, lw=1.4)                     # fixed support
        sheave(ax, x, 0, 0.13)
    dep = np.linspace(1.35, 0.30, nmem)
    for i in range(nmem):
        xm = 0.5 * (xs[i] + xs[i + 1])
        ax.plot([xm, xm], [CY, -dep[i]], color=BK, lw=1.2)             # engagement member
        sheave(ax, xm, -dep[i], 0.13)
    pts = [(xs[0], 0)]
    for i in range(nmem):
        pts.append((0.5 * (xs[i] + xs[i + 1]), -dep[i])); pts.append((xs[i + 1], 0))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=BK, lw=1.6, zorder=9)
    ax.plot([sgn * 0.16, xs[0]], [0, 0], color=BK, lw=1.6, zorder=9)  # rope to its anchor
    ax.plot(sgn * 0.16, 0, "s", ms=5, mfc="white", mec=BK, mew=1.2, zorder=12)
    num(ax, tag_anchor, (sgn * 0.16, 0.06), (sgn * 0.62, 0.88))
    num(ax, tag_sup, (xs[2], GY + 0.55), (sgn * 3.75, GY - 0.78))
    num(ax, tag_mem, (0.5 * (xs[1] + xs[2]), 0.62), (sgn * 3.3, 2.55))
    num(ax, tag_rope, (0.5 * (xs[3] + xs[4]), -dep[3] * 0.55), (sgn * 5.72, 0.95))
    return xs[-1]


xl = array(-1, 4, "engagement\nmembers", "array 1", "tension\nmember 1", "anchor 1")
xr = array(+1, 4, "engagement\nmembers", "array 2", "tension\nmember 2", "anchor 2")

# return runs to the fixed-ratio stage
ax.plot([xl, xl - 0.55, xl - 0.55, 6.3], [0, -0.45, -2.35, -2.35], color=BK, lw=1.4, ls=(0, (5, 2)))
ax.plot([xr, xr + 0.55, xr + 0.55, 6.3], [0, -0.45, -2.05, -2.05], color=BK, lw=1.4)
ax.text(0.0, -2.62, "return runs to the stage", fontsize=FS_S, ha="center", va="top")

# fixed-ratio stage block
ax.add_patch(FancyBboxPatch((6.3, -1.95), 2.2, 3.2, boxstyle="round,pad=0.06",
                            fc="white", ec=BK, lw=1.3, zorder=6))
ax.text(7.4, 0.78, "fixed-ratio\nstage", ha="center", va="center", fontsize=FS, zorder=7)
ax.text(7.4, -0.35, "detail:\nsee (b)", ha="center", va="center", fontsize=FS_S, style="italic", zorder=7)
ax.annotate("", (8.5, -1.5), (9.0, -2.6), arrowprops=dict(arrowstyle="<-", lw=1.4, color=BK))
ax.plot([8.5, 9.0], [-1.5, -1.5], color=BK, lw=1.6)
num(ax, "output\nmember", (8.78, -2.05), (9.65, -0.55))
ax.text(XA[0] + 0.1, YA[1] - 0.1, "(a)", ha="left", va="top", fontsize=9, weight="bold")

# ---------------------------------------------------------------- (b) stage detail
ax = fig.add_axes([(1 - WB) / 2, 0, WB, HB / H]); ax.axis("off")
ax.set_xlim(*XB); ax.set_ylim(*YB); ax.set_aspect("equal")

# k = 7 as reeved: no stationary sheave. Block 1 carries one sheave and the dead
# end, block 2 two sheaves; p = 3 travelling sheaves, n1 = 3, n2 = 4.
RS = 0.25
L1, L2, L3, L4 = 0.75, 0.25, -0.25, -0.75      # the four rope levels
A, B = (1.30, 0.50), (1.30, -0.50)             # the two sheaves of element 2
Sv = (6.00, 0.00)                              # sheave on element 1
rope = dict(color=BK, lw=1.35, zorder=9, solid_capstyle="round")


def arc(xc, yc, r, a0, a1):
    a = np.radians(np.linspace(a0, a1, 60))
    ax.plot(xc + r * np.cos(a), yc + r * np.sin(a), **rope)


ax.add_patch(Rectangle((0.95, -0.95), 0.70, 1.90, fc="white", ec=BK, lw=1.4, zorder=8))
ax.add_patch(Rectangle((5.55, -0.55), 0.90, 1.50, fc="white", ec=BK, lw=1.4, zorder=8))
for c in (A, B, Sv):
    sheave(ax, c[0], c[1], RS, lw=1.1)

ax.plot([5.55, A[0]], [L1, L1], **rope)                       # segment 1
ax.add_patch(Circle((5.55, L1), 0.09, fc=BK, ec=BK, zorder=12))
arc(A[0], A[1], RS, 90, 270)                                  # round element 2
ax.plot([A[0], Sv[0]], [L2, L2], **rope)                      # segment 2
arc(Sv[0], Sv[1], RS, 90, -90)                                # round element 1
ax.plot([Sv[0], B[0]], [L3, L3], **rope)                      # segment 3
arc(B[0], B[1], RS, 90, 270)                                  # round element 2
ax.plot([B[0], 8.35], [L4, L4], **rope)                       # segment 4, to the payload
ax.annotate("", (8.35, L4), (8.95, L4), arrowprops=dict(arrowstyle="->", lw=1.5, color=BK))
ax.text(9.15, L4, "output member", ha="left", va="center", fontsize=FS)

# the two tension members, pulling the elements apart
ax.plot([0.95, -0.55], [0, 0], color=BK, lw=1.9, zorder=8)
ax.annotate("", (-1.15, 0), (-0.55, 0), arrowprops=dict(arrowstyle="->", lw=1.9, color=BK))
ax.plot([6.45, 7.75], [0.45, 0.45], color=BK, lw=1.9, zorder=8)
ax.annotate("", (8.35, 0.45), (7.75, 0.45), arrowprops=dict(arrowstyle="->", lw=1.9, color=BK))

ax.text(1.30, 1.28, "block 2", ha="center", va="bottom", fontsize=FS)
ax.text(-0.35, 0.28, "tension member 2", ha="center", va="bottom", fontsize=FS_S, zorder=15)
ax.text(6.00, 1.28, "block 1", ha="center", va="bottom", fontsize=FS)
ax.text(8.05, 0.72, "tension member 1", ha="center", va="bottom", fontsize=FS_S, zorder=15)
ax.annotate("dead end on block 1", xy=(5.55, L1), xytext=(3.55, 2.05),
            fontsize=FS_S, ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color=BK, lw=0.6, shrinkB=4))
ax.text(1.30, 2.62, r"$n_2 = 4$ falls", ha="center", va="center", fontsize=FS)
ax.text(6.00, 2.62, r"$n_1 = 3$ falls", ha="center", va="center", fontsize=FS)
ax.text(3.65, 3.30, r"$n_1 + n_2 = k$", ha="center", va="center", fontsize=9.5)
ax.text(XB[0] + 0.1, YB[1] - 0.1, "(b)", ha="left", va="top", fontsize=9, weight="bold")

S.save(fig, "fig_arch")
