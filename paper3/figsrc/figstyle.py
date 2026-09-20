"""One style for every data figure in the manuscript.

Sans-serif (TeX Gyre Heros, the Helvetica clone that ships with TeX Live; Liberation
Sans / DejaVu Sans as fallbacks), 8 pt base, no gridlines, ticks inward, top and
right spines removed, one restrained colour-blind-safe palette with line styles
that survive greyscale printing. Figures are emitted as vector PDF at the
column widths of the elsarticle 12 pt preprint layout.

    import figstyle as S
    S.use()
    fig, ax = plt.subplots(figsize=(S.W_FULL, 3.0))
    S.save(fig, "fig_name")           # writes fig_name.pdf (and .png at 300 dpi)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")

W_FULL = 6.3      # elsarticle preprint 12 pt text width is 6.5 in; leave a margin
W_HALF = 3.1

# palette (Okabe–Ito)
BLUE = "#0072B2"
RED = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
GREY = "#7f7f7f"
LIGHT = "#bfbfbf"
BLACK = "black"

IDEAL = dict(color=BLACK, lw=1.1, ls=(0, (5, 2.5)))          # the ideal / target
DESIGN = dict(color=BLUE, lw=1.2, ls="-")                     # this design
SECOND = dict(color=RED, lw=1.2, ls=(0, (4, 1.5)))            # a second design
THIRD = dict(color=GREEN, lw=1.2, ls=(0, (1, 1.2)))
HATCH_A = "////"
HATCH_B = "...."
STROKE_FRACS = (0.0, 0.25, 0.5, 0.75, 1.0)
STROKE_GREYS = ("0.82", "0.66", "0.5", "0.32", "0.0")
STROKE_LS = ((0, (1, 1)), (0, (3, 1.5)), (0, (5, 1.5)), (0, (7, 1.5)), "-")


def use():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["TeX Gyre Heros", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "normal",
        "axes.labelsize": 8,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7,
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.direction": "in", "ytick.direction": "in",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False,
        "axes.titlelocation": "left",
        "axes.titlepad": 4,
        "mathtext.fontset": "custom",
        "mathtext.rm": "TeX Gyre Heros",
        "mathtext.it": "TeX Gyre Heros:italic",
        "mathtext.bf": "TeX Gyre Heros:bold",
        "mathtext.fallback": "stixsans",
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "hatch.linewidth": 0.5,
    })


def save(fig, name, png=True, pad=0.02):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name + ".pdf")
    fig.savefig(p, bbox_inches="tight", pad_inches=pad)
    if png:
        fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", pad_inches=pad, dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", name)
    return p


def twin_ok(ax):
    """A twin axis needs its right spine back."""
    ax.spines["right"].set_visible(True)
    return ax


def panel(ax, letter, text):
    ax.set_title("(%s) %s" % (letter, text), loc="left")
