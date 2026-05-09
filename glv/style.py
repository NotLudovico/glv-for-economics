import matplotlib as mpl
import matplotlib.pyplot as plt


# Muted, colorblind-aware palette for journal figures.
# Order chosen so single-line plots use a serious neutral first,
# and two-series plots get neutral + warm accent.
EDITORIAL_CYCLE = [
    "#1f2933",  # charcoal
    "#c2571a",  # burnt orange
    "#3b6fb6",  # muted blue
    "#2e7d4e",  # forest
    "#6b4f8c",  # plum
    "#7a7d83",  # slate
    "#a3492a",  # rust
    "#4a6f8a",  # steel
]


def apply_style() -> None:
    """Editorial style for journal figures (physics / economics).

    Clean sans-serif typography, muted colorblind-aware palette,
    full bounding box, inward ticks, no grid.

    Call once near the top of a notebook.
    """
    plt.style.use("default")

    mpl.rcParams.update({
        # Figure
        "figure.figsize": (6.4, 4.0),
        "figure.dpi": 120,
        "figure.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "savefig.transparent": False,

        # Fonts
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Helvetica Neue", "Helvetica", "Arial",
            "Liberation Sans", "DejaVu Sans",
        ],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "regular",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "regular",

        # Axes — full bounding box, thin
        "axes.facecolor": "white",
        "axes.edgecolor": "#000000",
        "axes.linewidth": 0.6,
        "axes.labelcolor": "#000000",
        "axes.titlecolor": "#000000",
        "axes.titlepad": 8,
        "axes.labelpad": 4,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.prop_cycle": mpl.cycler(color=EDITORIAL_CYCLE),
        "axes.axisbelow": True,
        "axes.xmargin": 0.02,
        "axes.ymargin": 0.05,

        # Ticks — inward, thin (PRL / Nature feel)
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.pad": 4,
        "ytick.major.pad": 4,

        # Grid — off by default; if enabled, very faint
        "axes.grid": False,
        "grid.color": "#cccccc",
        "grid.linestyle": "-",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.5,

        # Lines / markers
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "lines.markeredgewidth": 0.0,

        # Legend — minimal frame
        "legend.frameon": False,
        "legend.borderpad": 0.3,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "legend.labelcolor": "#000000",

        # Errorbar
        "errorbar.capsize": 2.5,

        # Colormap default
        "image.cmap": "cividis",
    })
