"""Make figures from phase_space_data.npz (produced by phase_space_compute.py). Re-run freely to
re-classify / re-style without recomputing. Thresholds are at the top so the boundaries are tunable.

Categories are the GENUINE phases only -- frozen/chaotic (the sigma_c=sqrt(2) dynamical transition),
shrinking/growing (the g_eff=0 sign change), and condensed (coexistence collapse). The growth RATE is a
continuous quantity (no transition between 'steady' and 'fast' growth), so it is shown as overlaid g_eff
contours rather than as discrete phases.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "phase_space_data.npz")

# ----- classification thresholds (tune here) -----
CHAOS_GSTD = 0.03    # a seed is chaotic if late-window growth std exceeds this
CHAOS_FRAC = 0.5     # a cell is chaotic if at least this fraction of seeds are
STATIONARY = 0.4     # and if median gstd_late/gstd_mid exceeds this (else relaxing, not chaotic)
COND_SURV  = 0.06    # condensed if median survivor fraction below this
SHRINK_G   = -0.05   # shrinking if g_eff below this (the only growth boundary; g_eff is continuous above)
G_CONTOURS = [1.0, 5.0, 20.0, 50.0]   # g_eff levels drawn as overlaid contours (magnitude, not a phase)


def aggregate(d):
    MUS, SIGS = d["mus"], d["sigs"]
    # restrict to computed sigma-rows (the sweep was stopped early; uncomputed rows are all-NaN)
    done = d["rows_done"].astype(bool) if "rows_done" in d.files else np.ones(len(SIGS), bool)
    SIGS = SIGS[done]
    med = lambda k: np.nanmedian(d[k][done], axis=2)
    g_eff = med("g_eff"); gl = med("gstd_late"); gm = med("gstd_mid")
    surv = med("surv_frac"); beta = med("beta")
    capped_frac = np.mean(d["capped"][done], axis=2) if "capped" in d.files else np.zeros_like(surv)
    chaos_frac = np.mean(d["gstd_late"][done] > CHAOS_GSTD, axis=2)
    ratio = np.where(gm > 1e-6, gl / gm, 0.0)
    chaotic = (chaos_frac >= CHAOS_FRAC) & (ratio >= STATIONARY)
    # condensed/explosive: coexistence collapses, or the cell was so stiff (explosive growth) it hit the
    # solver cap on most seeds (the fast-growth monopoly corner)
    condensed = (surv < COND_SURV) | (capped_frac > 0.5)
    # genuine phases: 0 frozen-shrink, 1 frozen-grow, 2 chaotic-shrink, 3 chaotic-grow, 4 condensed
    phase = np.zeros_like(g_eff, dtype=int)
    for j in range(len(SIGS)):
        for i in range(len(MUS)):
            if condensed[j, i]:
                phase[j, i] = 4
            elif chaotic[j, i]:
                phase[j, i] = 2 if g_eff[j, i] < SHRINK_G else 3
            else:
                phase[j, i] = 0 if g_eff[j, i] < SHRINK_G else 1
    return MUS, SIGS, g_eff, gl, surv, beta, chaos_frac, phase


def grid_edges(c):
    c = np.asarray(c, float); e = np.empty(c.size + 1)
    e[1:-1] = 0.5 * (c[:-1] + c[1:]); e[0] = c[0] - (c[1] - c[0]) / 2; e[-1] = c[-1] + (c[-1] - c[-2]) / 2
    return e


if __name__ == "__main__":
    d = np.load(DATA, allow_pickle=True)
    MUS, SIGS, g_eff, gl, surv, beta, chaos_frac, phase = aggregate(d)
    # thesis cooperative-positive convention: mu_chapter = -mu_sweep (competition is mu<0); reorder increasing
    order = np.argsort(-MUS)
    MUS = (-MUS)[order]
    g_eff = g_eff[:, order]; gl = gl[:, order]; surv = surv[:, order]
    beta = beta[:, order]; chaos_frac = chaos_frac[:, order]; phase = phase[:, order]
    me, se = grid_edges(MUS), grid_edges(SIGS)

    # ---- main phase diagram: genuine phases + g_eff magnitude as contours ----
    labels = ["frozen, shrinking", "frozen, growing", "chaotic, shrinking",
              "chaotic, growing", "condensed / explosive"]
    colors = ["#b8c4d0", "#cfe8df", "#3b5b8c", "#2a9d8f", "#d9a441"]
    cmap = ListedColormap(colors); norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)
    fig, ax = plt.subplots(figsize=(9.2, 6))
    ax.pcolormesh(me, se, phase, cmap=cmap, norm=norm, shading="flat", edgecolors="white", linewidth=0.3)
    # genuine boundaries
    ax.axhline(np.sqrt(2), color="#c1121f", ls="--", lw=1.6)
    ax.text(me[0] + 0.05, np.sqrt(2) + 0.04, r"$\sigma_c=\sqrt{2}$ (frozen/chaotic)", color="#c1121f", fontsize=9)
    ax.contour(MUS, SIGS, g_eff, levels=[0.0], colors="k", linewidths=1.8)
    # growth magnitude as continuous contours (NOT a phase boundary)
    cs = ax.contour(MUS, SIGS, g_eff, levels=G_CONTOURS, colors="0.25", linewidths=0.8, linestyles=":")
    ax.clabel(cs, fmt=lambda v: f"$g_{{\\rm eff}}{{=}}{v:g}$", fontsize=7, inline=True)
    OP = (-1.76, 1.75)   # locked operating point, chapter convention (mu=-1.76, sigma=1.75)
    ax.plot(*OP, marker="*", ms=22, mfc="white", mec="k", mew=1.4, zorder=6)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(facecolor=colors[i], edgecolor="k", label=labels[i]) for i in range(5)]
    handles += [Line2D([], [], color="k", lw=1.8, label=r"$g_{\rm eff}=0$ (shrink/grow)"),
                Line2D([], [], color="0.25", lw=0.8, ls=":", label=r"$g_{\rm eff}$ contours"),
                Line2D([], [], marker="*", ls="", mfc="white", mec="k", ms=14, label="operating point")]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.95)
    ax.set(xlabel=r"mean interaction $\mu$ (competitive for $\mu<0$)", ylabel=r"disorder $\sigma$",
           title="Phase diagram of the relative GLV")
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "phase_space_main.png"), dpi=140); plt.close()

    # ---- supporting heatmaps: g_eff, chaos fraction, beta (chaotic cells) ----
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    gmax = np.nanpercentile(np.abs(g_eff), 95)
    im0 = ax[0].pcolormesh(me, se, g_eff, cmap="RdBu_r", norm=TwoSlopeNorm(0, -gmax, gmax), shading="flat")
    ax[0].set_title(r"growth rate $g_{\mathrm{eff}}$ (blue=shrink, red=grow)"); fig.colorbar(im0, ax=ax[0])
    im1 = ax[1].pcolormesh(me, se, chaos_frac, cmap="viridis", vmin=0, vmax=1, shading="flat")
    ax[1].set_title("fraction of seeds chaotic"); fig.colorbar(im1, ax=ax[1])
    bshow = np.where(np.isin(phase, [2, 3]), beta, np.nan)
    im2 = ax[2].pcolormesh(me, se, bshow, cmap="magma", vmin=0, vmax=0.8, shading="flat")
    ax[2].set_title(r"size-volatility $\beta$ (chaotic cells)"); fig.colorbar(im2, ax=ax[2])
    for a in ax:
        a.axhline(np.sqrt(2), color="w", ls="--", lw=1); a.set(xlabel=r"$\mu$", ylabel=r"$\sigma$")
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "phase_space_fields.png"), dpi=140); plt.close()

    print("saved phase_space_main.png and phase_space_fields.png")
    print(f"grid {len(MUS)}x{len(SIGS)}; chaotic+growing (firm-growth region): {int(np.sum(phase == 3))} cells; "
          f"chaotic+shrinking: {int(np.sum(phase == 2))}; condensed: {int(np.sum(phase == 4))}")
