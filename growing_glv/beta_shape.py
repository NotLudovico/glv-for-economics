"""Honest characterization of the size-volatility relation (post-transient, T_BURN=35).
The relation is a plateau at small/mid S + a power-law decline for large S, so a single fit is
shape-spurious. Fit (a) full range and (b) the declining branch above the binned-volatility peak,
reporting R^2 for each, and show where a power law actually holds.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from explore import integrate

MU, SIGMA, LAM = 2.0, 2.0, 1e-3
S, TMAX, T_BURN, DT = 600, 80.0, 35.0, 0.5
SEED = 2024


def binned(Sbar, vol, nb=18):
    ok = (Sbar > 0) & (vol > 0) & np.isfinite(Sbar) & np.isfinite(vol)
    Sbar, vol = Sbar[ok], vol[ok]
    o = np.argsort(Sbar); P = np.array_split(np.arange(o.size), nb)
    bx = np.array([Sbar[o][p].mean() for p in P]); by = np.array([np.median(vol[o][p]) for p in P])
    m = (bx > 0) & (by > 0)
    return bx[m], by[m]


def fit(bx, by):
    if bx.size < 3:
        return np.nan, np.nan
    x, y = np.log10(bx), np.log10(by)
    c = np.polyfit(x, y, 1); yhat = np.polyval(c, x)
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(-c[0]), float(r2)


if __name__ == "__main__":
    t, W, lnM, _, _ = integrate(SEED, S, MU, SIGMA, lam=LAM, tmax=TMAX, n_eval=4000)
    sel = t >= T_BURN
    surv = np.where(W[:, sel].min(1) > 1e-6)[0]
    tg = np.arange(T_BURN, t[-1] + 1e-9, DT)
    lnS = np.array([np.interp(tg, t, np.log(np.maximum(S * W[i], 1e-12))) for i in surv])
    g = np.diff(lnS, axis=1)
    Sbar = np.exp(lnS).mean(1)
    vol = np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)
    live = (vol > 0) & np.isfinite(vol)
    Sbar, vol = Sbar[live], vol[live]

    bx, by = binned(Sbar, vol)
    peak = int(np.argmax(by))                       # plateau ends at the volatility peak
    b_full, r2_full = fit(bx, by)
    b_dec, r2_dec = fit(bx[peak:], by[peak:])       # declining branch only
    print(f"regime mu={MU} sigma={SIGMA} lam={LAM:g}, post-transient window [{T_BURN},{t[-1]:.0f}], survivors={surv.size}/{S}")
    print(f"  full range   : beta={b_full:.3f}  R2={r2_full:.3f}")
    print(f"  decline (S>={bx[peak]:.2f}): beta={b_dec:.3f}  R2={r2_dec:.3f}   ({by.size-peak} bins)")
    print(f"  plateau below peak: vol ~ const ~ {np.median(by[:peak+1]):.3f}  for S in [{bx[0]:.2g}, {bx[peak]:.2g}]")

    fig, ax = plt.subplots(figsize=(7.2, 5))
    ax.loglog(Sbar, vol, ".", ms=2, alpha=0.2, color="gray")
    ax.loglog(bx, by, "o", ms=5, color="#2a9d8f", label="bin median", zorder=3)
    xx = np.logspace(np.log10(bx[peak]), np.log10(bx[-1]), 50)
    ax.loglog(xx, by[peak] * (xx / bx[peak]) ** (-b_dec), "--", color="#c1121f", lw=1.8,
              label=fr"decline fit $\beta={b_dec:.2f}$, $R^2={r2_dec:.2f}$")
    ax.axhline(np.median(by[:peak + 1]), color="#264653", ls=":", lw=1.2,
               label=fr"small-S plateau (vol$\approx${np.median(by[:peak+1]):.2f})")
    ax.set(xlabel=r"time-average size $\bar S_i$", ylabel="MAD volatility",
           title=f"Size-volatility is plateau + decline, NOT one power law\n(full-range fit R2={r2_full:.2f} is shape-spurious)")
    ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "beta_shape.png"), dpi=120)
    print("\nsaved growing_glv/beta_shape.png")
