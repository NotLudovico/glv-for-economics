"""Is beta>0 reachable from interaction structure ALONE? Map (mu, sigma) at lam=0 (no floor),
measuring beta on survivors (natural survivorship conditioning) plus tent skew + coexistence.
Looking for a corner with beta>0 AND symmetric tent AND enough coexistence + growth.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from explore import integrate
from measure_msb import regrid_logS, binned_beta, msb_on_grid

S = 600
TMAX, T_BURN, DT = 45.0, 18.0, 0.5
SEED = 2024
MUS = [-1.0, 0.0, 1.0, 2.0, 3.0]
SIGS = [1.5, 2.0, 2.5]


def measure(mu, sigma):
    t, W, lnM, C_eff, ok = integrate(SEED, S, mu, sigma, tmax=TMAX, n_eval=3000, lam=0.0, floor_mode="share")
    g_eff = float(np.polyfit(t[t > T_BURN], lnM[t > T_BURN], 1)[0])
    win = t > T_BURN
    surv = W[:, win].min(1) > 1e-6                                   # survivors over the whole window
    tg = np.arange(T_BURN, TMAX + 1e-9, DT)
    lnS = regrid_logS(t, W[surv], tg) if surv.sum() >= 50 else None  # measure on survivors only
    if lnS is None:
        return dict(mu=mu, sigma=sigma, g_eff=g_eff, surv=int(surv.sum()),
                    beta=np.nan, skew=np.nan, exk=np.nan)
    Sbar, vol, gl = msb_on_grid(lnS)
    beta, _ = binned_beta(Sbar, vol)
    return dict(mu=mu, sigma=sigma, g_eff=g_eff, surv=int(surv.sum()),
                beta=beta, skew=float(skew(gl)), exk=float(kurtosis(gl)))


if __name__ == "__main__":
    grid = np.full((len(SIGS), len(MUS)), np.nan)
    print(f"lam=0 (pure interactions), beta on survivors, window [{T_BURN},{TMAX}] Dt={DT}, S={S}")
    print(f"{'mu':>5} {'sigma':>6} {'g_eff':>7} {'coexist':>9} {'beta':>8} {'skew':>7} {'exkurt':>8}")
    for j, sig in enumerate(SIGS):
        for i, mu in enumerate(MUS):
            r = measure(mu, sig)
            grid[j, i] = r["beta"]
            flag = "  <-- beta>0" if (np.isfinite(r["beta"]) and r["beta"] > 0.05) else ""
            bstr = f"{r['beta']:>8.3f}" if np.isfinite(r["beta"]) else f"{'n/a':>8}"
            sstr = f"{r['skew']:>7.2f}" if np.isfinite(r["skew"]) else f"{'n/a':>7}"
            estr = f"{r['exk']:>8.1f}" if np.isfinite(r["exk"]) else f"{'n/a':>8}"
            print(f"{mu:>5.1f} {sig:>6.1f} {r['g_eff']:>7.2f} {r['surv']:>5d}/{S} {bstr} {sstr} {estr}{flag}")

    fig, ax = plt.subplots(figsize=(7, 4.6))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-0.2, vmax=0.2,
                   extent=[min(MUS) - 0.5, max(MUS) + 0.5, min(SIGS) - 0.25, max(SIGS) + 0.25])
    ax.set(xlabel=r"$\mu$", ylabel=r"$\sigma$", title=r"size-volatility $\beta$  (red = $\beta>0$ empirical sign)")
    for j, sig in enumerate(SIGS):
        for i, mu in enumerate(MUS):
            if np.isfinite(grid[j, i]):
                ax.text(mu, sig, f"{grid[j, i]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label=r"$\beta$")
    plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "musigma_map.png"), dpi=120)
    print("\nsaved growing_glv/musigma_map.png  (empirical beta ~ +0.15..0.20; want red)")
