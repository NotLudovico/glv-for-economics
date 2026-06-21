"""Moran/empirical per-period delisting protocol (vs whole-window survivorship).

A firm contributes a growth increment g = Delta ln S only when LISTED at BOTH endpoints of that
increment (size above S_min). A declining firm contributes its decrements up to the period it drops
below S_min, then delists -> the left tail (crashers' decline) is restored WITHOUT a floor, while the
panel is still survivorship-conditioned (-> beta>0). Test at lam=0: does beta>0 AND skew~0 co-occur?

Sizes are S_i = N w_i with <S_i> = 1 (sum_i S_i = N over N firms), so S_min is a fraction of the
average firm size.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from explore import integrate
from measure_msb import regrid_logS, binned_beta

S = 600
TMAX, T_BURN, DT = 45.0, 18.0, 0.5
SEED = 2024
CORNERS = [(2.0, 2.0), (3.0, 2.5)]
SMINS = [0.01, 0.05, 0.1, 0.3]


def moran_msb(lnS, s_min, min_inc=5):
    Sarr = np.exp(lnS)                                   # relative sizes S_i = N w_i
    listed = Sarr > s_min
    g = np.diff(lnS, axis=1)
    pair = listed[:, :-1] & listed[:, 1:]                # increment counts only if listed both endpoints
    g_pool = g[pair]; g_pool = g_pool[np.isfinite(g_pool)]
    Sbar, vol = [], []
    for i in range(lnS.shape[0]):
        gi = g[i][pair[i]]; gi = gi[np.isfinite(gi)]
        if gi.size >= min_inc:
            Sbar.append(Sarr[i][listed[i]].mean())
            vol.append(np.sqrt(np.pi / 2) * np.abs(gi - gi.mean()).mean())
    Sbar, vol = np.array(Sbar), np.array(vol)
    beta, bb = binned_beta(Sbar, vol)
    return g_pool, Sbar, vol, beta, bb, float(listed.mean()), len(Sbar)


if __name__ == "__main__":
    best = None
    for mu, sig in CORNERS:
        t, W, lnM, C_eff, ok = integrate(SEED, S, mu, sig, tmax=TMAX, n_eval=3000, lam=0.0)
        tg = np.arange(T_BURN, TMAX + 1e-9, DT)
        lnS = regrid_logS(t, W, tg)                      # ALL firms (delisting handled per-period below)
        print(f"\nmu={mu} sigma={sig}  (lam=0, Moran per-period delisting)")
        print(f"{'S_min':>7} {'listed%':>8} {'n_beta':>7} {'beta':>8} {'skew':>7} {'exkurt':>8}")
        for sm in SMINS:
            g_pool, Sbar, vol, beta, bb, lf, nb = moran_msb(lnS, sm)
            print(f"{sm:>7.2f} {100*lf:>7.1f}% {nb:>7d} {beta:>8.3f} {skew(g_pool):>7.2f} {kurtosis(g_pool):>8.1f}")
            score = -(abs(beta - 0.175) + abs(skew(g_pool)))   # closeness to (beta=0.175, skew=0)
            if np.isfinite(beta) and (best is None or score > best[0]):
                best = (score, mu, sig, sm, g_pool, Sbar, vol, beta, bb)

    _, mu, sig, sm, g_pool, Sbar, vol, beta, bb = best
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].loglog(Sbar, vol, ".", ms=2, alpha=0.25, color="gray")
    if bb is not None:
        ax[0].loglog(bb[0], bb[1], "o", ms=5, color="#2a9d8f")
    ax[0].set(xlabel=r"$\bar S_i$", ylabel="volatility", title=fr"Size-volatility ($\beta={beta:.3f}$)")
    # MSB rescaling: subtract mean, divide by sqrt(pi/2)*MAD (NOT std; std is inflated by the fat tails)
    z = (g_pool - g_pool.mean()) / (np.sqrt(np.pi / 2) * np.abs(g_pool - g_pool.mean()).mean())
    h, e = np.histogram(z, bins=120, density=True); m = h > 0
    ax[1].semilogy(0.5 * (e[:-1] + e[1:])[m], h[m], color="#457b9d", label="growth")
    zz = np.linspace(-9, 9, 200)
    ax[1].semilogy(zz, 0.5 * np.sqrt(np.pi / 2) * np.exp(-np.abs(zz) * np.sqrt(np.pi / 2)), "k--", lw=1, label="Laplace")
    ax[1].set(xlabel=r"rescaled $\Delta\ln S$  $(\sqrt{\pi/2}\,\mathrm{MAD})$", ylabel="PDF",
              title=fr"Tent (Moran protocol, $\mu={mu},\sigma={sig}$, S_min={sm}, skew={skew(g_pool):.2f})")
    ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "moran_protocol.png"), dpi=120)
    print(f"\nbest (closest to beta=0.175, skew=0): mu={mu} sigma={sig} S_min={sm} "
          f"beta={beta:.3f} skew={skew(g_pool):.2f}  -> growing_glv/moran_protocol.png")
