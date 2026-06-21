"""How right-skewed is the tent, really? Moment skew over-weights extreme outliers on a fat-tailed
distribution. Add robust measures: Bowley quantile skew, left/right tail-magnitude ratios, and a
FOLDED-tail plot (right side vs |left side|) so the body/tail asymmetry is visible directly.
Measured at the best beta>0 regimes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from explore import integrate
from measure_msb import regrid_logS, msb_on_grid, binned_beta

S = 600
TMAX, T_BURN, DT = 45.0, 18.0, 0.5
SEED = 2024
REGIMES = [(3.0, 2.5, 3e-3), (2.0, 2.0, 0.0), (2.0, 2.0, 3e-3)]   # (mu, sigma, lam)


def growth_pool(mu, sigma, lam):
    t, W, lnM, *_ = integrate(SEED, S, mu, sigma, tmax=TMAX, n_eval=3000, lam=lam)
    win = t > T_BURN
    surv = W[:, win].min(1) > 1e-6
    tg = np.arange(T_BURN, TMAX + 1e-9, DT)
    lnS = regrid_logS(t, W[surv], tg)
    Sbar, vol, gl = msb_on_grid(lnS)
    beta, _ = binned_beta(Sbar, vol)
    return gl, beta, int(surv.sum())


def asym(g):
    g = g - np.median(g)                                   # center on median
    q = np.percentile(g, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    Q1, Q2, Q3 = q[3], q[4], q[5]
    bowley = (Q3 + Q1 - 2 * Q2) / (Q3 - Q1)                # robust skew, in [-1,1]
    # tail magnitude ratios: |right pct| / |left pct| (1.0 = symmetric)
    r90 = abs(q[7]) / abs(q[1]); r95 = abs(q[7]) / abs(q[1])
    rr = {p: abs(np.percentile(g, 100 - p)) / abs(np.percentile(g, p)) for p in [10, 5, 1]}
    mad = np.median(np.abs(g))
    # fraction beyond +/- 3 MAD on each side
    fr = (g > 3 * mad).mean(); fl = (g < -3 * mad).mean()
    return dict(moment=float(skew(g)), bowley=float(bowley),
                ratio10=rr[10], ratio5=rr[5], ratio1=rr[1],
                frac_R3=float(fr), frac_L3=float(fl))


if __name__ == "__main__":
    fig, axes = plt.subplots(1, len(REGIMES), figsize=(5 * len(REGIMES), 4.4))
    for ax, (mu, sig, lam) in zip(np.atleast_1d(axes), REGIMES):
        g, beta, ns = growth_pool(mu, sig, lam)
        a = asym(g)
        print(f"\nmu={mu} sigma={sig} lam={lam:g}  (beta={beta:.3f}, survivors={ns}/{S}, n_g={g.size})")
        print(f"  moment skew = {a['moment']:+.2f}   Bowley quantile skew = {a['bowley']:+.3f}  (|.|<0.1 ~ symmetric)")
        print(f"  tail magnitude ratio R/L:  90/10pct={a['ratio10']:.2f}  95/5pct={a['ratio5']:.2f}  99/1pct={a['ratio1']:.2f}  (1.0=symmetric)")
        print(f"  fraction beyond 3*MAD:  right={100*a['frac_R3']:.2f}%  left={100*a['frac_L3']:.2f}%")
        # folded-tail plot: right side vs |left side|, standardized by MAD
        gc = (g - np.median(g)) / np.median(np.abs(g - np.median(g)))
        R = gc[gc > 0]; L = -gc[gc < 0]
        bins = np.linspace(0, 12, 60)
        hR, _ = np.histogram(R, bins=bins, density=True); hL, _ = np.histogram(L, bins=bins, density=True)
        c = 0.5 * (bins[:-1] + bins[1:])
        ax.semilogy(c[hR > 0], hR[hR > 0], "-", color="#2a9d8f", label="right tail")
        ax.semilogy(c[hL > 0], hL[hL > 0], "--", color="#c1121f", label="|left tail|")
        ax.set(xlabel="|standardized growth| / MAD", ylabel="PDF",
               title=fr"$\mu={mu},\sigma={sig},\lambda={lam:g}$  ($\beta={beta:.2f}$, Bowley={a['bowley']:+.2f})")
        ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "asymmetry.png"), dpi=120)
    print("\nsaved growing_glv/asymmetry.png  (overlapping curves = symmetric; right above left = right-skew)")
