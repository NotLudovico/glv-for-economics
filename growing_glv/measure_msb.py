"""Rigorous MSB measurement on the relative-GLV growing+chaotic regime.

At a fixed good regime (sigma=2, mu=1, lam=1e-2) measure, in the STATIONARY window
(after the IC transient), on a sensible fixed Delta-t grid:
  - beta: size-volatility exponent (binned-median loglog of MAD-volatility vs mean size)
  - growth tent: skew, excess kurtosis, and a Dt-robust 99.9pct/MAD tail, swept over Dt
  - stationarity: early-vs-late skew/beta drift
Size is S_i = N w_i (shares); growth g_i = Delta ln S_i is idiosyncratic (M divides out).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from explore import integrate

S, MU, SIGMA, LAM = 600, 1.0, 2.0, 1e-2
TMAX, T_BURN = 45.0, 18.0                  # integrate to 45, measure in [18, 45] (post-transient)
SEED = 2024


def regrid_logS(t, W, t_grid):
    """log size ln(N w_i) interpolated onto t_grid -> (S, len(t_grid))."""
    lnS = np.log(np.maximum(W.shape[0] * W, 1e-12))
    return np.array([np.interp(t_grid, t, lnS[i]) for i in range(W.shape[0])])


def binned_beta(Sbar, vol, nb=18):
    ok = np.isfinite(Sbar) & np.isfinite(vol) & (Sbar > 0) & (vol > 0)
    Sbar, vol = Sbar[ok], vol[ok]
    if Sbar.size < 50:
        return np.nan, None
    o = np.argsort(Sbar); xs, ys = Sbar[o], vol[o]
    P = np.array_split(np.arange(xs.size), nb)
    bx = np.array([xs[p].mean() for p in P]); by = np.array([np.median(ys[p]) for p in P])
    m = (bx > 0) & (by > 0)
    beta = -np.polyfit(np.log10(bx[m]), np.log10(by[m]), 1)[0]
    return float(beta), (bx[m], by[m])


def msb_on_grid(lnS):
    """beta inputs + pooled growth from a (S, n) log-size grid."""
    g = np.diff(lnS, axis=1)
    Sbar = np.exp(lnS).mean(1)
    vol = np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)
    live = (Sbar > 0) & (vol > 0) & np.isfinite(vol)
    gl = g[live].ravel(); gl = gl[np.isfinite(gl)]
    return Sbar[live], vol[live], gl


if __name__ == "__main__":
    t, W, lnM, C_eff, ok = integrate(SEED, S, MU, SIGMA, tmax=TMAX, n_eval=3000, lam=LAM)
    g_eff = float(np.polyfit(t[t > T_BURN], lnM[t > T_BURN], 1)[0])
    print(f"regime: S={S} mu={MU} sigma={SIGMA} lam={LAM}  g_eff={g_eff:.3f}  ok={ok}")
    print(f"survivors in window: {int((W[:, t > T_BURN].min(1) > 1e-6).sum())}/{S}\n")

    print("Dt-robustness of the growth tent (pooled g = Delta ln S, window [18,45]):")
    print(f"{'Dt':>6} {'n_inc':>6} {'skew':>8} {'exkurt':>9} {'q99.9/MAD':>10} {'beta':>7}")
    rows = []
    for dt in [0.1, 0.25, 0.5, 1.0, 2.0]:
        tg = np.arange(T_BURN, TMAX + 1e-9, dt)
        if tg.size < 4:
            continue
        lnS = regrid_logS(t, W, tg)
        Sbar, vol, gl = msb_on_grid(lnS)
        beta, _ = binned_beta(Sbar, vol)
        g0 = gl - np.median(gl); mad = np.median(np.abs(g0))
        qtail = np.percentile(np.abs(g0), 99.9) / mad if mad > 0 else np.nan
        print(f"{dt:>6.2f} {lnS.shape[1]-1:>6d} {skew(gl):>8.2f} {kurtosis(gl):>9.1f} {qtail:>10.2f} {beta:>7.3f}")
        rows.append((dt, gl, Sbar, vol, beta))

    # stationarity: skew early vs late half of the window at Dt=0.5
    tg = np.arange(T_BURN, TMAX + 1e-9, 0.5); lnS = regrid_logS(t, W, tg); h = lnS.shape[1] // 2
    _, _, ge = msb_on_grid(lnS[:, :h + 1]); _, _, gl_ = msb_on_grid(lnS[:, h:])
    print(f"\nstationarity (Dt=0.5): skew early={skew(ge):.2f} late={skew(gl_):.2f}  "
          f"exkurt early={kurtosis(ge):.1f} late={kurtosis(gl_):.1f}")

    # figure at Dt=0.5: size-volatility + growth PDF vs Laplace
    dt, gl, Sbar, vol, beta = [r for r in rows if abs(r[0] - 0.5) < 1e-9][0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    _, bb = binned_beta(Sbar, vol)
    ax[0].loglog(Sbar, vol, ".", ms=2, alpha=0.2, color="gray")
    if bb is not None:
        ax[0].loglog(bb[0], bb[1], "o", ms=5, color="#2a9d8f")
    ax[0].set(xlabel=r"$\bar S_i$", ylabel=r"volatility", title=fr"Size-volatility ($\beta={beta:.3f}$)")
    z = (gl - gl.mean()) / gl.std()
    h_, e_ = np.histogram(z, bins=120, density=True); m_ = h_ > 0
    ax[1].semilogy(0.5 * (e_[:-1] + e_[1:])[m_], h_[m_], color="#457b9d", label="growth")
    zz = np.linspace(-9, 9, 200)
    ax[1].semilogy(zz, np.exp(-np.abs(zz) * np.sqrt(2)) / np.sqrt(2), "k--", lw=1, label="Laplace")
    ax[1].set(xlabel=r"standardized $\Delta\ln S$", ylabel="PDF",
              title=fr"Growth tent ($\Delta t=0.5$, skew={skew(gl):.2f}, exk={kurtosis(gl):.0f})")
    ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "msb_best.png"), dpi=120)
    print("\nsaved growing_glv/msb_best.png   (empirical MSB: beta~0.15-0.20, symmetric, fat tent)")
