"""Rigorous self-checks on the relative-GLV MSB claims (responding to three critiques):
  1. TRANSIENT: when does the IC relaxation end? (discard it before measuring)
  2. STATIONARITY: is the share churn PERSISTENT chaos or slowly RELAXING? -> compare the pooled growth
     variance AND cross-sectional spread Var(ln S) across successive late windows. Flat = persistent;
     decaying = relaxation (the trap this project hit before).
  3. BETA SHAPE: report R^2 of the size-volatility log-log fit (a slope with poor R^2 is shape-spurious;
     the binned points looked humped, not a clean power law).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from explore import integrate

MU, SIGMA, LAM = 2.0, 2.0, 1e-3
S, TMAX, DT = 600, 80.0, 0.5
SEED = 2024


def beta_r2(Sbar, vol, nb=18):
    ok = (Sbar > 0) & (vol > 0) & np.isfinite(Sbar) & np.isfinite(vol)
    Sbar, vol = Sbar[ok], vol[ok]
    if Sbar.size < 50:
        return np.nan, np.nan, None
    o = np.argsort(Sbar); P = np.array_split(np.arange(o.size), nb)
    bx = np.array([Sbar[o][p].mean() for p in P]); by = np.array([np.median(vol[o][p]) for p in P])
    m = (bx > 0) & (by > 0)
    x, y = np.log10(bx[m]), np.log10(by[m])
    c = np.polyfit(x, y, 1)
    yhat = np.polyval(c, x)
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(-c[0]), float(r2), (bx[m], by[m])


def window_msb(W, t, t0, t1, dt=DT):
    """size + growth on survivors-of-this-window, on a fixed dt grid in [t0,t1]."""
    N = W.shape[0]
    sel = (t >= t0) & (t <= t1)
    surv = np.where(W[:, sel].min(1) > 1e-6)[0]
    tg = np.arange(t0, t1 + 1e-9, dt)
    lnS = np.array([np.interp(tg, t, np.log(np.maximum(N * W[i], 1e-12))) for i in surv])
    g = np.diff(lnS, axis=1)
    Sbar = np.exp(lnS).mean(1)
    vol = np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)
    live = (vol > 0) & np.isfinite(vol)
    gl = g[live].ravel(); gl = gl[np.isfinite(gl)]
    return Sbar[live], vol[live], gl, surv.size


if __name__ == "__main__":
    t, W, lnM, _, _ = integrate(SEED, S, MU, SIGMA, lam=LAM, tmax=TMAX, n_eval=4000)

    # (2) cross-sectional spread Var(ln S) over time -> transient end + stationarity
    N = S
    lnS_all = np.log(np.maximum(N * W, 1e-12))
    alive = W > 1e-6
    var_xs = np.array([lnS_all[alive[:, k], k].var() if alive[:, k].sum() > 5 else np.nan
                       for k in range(t.size)])

    print(f"regime mu={MU} sigma={SIGMA} lam={LAM:g}, S={S}, TMAX={TMAX}\n")
    print("(1+2) STATIONARITY across successive 10-unit windows (pooled growth std + cross-sectional Var(lnS)):")
    print(f"  {'window':>12} {'growth_std':>11} {'Var(lnS)_mid':>13} {'beta':>7} {'R2':>6} {'survivors':>9}")
    wins = [(15, 25), (25, 35), (35, 45), (45, 55), (55, 65), (65, 75)]
    for (a, b) in wins:
        Sbar, vol, gl, ns = window_msb(W, t, a, b)
        beta, r2, _ = beta_r2(Sbar, vol)
        mid = (t >= (a + b) / 2 - 0.1) & (t <= (a + b) / 2 + 0.1)
        vm = np.nanmean(var_xs[mid])
        print(f"  [{a:>3},{b:>3}]    {gl.std():>11.3f} {vm:>13.3f} {beta:>7.3f} {r2:>6.2f} {ns:>6d}/{S}")

    # (3) beta + R2 on the full clean window, plus the actual shape
    Sbar, vol, gl, ns = window_msb(W, t, 30, 75)
    beta, r2, bb = beta_r2(Sbar, vol)
    print(f"\n(3) full clean window [30,75]: beta={beta:.3f}  R2={r2:.3f}  ({'OK power law' if r2 > 0.8 else 'POOR FIT -> shape not a clean power law'})")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].plot(t, var_xs, color="#264653")
    ax[0].axvline(30, ls=":", color="#c1121f", label="measurement start")
    ax[0].set(xlabel="t", ylabel=r"cross-sectional Var$(\ln S)$",
              title="Transient + stationarity (flat after start = persistent)"); ax[0].legend()
    ax[1].loglog(Sbar, vol, ".", ms=2, alpha=0.2, color="gray")
    ax[1].loglog(bb[0], bb[1], "o-", ms=5, color="#2a9d8f")
    ax[1].set(xlabel=r"$\bar S_i$", ylabel="volatility",
              title=fr"Size-volatility shape ($\beta={beta:.2f}$, $R^2={r2:.2f}$)")
    sd = [window_msb(W, t, a, b)[2].std() for (a, b) in wins]
    ax[2].plot([0.5 * (a + b) for (a, b) in wins], sd, "o-", color="#457b9d")
    ax[2].set(xlabel="window center t", ylabel="pooled growth std",
              title="Growth volatility vs time (flat = stationary)"); ax[2].set_ylim(bottom=0)
    plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "diagnostics.png"), dpi=120)
    print("\nsaved growing_glv/diagnostics.png")
