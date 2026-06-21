"""Long-time run to locate the TRUE burn-in and test genuine stationarity (a short window can make
slow relaxation look stationary). Integrate to t=300 and track, across 20-unit windows:
  - cross-sectional Var(ln S) and pooled growth std  (flat = stationary, drifting = still relaxing)
  - the decline-branch beta + R^2                     (does the size-volatility law stabilise, and where)
  - ln M(t) linearity                                 (is g_eff a true constant -> steady exponential growth)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from explore import integrate

MU, SIGMA, LAM = 2.0, 2.0, 1e-3
S, TMAX, DT = 600, 300.0, 6000
SEED = 2024


def decline_beta(W, t, t0, t1, dt=0.5):
    N = W.shape[0]
    surv = np.where(W[:, (t >= t0) & (t <= t1)].min(1) > 1e-6)[0]
    if surv.size < 50:
        return np.nan, np.nan, np.nan, surv.size
    tg = np.arange(t0, t1 + 1e-9, dt)
    lnS = np.array([np.interp(tg, t, np.log(np.maximum(N * W[i], 1e-12))) for i in surv])
    g = np.diff(lnS, axis=1)
    Sbar = np.exp(lnS).mean(1)
    vol = np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)
    live = (vol > 0) & np.isfinite(vol)
    Sbar, vol, gstd = Sbar[live], vol[live], g[live].ravel().std()
    o = np.argsort(Sbar); P = np.array_split(np.arange(o.size), 18)
    bx = np.array([Sbar[o][p].mean() for p in P]); by = np.array([np.median(vol[o][p]) for p in P])
    m = (bx > 0) & (by > 0); bx, by = bx[m], by[m]
    pk = int(np.argmax(by))
    if by.size - pk < 4:
        return np.nan, np.nan, gstd, surv.size
    x, y = np.log10(bx[pk:]), np.log10(by[pk:])
    c = np.polyfit(x, y, 1); r2 = 1 - np.sum((y - np.polyval(c, x)) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(-c[0]), float(r2), gstd, surv.size


if __name__ == "__main__":
    t, W, lnM, _, _ = integrate(SEED, S, MU, SIGMA, lam=LAM, tmax=TMAX, n_eval=DT)
    N = S
    lnS_all = np.log(np.maximum(N * W, 1e-12)); alive = W > 1e-6
    var_xs = np.array([lnS_all[alive[:, k], k].var() if alive[:, k].sum() > 5 else np.nan for k in range(t.size)])

    wins = [(a, a + 20) for a in range(20, int(TMAX) - 20, 20)]
    print(f"regime mu={MU} sigma={SIGMA} lam={LAM:g}, S={S}, TMAX={TMAX}\n")
    print(f"{'window':>12} {'growth_std':>11} {'Var(lnS)':>9} {'beta_decl':>10} {'R2':>6} {'surv':>6}")
    bs, gstds, centers = [], [], []
    for (a, b) in wins:
        beta, r2, gstd, ns = decline_beta(W, t, a, b)
        vm = np.nanmean(var_xs[(t >= a) & (t <= b)])
        print(f"  [{a:>3},{b:>3}]  {gstd:>11.3f} {vm:>9.2f} {beta:>10.3f} {r2:>6.2f} {ns:>5d}")
        bs.append(beta); gstds.append(gstd); centers.append((a + b) / 2)

    # ln M slope over first vs second half -> is g_eff constant?
    h = t.size // 2
    g1 = np.polyfit(t[:h], lnM[:h], 1)[0]; g2 = np.polyfit(t[h:], lnM[h:], 1)[0]
    print(f"\nln M slope: first half g_eff={g1:.3f}, second half g_eff={g2:.3f}  (equal -> steady exponential)")

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(t, var_xs, color="#264653"); ax[0, 0].set(xlabel="t", ylabel=r"Var$(\ln S)$", title="Cross-sectional spread (flat = stationary)")
    ax[0, 1].plot(t, lnM / np.log(10), color="#457b9d"); ax[0, 1].set(xlabel="t", ylabel=r"$\log_{10}M$", title=f"Economy (g_eff: {g1:.2f} -> {g2:.2f})")
    ax[1, 0].plot(centers, gstds, "o-", color="#c1121f"); ax[1, 0].set(xlabel="window center t", ylabel="pooled growth std", title="Growth volatility vs time", ylim=(0, None))
    ax[1, 1].plot(centers, bs, "o-", color="#2a9d8f"); ax[1, 1].set(xlabel="window center t", ylabel=r"decline-branch $\beta$", title="Size-volatility exponent vs time")
    plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "long_run.png"), dpi=120)
    print("\nsaved growing_glv/long_run.png")
