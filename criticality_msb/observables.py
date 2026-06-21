"""Analysis toolkit: derive MSB observables from a Run's stored log-abundance. Nothing re-simulates.

The Moran cross-section, the analysis horizon Δt, and the extinction floor are all chosen HERE at
analysis time (the dataset stores raw log-abundance), so the same runs support many analyses.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import skew, kurtosis


def moran_logS(logx):
    """Cross-sectional (Moran) normalized log-size: logS_i = logx_i - logsumexp_j(logx_j) + log(N)."""
    return logx - logsumexp(logx, axis=0, keepdims=True) + np.log(logx.shape[0])


def _step(run, dt):
    return max(int(round(dt / run.dt_store)), 1)              # snapshots per analysis horizon dt


def growth(run, dt=1.0, floor=None, burn_in=0.0):
    """Per-firm growth g = Δln S at horizon dt (a multiple of dt_store). Returns (logS_grid, g).
    burn_in: fraction of the trajectory's snapshots discarded from the start (the IC-relaxation transient)."""
    logx = run.logx if floor is None else np.maximum(run.logx, np.log(floor))
    if burn_in > 0.0:
        logx = logx[:, int(burn_in * logx.shape[1]):]
    logS = moran_logS(logx)[:, ::_step(run, dt)]
    return logS, np.diff(logS, axis=1)


def size_volatility(run, dt=1.0, floor=None, burn_in=0.0):
    """Per-firm time-average size Sbar and MAD growth-volatility at horizon dt."""
    logS, g = growth(run, dt, floor, burn_in)
    Sbar = np.exp(logS).mean(1)
    vol = np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)
    return Sbar, vol


def beta_fit(Sbar, vol, mask=None, nbins=18):
    """Binned-median size–volatility exponent β (vol ∝ Sbar^-β). NaN if too few firms."""
    if mask is not None:
        Sbar, vol = Sbar[mask], vol[mask]
    keep = (Sbar > 0) & (vol > 0) & np.isfinite(vol)
    Sbar, vol = Sbar[keep], vol[keep]
    if Sbar.size < 50:
        return np.nan
    o = np.argsort(Sbar); xs, ys = Sbar[o], vol[o]
    parts = np.array_split(np.arange(len(xs)), nbins)
    bx = np.array([xs[p].mean() for p in parts]); by = np.array([np.median(ys[p]) for p in parts])
    m = (bx > 0) & (by > 0)
    return -np.polyfit(np.log10(bx[m]), np.log10(by[m]), 1)[0] if m.sum() >= 2 else np.nan


def listed_size_volatility(run, dt=1.0, floor=1e-4, min_growths=2):
    """MSB-style per-interval listing (matches Compustat): a firm contributes its growth over [t, t+dt]
    only in years it is 'listed' — size > floor at BOTH endpoints — so a firm that dies drops out year-by-year
    and one that recovers re-enters. Per firm: average listed (normalized) size and MAD×√(π/2) volatility over
    its ≥min_growths listed growths. This is the faithful analog of MSB's '≥2 annual growth rates' sample,
    unlike the whole-window persistent_mask."""
    logS, g = growth(run, dt=dt)
    Slin = np.exp(logS)
    listed = (Slin[:, :-1] > floor) & (Slin[:, 1:] > floor)          # (S, n_inc): listed over interval i
    Sbar = np.full(g.shape[0], np.nan); vol = np.full(g.shape[0], np.nan)
    for i in range(g.shape[0]):
        m = listed[i]
        if m.sum() < min_growths:
            continue
        gi = g[i, m]
        Sbar[i] = Slin[i, :-1][m].mean()
        vol[i] = np.sqrt(np.pi / 2) * np.abs(gi - gi.mean()).mean()
    ok = np.isfinite(Sbar) & np.isfinite(vol)
    return Sbar[ok], vol[ok]


def delisted_size_volatility(run, dt=1.0, floor=1e-4, min_growths=2, burn_in=0.0):
    """Permanent-delisting (first-passage) listing — the most faithful to real firm panels: a firm
    contributes its history UP TO its first crash below `floor`, then is gone for good (a bankrupt firm
    stays dead). This keeps each firm's valid pre-death life but DROPS the GLV's unrealistic crash-and-
    recover growth that the per-interval rule (listed_size_volatility) wrongly includes."""
    logS, g = growth(run, dt=dt, burn_in=burn_in)
    Slin = np.exp(logS)
    Sbar = np.full(g.shape[0], np.nan); vol = np.full(g.shape[0], np.nan)
    for i in range(g.shape[0]):
        below = Slin[i] < floor
        if below.any():
            crash = int(np.argmax(below))              # first snapshot below the floor
            ng = crash - 1                             # growths strictly before it (between live snapshots)
            life = slice(0, crash)
        else:
            ng = g.shape[1]; life = slice(None)        # never crashes -> full history
        if ng < min_growths:
            continue
        gi = g[i, :ng]
        Sbar[i] = Slin[i, life].mean()
        vol[i] = np.sqrt(np.pi / 2) * np.abs(gi - gi.mean()).mean()
    ok = np.isfinite(Sbar) & np.isfinite(vol)
    return Sbar[ok], vol[ok]


def delisted_growth(run, dt=1.0, floor=1e-4, min_growths=2, burn_in=0.0):
    """Pooled growth rates over the listed (pre-first-crash) firm-years — the growth-distribution analog
    of delisted_size_volatility (same permanent-delist listing)."""
    logS, g = growth(run, dt=dt, burn_in=burn_in)
    Slin = np.exp(logS)
    out = []
    for i in range(g.shape[0]):
        below = Slin[i] < floor
        ng = (int(np.argmax(below)) - 1) if below.any() else g.shape[1]
        if ng >= min_growths:
            out.append(g[i, :ng])
    return np.concatenate(out) if out else np.array([])


def persistent_mask(run, alive=1e-4):
    """Firms that never dipped below `alive` over the run — the continuously-listed survivor cloud."""
    return run.minx > alive


def extinction_fractions(run, alive=1e-4, dead=1e-6):
    return {"persist": float((run.minx > alive).mean()),
            "crashed": float((run.minx < dead).mean()),
            "alive_end": float((np.exp(run.logx[:, -1]) > alive).mean())}


def robtail(g):
    """Robust quantile tail = 99.9th-pct(|g-med|)/MAD (Gaussian≈4.9, Laplace≈10, t₃≈17)."""
    g = g[np.isfinite(g)]
    med = np.median(g); mad = np.median(np.abs(g - med)) + 1e-12
    return float(np.percentile(np.abs(g - med), 99.9) / mad)


def tail_metrics(g):
    g = g[np.isfinite(g)]
    return {"skew": float(skew(g)), "exkurt": float(kurtosis(g)), "robtail": robtail(g), "n": int(g.size)}
