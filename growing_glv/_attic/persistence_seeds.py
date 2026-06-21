"""Are the persistent-chaos regimes robust across realizations (not a fragile seed-dependent sliver),
and do the MSB facts hold in the STATIONARY late window? For candidate regimes, run several seeds to
t=300 and measure in [200,280]: late growth std (persistence), decline-branch beta + R^2, Bowley (symmetry),
exkurt (fat tails), survivors.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from scipy.stats import kurtosis
from explore import integrate

S, TMAX, T0, T1 = 500, 300.0, 200.0, 280.0
REGIMES = [(3.0, 2.5, 1e-3), (3.0, 2.0, 1e-3), (2.0, 2.5, 0.0), (2.0, 2.0, 0.0)]
N_SEEDS = 4


def bowley(g):
    Q1, Q2, Q3 = np.percentile(g, [25, 50, 75])
    return (Q3 + Q1 - 2 * Q2) / (Q3 - Q1) if Q3 > Q1 else np.nan


def late_msb(W, t):
    N = W.shape[0]
    surv = np.where(W[:, (t >= T0) & (t <= T1)].min(1) > 1e-6)[0]
    if surv.size < 50:
        return None
    tg = np.arange(T0, T1 + 1e-9, 0.5)
    lnS = np.array([np.interp(tg, t, np.log(np.maximum(N * W[i], 1e-12))) for i in surv])
    g = np.diff(lnS, axis=1)
    Sbar = np.exp(lnS).mean(1)
    vol = np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)
    live = (vol > 0) & np.isfinite(vol)
    Sbar, vol = Sbar[live], vol[live]
    gl = g[live].ravel(); gstd = gl.std()
    o = np.argsort(Sbar); P = np.array_split(np.arange(o.size), 18)
    bx = np.array([Sbar[o][p].mean() for p in P]); by = np.array([np.median(vol[o][p]) for p in P])
    m = (bx > 0) & (by > 0); bx, by = bx[m], by[m]; pk = int(np.argmax(by))
    beta = r2 = np.nan
    if by.size - pk >= 4:
        x, y = np.log10(bx[pk:]), np.log10(by[pk:]); c = np.polyfit(x, y, 1)
        r2 = 1 - np.sum((y - np.polyval(c, x)) ** 2) / np.sum((y - y.mean()) ** 2); beta = -c[0]
    return dict(gstd=gstd, beta=float(beta), r2=float(r2), bowley=float(bowley(gl)),
                exk=float(kurtosis(gl)), surv=surv.size)


if __name__ == "__main__":
    seeds = np.random.default_rng(123).integers(0, 2**31 - 1, size=N_SEEDS)
    print(f"S={S}, stationary window [{T0:.0f},{T1:.0f}], {N_SEEDS} seeds. Persistence + MSB in the STATIONARY state.\n")
    for mu, sig, lam in REGIMES:
        rs = []
        for s in seeds:
            t, W, lnM, _, _ = integrate(int(s), S, mu, sig, lam=lam, tmax=TMAX, n_eval=6000)
            r = late_msb(W, t)
            if r:
                rs.append(r)
        print(f"mu={mu} sigma={sig} lam={lam:g}  ({len(rs)}/{N_SEEDS} seeds with >=50 late survivors):")
        if rs:
            def ms(k): v = np.array([r[k] for r in rs if np.isfinite(r[k])]); return (np.mean(v), np.std(v)) if v.size else (np.nan, np.nan)
            for k, lab in [("gstd", "late growth std (persistence)"), ("beta", "decline beta"),
                           ("r2", "R^2"), ("bowley", "Bowley (symmetry)"), ("exk", "exkurt (fat)"),
                           ("surv", "survivors")]:
                m_, s_ = ms(k); print(f"    {lab:>28}: {m_:.3f} +/- {s_:.3f}")
        print()
