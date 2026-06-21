"""Does the size-volatility exponent beta extrapolate to the empirical ~0.2 in the mean-field (N->inf)
limit? Measure beta(N) carefully at large N with RK45: decline-branch power-law fit (excluding the
small-firm floor plateau) + R^2, in the stationary window, on persistent seeds. Save raw per-seed betas
so the N->inf extrapolation can be fit afterward.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import kurtosis
from explore import build_alpha

MU, SIGMA, LAM = 3.0, 2.0, 1e-3
TMAX, T0, T1 = 250.0, 170.0, 240.0
NS = [800, 1600, 3200, 6400, 12800, 25600]
N_SEEDS = 6
PERSIST = 0.05


def integ(seed, S):
    alpha, _ = build_alpha(seed, S, MU, SIGMA)
    rng = np.random.default_rng(seed + 1); w0 = rng.uniform(0.5, 1.5, S); w0 /= w0.sum()
    def rhs(t, st):
        w = np.clip(st[:S], 0.0, None); s = w.sum()
        if s > 0: w = w / s
        f = 1.0 - S * w - S * (alpha @ w); fb = float(w @ f)
        return np.concatenate([w * (f - fb) + LAM * (1.0 / S - w), [fb]])
    n_eval = 1500
    r = solve_ivp(rhs, (0, TMAX), np.concatenate([w0, [0.0]]), method="RK45",
                  t_eval=np.linspace(0, TMAX, n_eval), rtol=1e-6, atol=1e-8)
    W = np.clip(r.y[:S], 0.0, None); W = W / W.sum(0, keepdims=True)
    return r.t, W


def decline_beta(W, t, S, t0=T0, t1=T1):
    surv = np.where(W[:, (t >= t0) & (t <= t1)].min(1) > 1e-6)[0]
    if surv.size < 50:
        return None
    tg = np.arange(t0, t1 + 1e-9, 0.5)
    lnS = np.array([np.interp(tg, t, np.log(np.maximum(S * W[i], 1e-12))) for i in surv])
    g = np.diff(lnS, axis=1); Sbar = np.exp(lnS).mean(1)
    vol = np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)
    live = (vol > 0) & np.isfinite(vol); Sbar, vol = Sbar[live], vol[live]; gstd = g[live].ravel().std()
    o = np.argsort(Sbar); P = np.array_split(np.arange(o.size), 20)
    bx = np.array([Sbar[o][p].mean() for p in P]); by = np.array([np.median(vol[o][p]) for p in P])
    m = (bx > 0) & (by > 0); bx, by = bx[m], by[m]; pk = int(np.argmax(by))
    if by.size - pk < 5:
        return dict(beta=np.nan, r2=np.nan, gstd=gstd, n=surv.size)
    x, y = np.log10(bx[pk:]), np.log10(by[pk:]); c = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - np.polyval(c, x)) ** 2) / np.sum((y - y.mean()) ** 2)
    return dict(beta=float(-c[0]), r2=float(r2), gstd=gstd, n=surv.size)


if __name__ == "__main__":
    seeds = np.random.default_rng(123).integers(0, 2**31 - 1, size=N_SEEDS)
    print(f"RK45 beta(N) scaling, mu={MU} sigma={SIGMA} lam={LAM:g}, window [{T0:.0f},{T1:.0f}], {N_SEEDS} seeds\n")
    print(f"{'N':>6} {'persist':>8} {'beta(decline)':>16} {'R2':>6}")
    out = {}
    for S in NS:
        betas, r2s = [], []
        for s in seeds:
            t, W = integ(int(s), S)
            r = decline_beta(W, t, S)
            if r and np.isfinite(r["beta"]) and r["gstd"] > PERSIST:
                betas.append(r["beta"]); r2s.append(r["r2"])
        out[S] = betas
        if betas:
            print(f"{S:>6} {len(betas)}/{N_SEEDS}   {np.mean(betas):>6.3f} +/- {np.std(betas):<5.3f}   {np.mean(r2s):>5.2f}", flush=True)
        else:
            print(f"{S:>6} 0/{N_SEEDS}   --", flush=True)
    np.savez(os.path.join(os.path.dirname(__file__), "beta_scaling.npz"),
             **{f"N{S}": np.array(out[S]) for S in NS})
    print("\nsaved beta_scaling.npz (per-seed betas) for N->inf extrapolation")
