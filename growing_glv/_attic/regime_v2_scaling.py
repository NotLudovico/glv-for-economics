"""Corrected regime mu=2.5, sigma=2, lam=1e-3 (GROWS: g_eff>0, AND chaotic). Confirm g_eff stays
positive and persistence robustifies as N grows, and measure beta(N) -> beta_infinity at this regime
(the mu=3,sigma=2 beta scaling was at a SHRINKING regime). Save per-seed betas for extrapolation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from scipy.integrate import solve_ivp
from explore import build_alpha

MU, SIGMA, LAM = 2.5, 2.0, 1e-3
TMAX = 250.0
NS = [1600, 3200, 6400, 12800, 25600]
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
    r = solve_ivp(rhs, (0, TMAX), np.concatenate([w0, [0.0]]), method="RK45",
                  t_eval=np.linspace(0, TMAX, 1500), rtol=1e-6, atol=1e-8)
    W = np.clip(r.y[:S], 0.0, None); W = W / W.sum(0, keepdims=True)
    return r.t, W, r.y[S]


def measure(t, W, lnM, S):
    g_eff = float(np.polyfit(t[t > 120], lnM[t > 120], 1)[0])
    surv = np.where(W[:, (t >= 170) & (t <= 240)].min(1) > 1e-6)[0]
    if surv.size < 50:
        return g_eff, 0.0, np.nan, np.nan
    tg = np.arange(170, 240 + 1e-9, 0.5)
    lnS = np.array([np.interp(tg, t, np.log(np.maximum(S * W[i], 1e-12))) for i in surv])
    g = np.diff(lnS, axis=1); Sbar = np.exp(lnS).mean(1)
    vol = np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)
    live = (vol > 0) & np.isfinite(vol); Sbar, vol = Sbar[live], vol[live]; gstd = g[live].ravel().std()
    o = np.argsort(Sbar); P = np.array_split(np.arange(o.size), 20)
    bx = np.array([Sbar[o][p].mean() for p in P]); by = np.array([np.median(vol[o][p]) for p in P])
    m = (bx > 0) & (by > 0); bx, by = bx[m], by[m]; pk = int(np.argmax(by))
    beta = np.nan
    if by.size - pk >= 5:
        beta = float(-np.polyfit(np.log10(bx[pk:]), np.log10(by[pk:]), 1)[0])
    return g_eff, gstd, beta, surv.size


if __name__ == "__main__":
    seeds = np.random.default_rng(123).integers(0, 2**31 - 1, size=N_SEEDS)
    print(f"CORRECTED regime mu={MU} sigma={SIGMA} lam={LAM:g}, {N_SEEDS} seeds\n")
    print(f"{'N':>6} {'g_eff(>0?)':>14} {'persist':>9} {'beta':>14}")
    out = {}
    for S in NS:
        gs, betas = [], []
        npers = 0
        for s in seeds:
            g_eff, gstd, beta, n = measure(*integ(int(s), S), S)
            gs.append(g_eff)
            if gstd > PERSIST and np.isfinite(beta):
                betas.append(beta); npers += 1
        out[S] = betas
        bstr = f"{np.mean(betas):>6.3f}+/-{np.std(betas):<5.3f}" if betas else f"{'--':>14}"
        print(f"{S:>6} {np.mean(gs):>7.2f}+/-{np.std(gs):<5.2f} {npers}/{N_SEEDS}   {bstr}", flush=True)
    np.savez(os.path.join(os.path.dirname(__file__), "regime_v2_scaling.npz"),
             **{f"N{S}": np.array(out[S]) for S in NS})
    print("\nsaved regime_v2_scaling.npz")
