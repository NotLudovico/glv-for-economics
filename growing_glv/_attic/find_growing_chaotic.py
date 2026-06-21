"""Find a regime that BOTH grows (g_eff>0) AND stays chaotic (persistent fluctuations) at large N.
The chapter regime mu=3,sigma=2 has persistent chaos but g_eff<0 (M shrinks). Scan (mu,sigma,lam),
measuring the aggregate growth rate g_eff over a long post-transient window, the persistence (late
growth std), the size-volatility beta, and coexistence -- and flag cells with g_eff>0 AND persistent.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from scipy.integrate import solve_ivp
from explore import build_alpha

S, TMAX = 3200, 250.0
MUS = [1.5, 2.0, 2.5, 3.0]
SIGS = [2.0, 2.5, 3.0]
LAMS = [0.0, 1e-3]
N_SEEDS = 3


def integ(seed, mu, sigma, lam):
    alpha, _ = build_alpha(seed, S, mu, sigma)
    rng = np.random.default_rng(seed + 1); w0 = rng.uniform(0.5, 1.5, S); w0 /= w0.sum()
    def rhs(t, st):
        w = np.clip(st[:S], 0.0, None); s = w.sum()
        if s > 0: w = w / s
        f = 1.0 - S * w - S * (alpha @ w); fb = float(w @ f)
        return np.concatenate([w * (f - fb) + lam * (1.0 / S - w), [fb]])
    r = solve_ivp(rhs, (0, TMAX), np.concatenate([w0, [0.0]]), method="RK45",
                  t_eval=np.linspace(0, TMAX, 1500), rtol=1e-6, atol=1e-8)
    W = np.clip(r.y[:S], 0.0, None); W = W / W.sum(0, keepdims=True)
    return r.t, W, r.y[S]


def measure(t, W, lnM):
    g_eff = float(np.polyfit(t[t > 120], lnM[t > 120], 1)[0])          # asymptotic growth rate of M
    surv = np.where(W[:, (t >= 170) & (t <= 240)].min(1) > 1e-6)[0]
    if surv.size < 50:
        return g_eff, 0.0, np.nan, surv.size
    tg = np.arange(170, 240 + 1e-9, 0.5)
    lnS = np.array([np.interp(tg, t, np.log(np.maximum(S * W[i], 1e-12))) for i in surv])
    g = np.diff(lnS, axis=1); Sbar = np.exp(lnS).mean(1)
    vol = np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)
    live = (vol > 0) & np.isfinite(vol); Sbar, vol = Sbar[live], vol[live]; gstd = g[live].ravel().std()
    o = np.argsort(Sbar); P = np.array_split(np.arange(o.size), 18)
    bx = np.array([Sbar[o][p].mean() for p in P]); by = np.array([np.median(vol[o][p]) for p in P])
    m = (bx > 0) & (by > 0); bx, by = bx[m], by[m]; pk = int(np.argmax(by))
    beta = np.nan
    if by.size - pk >= 4:
        beta = float(-np.polyfit(np.log10(bx[pk:]), np.log10(by[pk:]), 1)[0])
    return g_eff, gstd, beta, surv.size


if __name__ == "__main__":
    seeds = np.random.default_rng(123).integers(0, 2**31 - 1, size=N_SEEDS)
    print(f"N={S}, {N_SEEDS} seeds. Want g_eff>0 (GROWS) AND gstd>0.05 (CHAOTIC).\n")
    print(f"{'mu':>4} {'sig':>4} {'lam':>6} {'g_eff':>14} {'gstd':>13} {'beta':>13} {'surv':>7}  flag")
    for mu in MUS:
        for sig in SIGS:
            for lam in LAMS:
                gs, fs, bs, ns = [], [], [], []
                for s in seeds:
                    g_eff, gstd, beta, n = measure(*integ(int(s), mu, sig, lam))
                    gs.append(g_eff); fs.append(gstd); bs.append(beta); ns.append(n)
                gm, fm = np.mean(gs), np.mean(fs)
                bb = [b for b in bs if np.isfinite(b)]
                flag = "  <== GROWS+CHAOTIC" if (gm > 0.03 and fm > 0.05) else ""
                bstr = f"{np.mean(bb):>6.2f}+/-{np.std(bb):<4.2f}" if bb else f"{'--':>13}"
                print(f"{mu:>4.1f} {sig:>4.1f} {lam:>6.0e} {gm:>7.2f}+/-{np.std(gs):<5.2f} "
                      f"{fm:>7.3f}+/-{np.std(fs):<4.3f} {bstr} {int(np.mean(ns)):>6d}{flag}", flush=True)
