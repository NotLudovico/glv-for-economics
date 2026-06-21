"""Definitive large-N scan with RK45 (explicit -> fast at 10k; LSODA blew up on a dense Jacobian).
Does persistence robustify and beta shallow toward empirical as N -> 10000?
Per N (6 seeds): persistent fraction + persistent-state decline-beta/R^2, Bowley, exkurt.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import kurtosis
from explore import build_alpha

MU, SIGMA, LAM = 3.0, 2.0, 1e-3
TMAX, T0, T1 = 250.0, 170.0, 240.0
NS = [1600, 3200, 6400, 10000]
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
    n_eval = 2000
    r = solve_ivp(rhs, (0, TMAX), np.concatenate([w0, [0.0]]), method="RK45",
                  t_eval=np.linspace(0, TMAX, n_eval), rtol=1e-6, atol=1e-8)
    W = np.clip(r.y[:S], 0.0, None); W = W / W.sum(0, keepdims=True)
    return r.t, W


def late(W, t, S):
    surv = np.where(W[:, (t >= T0) & (t <= T1)].min(1) > 1e-6)[0]
    if surv.size < 40:
        return None
    tg = np.arange(T0, T1 + 1e-9, 0.5)
    lnS = np.array([np.interp(tg, t, np.log(np.maximum(S * W[i], 1e-12))) for i in surv])
    g = np.diff(lnS, axis=1); Sbar = np.exp(lnS).mean(1)
    vol = np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)
    live = (vol > 0) & np.isfinite(vol); Sbar, vol = Sbar[live], vol[live]
    gl = g[live].ravel(); gstd = gl.std()
    o = np.argsort(Sbar); P = np.array_split(np.arange(o.size), 18)
    bx = np.array([Sbar[o][p].mean() for p in P]); by = np.array([np.median(vol[o][p]) for p in P])
    m = (bx > 0) & (by > 0); bx, by = bx[m], by[m]; pk = int(np.argmax(by))
    beta = r2 = np.nan
    if by.size - pk >= 4:
        x, y = np.log10(bx[pk:]), np.log10(by[pk:]); c = np.polyfit(x, y, 1)
        r2 = 1 - np.sum((y - np.polyval(c, x)) ** 2) / np.sum((y - y.mean()) ** 2); beta = -c[0]
    q1, q2, q3 = np.percentile(gl, [25, 50, 75])
    return dict(gstd=gstd, beta=float(beta), r2=float(r2),
                bowley=float((q3 + q1 - 2 * q2) / (q3 - q1)), exk=float(kurtosis(gl)))


if __name__ == "__main__":
    print(f"RK45, mu={MU} sigma={SIGMA} lam={LAM:g}, window [{T0:.0f},{T1:.0f}], {N_SEEDS} seeds/N\n")
    print(f"{'N':>6} {'persist_frac':>13} {'gstd':>13} {'beta':>13} {'R2':>5} {'Bowley':>13} {'exkurt':>8}")
    seeds = np.random.default_rng(123).integers(0, 2**31 - 1, size=N_SEEDS)
    for S in NS:
        res = []
        for s in seeds:
            tt, WW = integ(int(s), S)
            res.append(late(WW, tt, S))
        res = [r for r in res if r]
        pers = [r for r in res if r["gstd"] > PERSIST]
        frac = len(pers) / max(1, len(res))
        def ms(k, rs): v = np.array([r[k] for r in rs if np.isfinite(r[k])]); return (np.mean(v), np.std(v)) if v.size else (np.nan, np.nan)
        if pers:
            gm, gs = ms("gstd", pers); bm, bs = ms("beta", pers); rm, _ = ms("r2", pers)
            wm, ws = ms("bowley", pers); em, _ = ms("exk", pers)
            print(f"{S:>6} {frac:>6.2f} ({len(pers)}/{len(res)}) {gm:>7.3f}+/-{gs:<4.3f} {bm:>6.2f}+/-{bs:<4.2f} "
                  f"{rm:>5.2f} {wm:>+7.2f}+/-{ws:<4.2f} {em:>8.1f}", flush=True)
        else:
            print(f"{S:>6} {frac:>6.2f} ({len(pers)}/{len(res)})  none persistent", flush=True)
