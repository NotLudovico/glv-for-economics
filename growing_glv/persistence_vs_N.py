"""Is persistent chaos a real phase (robustifies with N, mean-field limit) or a finite-size sliver?
For a candidate regime, scan N and report the FRACTION of seeds that stay persistent at long time, plus
the persistent-state MSB observables (decline beta + R^2, Bowley, exkurt). If the persistent fraction -> 1
as N grows, the chaotic phase is robust and the MSB-like tent is a genuine property.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import kurtosis
from explore import integrate

MU, SIGMA, LAM = 3.0, 2.0, 1e-3
TMAX, N_EVAL, T0, T1 = 250.0, 4000, 170.0, 240.0
NS = [200, 400, 800, 1600]
N_SEEDS = 5
PERSIST_THRESH = 0.05            # late growth std above this = persistent (frozen ~ 0.001)


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
                bowley=float((q3 + q1 - 2 * q2) / (q3 - q1)), exk=float(kurtosis(gl)), surv=surv.size)


if __name__ == "__main__":
    print(f"regime mu={MU} sigma={SIGMA} lam={LAM:g}, window [{T0:.0f},{T1:.0f}], {N_SEEDS} seeds/N, "
          f"persist if late growth std > {PERSIST_THRESH}\n")
    print(f"{'N':>6} {'persist_frac':>13} {'late_gstd(persist)':>19} {'beta':>13} {'R2':>6} {'Bowley':>13} {'exkurt':>11}")
    seeds = np.random.default_rng(123).integers(0, 2**31 - 1, size=N_SEEDS)
    for S in NS:
        res = []
        for s in seeds:
            t, W, lnM, _, _ = integrate(int(s), S, MU, SIGMA, lam=LAM, tmax=TMAX, n_eval=N_EVAL)
            r = late(W, t, S)
            if r:
                res.append(r)
        pers = [r for r in res if r["gstd"] > PERSIST_THRESH]
        frac = len(pers) / max(1, len(res))
        def ms(k, rs): v = np.array([r[k] for r in rs if np.isfinite(r[k])]); return (np.mean(v), np.std(v)) if v.size else (np.nan, np.nan)
        if pers:
            gm, gs = ms("gstd", pers); bm, bs = ms("beta", pers); rm, _ = ms("r2", pers)
            wm, ws = ms("bowley", pers); em, es = ms("exk", pers)
            print(f"{S:>6} {frac:>7.2f} ({len(pers)}/{len(res)}) {gm:>8.3f}+/-{gs:<4.3f}  {bm:>6.2f}+/-{bs:<4.2f} "
                  f"{rm:>6.2f} {wm:>+7.2f}+/-{ws:<4.2f} {em:>7.1f}")
        else:
            print(f"{S:>6} {frac:>7.2f} ({len(pers)}/{len(res)})  -- none persistent --")
