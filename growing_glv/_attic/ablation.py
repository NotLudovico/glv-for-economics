"""ABLATION: do the firm-growth stylized facts REQUIRE the disordered interactions?
Hold mu, lambda, graph, N fixed; dial the DISORDER sigma from 0 -> 2. There is NO injected noise in this
model, so any persistent fluctuation / tent / size-volatility scaling can only come from the disordered
interactions. If they vanish at sigma=0 and only switch on above sigma_c~sqrt(2), the stylized facts are
INTERACTION-GENERATED (the thesis claim) -- not noise-driven (cf. the Langevin-GLV variant where beta
survived W=0) and not Bouchaud-Mezard-in-disguise.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import kurtosis
from explore import build_alpha

MU, LAM, S = 3.0, 1e-3, 3200
TMAX, T0, T1 = 250.0, 170.0, 240.0
SIGMAS = [0.0, 0.5, 1.0, 1.41, 1.7, 2.0]
N_SEEDS = 4


def integ(seed, sigma):
    alpha, _ = build_alpha(seed, S, MU, sigma)        # sigma=0 -> pure mean interaction mu/C, NO disorder
    rng = np.random.default_rng(seed + 1); w0 = rng.uniform(0.5, 1.5, S); w0 /= w0.sum()
    def rhs(t, st):
        w = np.clip(st[:S], 0.0, None); s = w.sum()
        if s > 0: w = w / s
        f = 1.0 - S * w - S * (alpha @ w); fb = float(w @ f)
        return np.concatenate([w * (f - fb) + LAM * (1.0 / S - w), [fb]])
    r = solve_ivp(rhs, (0, TMAX), np.concatenate([w0, [0.0]]), method="RK45",
                  t_eval=np.linspace(0, TMAX, 1500), rtol=1e-6, atol=1e-8)
    W = np.clip(r.y[:S], 0.0, None); W = W / W.sum(0, keepdims=True)
    return r.t, W


def measure(W, t):
    surv = np.where(W[:, (t >= T0) & (t <= T1)].min(1) > 1e-6)[0]
    if surv.size < 50:
        return dict(gstd=0.0, beta=np.nan, r2=np.nan, bowley=np.nan, exk=np.nan, n=surv.size)
    tg = np.arange(T0, T1 + 1e-9, 0.5)
    lnS = np.array([np.interp(tg, t, np.log(np.maximum(S * W[i], 1e-12))) for i in surv])
    g = np.diff(lnS, axis=1); Sbar = np.exp(lnS).mean(1)
    vol = np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)
    live = (vol > 0) & np.isfinite(vol); Sbar, vol = Sbar[live], vol[live]
    gl = g[live].ravel(); gstd = gl.std()
    o = np.argsort(Sbar); P = np.array_split(np.arange(o.size), 20)
    bx = np.array([Sbar[o][p].mean() for p in P]); by = np.array([np.median(vol[o][p]) for p in P])
    m = (bx > 0) & (by > 0); bx, by = bx[m], by[m]; pk = int(np.argmax(by))
    beta = r2 = np.nan
    if by.size - pk >= 5:
        x, y = np.log10(bx[pk:]), np.log10(by[pk:]); c = np.polyfit(x, y, 1)
        r2 = 1 - np.sum((y - np.polyval(c, x)) ** 2) / np.sum((y - y.mean()) ** 2); beta = -c[0]
    q1, q2, q3 = np.percentile(gl, [25, 50, 75])
    return dict(gstd=gstd, beta=float(beta), r2=float(r2),
                bowley=float((q3 + q1 - 2 * q2) / (q3 - q1)), exk=float(kurtosis(gl)), n=surv.size)


if __name__ == "__main__":
    seeds = np.random.default_rng(123).integers(0, 2**31 - 1, size=N_SEEDS)
    print(f"ABLATION: dial disorder sigma at mu={MU}, lam={LAM:g}, N={S}, {N_SEEDS} seeds. "
          f"(sigma=0 -> no disorder)\n")
    print(f"{'sigma':>6} {'fluct(gstd)':>12} {'beta':>14} {'R2':>6} {'Bowley':>13} {'exkurt':>8} {'tent?':>6}")
    for sig in SIGMAS:
        res = []
        for s in seeds:
            tt, WW = integ(int(s), sig)
            res.append(measure(WW, tt))
        gm = np.mean([r["gstd"] for r in res])
        bs = [r["beta"] for r in res if np.isfinite(r["beta"])]
        ws = [r["bowley"] for r in res if np.isfinite(r["bowley"])]
        es = [r["exk"] for r in res if np.isfinite(r["exk"])]
        rs = [r["r2"] for r in res if np.isfinite(r["r2"])]
        bstr = f"{np.mean(bs):>6.2f}+/-{np.std(bs):<4.2f}" if bs else f"{'--':>12}"
        wstr = f"{np.mean(ws):>+6.2f}+/-{np.std(ws):<4.2f}" if ws else f"{'--':>12}"
        estr = f"{np.mean(es):>8.1f}" if es else f"{'--':>8}"
        rstr = f"{np.mean(rs):>5.2f}" if rs else f"{'--':>5}"
        tent = "yes" if (gm > 0.02 and es and np.mean(es) > 2) else "no"
        print(f"{sig:>6.2f} {gm:>12.4f} {bstr} {rstr} {wstr} {estr} {tent:>6}", flush=True)
    print("\nReading: sigma=0 frozen (gstd~0, no beta/tent) + facts switch on above sigma_c~1.41 => "
          "stylized facts are INTERACTION-generated, not noise-driven.")
