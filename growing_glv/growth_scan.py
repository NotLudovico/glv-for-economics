"""Lower-mu growth-robustness scan: find the band point that grows in the MOST seeds while keeping the MSB facts.

OP-A (sigma=1.90, mu=2.35) reproduces the facts but growth is near-critical (~70% of seeds grow). Lower mu along
the band grows harder; the question is whether a point grows in (near-)all seeds while keeping beta>0 / symmetry /
fat tails / coexistence. We sweep candidate (sigma, mu) points, 30 seeds each at N=4000, and report for each:
  growing-fraction, persistent-fraction, FULL-signature fraction, and the medians.
Same measurement formulas as robustness_intermediate.py (decline-branch beta, MAD volatility) for comparability.

    uv run --directory /path/to/glv python growing_glv/growth_scan.py [--smoke]
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import kurtosis
from joblib import Parallel, delayed
from explore import build_alpha

SMOKE = "--smoke" in sys.argv

# candidate band points (sweep convention: mu>0 competitive). Lower mu -> stronger growth, less coexistence.
POINTS = [(1.75, 1.56), (1.75, 1.76), (1.90, 1.96), (1.90, 2.15), (1.90, 2.35), (2.05, 2.74)]
N        = 4000
N_SEEDS  = 30
TMAX     = 400.0
N_EVAL   = 1600
RTOL, ATOL = 1e-4, 1e-7
MAX_NFEV = 80000
LAM      = 1e-3
N_JOBS   = 3
MID_WIN  = (240.0, 310.0)
LATE_WIN = (320.0, 390.0)
DT       = 0.5
OUT      = os.path.join(os.path.dirname(__file__), "growth_scan.npz")

if SMOKE:
    POINTS = [(1.90, 2.15), (1.90, 2.35)]; N = 400; N_SEEDS = 4; TMAX = 120.0; N_EVAL = 600
    MID_WIN = (60.0, 90.0); LATE_WIN = (95.0, 118.0)
TGRID = np.linspace(0.0, TMAX, N_EVAL)


class _Capped(Exception):
    pass


def _decline_beta(Sbar, vol):
    live = (vol > 0) & np.isfinite(vol) & (Sbar > 0)
    Sbar, vol = Sbar[live], vol[live]
    if Sbar.size < 50:
        return np.nan, np.nan
    o = np.argsort(Sbar); P = np.array_split(np.arange(o.size), 20)
    bx = np.array([Sbar[o][p].mean() for p in P]); by = np.array([np.median(vol[o][p]) for p in P])
    m = (bx > 0) & (by > 0); bx, by = bx[m], by[m]
    if bx.size <= 5:
        return np.nan, np.nan
    pk = int(np.argmax(by))
    if by.size - pk < 5:
        return np.nan, np.nan
    c = np.polyfit(np.log10(bx[pk:]), np.log10(by[pk:]), 1)
    yh = np.polyval(c, np.log10(bx[pk:])); ss = np.sum((np.log10(by[pk:]) - np.log10(by[pk:]).mean()) ** 2)
    r2 = float(1 - np.sum((np.log10(by[pk:]) - yh) ** 2) / ss) if ss > 0 else np.nan
    return float(-c[0]), r2


def simulate(sigma, mu, seed):
    alpha, _ = build_alpha(seed, N, mu, sigma)
    rng = np.random.default_rng(seed + 1); w0 = rng.uniform(0.5, 1.5, N); w0 /= w0.sum()
    nev = [0]
    def rhs(t, st):
        nev[0] += 1
        if nev[0] > MAX_NFEV:
            raise _Capped()
        w = np.clip(st[:N], 0.0, None); s = w.sum()
        if s > 0:
            w = w / s
        f = 1.0 - N * w - N * (alpha @ w); fb = float(w @ f)
        return np.concatenate([w * (f - fb) + LAM * (1.0 / N - w), [fb]])
    rec = dict(sigma=sigma, mu=mu, seed=int(seed), capped=False,
               g_eff=np.nan, surv=np.nan, persistent=False, beta=np.nan, r2=np.nan, bowley=np.nan, exk=np.nan)
    try:
        r = solve_ivp(rhs, (0, TMAX), np.concatenate([w0, [0.0]]), method="RK45",
                      t_eval=TGRID, rtol=RTOL, atol=ATOL)
    except _Capped:
        rec["capped"] = True; return rec
    t = r.t; W = np.clip(r.y[:N], 0, None); W = W / W.sum(0, keepdims=True); lnM = r.y[N]
    lh = t > 0.5 * t[-1]; rec["g_eff"] = float(np.polyfit(t[lh], lnM[lh], 1)[0])
    surv = np.where(W[:, (t >= LATE_WIN[0]) & (t <= LATE_WIN[1])].min(1) > 1e-6)[0]
    rec["surv"] = float(surv.size / N)
    if surv.size >= 50:
        tg = np.arange(LATE_WIN[0], LATE_WIN[1] + 1e-9, DT)
        lnS = np.array([np.interp(tg, t, np.log(np.maximum(N * W[i], 1e-12))) for i in surv])
        g = np.diff(lnS, axis=1)
        rec["persistent"] = bool(g.std() > 0.05)
        Sbar = np.exp(lnS).mean(1); vol = np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)
        rec["beta"], rec["r2"] = _decline_beta(Sbar, vol)
        flat = g.ravel(); flat = flat[np.isfinite(flat)]
        q1, q2, q3 = np.percentile(flat, [25, 50, 75])
        rec["bowley"] = float((q3 + q1 - 2 * q2) / (q3 - q1)) if q3 > q1 else np.nan
        rec["exk"] = float(kurtosis(flat))
    return rec


if __name__ == "__main__":
    import time
    print(f"[growth_scan] smoke={SMOKE} | {len(POINTS)} points x {N_SEEDS} seeds, N={N}, jobs={N_JOBS}", flush=True)
    t0 = time.time()
    jobs = [(s, m, sd) for (s, m) in POINTS for sd in range(N_SEEDS)]
    recs = Parallel(n_jobs=N_JOBS, backend="loky", verbose=6)(delayed(simulate)(s, m, sd) for (s, m, sd) in jobs)
    print(f"[growth_scan] integrated in {(time.time()-t0)/60:.1f} min", flush=True)

    print("\n================ growth-robustness across band points (N=%d, %d seeds) ================" % (N, N_SEEDS))
    print("point sorted by GROWING fraction; FULL = grow & persist & beta>0 & r2>.8 & |bow|<.1 & exk>3\n")
    rows, out = [], {}
    for (s, m) in POINTS:
        g = [r for r in recs if r["sigma"] == s and r["mu"] == m and not r["capped"]]
        if not g:
            continue
        gef = np.array([r["g_eff"] for r in g]); per = np.array([r["persistent"] for r in g])
        be = np.array([r["beta"] for r in g]); r2 = np.array([r["r2"] for r in g])
        bw = np.array([r["bowley"] for r in g]); ek = np.array([r["exk"] for r in g]); sv = np.array([r["surv"] for r in g])
        full = (gef > 0) & per & (be > 0) & (r2 > 0.8) & (np.abs(bw) < 0.10) & (ek > 3)
        def med(a): a = a[np.isfinite(a)]; return np.median(a) if a.size else np.nan
        cleanb = be[(r2 > 0.8) & np.isfinite(be)]
        rows.append(dict(sigma=s, mu=m, grow=float(np.mean(gef > 0)), persist=float(np.mean(per)),
                         full=float(np.mean(full)), g_eff=med(gef), surv=med(sv),
                         beta_clean=(np.median(cleanb) if cleanb.size else np.nan),
                         bowley=med(bw), exk=med(ek), ncap=sum(r["capped"] for r in recs if r["sigma"] == s and r["mu"] == m)))
        out[f"s{s}_m{m}_geff"] = gef; out[f"s{s}_m{m}_full"] = full.astype(float)
    rows.sort(key=lambda d: -d["grow"])
    print(f"{'sigma':>5} {'mu':>5} | {'grow':>5} {'persist':>7} {'FULL':>5} | {'g_eff':>6} {'surv':>5} "
          f"{'beta*':>6} {'bow':>6} {'exk':>5} {'cap':>4}")
    for d in rows:
        print(f"{d['sigma']:>5.2f} {d['mu']:>5.2f} | {d['grow']:>5.2f} {d['persist']:>7.2f} {d['full']:>5.2f} | "
              f"{d['g_eff']:>+6.2f} {d['surv']:>5.2f} {d['beta_clean']:>+6.2f} {d['bowley']:>+6.2f} {d['exk']:>5.0f} {d['ncap']:>4d}")
    # recommendation: maximize growing fraction subject to a usable full-signature fraction
    usable = [d for d in rows if d["full"] >= 0.30]
    best = max(usable, key=lambda d: d["grow"]) if usable else max(rows, key=lambda d: d["full"])
    print(f"\n[recommend] best growing point with FULL>=0.30: sigma={best['sigma']}, mu={best['mu']} "
          f"(grow={best['grow']:.2f}, full={best['full']:.2f}, beta*={best['beta_clean']:+.2f})")
    np.savez(OUT, points=np.array(POINTS), **out)
    print(f"[growth_scan] saved -> {OUT}  ({(time.time()-t0)/60:.1f} min)", flush=True)
