"""Generate ALL Chapter-4 figure data at the LOCKED operating point of the relative GLV.

Regime: sigma=1.75, mu=1.76 (sweep convention) = chapter mu=-1.76. One dataset-oriented run; ch4_plots.py turns
ch4_data.npz into the figures (single-run churn, MSB size-volatility + tent, mean-field persistence+beta-vs-N,
beta extrapolation, freeze-vs-persist trajectories). Sweep + growth-scan are stopped, so full cores are free.

    uv run --directory /path/to/glv python growing_glv/ch4_data.py [--smoke]
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import kurtosis
from joblib import Parallel, delayed
from explore import build_alpha

SMOKE = "--smoke" in sys.argv
SIGMA, MU, LAM = 1.75, 1.76, 1e-3
TMAX, N_EVAL = 400.0, 1600
RTOL, ATOL, MAX_NFEV = 1e-4, 1e-7, 80000
MID_WIN, LATE_WIN, DT = (240.0, 310.0), (320.0, 390.0), 0.5
N_JOBS = 6
SHARE_SAMPLE = 30

N_SHOWCASE = 6400                 # single-run churn figure
MSB_N, MSB_SEEDS = 6400, 8        # size-volatility + tent, pooled
NSCAN, NSCAN_SEEDS = [800, 1600, 3200, 6400, 12800], 8   # mean-field persistence + beta(N)
TRAJ_N, TRAJ_SEEDS = 600, 12      # freeze vs persist (finite-size)
OUT = os.path.join(os.path.dirname(__file__), "ch4_data.npz")

if SMOKE:
    N_SHOWCASE = 400; MSB_N, MSB_SEEDS = 400, 3; NSCAN, NSCAN_SEEDS = [400, 800], 2
    TRAJ_N, TRAJ_SEEDS = 300, 3; TMAX, N_EVAL = 120.0, 600
    MID_WIN, LATE_WIN = (60.0, 90.0), (95.0, 118.0)
TGRID = np.linspace(0.0, TMAX, N_EVAL)


class _Capped(Exception):
    pass


def _decline_beta(Sbar, vol):
    """sigma(S) ~ S^-beta on the large-S decline branch; returns beta, r2, binned (bx, by), peak index."""
    live = (vol > 0) & np.isfinite(vol) & (Sbar > 0)
    Sbar, vol = Sbar[live], vol[live]
    if Sbar.size < 50:
        return np.nan, np.nan, np.array([]), np.array([]), 0
    o = np.argsort(Sbar); P = np.array_split(np.arange(o.size), 20)
    bx = np.array([Sbar[o][p].mean() for p in P]); by = np.array([np.median(vol[o][p]) for p in P])
    m = (bx > 0) & (by > 0); bx, by = bx[m], by[m]
    if bx.size <= 5:
        return np.nan, np.nan, bx, by, 0
    pk = int(np.argmax(by))
    if by.size - pk < 5:
        return np.nan, np.nan, bx, by, pk
    c = np.polyfit(np.log10(bx[pk:]), np.log10(by[pk:]), 1)
    yh = np.polyval(c, np.log10(bx[pk:])); ss = np.sum((np.log10(by[pk:]) - np.log10(by[pk:]).mean()) ** 2)
    r2 = float(1 - np.sum((np.log10(by[pk:]) - yh) ** 2) / ss) if ss > 0 else np.nan
    return float(-c[0]), r2, bx, by, pk


def simulate(N, seed, store_shares=False):
    alpha, _ = build_alpha(seed, N, MU, SIGMA)
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
    rec = dict(N=N, seed=int(seed), capped=False, g_eff=np.nan, surv=np.nan, gstd_late=0.0,
               persistent=False, beta=np.nan, r2=np.nan)
    try:
        r = solve_ivp(rhs, (0, TMAX), np.concatenate([w0, [0.0]]), method="RK45",
                      t_eval=TGRID, rtol=RTOL, atol=ATOL)
    except _Capped:
        rec["capped"] = True; return rec
    t = r.t; W = np.clip(r.y[:N], 0, None); W = W / W.sum(0, keepdims=True); lnM = r.y[N]
    lh = t > 0.5 * t[-1]; rec["g_eff"] = float(np.polyfit(t[lh], lnM[lh], 1)[0])
    sm = np.where(W[:, (t >= MID_WIN[0]) & (t <= MID_WIN[1])].min(1) > 1e-6)[0]
    if sm.size:
        tgm = np.arange(MID_WIN[0], MID_WIN[1] + 1e-9, DT)
        gm = np.diff(np.array([np.interp(tgm, t, np.log(np.maximum(N * W[i], 1e-12))) for i in sm]), axis=1)
        rec["gstd_mid"] = float(gm.std())
    surv = np.where(W[:, (t >= LATE_WIN[0]) & (t <= LATE_WIN[1])].min(1) > 1e-6)[0]
    rec["surv"] = float(surv.size / N)
    if surv.size >= 50:
        tg = np.arange(LATE_WIN[0], LATE_WIN[1] + 1e-9, DT)
        lnS = np.array([np.interp(tg, t, np.log(np.maximum(N * W[i], 1e-12))) for i in surv])
        g = np.diff(lnS, axis=1)
        rec["gstd_late"] = float(g.std()); rec["persistent"] = bool(g.std() > 0.05)
        Sbar = np.exp(lnS).mean(1); vol = np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)
        rec["beta"], rec["r2"], rec["bin_S"], rec["bin_vol"], rec["pk"] = _decline_beta(Sbar, vol)
        live = (vol > 0) & np.isfinite(vol) & (Sbar > 0)
        rec["sv_S"] = Sbar[live].astype(np.float32); rec["sv_vol"] = vol[live].astype(np.float32)
        # MSB rescaling: subtract mean, divide by sqrt(pi/2)*MAD (NOT std; std is inflated by the fat tails)
        flat = g.ravel(); flat = flat[np.isfinite(flat)]; scale = np.sqrt(np.pi / 2) * np.abs(flat - flat.mean()).mean()
        z = (flat - flat.mean()) / (scale if scale > 0 else 1.0)
        rec["growth_z"] = (np.random.default_rng(seed + 9).choice(z, 8000, replace=False)
                           if z.size > 8000 else z).astype(np.float32)
    if store_shares:
        end = W[:, -1]
        rng2 = np.random.default_rng(seed + 7)
        order = np.argsort(end); top = order[-(SHARE_SAMPLE // 2):]
        rnd = rng2.choice(order[:-(SHARE_SAMPLE // 2)], SHARE_SAMPLE - top.size, replace=False)
        idx = np.concatenate([top, rnd])   # exactly SHARE_SAMPLE distinct firms
        rec["t"] = t.astype(np.float32); rec["lnM"] = lnM.astype(np.float32)
        rec["share_S"] = (N * W[idx]).astype(np.float32)
    return rec


if __name__ == "__main__":
    import time
    print(f"[ch4_data] smoke={SMOKE} | regime sigma={SIGMA} mu={MU} (chapter mu={-MU}) | jobs={N_JOBS}", flush=True)
    t0 = time.time(); out = {}

    print("[ch4_data] (1/4) showcase single run ...", flush=True)
    sc = simulate(N_SHOWCASE, 0, store_shares=True)
    for k in ("t", "lnM", "share_S"):
        out["sc_" + k] = sc.get(k, np.array([]))
    out["sc_g_eff"] = sc["g_eff"]; out["sc_surv"] = sc["surv"]; out["sc_N"] = N_SHOWCASE

    print("[ch4_data] (2/4) MSB size-volatility + tent ...", flush=True)
    msb = Parallel(n_jobs=N_JOBS, backend="loky")(delayed(simulate)(MSB_N, s) for s in range(MSB_SEEDS))
    msb = [r for r in msb if not r["capped"] and r["persistent"] and "sv_S" in r and np.isfinite(r["beta"])]
    out["msb_z"] = np.concatenate([r["growth_z"] for r in msb]) if msb else np.array([])
    def _bowley(f):
        q1, q2, q3 = np.percentile(f, [25, 50, 75]); return (q3 + q1 - 2 * q2) / (q3 - q1)
    out["msb_bowley"] = np.nanmedian([_bowley(r["growth_z"]) for r in msb]) if msb else np.nan
    out["msb_exk"] = np.nanmedian([kurtosis(r["growth_z"]) for r in msb]) if msb else np.nan
    out["msb_betas"] = np.array([r["beta"] for r in msb])
    # representative SINGLE economy (best survival) for the size-volatility figure: shows the plateau + decline,
    # decline-branch beta (NOT pooled across economies, which washes out the floor plateau and flattens the slope)
    if msb:
        best = max(msb, key=lambda r: r["surv"])
        out["msb_sv_S"] = best["sv_S"]; out["msb_sv_vol"] = best["sv_vol"]
        out["msb_bin_S"] = best["bin_S"]; out["msb_bin_vol"] = best["bin_vol"]; out["msb_pk"] = best["pk"]
        out["msb_beta"] = best["beta"]; out["msb_r2"] = best["r2"]; out["msb_surv"] = best["surv"]
    else:
        for k in ("msb_sv_S", "msb_sv_vol", "msb_bin_S", "msb_bin_vol"):
            out[k] = np.array([])
        out["msb_beta"] = np.nan; out["msb_r2"] = np.nan; out["msb_pk"] = 0; out["msb_surv"] = np.nan
    out["msb_N"] = MSB_N

    print("[ch4_data] (3/4) N-scan (persistence + beta) ...", flush=True)
    jobs = [(N, s) for N in NSCAN for s in range(NSCAN_SEEDS)]
    ns = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(delayed(simulate)(N, s) for (N, s) in jobs)
    Ns, pf, bmed, blo, bhi = [], [], [], [], []
    for N in NSCAN:
        g = [r for r in ns if r["N"] == N and not r["capped"]]
        if not g:
            continue
        persist = [r for r in g if r["persistent"]]
        clean = [r["beta"] for r in persist if np.isfinite(r["beta"]) and r["r2"] > 0.9]
        Ns.append(N); pf.append(len(persist) / len(g))
        bmed.append(np.median(clean) if clean else np.nan)
        blo.append(np.percentile(clean, 25) if clean else np.nan)
        bhi.append(np.percentile(clean, 75) if clean else np.nan)
    Ns = np.array(Ns, float); bmed = np.array(bmed)
    out["ns_N"] = Ns; out["ns_persistf"] = np.array(pf)
    out["ns_beta"] = bmed; out["ns_blo"] = np.array(blo); out["ns_bhi"] = np.array(bhi)
    # extrapolation beta = beta_inf + c / sqrt(N) on the finite points
    fin = np.isfinite(bmed)
    if fin.sum() >= 3:
        x = 1.0 / np.sqrt(Ns[fin]); A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, bmed[fin], rcond=None)
        out["ns_beta_inf"] = float(coef[0]); out["ns_beta_slope"] = float(coef[1])
    else:
        out["ns_beta_inf"] = np.nan; out["ns_beta_slope"] = np.nan

    print("[ch4_data] (4/4) freeze-vs-persist trajectories ...", flush=True)
    tr = Parallel(n_jobs=N_JOBS, backend="loky")(delayed(simulate)(TRAJ_N, s, True) for s in range(TRAJ_SEEDS))
    tr = [r for r in tr if not r["capped"] and "share_S" in r]
    out["traj_t"] = tr[0]["t"] if tr else np.array([])
    out["traj_shares"] = np.stack([r["share_S"] for r in tr]) if tr else np.array([])
    out["traj_persistent"] = np.array([r["persistent"] for r in tr])
    out["traj_gstd"] = np.array([r["gstd_late"] for r in tr])
    out["traj_N"] = TRAJ_N

    np.savez(OUT, **out)
    print(f"\n[ch4_data] N-scan beta(N): " +
          ", ".join(f"N={int(n)}:{b:+.2f}" for n, b in zip(Ns, bmed)) +
          f"  -> beta_inf={out['ns_beta_inf']:+.3f}", flush=True)
    print(f"[ch4_data] MSB pooled beta={out['msb_beta']:+.2f} (r2={out['msb_r2']:.2f}), "
          f"bowley={out['msb_bowley']:+.2f}, exk={out['msb_exk']:.1f}; showcase g_eff={sc['g_eff']:+.2f}", flush=True)
    print(f"[ch4_data] saved -> {OUT}  ({(time.time()-t0)/60:.1f} min)", flush=True)
