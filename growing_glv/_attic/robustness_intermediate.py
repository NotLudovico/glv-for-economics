"""Robustness checks for the two intermediate operating points of the relative GLV.

Hardens the phase-sweep front-runners
    OP-A: sigma=1.90, mu=2.35   (cleaner fit, beta~0.40, higher coexistence)
    OP-B: sigma=2.05, mu=2.94   (lower beta~0.28, near the g_eff growth edge)
against the three things that have burned this project before:

  1. N-SCALING of beta   -- the honesty must-fix. The thesis eq (5.4) beta_inf=0.23 was measured at the
                            SHRINKING mu=3,sigma=2 regime. Re-derive beta(N) HERE, on R^2>0.9 PERSISTENT
                            seeds only, and report the persistence fraction -> 1 with N.
  2. SEED robustness     -- many seeds at one N: what fraction give the full MSB signature, and the spread.
  3. Dt robustness       -- vary the growth-rate sampling horizon; beta / tails must be FLAT, not the
                            adaptive-Dt aliasing artifact of chaotic_glv.

DATASET-ORIENTED: each (op, N, seed) is integrated ONCE; the late-window survivor log-sizes are stored on a
fine grid so beta, the tent, and the Dt sweep are all computed afterwards without re-simulating. Results ->
robustness_intermediate.npz (+ .png). Runs at N_JOBS=3 to leave the phase sweep its cores.

    uv run --directory /path/to/glv python growing_glv/robustness_intermediate.py [--smoke]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import kurtosis
from joblib import Parallel, delayed
from explore import build_alpha            # same disorder/graph as the phase sweep (faithful)

SMOKE = "--smoke" in sys.argv

# ----------------------------- CONFIG -----------------------------
OPS = {"A_s1.90_m2.35": dict(sigma=1.90, mu=2.35),
       "B_s2.05_m2.94": dict(sigma=2.05, mu=2.94)}
N_SCALE   = [1600, 3200, 6400, 12800]      # N-scaling axis (25600 deferred until the sweep frees cores)
N_SCALE_SEEDS = 6
N_SEED    = 4000                           # system size for seed- and Dt-robustness
N_SEED_SEEDS = 40                          # seeds for the seed-robustness distribution
DT_LIST   = [0.25, 0.5, 1.0, 2.0]          # growth-rate sampling horizons (0.25 may alias -> flagged)
LAM       = 1e-3
TMAX      = 400.0
N_EVAL    = 1600
RTOL, ATOL = 1e-4, 1e-7
MAX_NFEV  = 80000
N_JOBS    = 3
MID_WIN   = (240.0, 310.0)
LATE_WIN  = (320.0, 390.0)
DT_BASE   = 0.5                            # base step for the size-vol/tent measurement (matches the sweep)
FINE_DT   = 0.25                           # fine grid stored for the Dt sweep
SURV_CAP  = 1500                           # cap stored survivors (memory) for the Dt dataset
OUT       = os.path.join(os.path.dirname(__file__), "robustness_intermediate.npz")

if SMOKE:                                  # tiny grid to validate the code path fast
    N_SCALE = [400, 800]; N_SCALE_SEEDS = 2
    N_SEED = 400; N_SEED_SEEDS = 4; TMAX = 120.0; N_EVAL = 600
    MID_WIN = (60.0, 90.0); LATE_WIN = (95.0, 118.0); SURV_CAP = 400
# ------------------------------------------------------------------
TGRID = np.linspace(0.0, TMAX, N_EVAL)


class _Capped(Exception):
    pass


def _decline_beta(Sbar, vol):
    """Fit sigma(S) ~ S^-beta on the large-S DECLINE branch only (above the small-firm floor plateau)."""
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


def _window(W, t, N, w0, w1, dt):
    """Survivor indices + their log-sizes ln S (S=N w) on a regular grid of step dt over [w0,w1]."""
    surv = np.where(W[:, (t >= w0) & (t <= w1)].min(1) > 1e-6)[0]
    tg = np.arange(w0, w1 + 1e-9, dt)
    if surv.size == 0:
        return surv, np.empty((0, tg.size)), tg
    lnS = np.array([np.interp(tg, t, np.log(np.maximum(N * W[i], 1e-12))) for i in surv])
    return surv, lnS, tg


def simulate(op, N, seed, store_fine):
    p = OPS[op]
    alpha, _ = build_alpha(seed, N, p["mu"], p["sigma"])
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
    rec = dict(op=op, N=N, seed=int(seed), capped=False)
    try:
        r = solve_ivp(rhs, (0, TMAX), np.concatenate([w0, [0.0]]), method="RK45",
                      t_eval=TGRID, rtol=RTOL, atol=ATOL)
    except _Capped:
        rec["capped"] = True; return rec
    t = r.t; W = np.clip(r.y[:N], 0, None); W = W / W.sum(0, keepdims=True); lnM = r.y[N]
    lh = t > 0.5 * t[-1]; rec["g_eff"] = float(np.polyfit(t[lh], lnM[lh], 1)[0])
    # stationarity scalars
    sm, lnSm, _ = _window(W, t, N, *MID_WIN, DT_BASE); gm = np.diff(lnSm, 1) if sm.size else np.empty((0, 0))
    sl, lnSl, _ = _window(W, t, N, *LATE_WIN, DT_BASE); gl = np.diff(lnSl, 1) if sl.size else np.empty((0, 0))
    rec["gstd_mid"] = float(gm.std()) if gm.size > 50 else 0.0
    rec["gstd_late"] = float(gl.std()) if gl.size > 50 else 0.0
    rec["surv_frac"] = float(sl.size / N)
    rec["persistent"] = bool(rec["gstd_late"] > 0.05)
    # base-Dt size-volatility + beta + tent
    if sl.size >= 50:
        Sbar = np.exp(lnSl).mean(1)
        vol = np.sqrt(np.pi / 2) * np.abs(gl - gl.mean(1, keepdims=True)).mean(1)
        b, r2 = _decline_beta(Sbar, vol); rec["beta"] = b; rec["r2"] = r2
        flat = gl.ravel(); flat = flat[np.isfinite(flat)]
        q1, q2, q3 = np.percentile(flat, [25, 50, 75])
        rec["bowley"] = float((q3 + q1 - 2 * q2) / (q3 - q1)) if q3 > q1 else np.nan
        rec["exkurt"] = float(kurtosis(flat))
    else:
        rec["beta"] = rec["r2"] = rec["bowley"] = rec["exkurt"] = np.nan
    # fine-grid late-window log-sizes for the Dt sweep (only stored where requested)
    if store_fine and sl.size >= 50:
        sf, lnSf, _ = _window(W, t, N, *LATE_WIN, FINE_DT)
        if sf.size > SURV_CAP:
            keep = np.random.default_rng(seed + 3).choice(sf.size, SURV_CAP, replace=False)
            lnSf = lnSf[keep]
        rec["fine_lnS"] = lnSf.astype(np.float32)
    return rec


def beta_at_dt(fine_lnS, dt):
    """Recompute (beta, r2, bowley, exkurt) from a fine ln S matrix subsampled to step dt."""
    step = max(int(round(dt / FINE_DT)), 1)
    lnS = fine_lnS[:, ::step]
    if lnS.shape[1] < 3:
        return dict(beta=np.nan, r2=np.nan, bowley=np.nan, exkurt=np.nan)
    g = np.diff(lnS, axis=1)
    Sbar = np.exp(lnS).mean(1)
    vol = np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)
    b, r2 = _decline_beta(Sbar, vol)
    flat = g.ravel(); flat = flat[np.isfinite(flat)]
    q1, q2, q3 = np.percentile(flat, [25, 50, 75])
    bow = float((q3 + q1 - 2 * q2) / (q3 - q1)) if q3 > q1 else np.nan
    return dict(beta=b, r2=r2, bowley=bow, exkurt=float(kurtosis(flat)))


if __name__ == "__main__":
    import time
    print(f"[robustness] smoke={SMOKE} ops={list(OPS)} | N-scale {N_SCALE}x{N_SCALE_SEEDS} | "
          f"seed N={N_SEED}x{N_SEED_SEEDS} | Dt {DT_LIST} | jobs={N_JOBS}", flush=True)
    t0 = time.time()

    # ---- jobs: N=N_SEED group stores fine grid (serves seed + Dt + the N_SEED point of N-scaling) ----
    jobs = []
    for op in OPS:
        for sd in range(N_SEED_SEEDS):
            jobs.append((op, N_SEED, sd, True))
        for N in N_SCALE:
            if N == N_SEED:
                continue
            for sd in range(N_SCALE_SEEDS):
                jobs.append((op, N, sd, False))
    print(f"[robustness] {len(jobs)} runs queued", flush=True)
    recs = Parallel(n_jobs=N_JOBS, backend="loky", verbose=8)(
        delayed(simulate)(op, N, sd, sf) for (op, N, sd, sf) in jobs)
    recs = [r for r in recs if r is not None]
    print(f"[robustness] done integrating in {(time.time()-t0)/60:.1f} min "
          f"({sum(r['capped'] for r in recs)} capped)", flush=True)

    # ===================== analyses =====================
    def grp(op, N):
        return [r for r in recs if r["op"] == op and r["N"] == N and not r["capped"]]

    out = {}
    print("\n================ 1) N-SCALING of beta (R^2>0.9 PERSISTENT seeds) ================")
    for op in OPS:
        Ns, bmed, blo, bhi, pf, nseed = [], [], [], [], [], []
        for N in sorted(set(N_SCALE + [N_SEED])):
            g = grp(op, N)
            if not g:
                continue
            persist = [r for r in g if r["persistent"]]
            clean = [r["beta"] for r in persist if np.isfinite(r["beta"]) and r["r2"] > 0.9]
            Ns.append(N); pf.append(len(persist) / len(g)); nseed.append(len(clean))
            if clean:
                bmed.append(np.median(clean)); blo.append(np.percentile(clean, 25)); bhi.append(np.percentile(clean, 75))
            else:
                bmed.append(np.nan); blo.append(np.nan); bhi.append(np.nan)
        out[f"{op}_Ns"] = np.array(Ns); out[f"{op}_beta"] = np.array(bmed)
        out[f"{op}_blo"] = np.array(blo); out[f"{op}_bhi"] = np.array(bhi); out[f"{op}_persistf"] = np.array(pf)
        print(f"\n  {op}:")
        for k, N in enumerate(Ns):
            print(f"    N={N:6d}  persist_frac={pf[k]:.2f}  clean_seeds={nseed[k]:2d}  "
                  f"beta={bmed[k]:+.3f} [{blo[k]:+.2f},{bhi[k]:+.2f}]")

    print("\n================ 2) SEED robustness (N=%d) ================" % N_SEED)
    for op in OPS:
        g = grp(op, N_SEED)
        gef = np.array([r["g_eff"] for r in g]); per = np.array([r["persistent"] for r in g])
        be = np.array([r["beta"] for r in g]); r2 = np.array([r["r2"] for r in g])
        bw = np.array([r["bowley"] for r in g]); ek = np.array([r["exkurt"] for r in g])
        full = (gef > 0) & per & (be > 0) & (r2 > 0.8) & (np.abs(bw) < 0.10) & (ek > 3)
        out[f"{op}_seed_geff"] = gef; out[f"{op}_seed_beta"] = be; out[f"{op}_seed_r2"] = r2
        out[f"{op}_seed_bowley"] = bw; out[f"{op}_seed_exk"] = ek; out[f"{op}_seed_persist"] = per
        def ms(a): a = a[np.isfinite(a)]; return (np.nan, np.nan, np.nan) if not a.size else (np.median(a), np.percentile(a, 25), np.percentile(a, 75))
        print(f"\n  {op}  (n={len(g)} seeds):")
        print(f"    growing frac      = {np.mean(gef>0):.2f}")
        print(f"    persistent frac   = {np.mean(per):.2f}")
        print(f"    FULL-signature frac = {np.mean(full):.2f}  (grow & persist & beta>0 & r2>.8 & |bow|<.1 & exk>3)")
        for nm, a in [("g_eff", gef), ("beta", be), ("r2", r2), ("bowley", bw), ("exkurt", ek)]:
            m, lo, hi = ms(a); print(f"    {nm:7s} median={m:+.3f}  IQR=[{lo:+.2f},{hi:+.2f}]")

    print("\n================ 3) Dt robustness (N=%d, persistent seeds) ================" % N_SEED)
    for op in OPS:
        g = [r for r in grp(op, N_SEED) if r["persistent"] and "fine_lnS" in r]
        print(f"\n  {op}  ({len(g)} persistent seeds):")
        for dt in DT_LIST:
            vals = [beta_at_dt(r["fine_lnS"], dt) for r in g]
            bb = np.array([v["beta"] for v in vals]); bw = np.array([v["bowley"] for v in vals])
            ek = np.array([v["exkurt"] for v in vals]); rr = np.array([v["r2"] for v in vals])
            def med(a): a = a[np.isfinite(a)]; return np.median(a) if a.size else np.nan
            flag = "  (may alias)" if dt <= FINE_DT else ""
            print(f"    Dt={dt:.2f}: beta={med(bb):+.3f} (r2={med(rr):.2f})  bowley={med(bw):+.3f}  exkurt={med(ek):.1f}{flag}")
            out[f"{op}_dt{dt}_beta"] = bb; out[f"{op}_dt{dt}_bowley"] = bw; out[f"{op}_dt{dt}_exk"] = ek

    out["DT_LIST"] = np.array(DT_LIST); out["ops"] = np.array(list(OPS))
    np.savez(OUT, **out)
    print(f"\n[robustness] saved -> {OUT}   ({(time.time()-t0)/60:.1f} min total)", flush=True)
