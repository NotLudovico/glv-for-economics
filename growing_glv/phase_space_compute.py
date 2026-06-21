"""Overnight phase-space computation for the relative GLV.

Sweeps a (mu, sigma) grid with several seeds, integrating each cell with RK45 and storing a rich set of
per-(sigma, mu, seed) observables to phase_space_data.npz. The companion phase_space_plot.py turns that
dataset into figures, so plotting and re-classification cost nothing once this has run.

RESUMABLE: results are checkpointed after every sigma-row. Re-running picks up where it stopped, so an
interrupt or crash loses at most one row. Delete phase_space_data.npz to start fresh.

Edit the CONFIG block, then run:
    uv run --directory /path/to/glv python growing_glv/phase_space_compute.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import kurtosis
from joblib import Parallel, delayed
from explore import build_alpha

# ----------------------------- CONFIG (edit me) -----------------------------
N        = 4000                       # system size. NOTE cost ~ N^2.5: chaotic cells take ~11s at N=4000
                                      #   but ~3min at N=8000 (RK45 takes far more, costlier steps). 4000 is the
                                      #   feasible sweet spot where the phases are still clearly resolved.
MUS      = np.round(np.linspace(0.0, 4.5, 24), 3)   # mean competition axis
SIGS     = np.round(np.linspace(0.4, 3.4, 21), 3)   # disorder axis
N_SEEDS  = 15                         # realizations per cell (more = crisper boundaries)
LAM      = 1e-3                       # immigration floor
TMAX     = 400.0                      # integration horizon (long enough to separate chaos from relaxation at N=4000)
N_EVAL   = 1600
RTOL     = 1e-4                       # loose: phase diagram needs trajectory STATISTICS, not accurate paths
                                      #   (benchmark: gstd/beta unchanged vs 1e-6, but far fewer solver steps)
MAX_NFEV = 80000                      # hard cap on RHS evals/run. Chaotic cells need ~41k (finish); the
                                      #   stiff explosive/condensed corner needs >>this -> capped + flagged,
                                      #   which bounds total runtime (that corner only needs a label, not stats).
N_JOBS   = 6                          # joblib workers (you have 10 cores; 6 leaves comfortable headroom)
OUT      = os.path.join(os.path.dirname(__file__), "phase_space_data.npz")
MID_WIN  = (240.0, 310.0)            # mid window  (for the stationarity ratio)
LATE_WIN = (320.0, 390.0)            # late window (chaos amplitude + observables)
# ----------------------------------------------------------------------------

OBS = ["g_eff", "gstd_late", "gstd_mid", "surv_frac", "beta", "r2", "bowley", "exkurt", "capped"]


class _Capped(Exception):
    """Raised inside the RHS once MAX_NFEV is exceeded, to abort a too-stiff (explosive) cell."""
    pass


def _wstd(W, t, S, w0, w1):
    surv = np.where(W[:, (t >= w0) & (t <= w1)].min(1) > 1e-6)[0]
    if surv.size < 30:
        return 0.0, surv
    tg = np.arange(w0, w1 + 1e-9, 0.5)
    lnS = np.array([np.interp(tg, t, np.log(np.maximum(S * W[i], 1e-12))) for i in surv])
    return float(np.diff(lnS, axis=1).ravel().std()), surv


def measure(seed, mu, sigma):
    alpha, _ = build_alpha(seed, N, mu, sigma)
    rng = np.random.default_rng(seed + 1); w0 = rng.uniform(0.5, 1.5, N); w0 /= w0.sum()
    nev = [0]
    def rhs(t, st):
        nev[0] += 1
        if nev[0] > MAX_NFEV:                    # too stiff (explosive corner): abort, label later
            raise _Capped()
        w = np.clip(st[:N], 0.0, None); s = w.sum()
        if s > 0: w = w / s
        f = 1.0 - N * w - N * (alpha @ w); fb = float(w @ f)
        return np.concatenate([w * (f - fb) + LAM * (1.0 / N - w), [fb]])
    out = {k: np.nan for k in OBS}
    out["capped"] = 0.0
    try:
        r = solve_ivp(rhs, (0, TMAX), np.concatenate([w0, [0.0]]), method="RK45",
                      t_eval=np.linspace(0, TMAX, N_EVAL), rtol=RTOL, atol=1e-7)
    except _Capped:
        # explosive/condensed corner -- only a label is needed, not accurate observables
        out["capped"] = 1.0; out["gstd_late"] = 0.0; out["surv_frac"] = 0.0
        return out
    W = np.clip(r.y[:N], 0.0, None); W = W / W.sum(0, keepdims=True); t = r.t; lnM = r.y[N]
    treached = float(t[-1])
    late_t = t > 0.5 * treached
    if late_t.sum() > 2:
        out["g_eff"] = float(np.polyfit(t[late_t], lnM[late_t], 1)[0])
    if treached < LATE_WIN[1]:                  # condensed early -> no fluctuating late window
        out["gstd_late"] = 0.0; out["gstd_mid"] = 0.0
        out["surv_frac"] = float((W[:, -1] > 1e-6).mean())
        return out
    out["gstd_mid"], _ = _wstd(W, t, N, *MID_WIN)
    gstd_late, surv = _wstd(W, t, N, *LATE_WIN)
    out["gstd_late"] = gstd_late
    out["surv_frac"] = surv.size / N
    if surv.size >= 50:
        tg = np.arange(LATE_WIN[0], LATE_WIN[1] + 1e-9, 0.5)
        lnS = np.array([np.interp(tg, t, np.log(np.maximum(N * W[i], 1e-12))) for i in surv])
        g = np.diff(lnS, axis=1); Sbar = np.exp(lnS).mean(1)
        vol = np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)
        live = (vol > 0) & np.isfinite(vol); Sbar, vol = Sbar[live], vol[live]
        gl = g[live].ravel()
        if gl.size > 10:
            q1, q2, q3 = np.percentile(gl, [25, 50, 75])
            out["bowley"] = float((q3 + q1 - 2 * q2) / (q3 - q1)) if q3 > q1 else np.nan
            out["exkurt"] = float(kurtosis(gl))
        o = np.argsort(Sbar); P = np.array_split(np.arange(o.size), 20)
        bx = np.array([Sbar[o][p].mean() for p in P]); by = np.array([np.median(vol[o][p]) for p in P])
        m = (bx > 0) & (by > 0); bx, by = bx[m], by[m]
        if bx.size > 5:
            pk = int(np.argmax(by))
            if by.size - pk >= 5:
                c = np.polyfit(np.log10(bx[pk:]), np.log10(by[pk:]), 1)
                yh = np.polyval(c, np.log10(bx[pk:]))
                ss = np.sum((np.log10(by[pk:]) - np.log10(by[pk:]).mean()) ** 2)
                out["beta"] = float(-c[0])
                out["r2"] = float(1 - np.sum((np.log10(by[pk:]) - yh) ** 2) / ss) if ss > 0 else np.nan
    return out


def load_or_init():
    arrs = {k: np.full((len(SIGS), len(MUS), N_SEEDS), np.nan) for k in OBS}
    done = np.zeros(len(SIGS), dtype=bool)
    if os.path.exists(OUT):
        d = np.load(OUT, allow_pickle=True)
        if list(d["mus"]) == list(MUS) and list(d["sigs"]) == list(SIGS) and int(d["n_seeds"]) == N_SEEDS:
            for k in OBS:
                arrs[k] = d[k]
            done = d["rows_done"]
            print(f"resuming: {int(done.sum())}/{len(SIGS)} sigma-rows already done")
        else:
            print("existing file has a different grid -> starting fresh (rename it to keep)")
    return arrs, done


if __name__ == "__main__":
    import time
    seeds = np.random.default_rng(2024).integers(0, 2**31 - 1, size=N_SEEDS)
    arrs, done = load_or_init()
    print(f"[start] grid {len(MUS)}x{len(SIGS)} x {N_SEEDS} seeds, N={N}, TMAX={TMAX}, jobs={N_JOBS}, "
          f"cap={MAX_NFEV}; {len(SIGS) - int(done.sum())} sigma-rows to do ({len(MUS) * N_SEEDS} runs each)", flush=True)
    t_start = time.time(); rows_done_now = 0; rows_left = int((~done).sum())
    for j, sig in enumerate(SIGS):
        if done[j]:
            continue
        t0 = time.time()
        jobs = [(int(s), float(mu)) for mu in MUS for s in seeds]
        # verbose=10 -> joblib prints task-completion progress WITHIN the row, so you see liveness
        # during the ~minutes each row takes, not just one line per row.
        res = Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
            delayed(measure)(sd, mu, float(sig)) for sd, mu in jobs)
        for idx, (mu_i) in enumerate([i for i in range(len(MUS)) for _ in range(N_SEEDS)]):
            seed_i = idx % N_SEEDS
            for k in OBS:
                arrs[k][j, mu_i, seed_i] = res[idx][k]
        done[j] = True
        np.savez(OUT, mus=MUS, sigs=SIGS, n_seeds=N_SEEDS, rows_done=done, **arrs)
        rows_done_now += 1
        elapsed = time.time() - t_start
        eta = elapsed / rows_done_now * (rows_left - rows_done_now)
        chaotic = float(np.mean(arrs["gstd_late"][j] > 0.03))
        capped = float(np.mean(arrs["capped"][j]))
        print(f"[row {j+1}/{len(SIGS)}] sigma={sig:.2f} done in {time.time()-t0:.0f}s | "
              f"chaotic={chaotic:.2f} capped={capped:.2f} med g_eff={np.nanmedian(arrs['g_eff'][j]):+.1f} | "
              f"elapsed {elapsed/60:.0f}m, ETA {eta/60:.0f}m", flush=True)
    print(f"\n[done] ALL ROWS COMPLETE -> {OUT}", flush=True)
