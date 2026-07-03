"""Fluctuating-phase, degree-resolved TEMPORAL volatility -- the decisive test.

The relaxed-phase check (hdmft_validate_sim.py) validated the HDMFT structure but
could not give MSB beta (no temporal dynamics). beta lives in the fluctuating phase
(sigma>sigma_c), exactly where beta_limit.py sees the pooled beta(size) CRASH with N.

Here we run the ch4 protocol (sigma=1.75, mu=1.76, RK45, late-window MAD vol) but
resolve every survivor's temporal volatility vol_i = sqrt(pi/2) MAD(dln S_i) by its
NODE DEGREE k_i as well as its size. Three questions, one run:
  Q1  size-degree map in the fluctuating phase (do hubs stay small? is survival
      degree-selective -> do survivors span a narrow degree range?)
  Q2  does temporal vol depend on degree (vol ~ k^? )
  Q3  pooled beta(size) vs N  -- reproduce the crash
  Q4  degree-resolved beta (vol vs size across degree classes) vs N -- is the
      degree structure N-stable even while the pooled fit crashes?

If Q4 is N-stable while Q3 crashes, the crash is a pooling artifact and the
degree-resolved relation is the real, stable beta. If Q4 also crashes, the
fluctuating relative GLV genuinely washes out and beta is a finite-size effect.
"""
import numpy as np
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed

from explore import build_alpha

MU, SIGMA, LAM = 1.76, 1.75, 1e-3
TMAX, N_EVAL = 400.0, 1600
RTOL, ATOL, MAX_NFEV = 1e-4, 1e-7, 80000
LATE = (320.0, 390.0)
DT = 0.5
TGRID = np.linspace(0.0, TMAX, N_EVAL)


class _Capped(Exception):
    pass


def simulate(N, seed):
    """Per-survivor (size, temporal vol, degree) at the ch4 fluctuating operating point."""
    alpha, _ = build_alpha(seed, N, MU, SIGMA)
    deg = np.diff(alpha.indptr)                       # node degree from CSR row pointer
    rng = np.random.default_rng(seed + 1)
    w0 = rng.uniform(0.5, 1.5, N); w0 /= w0.sum()
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

    try:
        r = solve_ivp(rhs, (0, TMAX), np.concatenate([w0, [0.0]]), method="RK45",
                      t_eval=TGRID, rtol=RTOL, atol=ATOL)
    except _Capped:
        return None
    t = r.t; W = np.clip(r.y[:N], 0, None); W = W / W.sum(0, keepdims=True)
    g_eff = float(np.polyfit(t[t > 0.5 * TMAX], r.y[N][t > 0.5 * TMAX], 1)[0])
    surv = np.where(W[:, (t >= LATE[0]) & (t <= LATE[1])].min(1) > 1e-6)[0]
    if surv.size < 50:
        return dict(empty=True, surv=surv.size / N, N=N, g_eff=g_eff, deg=deg)
    tg = np.arange(LATE[0], LATE[1] + 1e-9, DT)
    lnS = np.array([np.interp(tg, t, np.log(np.maximum(N * W[i], 1e-12))) for i in surv])
    g = np.diff(lnS, axis=1)
    Sbar = np.exp(lnS).mean(1)
    vol = np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)
    return dict(empty=False, N=N, g_eff=g_eff, surv=surv.size / N,
                S=Sbar, vol=vol, k=deg[surv], k_all=deg)


def decline_beta(S, vol, nb=20):
    """ch4 pooled beta: median vol in size bins, log-log fit from the peak bin onward."""
    live = (vol > 0) & np.isfinite(vol) & (S > 0)
    S, vol = S[live], vol[live]
    if S.size < 50:
        return np.nan, np.nan
    o = np.argsort(S); P = np.array_split(np.arange(o.size), nb)
    bx = np.array([S[o][p].mean() for p in P]); by = np.array([np.median(vol[o][p]) for p in P])
    m = (bx > 0) & (by > 0); bx, by = bx[m], by[m]
    if bx.size <= 5:
        return np.nan, np.nan
    pk = int(np.argmax(by))
    if by.size - pk < 5:
        return np.nan, np.nan
    c, res, *_ = np.polyfit(np.log10(bx[pk:]), np.log10(by[pk:]), 1, full=True)
    yl = np.log10(by[pk:]); ss = np.sum((yl - yl.mean()) ** 2)
    r2 = float(1 - (res[0] if res.size else 0.0) / ss) if ss > 0 else np.nan
    return float(-c[0]), r2


def degree_bins(k, S, vol, nb=12):
    """Median vol & mean size per log-degree bin (survivors only)."""
    edges = np.geomspace(k.min(), k.max() + 1, nb + 1)
    bk, bS, bv, bn = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (k >= lo) & (k < hi)
        if m.sum() >= 15:
            bk.append(k[m].mean()); bS.append(S[m].mean()); bv.append(np.median(vol[m])); bn.append(int(m.sum()))
    return map(np.array, (bk, bS, bv, bn))


def slope(x, y):
    if len(x) < 3:
        return np.nan, np.nan
    lx, ly = np.log(x), np.log(y)
    c = np.polyfit(lx, ly, 1)
    r2 = 1 - np.sum((ly - np.polyval(c, lx)) ** 2) / np.sum((ly - ly.mean()) ** 2)
    return float(c[0]), float(r2)


if __name__ == "__main__":
    print(f"fluctuating-phase degree-resolved vol  mu={MU} sigma={SIGMA} lam={LAM}\n")
    for N, nseed in [(4000, 6), (12800, 4)]:
        res = Parallel(n_jobs=6)(delayed(simulate)(N, s) for s in range(nseed))
        res = [r for r in res if r is not None and not r.get("empty")]
        if not res:
            print(f"  N={N}: all seeds empty/capped"); continue
        S = np.concatenate([r["S"] for r in res])
        vol = np.concatenate([r["vol"] for r in res])
        k = np.concatenate([r["k"] for r in res])
        k_all = np.concatenate([r["k_all"] for r in res])
        surv = np.mean([r["surv"] for r in res]); g_eff = np.mean([r["g_eff"] for r in res])

        b_pool, r2_pool = decline_beta(S, vol)
        bk, bS, bv, bn = degree_bins(k, S, vol)
        s_vk, r2_vk = slope(bk, bv)                       # vol ~ degree^s_vk
        s_vs, r2_vs = slope(bS, bv)                       # vol ~ size^s_vs  (degree-resolved beta = -s_vs)
        s_sk, _ = slope(bk, bS)                           # size ~ degree^s_sk (size-degree map)

        # CLINCHER: beta WITHIN a single (narrow) degree band -> is beta a within-firm effect?
        kmed = np.median(k)
        band = (k >= 0.8 * kmed) & (k <= 1.25 * kmed)        # firms of ~equal degree
        b_in, r2_in = decline_beta(S[band], vol[band])
        # do the big hubs break MSB? vol of top-decile-degree survivors vs the bulk
        hub = k >= np.quantile(k, 0.9)
        v_hub, v_bulk = np.median(vol[hub]), np.median(vol[~hub])
        S_hub, S_bulk = np.median(S[hub]), np.median(S[~hub])

        print(f"N={N}  seeds={len(res)}  surv={surv:.3f}  g_eff={g_eff:+.3f}  "
              f"survivors={S.size}  deg(all) [{k_all.min()},{k_all.max()}]  deg(surv) [{k.min()},{k.max()}]")
        print(f"   Q1 size~deg slope = {s_sk:+.2f}   (hubs small if <0)")
        print(f"   Q2 vol ~deg slope = {s_vk:+.2f}  (r2={r2_vk:.2f})  -> vol degree-independent if ~0")
        print(f"   Q3 pooled    beta(size)   = {b_pool:+.3f}  (r2={r2_pool:.2f})   <- crash metric")
        print(f"   Q4 across-degree beta     = {-s_vs:+.3f}  (r2={r2_vs:.2f})   <- degree-driven?")
        print(f"   Q5 within-band beta (k~{kmed:.0f}, n={band.sum()}) = {b_in:+.3f}  (r2={r2_in:.2f})"
              f"   <- single-firm effect if ~0.2")
        print(f"      hubs(top10% deg): vol={v_hub:.3f} S={S_hub:.2f}  vs bulk: vol={v_bulk:.3f} S={S_bulk:.2f}"
              f"  -> big hubs as volatile as bulk breaks MSB" + "\n")
