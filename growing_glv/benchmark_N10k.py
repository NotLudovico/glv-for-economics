"""Benchmark a single N=10000 long run: wall time + a quick persistence read, to decide whether a
multi-seed long run at 10k is feasible (and whether LSODA stays fast or blows up on a dense Jacobian).
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from explore import integrate

S, MU, SIGMA, LAM = 10000, 3.0, 2.0, 1e-3
TMAX, N_EVAL = 150.0, 3000


def gstd(W, t, t0, t1):
    surv = np.where(W[:, (t >= t0) & (t <= t1)].min(1) > 1e-6)[0]
    if surv.size < 40:
        return np.nan, surv.size
    tg = np.arange(t0, t1 + 1e-9, 0.5)
    lnS = np.array([np.interp(tg, t, np.log(np.maximum(S * W[i], 1e-12))) for i in surv])
    return float(np.diff(lnS, axis=1).ravel().std()), surv.size


if __name__ == "__main__":
    seed = int(np.random.default_rng(123).integers(0, 2**31 - 1, size=4)[1])  # a persistent seed at smaller N
    print(f"benchmarking N={S}, mu={MU} sigma={SIGMA} lam={LAM:g}, tmax={TMAX}, seed={seed%10000} ...", flush=True)
    t0 = time.time()
    t, W, lnM, _, _ = integrate(seed, S, MU, SIGMA, lam=LAM, tmax=TMAX, n_eval=N_EVAL)
    wall = time.time() - t0
    em, ne = gstd(W, t, 60, 90)
    lm, nl = gstd(W, t, 120, 145)
    print(f"wall time: {wall:.1f} s for tmax={TMAX}  (-> ~{wall/TMAX*300:.0f} s per tmax=300 run)")
    print(f"growth std [60,90]={em:.3f} (n={ne}),  [120,145]={lm:.3f} (n={nl})  ratio={lm/em if em else float('nan'):.2f}")
    print(f"g_eff (late)={np.polyfit(t[t>80], lnM[t>80], 1)[0]:.3f}")
