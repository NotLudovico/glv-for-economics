"""Does the relative GLV have ANY genuinely persistent-chaos regime, or does it relax to a fixed point
everywhere (like the original model)? Integrate each (mu,sigma,lam) to t=300 and compare the pooled
growth std in an EARLY window [40,80] vs a LATE window [200,280]. ratio = late/early:
  ~1   -> persistent chaos (stationary)         <- what we need for a real MSB tent
  ->0  -> relaxation transient (shares freeze)   <- the trap (tent/beta are transient artifacts)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from explore import integrate

S, TMAX = 400, 300.0
SEED = 2024


def growth_std(W, t, t0, t1, dt=0.5):
    N = W.shape[0]
    surv = np.where(W[:, (t >= t0) & (t <= t1)].min(1) > 1e-6)[0]
    if surv.size < 30:
        return np.nan, surv.size
    tg = np.arange(t0, t1 + 1e-9, dt)
    lnS = np.array([np.interp(tg, t, np.log(np.maximum(N * W[i], 1e-12))) for i in surv])
    g = np.diff(lnS, axis=1)
    return float(g.ravel().std()), surv.size


if __name__ == "__main__":
    print(f"S={S}, TMAX={TMAX}. growth std early [40,80] vs late [200,280]; ratio->0 = relaxes, ~1 = persistent.\n")
    print(f"{'mu':>4} {'sig':>4} {'lam':>7} {'early_std':>10} {'late_std':>9} {'ratio':>7} {'surv_late':>10}")
    for mu in [2.0, 3.0]:
        for sig in [2.0, 2.5, 3.0]:
            for lam in [0.0, 1e-3]:
                t, W, lnM, _, _ = integrate(SEED, S, mu, sig, lam=lam, tmax=TMAX, n_eval=6000)
                e, ne = growth_std(W, t, 40, 80)
                l, nl = growth_std(W, t, 200, 280)
                ratio = (l / e) if (np.isfinite(e) and e > 0 and np.isfinite(l)) else np.nan
                tag = ""
                if np.isfinite(ratio):
                    tag = "  <- persistent" if ratio > 0.5 else ("  relaxes" if ratio < 0.1 else "")
                print(f"{mu:>4.1f} {sig:>4.1f} {lam:>7.0e} {e:>10.3f} {l:>9.3f} {ratio:>7.2f} {nl:>7d}{tag}")
