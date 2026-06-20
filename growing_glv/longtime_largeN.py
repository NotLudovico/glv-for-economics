"""Rule out the finite-size relaxation trap: at large N, is the 'persistent' state GENUINELY stationary at
very long time, or a slow transient whose relaxation timescale just grew with N? Integrate to t=600 and
compare growth std in a mid-late window [170,240] vs a very-late window [470,540]. ratio ~1 = genuine;
ratio -> 0 = it was still relaxing (the trap, scaled by N).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from explore import integrate

MU, SIGMA, LAM = 3.0, 2.0, 1e-3
TMAX, N_EVAL = 600.0, 8000


def gstd(W, t, S, t0, t1):
    surv = np.where(W[:, (t >= t0) & (t <= t1)].min(1) > 1e-6)[0]
    if surv.size < 40:
        return np.nan, surv.size
    tg = np.arange(t0, t1 + 1e-9, 0.5)
    lnS = np.array([np.interp(tg, t, np.log(np.maximum(S * W[i], 1e-12))) for i in surv])
    return float(np.diff(lnS, axis=1).ravel().std()), surv.size


if __name__ == "__main__":
    seeds = np.random.default_rng(123).integers(0, 2**31 - 1, size=3)
    print(f"regime mu={MU} sigma={SIGMA} lam={LAM:g}, TMAX={TMAX}. mid=[170,240] vs late=[470,540].")
    print(f"ratio late/mid ~1 = genuinely persistent; ->0 = slow transient (finite-size trap).\n")
    print(f"{'N':>6} {'seed':>6} {'mid_gstd':>9} {'late_gstd':>10} {'ratio':>7} {'surv_late':>10}")
    for S in [400, 1600]:
        for s in seeds:
            t, W, lnM, _, _ = integrate(int(s), S, MU, SIGMA, lam=LAM, tmax=TMAX, n_eval=N_EVAL)
            gm, _ = gstd(W, t, S, 170, 240)
            gl, nl = gstd(W, t, S, 470, 540)
            ratio = gl / gm if (np.isfinite(gm) and gm > 0 and np.isfinite(gl)) else np.nan
            tag = ""
            if np.isfinite(ratio):
                tag = "  genuine" if ratio > 0.5 else ("  RELAXES (trap)" if ratio < 0.15 else "  decaying")
            print(f"{S:>6} {int(s)%10000:>6} {gm:>9.3f} {gl:>10.3f} {ratio:>7.2f} {nl:>8d}{tag}")
