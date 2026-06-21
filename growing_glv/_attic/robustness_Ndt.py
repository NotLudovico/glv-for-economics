"""Finite-size (N) and Delta-t robustness of the MSB facts at the round headline regime
(mu=2, sigma=2, lam=1e-3). If beta/Bowley/exkurt drift with N they'd be a finite-size artifact; if
they drift with Dt they'd be a sampling artifact. We want them STABLE (the defensibility traps that
bit us earlier this session: finite-size blow-up + adaptive-Dt fat tails).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from scipy.stats import kurtosis
from explore import integrate
from measure_msb import regrid_logS, msb_on_grid, binned_beta

MU, SIGMA, LAM = 2.0, 2.0, 1e-3
TMAX, T_BURN = 45.0, 18.0


def bowley(g):
    Q1, Q2, Q3 = np.percentile(g, [25, 50, 75])
    return (Q3 + Q1 - 2 * Q2) / (Q3 - Q1) if Q3 > Q1 else np.nan


def measure(seed, S, dt):
    t, W, lnM, *_ = integrate(seed, S, MU, SIGMA, tmax=TMAX, n_eval=3000, lam=LAM)
    g_eff = float(np.polyfit(t[t > T_BURN], lnM[t > T_BURN], 1)[0])
    win = t > T_BURN
    surv = W[:, win].min(1) > 1e-6
    alive = float((W[:, win] > 1e-6).sum(0).mean()) / S
    if surv.sum() < 50:
        return dict(g_eff=g_eff, beta=np.nan, bowley=np.nan, exk=np.nan, alive=alive)
    tg = np.arange(T_BURN, TMAX + 1e-9, dt)
    lnS = regrid_logS(t, W[surv], tg)
    Sbar, vol, gl = msb_on_grid(lnS)
    beta, _ = binned_beta(Sbar, vol)
    return dict(g_eff=g_eff, beta=beta, bowley=float(bowley(gl)), exk=float(kurtosis(gl)), alive=alive)


def agg(res, key):
    v = np.array([r[key] for r in res if np.isfinite(r[key])])
    return (np.mean(v), np.std(v), v.size) if v.size else (np.nan, np.nan, 0)


if __name__ == "__main__":
    print(f"headline regime: mu={MU} sigma={SIGMA} lam={LAM:g}\n")
    seeds = np.random.default_rng(7).integers(0, 2**31 - 1, size=3)

    print("FINITE-SIZE (Dt=0.5, 3 seeds):")
    print(f"{'N':>6} {'g_eff':>12} {'beta':>12} {'Bowley':>12} {'exkurt':>11} {'alive%':>7}")
    for S in [300, 600, 1200]:
        res = [measure(int(s), S, 0.5) for s in seeds]
        gm = agg(res, "g_eff"); bm = agg(res, "beta"); wm = agg(res, "bowley"); em = agg(res, "exk")
        am = agg(res, "alive")
        print(f"{S:>6} {gm[0]:>6.2f}+/-{gm[1]:<4.2f} {bm[0]:>6.2f}+/-{bm[1]:<4.2f} "
              f"{wm[0]:>+6.2f}+/-{wm[1]:<4.2f} {em[0]:>6.1f}+/-{em[1]:<4.1f} {100*am[0]:>6.0f}%")

    print("\nDelta-t ROBUSTNESS (S=600, single seed, same trajectory re-gridded):")
    print(f"{'Dt':>6} {'beta':>8} {'Bowley':>8} {'exkurt':>9}")
    sd = int(seeds[0])
    t, W, lnM, *_ = integrate(sd, 600, MU, SIGMA, tmax=TMAX, n_eval=3000, lam=LAM)
    win = t > T_BURN; surv = W[:, win].min(1) > 1e-6
    for dt in [0.25, 0.5, 1.0, 2.0]:
        tg = np.arange(T_BURN, TMAX + 1e-9, dt)
        lnS = regrid_logS(t, W[surv], tg)
        Sbar, vol, gl = msb_on_grid(lnS)
        beta, _ = binned_beta(Sbar, vol)
        print(f"{dt:>6.2f} {beta:>8.3f} {bowley(gl):>+8.3f} {kurtosis(gl):>9.1f}")

    print("\nWant: beta/Bowley stable across N and Dt (sign + ~magnitude); exkurt may shrink with Dt (normal).")
