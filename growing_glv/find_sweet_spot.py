"""Find the best joint operating point: scan (mu, sigma, lam) and score each on the full MSB trifecta
  - beta in [0.15, 0.20] (size-volatility, on survivors)
  - Bowley quantile skew ~ 0 (robust symmetry; moment skew is outlier-noisy on fat tails)
  - coexistence as high as possible
  - fat-tailed (excess kurtosis > 0) and growing (g_eff > 0)
Rank the valid candidates so we can pick the cleanest single regime to validate next.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import kurtosis
from explore import integrate
from measure_msb import regrid_logS, msb_on_grid, binned_beta

S = 600
TMAX, T_BURN, DT = 45.0, 18.0, 0.5
SEED = 2024
MUS = [2.0, 2.5, 3.0, 3.5]
SIGS = [2.0, 2.5, 3.0]
LAMS = [0.0, 1e-3, 3e-3, 1e-2]


def bowley(g):
    Q1, Q2, Q3 = np.percentile(g, [25, 50, 75])
    return (Q3 + Q1 - 2 * Q2) / (Q3 - Q1) if Q3 > Q1 else np.nan


def evaluate(mu, sigma, lam):
    t, W, lnM, *_ = integrate(SEED, S, mu, sigma, tmax=TMAX, n_eval=3000, lam=lam)
    g_eff = float(np.polyfit(t[t > T_BURN], lnM[t > T_BURN], 1)[0])
    win = t > T_BURN
    surv = W[:, win].min(1) > 1e-6                              # survive the WHOLE window (balanced panel)
    alive = float((W[:, win] > 1e-6).sum(0).mean())            # mean firms LIVE at any snapshot (instantaneous)
    if surv.sum() < 50:
        return dict(mu=mu, sigma=sigma, lam=lam, g_eff=g_eff, coexist=int(surv.sum()), alive=alive,
                    beta=np.nan, bowley=np.nan, exk=np.nan)
    tg = np.arange(T_BURN, TMAX + 1e-9, DT)
    lnS = regrid_logS(t, W[surv], tg)
    Sbar, vol, gl = msb_on_grid(lnS)
    beta, _ = binned_beta(Sbar, vol)
    return dict(mu=mu, sigma=sigma, lam=lam, g_eff=g_eff, coexist=int(surv.sum()), alive=alive,
                beta=beta, bowley=float(bowley(gl)), exk=float(kurtosis(gl)))


if __name__ == "__main__":
    rows = [evaluate(mu, sig, lam) for mu in MUS for sig in SIGS for lam in LAMS]

    # show ALL reasonable points ranked, with fatness + instantaneous alive, so the tradeoff is visible
    show = [r for r in rows if np.isfinite(r["beta"]) and r["g_eff"] > 0.05
            and 0.10 <= r["beta"] <= 0.30 and abs(r["bowley"]) <= 0.15]
    # combined score: in-band beta, symmetric, fat, and as many live firms as possible
    def score(r):
        return (-abs(r["beta"] - 0.175) - abs(r["bowley"]) - 0.3 * max(0, 3 - r["exk"])
                + 0.002 * r["alive"])
    show.sort(key=score, reverse=True)

    print(f"scanned {len(rows)} points; {len(show)} have beta in [0.10,0.30] & |Bowley|<=0.15 & growing\n")
    print("RANKED (best joint trifecta first; alive = mean firms live per snapshot, fat needs exkurt>~1):")
    print(f"{'mu':>4} {'sig':>4} {'lam':>7} {'beta':>7} {'Bowley':>7} {'exkurt':>7} {'survive':>8} {'alive':>7} {'g_eff':>7}")
    for r in show[:10]:
        print(f"{r['mu']:>4.1f} {r['sigma']:>4.1f} {r['lam']:>7.0e} {r['beta']:>7.3f} {r['bowley']:>+7.3f} "
              f"{r['exk']:>7.1f} {r['coexist']:>4d}/{S} {r['alive']:>7.0f} {r['g_eff']:>7.2f}")

    print("\nFULL GRID (beta / Bowley / exkurt / whole-window-survivors):")
    for mu in MUS:
        for sig in SIGS:
            cells = []
            for lam in LAMS:
                r = next(x for x in rows if x["mu"] == mu and x["sigma"] == sig and x["lam"] == lam)
                b = f"{r['beta']:.2f}" if np.isfinite(r["beta"]) else " na "
                e = f"{r['exk']:.0f}" if np.isfinite(r["exk"]) else "na"
                cells.append(f"l={lam:.0e}:b={b} e={e} cx={r['coexist']}")
            print(f"  mu={mu} sig={sig}: " + " | ".join(cells))
