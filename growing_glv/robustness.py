"""Defensibility check: are the MSB facts GENERIC (robust across parameters + realizations), or a
fine-tuned coincidence? For a grid of round/principled parameters, run several independent seeds each
and report mean +/- std of:
  - g_eff  (growing economy, want > 0 everywhere)
  - beta   (size-volatility on survivors, want > 0 everywhere -- magnitude not fine-tuned)
  - Bowley (robust skew, want ~ 0 everywhere -- symmetric tent)
  - exkurt (fat tails, want > 0 everywhere)
The claim is genericity: the phenomenology holds across the whole chaotic regime, not at one magic point.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import kurtosis
from explore import integrate
from measure_msb import regrid_logS, msb_on_grid, binned_beta

S = 600
TMAX, T_BURN, DT = 45.0, 18.0, 0.5
N_SEEDS = 5
MUS = [2.0, 3.0]
SIGS = [2.0, 2.5]
LAMS = [0.0, 1e-3]


def bowley(g):
    Q1, Q2, Q3 = np.percentile(g, [25, 50, 75])
    return (Q3 + Q1 - 2 * Q2) / (Q3 - Q1) if Q3 > Q1 else np.nan


def one(seed, mu, sigma, lam):
    t, W, lnM, *_ = integrate(seed, S, mu, sigma, tmax=TMAX, n_eval=3000, lam=lam)
    g_eff = float(np.polyfit(t[t > T_BURN], lnM[t > T_BURN], 1)[0])
    win = t > T_BURN
    surv = W[:, win].min(1) > 1e-6
    if surv.sum() < 50:
        return dict(g_eff=g_eff, beta=np.nan, bowley=np.nan, exk=np.nan, alive=float((W[:, win] > 1e-6).sum(0).mean()))
    tg = np.arange(T_BURN, TMAX + 1e-9, DT)
    lnS = regrid_logS(t, W[surv], tg)
    Sbar, vol, gl = msb_on_grid(lnS)
    beta, _ = binned_beta(Sbar, vol)
    return dict(g_eff=g_eff, beta=beta, bowley=float(bowley(gl)), exk=float(kurtosis(gl)),
                alive=float((W[:, win] > 1e-6).sum(0).mean()))


def agg(vals):
    v = np.array([x for x in vals if np.isfinite(x)])
    return (np.mean(v), np.std(v), v.size) if v.size else (np.nan, np.nan, 0)


if __name__ == "__main__":
    base_seeds = np.random.default_rng(99).integers(0, 2**31 - 1, size=N_SEEDS)
    print(f"S={S}, {N_SEEDS} seeds per cell. Genericity of the MSB facts across round parameters.\n")
    print(f"{'mu':>4} {'sig':>4} {'lam':>6} | {'g_eff':>13} {'beta':>13} {'Bowley':>13} {'exkurt':>12} {'alive':>6}")
    all_beta, all_bow, all_geff, all_exk = [], [], [], []
    for mu in MUS:
        for sig in SIGS:
            for lam in LAMS:
                res = [one(int(s), mu, sig, lam) for s in base_seeds]
                gm, gs, _ = agg([r["g_eff"] for r in res])
                bm, bs, nb = agg([r["beta"] for r in res])
                wm, ws, _ = agg([r["bowley"] for r in res])
                em, es, _ = agg([r["exk"] for r in res])
                am, _, _ = agg([r["alive"] for r in res])
                print(f"{mu:>4.1f} {sig:>4.1f} {lam:>6.0e} | {gm:>6.2f}+/-{gs:<4.2f} {bm:>6.2f}+/-{bs:<4.2f} "
                      f"{wm:>+6.2f}+/-{ws:<4.2f} {em:>6.1f}+/-{es:<4.1f} {am:>6.0f}  (n_beta={nb})")
                if np.isfinite(bm): all_beta.append(bm); all_bow.append(wm); all_geff.append(gm); all_exk.append(em)

    print("\nVERDICT across all measurable cells:")
    print(f"  g_eff  > 0 (growing):     {all(g > 0 for g in all_geff)}  range [{min(all_geff):.2f}, {max(all_geff):.2f}]")
    print(f"  beta   > 0 (right sign):  {all(b > 0 for b in all_beta)}  range [{min(all_beta):.2f}, {max(all_beta):.2f}]")
    print(f"  |Bowley| < 0.1 (symmetric): {all(abs(w) < 0.1 for w in all_bow)}  range [{min(all_bow):+.2f}, {max(all_bow):+.2f}]")
    print(f"  exkurt > 0 (fat tails):   {all(e > 0 for e in all_exk)}  range [{min(all_exk):.1f}, {max(all_exk):.1f}]")
