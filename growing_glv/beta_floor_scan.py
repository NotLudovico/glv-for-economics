"""Can beta > 0 co-exist with the symmetric tent? Sweep the immigration floor strength lam
for BOTH floor mechanisms (persistent share floor vs decaying Bouchaud-Mezard additive), at the
growing+chaotic regime (mu=1, sigma=2), measuring beta, tent skew/kurtosis, and coexistence together.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from explore import integrate
from measure_msb import regrid_logS, binned_beta, msb_on_grid

S, MU, SIGMA = 600, 1.0, 2.0
TMAX, T_BURN, DT = 45.0, 18.0, 0.5
SEED = 2024
LAMS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]


def measure(lam, floor_mode):
    t, W, lnM, C_eff, ok = integrate(SEED, S, MU, SIGMA, tmax=TMAX, n_eval=3000, lam=lam, floor_mode=floor_mode)
    g_eff = float(np.polyfit(t[t > T_BURN], lnM[t > T_BURN], 1)[0])
    surv = int((W[:, t > T_BURN].min(1) > 1e-6).sum())
    tg = np.arange(T_BURN, TMAX + 1e-9, DT)
    lnS = regrid_logS(t, W, tg)
    Sbar, vol, gl = msb_on_grid(lnS)
    beta, _ = binned_beta(Sbar, vol)
    return dict(lam=lam, mode=floor_mode, ok=ok, g_eff=g_eff, surv=surv,
                beta=beta, skew=float(skew(gl)), exk=float(kurtosis(gl)))


if __name__ == "__main__":
    results = {}
    for mode in ["share", "bm"]:
        print(f"\nfloor_mode = {mode}   (regime S={S} mu={MU} sigma={SIGMA}, window [{T_BURN},{TMAX}] Dt={DT})")
        print(f"{'lam':>8} {'beta':>8} {'skew':>7} {'exkurt':>8} {'coexist':>9} {'g_eff':>7}")
        rows = []
        for lam in LAMS:
            r = measure(lam, mode)
            rows.append(r)
            print(f"{lam:>8.0e} {r['beta']:>8.3f} {r['skew']:>7.2f} {r['exk']:>8.1f} "
                  f"{r['surv']:>5d}/{S} {r['g_eff']:>7.2f}")
        results[mode] = rows

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for mode, mk in [("share", "o-"), ("bm", "s--")]:
        rs = results[mode]
        lams = [max(r["lam"], 3e-5) for r in rs]               # 0 -> small for log axis
        ax[0].semilogx(lams, [r["beta"] for r in rs], mk, label=mode)
        ax[1].semilogx(lams, [r["surv"] / S for r in rs], mk, label=mode)
    ax[0].axhspan(0.15, 0.20, color="#2a9d8f", alpha=0.15, label="empirical beta")
    ax[0].axhline(0, color="#888", lw=0.8, ls=":")
    ax[0].set(xlabel=r"immigration $\lambda$", ylabel=r"size-volatility $\beta$",
              title="beta vs floor strength"); ax[0].legend()
    ax[1].set(xlabel=r"immigration $\lambda$", ylabel="coexisting fraction",
              title="coexistence vs floor strength"); ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "beta_floor_scan.png"), dpi=120)
    print("\nsaved growing_glv/beta_floor_scan.png")
    print("Looking for: a lam where beta>0 (toward +0.2) AND coexistence stays high AND skew stays ~0.")
