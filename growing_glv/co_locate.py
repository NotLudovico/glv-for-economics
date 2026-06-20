"""Co-location search: can beta ~ +0.2 AND skew ~ 0 happen together?
Scan the floor lam at two competitive corners, measuring BOTH beta and skew on the survivor set
(persistent firms = empirical panel). Find an intermediate lam where the left tail is restored
(skew -> 0) before the floor pins beta negative.
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

S = 600
TMAX, T_BURN, DT = 45.0, 18.0, 0.5
SEED = 2024
LAMS = [0.0, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
CORNERS = [(2.0, 2.0), (3.0, 2.5)]


def measure(mu, sigma, lam):
    t, W, lnM, C_eff, ok = integrate(SEED, S, mu, sigma, tmax=TMAX, n_eval=3000, lam=lam, floor_mode="share")
    win = t > T_BURN
    surv = W[:, win].min(1) > 1e-6
    if surv.sum() < 50:
        return dict(surv=int(surv.sum()), beta=np.nan, skew=np.nan, exk=np.nan)
    tg = np.arange(T_BURN, TMAX + 1e-9, DT)
    lnS = regrid_logS(t, W[surv], tg)
    Sbar, vol, gl = msb_on_grid(lnS)
    beta, _ = binned_beta(Sbar, vol)
    return dict(surv=int(surv.sum()), beta=beta, skew=float(skew(gl)), exk=float(kurtosis(gl)))


if __name__ == "__main__":
    fig, axes = plt.subplots(1, len(CORNERS), figsize=(6 * len(CORNERS), 4.6))
    for ax, (mu, sig) in zip(np.atleast_1d(axes), CORNERS):
        print(f"\ncorner mu={mu} sigma={sig}  (beta & skew on survivors vs lam)")
        print(f"{'lam':>8} {'surv':>8} {'beta':>8} {'skew':>7} {'exkurt':>8}")
        lams, betas, skews = [], [], []
        for lam in LAMS:
            r = measure(mu, sig, lam)
            bstr = f"{r['beta']:>8.3f}" if np.isfinite(r["beta"]) else f"{'n/a':>8}"
            sstr = f"{r['skew']:>7.2f}" if np.isfinite(r["skew"]) else f"{'n/a':>7}"
            estr = f"{r['exk']:>8.1f}" if np.isfinite(r["exk"]) else f"{'n/a':>8}"
            print(f"{lam:>8.0e} {r['surv']:>5d}/{S} {bstr} {sstr} {estr}")
            lams.append(max(lam, 1.5e-5)); betas.append(r["beta"]); skews.append(r["skew"])
        ax.axhspan(0.15, 0.20, color="#2a9d8f", alpha=0.15)
        ax.axhline(0, color="#888", lw=0.8, ls=":")
        ax.semilogx(lams, betas, "o-", color="#264653", label=r"$\beta$ (want 0.15-0.20)")
        ax.semilogx(lams, skews, "s--", color="#c1121f", label="skew (want 0)")
        ax.set(xlabel=r"floor $\lambda$", title=fr"$\mu={mu}, \sigma={sig}$"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "co_locate.png"), dpi=120)
    print("\nsaved growing_glv/co_locate.png  (trifecta = beta in band AND skew~0 at the same lam)")
