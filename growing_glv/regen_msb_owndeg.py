"""Regenerate growing_stationary_msb.png under the OWN-DEGREE normalization.

The mean-degree normalization is gone from the thesis; this figure (size--volatility
relation + growth tent) must therefore be the own-degree one. Reuses the validated
own-degree model/MSB code in ../relative-glv (kind='powerlaw_owndeg') at the locked
operating point mu=1.76 (=chapter mu=-1.76), sigma=1.75, N=6400.

    uv run python growing_glv/regen_msb_owndeg.py [--smoke]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "relative-glv"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import laplace, norm
from joblib import Parallel, delayed
from relative_glv.model import coupling, integrate, growth_rate
from relative_glv.msb import size_volatility, tent_stats, rescale

THESIS = os.path.join(os.path.dirname(__file__), "..", "thesis")
SMOKE = "--smoke" in sys.argv
MU, SIGMA, LAM = 1.76, 1.75, 1e-3
N, SEEDS = (800, 3) if SMOKE else (6400, 8)
TMAX, N_EVAL = (120.0, 600) if SMOKE else (400.0, 800)
LATE, DT = ((95.0, 118.0) if SMOKE else (320.0, 390.0)), 0.5


def simulate(seed):
    a = coupling(N, MU, SIGMA, kind="powerlaw_owndeg", seed=seed)
    r = integrate(a, tmax=TMAX, n_eval=N_EVAL, lam=LAM, seed=seed,
                  method="RK45", rtol=1e-4, atol=1e-7)
    if not r["success"]:
        return None
    sv = size_volatility(r["W"], r["t"], window=LATE, dt=DT, n_bins=20)
    if sv["growth"].size == 0 or not np.isfinite(sv["beta"]):
        return None
    sv["g_eff"] = growth_rate(r["t"], r["lnM"], frac=0.3)
    return sv


if __name__ == "__main__":
    recs = [r for r in Parallel(n_jobs=6, backend="loky")(
        delayed(simulate)(s) for s in range(SEEDS)) if r is not None]
    assert recs, "no economies survived integration"

    # left panel: the economy whose beta is closest to the median, so the displayed exponent
    # is representative rather than a best-fit outlier
    beta_med0 = float(np.median([r["beta"] for r in recs]))
    best = min(recs, key=lambda r: abs(r["beta"] - beta_med0))
    # tent: rescale EACH economy by its own MAD before pooling, so cross-economy scale
    # differences do not inflate the kurtosis; report the median per-economy shape (the ch4
    # convention, consistent with the ablation table and the conditional figure)
    z = np.concatenate([rescale(r["growth"]) for r in recs])
    per_econ = [tent_stats(r["growth"]) for r in recs]
    ts = dict(bowley=float(np.median([t["bowley"] for t in per_econ])),
              exkurt=float(np.median([t["exkurt"] for t in per_econ])))
    beta_med = float(np.median([r["beta"] for r in recs]))
    print(f"[regen owndeg] economies={len(recs)} N={N}  beta(shown)={best['beta']:.2f} "
          f"r2={best['r2']:.2f}  beta(median)={beta_med:.2f}  "
          f"bowley(med)={ts['bowley']:+.2f} exk(med)={ts['exkurt']:.0f}", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    svS, svV = best["Sbar"], best["vol"]
    bx, by, pk = best["bin_S"], best["bin_vol"], int(best["pk"])
    live = (svV > 0) & np.isfinite(svV) & (svS > 0)
    ax[0].scatter(svS[live], svV[live], s=5, alpha=0.15, color="0.6")
    ax[0].plot(bx, by, "o-", color="#2a9d8f", ms=5, label="binned median")
    c = np.polyfit(np.log10(bx[pk:]), np.log10(by[pk:]), 1)
    xx = np.array([bx[pk], bx[-1]])
    ax[0].plot(xx, 10 ** np.polyval(c, np.log10(xx)), color="#e76f51", ls="--", lw=2,
               label=fr"decline fit $\beta={best['beta']:.2f}$ ($R^2={best['r2']:.2f}$)")
    ax[0].axvspan(bx[0], bx[pk], color="0.85", alpha=0.5)
    ax[0].text(bx[0], by.min(), " floor\n plateau", fontsize=8, va="bottom", color="0.4")
    ax[0].set(xscale="log", yscale="log", xlabel=r"firm size $\bar S$",
              ylabel=r"growth volatility $\sigma(\bar S)$",
              title="Size--volatility relation (one economy)")
    ax[0].legend(fontsize=9)

    xx = np.linspace(-8, 8, 400)
    ax[1].hist(z, bins=160, density=True, color="#2a9d8f", alpha=0.6, label="growth rates")
    ax[1].plot(xx, laplace.pdf(xx, scale=np.sqrt(2 / np.pi)), color="#e76f51", ls="--", lw=1.5, label="Laplace")
    ax[1].plot(xx, norm.pdf(xx), "k:", lw=1.2, label="Gaussian")
    ax[1].set_yscale("log"); ax[1].set_ylim(1e-4, None); ax[1].set_xlim(-20, 20)
    ax[1].set(xlabel=r"rescaled growth rate $\Delta\ln S$ ($\sqrt{\pi/2}\,$MAD)", ylabel="PDF",
              title=fr"Growth distribution (Bowley $={ts['bowley']:.2f}$, exc.\ kurt $={ts['exkurt']:.0f}$)")
    ax[1].legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(THESIS, "growing_stationary_msb.png")
    plt.savefig(out, dpi=140); plt.close()
    print(f"[regen owndeg] wrote {out}", flush=True)
