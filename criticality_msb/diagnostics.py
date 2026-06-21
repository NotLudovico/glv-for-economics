"""Visual diagnostics for a candidate regime: firm trajectories, the size–volatility fit, and the growth
PDF. Default f=0.9, λ=1e-3 (the principled β-match), floor=λ. Edit F/LAM to inspect others.
    uv run python -m criticality_msb.diagnostics
"""
import argparse
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

from criticality_msb import config as C
from criticality_msb.dataset import Run
from criticality_msb import observables as O

_ap = argparse.ArgumentParser()
_ap.add_argument("--f", default="0.9")
_ap.add_argument("--lam", default="0.001")
_args = _ap.parse_args()
F, LAM_S, LAM = _args.f, _args.lam, float(_args.lam)
runs = [Run(p) for p in glob.glob(f"{C.DATA_DIR}/run_*lam={LAM_S}_*f={F}.npz")]
r0 = runs[0]
print(f"f={F}, λ={LAM:g}, floor=λ, {len(runs)} seeds; trajectories from seed={r0.seed}, σ_c={r0.sigma_c:.3f}")

fig = plt.figure(figsize=(17, 5))

# --- A: raw abundance trajectories ---
axA = fig.add_subplot(1, 3, 1)
x = np.exp(r0.logx)                                              # raw abundance x_i(t)
idx = np.random.default_rng(0).choice(x.shape[0], 80, replace=False)
axA.semilogy(r0.t_snap, x[idx].T, lw=0.5, alpha=0.5)
axA.axhline(LAM, color="r", ls="--", lw=1.2, label=f"immigration floor λ={LAM:g}")
axA.axhline(1.0, color="k", ls=":", lw=1, label="carrying capacity ~1")
axA.set(xlabel="time", ylabel=r"abundance $x_i$ (log)", ylim=(LAM / 3, 5),
        title=f"trajectories (80 of {x.shape[0]} firms)")
axA.legend(fontsize=8)

# --- B: size-volatility fit ---
axB = fig.add_subplot(1, 3, 2)
Sb, Vl = [], []
for r in runs:
    s, v = O.delisted_size_volatility(r, dt=1.0, floor=LAM, min_growths=2)
    Sb.append(s); Vl.append(v)
Sb = np.concatenate(Sb); Vl = np.concatenate(Vl)
keep = (Sb > 0) & (Vl > 0) & np.isfinite(Vl); Sb, Vl = Sb[keep], Vl[keep]
o = np.argsort(Sb); xs, ys = Sb[o], Vl[o]; parts = np.array_split(np.arange(len(xs)), 18)
bx = np.array([xs[p].mean() for p in parts]); by = np.array([np.median(ys[p]) for p in parts])
m = (bx > 0) & (by > 0)
coef = np.polyfit(np.log10(bx[m]), np.log10(by[m]), 1); beta = -coef[0]
mid = len(bx[m]) // 2
axB.loglog(Sb, Vl, ".", ms=1, alpha=0.08, color="gray")
axB.loglog(bx[m], by[m], "o", ms=6, color="#2a9d8f", label="binned median")
axB.loglog(bx[m], 10 ** np.polyval(coef, np.log10(bx[m])), "-", color="#264653", lw=2, label=fr"fit  β={beta:.3f}")
axB.loglog(bx[m], by[m][mid] * (bx[m] / bx[m][mid]) ** (-0.204), "r--", lw=1.4, label="empirical β=0.204")
axB.set(xlabel=r"avg size $\bar S_i$", ylabel=r"volatility $\sigma_i$", title="size–volatility fit")
axB.legend(fontsize=8)

# --- C: growth PDF ---
axC = fig.add_subplot(1, 3, 3)
g = np.concatenate([O.delisted_growth(r, dt=1.0, floor=LAM, min_growths=2) for r in runs])
g = g[np.isfinite(g)]
gd = g[np.abs(g) < 10]                                            # clip extreme outliers for display only
e = np.linspace(-8, 8, 100); cc = 0.5 * (e[:-1] + e[1:])
d, _ = np.histogram(gd, bins=e, density=True)
axC.semilogy(cc, np.where(d > 0, d, np.nan), "o", ms=3, color="#1d3557", label="growth $g$")
xx = np.linspace(-8, 8, 300); sd = gd.std(); med = np.median(gd); bl = np.mean(np.abs(gd - med))
axC.semilogy(xx, np.exp(-0.5 * ((xx - gd.mean()) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), color="#457b9d", label="Gaussian")
axC.semilogy(xx, np.exp(-np.abs(xx - med) / bl) / (2 * bl), "r--", label="Laplace (symmetric)")
axC.set(xlabel=r"$g=\Delta\ln S$", ylabel="density",
        title=f"growth PDF  (skew {skew(g):+.1f}, exkurt {kurtosis(g):.0f})")
axC.legend(fontsize=8)

fig.suptitle(f"candidate regime: f={F}, λ={LAM:g}, floor=λ", y=1.02)
plt.tight_layout()
out = f"{C.FIG_DIR}/diagnostics_f{F}_lam{LAM_S}.png"
plt.savefig(out, dpi=120)
print(f"β={beta:.3f} (empirical 0.204)   growth: skew={skew(g):+.2f}, exkurt={kurtosis(g):.1f}, robtail={O.robtail(g):.0f}")
print("saved", out)
