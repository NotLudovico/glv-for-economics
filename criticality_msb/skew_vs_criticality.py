"""Test the hypothesis (from the thesis regime): moving CLOSER to criticality (f=0.95 vs 0.9) symmetrizes
the growth distribution while keeping β≈0.20. Immigration, floor=λ, 50 seeds.
    uv run python -m criticality_msb.skew_vs_criticality
"""
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

from criticality_msb import config as C
from criticality_msb.dataset import Run
from criticality_msb import observables as O

LAMS = [("0.001", 1e-3), ("0.01", 1e-2), ("0.1", 1e-1)]
FS = ["0.9", "0.95"]


def load(lam, f):
    return [Run(p) for p in glob.glob(f"{C.DATA_DIR}/run_*lam={lam}_*f={f}.npz")]


def beta(rs, floor):
    Sb, Vl = [], []
    for r in rs:
        s, v = O.delisted_size_volatility(r, dt=1.0, floor=floor, min_growths=2)
        Sb.append(s); Vl.append(v)
    return O.beta_fit(np.concatenate(Sb), np.concatenate(Vl))


def gpool(rs, floor):
    g = np.concatenate([O.delisted_growth(r, dt=1.0, floor=floor, min_growths=2) for r in rs])
    return g[np.isfinite(g)]


print("Closer to criticality (f=0.95) — does it symmetrize while keeping β? (floor=λ, empirical β=0.204)\n")
print(f"{'λ':>8}{'β(0.9)':>9}{'β(0.95)':>9}{'skew(0.9)':>11}{'skew(0.95)':>12}{'robtail(0.95)':>14}")
fig, (axS, axP) = plt.subplots(1, 2, figsize=(13, 5))
for ls, lv in LAMS:
    rs = {f: load(ls, f) for f in FS}
    if not rs["0.95"]:
        continue
    b = {f: beta(rs[f], lv) for f in FS}
    g = {f: gpool(rs[f], lv) for f in FS}
    print(f"{ls:>8}{b['0.9']:>9.3f}{b['0.95']:>9.3f}{skew(g['0.9']):>11.2f}{skew(g['0.95']):>12.2f}{O.robtail(g['0.95']):>14.0f}")
    axS.plot([0.9, 0.95], [skew(g["0.9"]), skew(g["0.95"])], "o-", label=f"λ={ls}")

axS.axhline(0, color="k", lw=0.6, label="symmetric")
axS.set(xlabel=r"$f=\sigma/\sigma_c$", ylabel="growth-rate skew", title="skew vs criticality"); axS.legend(fontsize=8)

e = np.linspace(-8, 8, 90); cc = 0.5 * (e[:-1] + e[1:])
for f, col in zip(FS, ["#457b9d", "#c1121f"]):
    g = gpool(load("0.001", f), 1e-3); g = g[np.abs(g) < 10]
    d, _ = np.histogram(g, bins=e, density=True)
    axP.semilogy(cc, np.where(d > 0, d, np.nan), "-o", ms=2, color=col, label=f"f={f} (skew {skew(g):+.1f})")
axP.set(xlabel=r"$g=\Delta\ln S$", ylabel="density", title="growth PDF: f=0.9 vs 0.95 (λ=1e-3)"); axP.legend(fontsize=8)
plt.tight_layout()
out = f"{C.FIG_DIR}/skew_vs_criticality.png"
plt.savefig(out, dpi=120)
print("\nsaved", out)
