"""Robustness of the candidate match: near-critical f=0.9, immigration, delisting floor = λ.
Three tests: β vs Δt (sampling), β vs sample cut ≥2/≥20 growths (MSB's own robustness check — the data is
stable, our model was NOT at the wrong floor), and the growth distribution (the 2nd MSB fact).
    uv run python -m criticality_msb.verify_candidate
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

F = "0.9"
LAMS = [("0.001", 1e-3), ("0.01", 1e-2), ("0.1", 1e-1)]
DTS = [0.25, 0.5, 1.0, 2.0, 4.0]


def load(lam):
    return [Run(p) for p in glob.glob(f"{C.DATA_DIR}/run_*lam={lam}_*f={F}.npz")]


def beta(rs, floor, dt, ming):
    Sb, Vl = [], []
    for r in rs:
        s, v = O.delisted_size_volatility(r, dt=dt, floor=floor, min_growths=ming)
        Sb.append(s); Vl.append(v)
    return O.beta_fit(np.concatenate(Sb), np.concatenate(Vl))


print(f"CANDIDATE ROBUSTNESS — f={F}, floor=λ.  Empirical β=0.204±0.01\n")
print("β vs Δt  (min_growths=2):")
fig, (axB, axP) = plt.subplots(1, 2, figsize=(13, 5))
for ls, lv in LAMS:
    rs = load(ls)
    bs = [beta(rs, lv, dt, 2) for dt in DTS]
    print(f"  λ={ls}:  " + "  ".join(f"Δt={dt}:{b:.3f}" for dt, b in zip(DTS, bs)))
    axB.plot(DTS, bs, "o-", label=f"λ={ls}")

print("\nβ vs sample cut  (Δt=1) — MSB check ≥2 vs ≥20 (real data: stable):")
for ls, lv in LAMS:
    rs = load(ls)
    print(f"  λ={ls}:  ≥2 growths: {beta(rs, lv, 1.0, 2):.3f}   ≥20 growths: {beta(rs, lv, 1.0, 20):.3f}")

print("\ngrowth distribution of listed firms  (Δt=1, floor=λ):")
e = np.linspace(-8, 8, 80); cc = 0.5 * (e[:-1] + e[1:])
for ls, lv in LAMS:
    rs = load(ls)
    g = np.concatenate([O.delisted_growth(r, dt=1.0, floor=lv, min_growths=2) for r in rs])
    g = g[np.isfinite(g)]
    print(f"  λ={ls}:  skew={skew(g):+.2f}  exkurt={kurtosis(g):.1f}  robtail={O.robtail(g):.0f}  (Laplace robtail≈10)")
    d, _ = np.histogram(g, bins=e, density=True)
    axP.semilogy(cc, np.where(d > 0, d, np.nan), "-", label=f"λ={ls}")

axB.axhspan(0.15, 0.20, color="gray", alpha=0.25, label="empirical MSB")
axB.set(xscale="log", xlabel=r"$\Delta t$", ylabel=r"$\beta$ (faithful)", title=f"β vs Δt  (f={F}, floor=λ)"); axB.legend()
g0 = np.concatenate([O.delisted_growth(r, dt=1.0, floor=1e-2, min_growths=2) for r in load("0.01")]); g0 = g0[np.isfinite(g0)]
xx = np.linspace(-8, 8, 200); sd = g0.std()
axP.semilogy(xx, np.exp(-0.5 * (xx / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), "k--", lw=1, label="Gaussian")
axP.set(xlabel=r"$g=\Delta\ln S$", ylabel="density", title="growth PDF (listed firms)"); axP.legend(fontsize=8)
plt.tight_layout()
out = f"{C.FIG_DIR}/candidate_robustness.png"
plt.savefig(out, dpi=120)
print("\nsaved", out)
