"""μ-scan analysis: β_persist and the persistent-firm tail vs μ at fixed f, under the locked empirical
protocol (listed firms, Δt=1, MAD volatility). Question: does any μ bring β into 0.15–0.20 *and* keep
a fat, near-symmetric tail?  uv run python -m criticality_msb.analyze_mu
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from criticality_msb import config as C
from criticality_msb.dataset import load_runs
from criticality_msb import observables as O

DT, ALIVE = 1.0, 1e-4
F_SHOW = C.SWEEP["f_values"]

runs = load_runs(C.DATA_DIR)
mus = sorted({round(r.mu, 3) for r in runs})
print(f"{len(runs)} runs | μ present: {mus} | f shown: {F_SHOW}")

fig, (axB, axT) = plt.subplots(1, 2, figsize=(13, 5))
for f, col in zip(F_SHOW, ["#2a9d8f", "#c1121f"]):
    xs, bm, bs, sk, rt = [], [], [], [], []
    print(f"\nf={f}:")
    for mu in mus:
        rf = [r for r in runs if abs(r.f - f) < 1e-9 and abs(r.mu - mu) < 1e-6 and not r.diverged]
        if not rf:
            continue
        bp = np.array([O.beta_fit(*O.size_volatility(r, dt=DT), mask=O.persistent_mask(r, alive=ALIVE)) for r in rf])
        bp = bp[np.isfinite(bp)]
        gp = np.concatenate([O.growth(r, dt=DT)[1][O.persistent_mask(r, alive=ALIVE)].ravel() for r in rf])
        m = O.tail_metrics(gp[np.isfinite(gp)])
        xs.append(mu); bm.append(bp.mean()); bs.append(bp.std()); sk.append(m["skew"]); rt.append(m["robtail"])
        print(f"  μ={mu:>4}: β_persist={bp.mean():.3f}±{bp.std():.2f} (n={bp.size})  skew={m['skew']:+.2f}  robtail={m['robtail']:.0f}")
    axB.errorbar(xs, bm, yerr=bs, fmt="o-", color=col, capsize=3, label=f"f={f}")
    axT.plot(xs, rt, "o-", color=col, label=f"f={f}")

axB.axhspan(0.15, 0.20, color="gray", alpha=0.25, label="empirical MSB")
axB.set(xlabel=r"$\mu$ (mean competition)", ylabel=r"$\beta_\mathrm{persist}$", title="β vs μ — does any μ enter the band?")
axB.set_xscale("log"); axB.legend()
axT.axhline(10, color="k", ls=":", lw=0.8, label="Laplace ≈ 10")
axT.set(xlabel=r"$\mu$", ylabel="robust tail (99.9pct/MAD)", title="persistent-firm tail vs μ")
axT.set_xscale("log"); axT.legend()
plt.tight_layout()
os.makedirs(C.FIG_DIR, exist_ok=True)
out = os.path.join(C.FIG_DIR, "mu_scan.png")
plt.savefig(out, dpi=120)
print("\nsaved", out)
