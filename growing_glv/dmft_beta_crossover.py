"""Why the MSB exponent beta drifts with horizon: it is a mean-reversion crossover.

The two-time DMFT single-site process produces a size-volatility relation purely from
self-limitation (-y) + noise, with NO cross-firm correlation. But it is NOT a clean
power law and beta is NOT a fixed exponent:

  Left   the vol-size relation is CONVEX (log-log slope steepens with size), so any
         single beta is a chord of a curved relation, not a scale-free exponent.
  Centre beta collapses as a universal function of h/tau, where tau is the firm's
         size-memory time: ~0.55 instantaneous, crossing the empirical 0.2 near
         h ~ 1.3 tau, -> 0 as the start size is forgotten. tau differs ~3x across the
         sigma shown, yet the curves coincide.
  Right  tau diverges as sigma -> sigma_c = sqrt2 (critical slowing of the relaxation).

So beta is set by horizon / memory-time, which is why it has been so slippery: short
horizon -> steep (~0.5-0.7), long horizon / large N -> washed out (~0.1). One picture.

Run: uv run python growing_glv/dmft_beta_crossover.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dmft_solver import sigma_c
from dmft_twotime import solve_twotime, msb_beta, size_memory_time

sc = sigma_c(0.0)
MU = -0.5
# long window so we can reach h ~ 1.5 tau in the fluctuating phase
KW = dict(Nt=480, dt=0.2, late=0.45, n=5000, seed=0)

sigmas = [1.6, 2.0, 2.5]
col = {1.6: "#e07a18", 2.0: "#c1121f", 2.5: "#7b2cbf"}
R = {s: solve_twotime(MU, s, **KW) for s in sigmas}
TAU = {s: size_memory_time(R[s])[0] for s in sigmas}
print("tau:", {s: round(TAU[s], 1) for s in sigmas})

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

# ---- Left: the vol-size relation is convex, not a power law ----
for s in sigmas:
    b = msb_beta(R[s], h_units=0.3 * TAU[s])           # short horizon: curvature is sharpest
    lx, ly = b["lx"] - b["lx"].mean(), b["ly"] - b["ly"].mean()
    ax[0].plot(lx, ly, "o-", color=col[s], ms=5, label=rf"$\sigma={s}$")
xx = np.array([-2.2, 2.2])
ax[0].plot(xx, -0.2 * xx, "k--", lw=1.2, label=r"power law, slope $-0.2$")
ax[0].set(xlabel="log size (centred)", ylabel="log std growth (centred)",
          title="Slope steepens with size (no single $\\beta$)")
ax[0].legend(fontsize=8)

# ---- Centre: beta collapses on h/tau (the mean-reversion crossover) ----
hf = np.linspace(0.15, 1.55, 13)
for s in sigmas:
    bs = [msb_beta(R[s], h_units=f * TAU[s])["beta"] for f in hf]
    bs = np.array(bs); m = np.isfinite(bs)
    ax[1].plot(hf[m], bs[m], "o-", color=col[s], ms=5,
               label=rf"$\sigma={s}$  ($\tau={TAU[s]:.0f}$)")
ax[1].axhspan(0.19, 0.21, color="0.82", zorder=0)
ax[1].text(0.18, 0.215, r"empirical $\beta\approx0.2$", fontsize=8, color="0.4")
ax[1].axvline(1.0, color="grey", ls=":", lw=1)
ax[1].text(1.02, 0.5, r"$h\approx\tau$", fontsize=8, color="0.4")
ax[1].set(xlabel=r"horizon / memory time   $h/\tau$", ylabel=r"MSB exponent $\beta$",
          title=r"$\beta$ collapses on $h/\tau$", ylim=(0, 0.62))
ax[1].legend(fontsize=8)

# ---- Right: tau diverges at sigma_c (critical slowing) ----
sig_c = [1.7, 1.85, 2.0, 2.2, 2.5]
tau_c = []
for s in sig_c:
    r = R[s] if s in R else solve_twotime(MU, s, **KW)
    tau_c.append(size_memory_time(r)[0])
ax[2].plot(sig_c, tau_c, "s-", color="#1f6f8b", ms=7)
ax[2].axvline(sc, color="#c1121f", ls="--", lw=1.5)
ax[2].text(sc + 0.02, max(tau_c) * 0.55, r"$\sigma_c=\sqrt{2}$", color="#c1121f", fontsize=9)
ax[2].annotate(r"$\tau\to\infty$ at $\sigma_c$", (1.7, tau_c[0]), (1.95, tau_c[0] * 0.9),
               fontsize=8, color="0.3", arrowprops=dict(arrowstyle="->", color="0.5"))
ax[2].set(xlabel=r"disorder $\sigma$", ylabel=r"size-memory time $\tau$",
          title="Critical slowing of the relaxation")

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "dmft_beta_crossover.png")
plt.savefig(out, dpi=120)
print(f"saved {out}")
print("beta at h/tau=1.0 (empirical-band crossing):",
      {s: round(msb_beta(R[s], h_units=1.0 * TAU[s])["beta"], 3) for s in sigmas})
