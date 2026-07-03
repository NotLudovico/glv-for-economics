"""Predicted phase structure of the relative GLV from the DMFT solver (no simulation).

Panel A: the (mu, sigma) plane. The relaxed<->fluctuating boundary is the horizontal line
sigma_c=sqrt(2) (mu-independent, because mu is a uniform fitness shift that cancels in the
replicator). The growth contour g*=0 is mu = g0(sigma), valid in the relaxed phase; it
splits relaxed-growing (left) from relaxed-shrinking (right).
Panel B: the mu-independent order parameters g0(sigma) and phi(sigma); above sigma_c the
fixed point is unstable so they are drawn dashed (the true rate needs the two-time DMFT).

Run: uv run python growing_glv/dmft_phase.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dmft_solver import solve_fixed_point, sigma_c

sc = sigma_c(0.0)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))

# ---- Panel A: (mu, sigma) phase diagram ----
mu_lo, mu_hi, sig_max = -2.0, 0.5, 2.5
sig_relaxed = np.linspace(0.02, sc, 200)
# doc convention (mu<0 competitive): g*=g0+mu, so the growth boundary g*=0 is at mu=-g0(sigma).
g0_curve = np.array([solve_fixed_point(0.0, float(s))["g0"] for s in sig_relaxed])

ax[0].fill_between([mu_lo, mu_hi], sc, sig_max, color="0.88", zorder=0)           # fluctuating band
ax[0].plot(-g0_curve, sig_relaxed, color="#1f6f8b", lw=2, zorder=3,
           label=r"$g^*=0$  ($\mu=-g_0(\sigma)$)")
ax[0].axhline(sc, color="#c1121f", ls="--", lw=1.8, zorder=2,
              label=r"$\sigma_c=\sqrt{2}$ (chaos onset)")
ax[0].plot(-1.76, 1.75, "k*", ms=13, zorder=4)                                    # operating point
ax[0].annotate(r"operating point $\mu=-1.76$", (-1.76, 1.75), (-1.95, 2.10),
               fontsize=8, color="k")
ax[0].text(-1.4, 0.5, "relaxed,\nshrinking\n$g^*<0$", fontsize=9, color="#3b5b8c", ha="center")
ax[0].text(0.22, 0.95, "relaxed,\ngrowing\n$g^*>0$", fontsize=9, color="#2a9d8f", ha="center")
ax[0].text(-0.8, 1.85, "fluctuating phase\n(MSB tent; rate via\ntwo-time DMFT)",
           fontsize=9, color="0.3", ha="center")
ax[0].set(xlabel=r"mean interaction $\mu$ (competitive $\mu<0$)", ylabel=r"disorder $\sigma$",
          xlim=(mu_lo, mu_hi), ylim=(0, sig_max), title="Predicted phase structure")
ax[0].legend(loc="lower right", fontsize=8, framealpha=0.95)

# ---- Panel B: mu-independent order parameters vs sigma ----
sig = np.linspace(0.05, 2.4, 240)
sol = [solve_fixed_point(0.0, float(s)) for s in sig]
g0 = np.array([d["g0"] for d in sol]); phi = np.array([d["phi"] for d in sol])
rel = sig < sc

ax[1].plot(sig[rel], g0[rel], color="#1f6f8b", lw=2, label=r"$g_0(\sigma)$ (growth)")
ax[1].plot(sig[~rel], g0[~rel], color="#1f6f8b", lw=2, ls=":")
ax[1].plot(sig[rel], phi[rel], color="#e07a18", lw=2, label=r"$\varphi(\sigma)$ (survival)")
ax[1].plot(sig[~rel], phi[~rel], color="#e07a18", lw=2, ls=":")
ax[1].axvline(sc, color="#c1121f", ls="--", lw=1.6)
ax[1].text(sc + 0.04, 0.05, r"$\sigma_c=\sqrt{2}$", color="#c1121f", fontsize=9)
ax[1].text(1.85, 0.9, "unstable FP\n(dotted)", fontsize=8, color="0.4")
ax[1].set(xlabel=r"disorder $\sigma$", ylabel="order parameter",
          xlim=(0, 2.4), ylim=(0, 1.05), title=r"Relaxed-phase observables (any $\mu$)")
ax[1].legend(loc="center right", fontsize=8)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "dmft_phase.png")
plt.savefig(out, dpi=120)
print(f"saved {out}  (sigma_c={sc:.4f}, g0 at sigma_c={solve_fixed_point(0.0, sc)['g0']:.3f})")
