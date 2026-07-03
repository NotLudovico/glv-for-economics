"""Figure for the two-time DMFT (fluctuating phase, gamma=0).

Three panels:
  Left   the share autocorrelation C(tau) freezes below sigma_c (relaxed) and
         decorrelates above it (fluctuating) -- the object the static theory misses.
  Centre survival phi: the two-time DMFT tracks the direct sim, while the static
         fixed point overshoots above sigma_c (it is the unstable FP there).
  Right  growth rate g: the two-time DMFT tracks the sim and the static FP overshoots
         above sigma_c, the same story as survival.

(The MSB exponent beta is a horizon-dependent crossover, not a single number, so it gets
its own figure -- dmft_beta_crossover.py -- rather than a misleading beta-vs-sigma panel
at one fixed horizon.) Run: uv run python growing_glv/dmft_twotime_fig.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dmft_solver import solve_fixed_point, sigma_c
from dmft_twotime import solve_twotime, msb_beta

sc = sigma_c(0.0)
MU = -0.5
NPZ = os.path.join(os.path.dirname(__file__), "dmft_validation.npz")
d = np.load(NPZ); sim = {float(s): r for s, r in zip(d["sigmas"], d["sim"])}

sig_2t = [0.7, 1.0, 1.2, 1.6, 2.0, 2.5]
sol = {s: solve_twotime(MU, s, seed=0) for s in sig_2t}

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

# ---- Left: autocorrelation freeze vs decorrelation ----
for s, col in [(1.0, "#1f6f8b"), (1.6, "#e07a18"), (2.0, "#c1121f"), (2.5, "#7b2cbf")]:
    r = sol[s]; tau = np.arange(len(r["Ctau"])) * r["dt"]
    conn = (r["Ctau"] - 1.0) / (r["Ctau"][0] - 1.0)
    lab = rf"$\sigma={s}$" + (" (relaxed)" if s < sc else "")
    ax[0].plot(tau, conn, color=col, lw=2, label=lab)
ax[0].axhline(0, color="grey", lw=0.5)
ax[0].set(xlabel=r"lag $\tau$", ylabel=r"$[C(\tau)-1]/[C(0)-1]$",
          title="Share autocorrelation", ylim=(-0.05, 1.05))
ax[0].legend(fontsize=8)

# ---- Centre: survival, two-time vs sim vs static FP ----
sg = np.linspace(0.4, 2.5, 120)
phi_stat = np.array([solve_fixed_point(-MU, float(s))["phi"] for s in sg])
ax[1].plot(sg[sg < sc], phi_stat[sg < sc], color="#c1121f", lw=2, label=r"static FP $\varphi$")
ax[1].plot(sg[sg >= sc], phi_stat[sg >= sc], color="#c1121f", lw=2, ls=":",
           label=r"static FP (unstable)")
ax[1].plot(sig_2t, [sol[s]["surv"] for s in sig_2t], "s", color="#1f6f8b", ms=8,
           label="two-time DMFT")
ssig = [s for s in sig_2t if s in sim]
ax[1].plot(ssig, [sim[s][1] for s in ssig], "o", color="#2a9d8f", ms=7, label="direct sim")
ax[1].axvline(sc, color="k", ls=":", lw=1)
ax[1].text(sc + 0.03, 0.92, r"$\sigma_c=\sqrt{2}$", fontsize=9)
ax[1].set(xlabel=r"disorder $\sigma$", ylabel=r"surviving fraction $\varphi$",
          title="Survival: two-time tracks the sim", ylim=(0, 1.02))
ax[1].legend(fontsize=8)

# ---- Right: growth rate g, two-time vs sim vs static FP ----
g_stat = np.array([solve_fixed_point(-MU, float(s))["gstar"] for s in sg])
ax[2].plot(sg[sg < sc], g_stat[sg < sc], color="#c1121f", lw=2, label=r"static FP $g^*$")
ax[2].plot(sg[sg >= sc], g_stat[sg >= sc], color="#c1121f", lw=2, ls=":",
           label="static FP (unstable)")
ax[2].plot(sig_2t, [sol[s]["g_eff"] for s in sig_2t], "s", color="#1f6f8b", ms=8,
           label="two-time DMFT")
ax[2].plot(ssig, [sim[s][0] for s in ssig], "o", color="#2a9d8f", ms=7, label="direct sim")
ax[2].axvline(sc, color="k", ls=":", lw=1)
ax[2].axhline(0, color="grey", lw=0.5)
ax[2].text(sc + 0.03, ax[2].get_ylim()[1] * 0.1, r"$\sigma_c=\sqrt{2}$", fontsize=9)
ax[2].set(xlabel=r"disorder $\sigma$", ylabel=r"growth rate $g$",
          title="Growth: two-time tracks the sim")
ax[2].legend(fontsize=8, loc="upper left")

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "dmft_twotime.png")
plt.savefig(out, dpi=120)
print(f"saved {out}")
