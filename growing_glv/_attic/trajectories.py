"""Long-time firm-size trajectories S_i(t) = N w_i(t). Contrast a relaxing realization (shares freeze
to a fixed point) with a persistent one (shares keep churning), so the dynamics are visible directly.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from explore import integrate

TMAX, N_EVAL = 300.0, 6000
N_SHOW = 45                                   # firms to draw


def panel(ax, seed, S, mu, sigma, lam, title):
    t, W, lnM, _, _ = integrate(seed, S, mu, sigma, lam=lam, tmax=TMAX, n_eval=N_EVAL)
    Sz = S * W                                # relative sizes (mean 1)
    rng = np.random.default_rng(0)
    idx = rng.choice(S, N_SHOW, replace=False)
    for i in idx:
        ax.plot(t, np.log10(np.maximum(Sz[i], 1e-3)), lw=0.6, alpha=0.8)
    # late-window growth std to label persistence
    surv = np.where(W[:, (t >= 200) & (t <= 280)].min(1) > 1e-6)[0]
    tg = np.arange(200, 280, 0.5)
    g = np.diff(np.array([np.interp(tg, t, np.log(np.maximum(S * W[i], 1e-12))) for i in surv]), axis=1)
    gstd = g.ravel().std() if surv.size > 5 else np.nan
    ax.set(xlabel="t", ylabel=r"$\log_{10} S_i$  ($S_i = N w_i$)",
           title=f"{title}\nlate-window growth std = {gstd:.3f}")
    ax.axvspan(18, 75, color="#888", alpha=0.12)


if __name__ == "__main__":
    seeds = np.random.default_rng(123).integers(0, 2**31 - 1, size=4)   # seed[3] was persistent for mu=3,sig=2
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
    panel(ax[0], 2024, 600, 2.0, 2.0, 1e-3, "Headline regime (mu=2, sigma=2, lam=1e-3): shares FREEZE")
    panel(ax[1], int(seeds[3]), 500, 3.0, 2.0, 1e-3, "A persistent realization (mu=3, sigma=2, lam=1e-3): churn survives")
    ax[0].text(0.5, 0.02, "grey band = old short measurement window", transform=ax[0].transAxes,
               ha="center", fontsize=8, color="#555")
    plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "trajectories.png"), dpi=120)
    print("saved growing_glv/trajectories.png")
