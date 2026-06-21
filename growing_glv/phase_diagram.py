"""Phase diagram of the relative GLV in the (mu, sigma) plane (mu = mean COMPETITION, sim convention).
Two control parameters set the structure:
  - disorder sigma: shape relaxes to a fixed point (frozen) below sigma_c ~ sqrt(2), chaotic above;
  - competition mu: the aggregate growth rate g_eff = d ln M/dt is negative (economy SHRINKS) for strong
    competition, small positive (STEADY growth) near the boundary, and large (FAST growth) for weak
    competition / cooperation.
We measure g_eff and the persistent-fluctuation amplitude over a grid and classify each cell, so the
region where the firm-growth facts live (chaotic AND growing) is located honestly rather than fine-tuned.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from explore import build_alpha

S, LAM, TMAX = 2000, 1e-3, 200.0
MUS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
SIGS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
N_SEEDS = 2


def cell(seed, mu, sigma):
    alpha, _ = build_alpha(seed, S, mu, sigma)
    rng = np.random.default_rng(seed + 1); w0 = rng.uniform(0.5, 1.5, S); w0 /= w0.sum()
    def rhs(t, st):
        w = np.clip(st[:S], 0.0, None); s = w.sum()
        if s > 0: w = w / s
        f = 1.0 - S * w - S * (alpha @ w); fb = float(w @ f)
        return np.concatenate([w * (f - fb) + LAM * (1.0 / S - w), [fb]])
    r = solve_ivp(rhs, (0, TMAX), np.concatenate([w0, [0.0]]), method="RK45",
                  t_eval=np.linspace(0, TMAX, 1200), rtol=1e-6, atol=1e-8)
    W = np.clip(r.y[:S], 0.0, None); W = W / W.sum(0, keepdims=True); t = r.t; lnM = r.y[S]
    g_eff = float(np.polyfit(t[t > 100], lnM[t > 100], 1)[0])
    surv = np.where(W[:, (t >= 120) & (t <= 190)].min(1) > 1e-6)[0]
    if surv.size < 30:
        return g_eff, 0.0
    tg = np.arange(120, 190 + 1e-9, 0.5)
    lnS = np.array([np.interp(tg, t, np.log(np.maximum(S * W[i], 1e-12))) for i in surv])
    return g_eff, float(np.diff(lnS, axis=1).ravel().std())


if __name__ == "__main__":
    seeds = np.random.default_rng(7).integers(0, 2**31 - 1, size=N_SEEDS)
    G = np.zeros((len(SIGS), len(MUS))); F = np.zeros_like(G)
    print(f"phase diagram, N={S}, lam={LAM:g}, {N_SEEDS} seeds. g_eff (growth) | gstd (chaos)")
    for j, sig in enumerate(SIGS):
        row = []
        for i, mu in enumerate(MUS):
            gg, ff = np.mean([cell(int(s), mu, sig) for s in seeds], axis=0)
            G[j, i] = gg; F[j, i] = ff
            row.append(f"({gg:+.1f},{ff:.2f})")
        print(f"sig={sig}: " + " ".join(row), flush=True)
    np.savez(os.path.join(os.path.dirname(__file__), "phase_diagram.npz"), G=G, F=F, MUS=MUS, SIGS=SIGS)

    # classify: frozen if gstd<0.02; among growing g_eff>0.05; shrinking g_eff<-0.05; fast g_eff>5
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for j, sig in enumerate(SIGS):
        for i, mu in enumerate(MUS):
            g, f = G[j, i], F[j, i]
            chaotic = f > 0.02
            if g < -0.05:
                c, lab = "#3b5b8c", "shrinking"
            elif g > 5:
                c, lab = "#7b2d8e", "fast growth"
            else:
                c, lab = "#2a9d8f", "steady growth"
            ax.scatter(mu, sig, s=260, c=c, marker=("o" if chaotic else "s"),
                       edgecolors="k", linewidths=0.5, zorder=3)
    ax.axhline(np.sqrt(2), color="#c1121f", ls="--", lw=1.4, zorder=1)
    ax.text(0.05, np.sqrt(2) + 0.05, r"$\sigma_c=\sqrt{2}$ (chaos onset)", color="#c1121f", fontsize=9)
    from matplotlib.lines import Line2D
    leg = [Line2D([], [], marker="o", color="w", markerfacecolor="#888", markeredgecolor="k", label="chaotic (circle)"),
           Line2D([], [], marker="s", color="w", markerfacecolor="#888", markeredgecolor="k", label="frozen (square)"),
           Line2D([], [], marker="o", color="w", markerfacecolor="#3b5b8c", label="shrinking"),
           Line2D([], [], marker="o", color="w", markerfacecolor="#2a9d8f", label="steady growth"),
           Line2D([], [], marker="o", color="w", markerfacecolor="#7b2d8e", label="fast growth")]
    ax.legend(handles=leg, loc="upper left", fontsize=8, framealpha=0.95)
    ax.set(xlabel=r"mean competition $\mu$", ylabel=r"disorder $\sigma$",
           title="Phase diagram of the relative GLV (marker: chaotic/frozen; colour: growth)")
    plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "phase_diagram.png"), dpi=120)
    print("\nsaved phase_diagram.png")
