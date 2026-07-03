"""Targeted sim check: does the degree-resolved HDMFT (hdmft_solver.py) match the
relative-GLV simulation in the RELAXED phase?  Isolates 'phase vs measurement' as
the cause of the theory(stable beta)/sim(crashing beta) gap.

Lever 1 predicts a cross-sectional (size, dispersion)-by-degree relation with a
stable beta~0.25 at strong competition, sigma<sigma_c. Here we run build_alpha +
integrate at exactly that point (mu=2, sigma=0.8, lam=0 -> pure fixed point), bin
the relaxed relative sizes S_i=N w_i by node degree k_i, and compare the
survivor-conditioned (mean, std)-per-degree-class to the HDMFT.

If sim and theory agree here, the gap with beta_limit.py (sigma=1.75-2) is the
PHASE (relaxed vs fluctuating) -> needs the two-time HDMFT. If they disagree, the
measurement/mapping is the problem.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from explore import build_alpha, integrate, summarize, ALPHA_PL
from hdmft_solver import solve_hdmft, loglog_slope

ALPHA = ALPHA_PL - 1.0                      # P(k)~k^-2.5 in build_alpha  <->  paper alpha=1.5
FLOOR = 1e-2                                 # survivor threshold in relative-size units (<S>=1)


def run_sim(mu, sigma, N, seeds, tmax=50.0):
    """Relaxed-phase relative sizes S_i and degrees k_i, pooled over seeds."""
    S_all, k_all, chaos, gdrift, Cs = [], [], [], [], []
    for sd in seeds:
        alpha, C_eff = build_alpha(sd, N, mu, sigma)
        deg = np.diff(alpha.indptr)                          # CSR row pointer -> node degree
        t, W, lnM, _, ok = integrate(sd, N, mu, sigma, tmax=tmax, n_eval=300, lam=0.0)
        s = summarize(t, W, lnM)
        S_all.append(N * W[:, -1])                           # relative size at relaxation
        k_all.append(deg)
        chaos.append(s["chaos"]); gdrift.append(s["g_drift"]); Cs.append(C_eff)
    return (np.concatenate(S_all), np.concatenate(k_all),
            float(np.mean(chaos)), float(np.mean(gdrift)), float(np.mean(Cs)))


def binned_by_degree(S, k, nbins=14):
    """Survivor-conditioned mean & std of size per log-degree bin."""
    surv = S > FLOOR
    edges = np.geomspace(k.min(), k.max() + 1, nbins + 1)
    mk, mS, sS = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = surv & (k >= lo) & (k < hi)
        if m.sum() >= 20:                                    # enough survivors for a stable std
            mk.append(k[m].mean()); mS.append(S[m].mean()); sS.append(S[m].std())
    return np.array(mk), np.array(mS), np.array(sS), float(surv.mean())


def theory_point(mu, sigma, N, C, kmax):
    """HDMFT survivor-conditioned (mean, std) per degree class at matched C, kmax."""
    r = solve_hdmft(mu, sigma, alpha=ALPHA, C=C, N=N, kern="cap", normalize=True, kmax=kmax)
    phi = np.maximum(r["phi"], 1e-12)
    mean_s = r["mean"] / phi                                 # E[y | survive]
    msq_s = r["msq"] / phi
    std_s = np.sqrt(np.maximum(msq_s - mean_s ** 2, 0.0))
    surv = r["phi"] > 0.02
    return r["k"], mean_s, std_s, surv, r["phi_tot"], r


if __name__ == "__main__":
    MU, SIGMA = 2.0, 0.8
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    print(f"relaxed-phase check  mu={MU}, sigma={SIGMA}, alpha={ALPHA}, P(k)~k^-{ALPHA_PL}\n")
    print(f"  {'N':>6} {'seeds':>5} {'C_eff':>6} {'chaos':>7} {'g_drift':>7} "
          f"{'phi_sim':>7} {'phi_th':>6} {'beta_sim':>8} {'beta_th':>7}")

    for N, nseed, color in [(4000, 8, "tab:blue"), (12800, 4, "tab:red")]:
        seeds = list(range(nseed))
        S, k, chaos, gdrift, C_eff = run_sim(MU, SIGMA, N, seeds)
        mk, mS, sS, phi_sim = binned_by_degree(S, k)
        b_sim, r2_sim = loglog_slope(mS, sS, np.ones(mS.size, bool))   # std ~ size^slope
        beta_sim = -b_sim

        kt, mean_t, std_t, surv_t, phi_th, r = theory_point(MU, SIGMA, N, C_eff, kmax=float(k.max()))
        b_th, _ = loglog_slope(mean_t, std_t, surv_t)
        beta_th = -b_th

        print(f"  {N:>6} {nseed:>5} {C_eff:>6.1f} {chaos:>7.4f} {gdrift:>7.4f} "
              f"{phi_sim:>7.3f} {phi_th:>6.3f} {beta_sim:>8.3f} {beta_th:>7.3f}")

        # size-degree map: sim survivor-mean vs theory
        axes[0].loglog(mk, mS, "o", color=color, label=f"sim N={N}")
        axes[0].loglog(kt[surv_t], mean_t[surv_t], "-", color=color, alpha=0.6, label=f"HDMFT N={N}")
        # dispersion vs size: the beta relation
        axes[1].loglog(mS, sS, "o", color=color, label=f"sim N={N} (b={beta_sim:.2f})")
        axes[1].loglog(mean_t[surv_t], std_t[surv_t], "-", color=color, alpha=0.6,
                       label=f"HDMFT (b={beta_th:.2f})")

    axes[0].set(xlabel="degree k", ylabel="survivor mean size  <S | k>", title="size-degree map")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="survivor mean size", ylabel="survivor std (cross-sectional vol)",
                title=f"dispersion-size relation (beta = -slope)")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    out = "/Users/ludovicofurlanetto/Code/glv/growing_glv/hdmft_validate_sim.png"
    plt.savefig(out, dpi=110)
    print(f"\n  chaos~0 confirms relaxed phase.  beta_sim ~ beta_th => HDMFT validated; gap is the PHASE.")
    print(f"  -> {out}")
