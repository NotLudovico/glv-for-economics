"""Exploration of the RELATIVE (scale-invariant) GLV:

    ẋ_i = x_i [ 1 − x_i/m − (α x)_i/m ] ,   m = ⟨x⟩ = M/N    (Roy competition sign)

Interactions act on the MEAN firm, so the cooperative term is O(M) not O(M²):
the aggregate grows EXPONENTIALLY in physical time (no finite-time blow-up).
Integrated in the share + log-scale split (w on the simplex, lnM separate) so
nothing overflows and physical time is unbounded:

    x_i = M w_i,   f_i = 1 − N w_i − N (α w)_i,   <f> = Σ w_i f_i = g_eff
    ẇ_i = w_i (f_i − <f>)              (replicator on the simplex)
    d lnM/dt = g_eff                   (aggregate growth rate)

Goal of this first pass: MAP where g_eff is small-positive (steady growth) AND
the shares stay chaotic (persistent fluctuations), vs collapse / freeze.
"""
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GAMMA = 0.0
ALPHA_PL, MEAN_DEGREE = 2.5, 100      # power-law degree exponent, mean degree


def build_alpha(seed, S, mu, sigma):
    """Roy disorder on a power-law configuration-model graph: per-edge mean mu/C, var sigma^2/C."""
    rng = np.random.default_rng(seed)
    C = min(MEAN_DEGREE, S - 1)
    kmin = C * (ALPHA_PL - 2) / (ALPHA_PL - 1)
    deg = np.maximum((kmin * (1 - rng.uniform(size=S)) ** (-1 / (ALPHA_PL - 1))).round().astype(int), 1)
    if deg.sum() % 2:
        deg[deg.argmin()] += 1
    G = nx.Graph(nx.configuration_model(deg.tolist(), seed=int(seed)))
    G.remove_edges_from(nx.selfloop_edges(G))
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float); A.data[:] = 1.0
    C_eff = float(np.asarray(A.sum(axis=1)).mean())
    Au = sparse.triu(A, k=1).tocoo(); ei, ej = Au.row, Au.col; nE = ei.size
    a = rng.normal(0, 1, nE); b = rng.normal(0, 1, nE)
    sym = (a + b) / np.sqrt(2); anti = (a - b) / np.sqrt(2)
    scale = sigma / np.sqrt(2 * C_eff)
    w_ij = mu / C_eff + scale * (np.sqrt(1 + GAMMA) * sym + np.sqrt(1 - GAMMA) * anti)
    w_ji = mu / C_eff + scale * (np.sqrt(1 + GAMMA) * sym - np.sqrt(1 - GAMMA) * anti)
    rows = np.concatenate([ei, ej]); cols = np.concatenate([ej, ei])
    alpha = sparse.csr_array((np.concatenate([w_ij, w_ji]), (rows, cols)), shape=(S, S))
    return alpha, C_eff


def integrate(seed, S, mu, sigma, tmax=30.0, n_eval=1500, lam=0.0, floor_mode="share"):
    """floor_mode:
      'share' -> persistent share floor  lam*(1/S - w)        (scales with the economy)
      'bm'    -> additive Bouchaud-Mezard immigration +lam to abundance, which in share
                 variables is (lam/M)*(1 - S*w) and DECAYS as M grows (washes out late)."""
    alpha, C_eff = build_alpha(seed, S, mu, sigma)
    rng = np.random.default_rng(seed + 1)
    w0 = rng.uniform(0.5, 1.5, S); w0 /= w0.sum()

    def rhs(t, state):
        w = np.clip(state[:S], 0.0, None)
        s = w.sum()
        if s > 0:
            w = w / s                                  # reproject to the simplex
        f = 1.0 - S * w - S * (alpha @ w)              # N = S; relative self-limitation + interactions
        fbar = float(w @ f)
        if floor_mode == "bm":
            Minv = np.exp(-state[S])                   # 1/M; additive immigration, decays as economy grows
            dw = w * (f - fbar) + lam * Minv * (1.0 - S * w)
            dlnM = fbar + S * lam * Minv
        else:
            dw = w * (f - fbar) + lam * (1.0 / S - w)  # persistent share floor (mass-conserving on simplex)
            dlnM = fbar
        return np.concatenate([dw, [dlnM]])

    state0 = np.concatenate([w0, [0.0]])               # lnM(0) = 0
    r = solve_ivp(rhs, (0.0, tmax), state0, method="LSODA", t_eval=np.linspace(0, tmax, n_eval),
                  rtol=1e-7, atol=1e-9)
    t = r.t
    W = np.clip(r.y[:S], 0.0, None)
    W = W / W.sum(0, keepdims=True)                    # normalized shares (S, n_t)
    lnM = r.y[S]
    return t, W, lnM, C_eff, r.success


def summarize(t, W, lnM):
    """g_eff (growth rate, stationarity), chaos (persistent share fluctuation), survivors."""
    n = t.size
    lo, hi = n // 2, n                                 # late half
    # growth rate from lnM slope over the late window
    tt = t[lo:hi]
    g_eff = float(np.polyfit(tt, lnM[lo:hi], 1)[0])
    # split late window in two, compare growth rate -> stationary?
    mid = (lo + hi) // 2
    g1 = float(np.polyfit(t[lo:mid], lnM[lo:mid], 1)[0])
    g2 = float(np.polyfit(t[mid:hi], lnM[mid:hi], 1)[0])
    # chaos: per-species std of Δln(share) in the late window, median over survivors
    floor = 1e-6
    surv = W[:, lo:hi].min(1) > floor
    if surv.sum() > 5:
        lnW = np.log(W[surv][:, lo:hi])
        dln = np.diff(lnW, axis=1)
        chaos = float(np.median(np.abs(dln).mean(1)))  # mean |increment| per species, median over species
    else:
        chaos = 0.0
    return dict(g_eff=g_eff, g_drift=abs(g2 - g1), chaos=chaos,
                survivors=int(surv.sum()), n_surv_end=int((W[:, -1] > floor).sum()))


if __name__ == "__main__":
    S = 400
    mus = [-1.0, 0.0, 1.0, 2.0]
    sigmas = [2.0, 3.0, 4.0]
    LAM = 1e-2                                          # immigration floor to fight condensation
    seed = 12345
    print(f"S={S}  lam={LAM}  (g_eff = aggregate growth rate; chaos = persistent share fluctuation)")
    best = None
    for sig in sigmas:
        print(f"\nsigma={sig}")
        for mu in mus:
            t, W, lnM, C_eff, ok = integrate(seed, S, mu, sig, lam=LAM)
            s = summarize(t, W, lnM)
            growing_chaotic = s["g_eff"] > 0.05 and s["chaos"] > 1e-3
            flag = "  <-- growing + chaotic" if growing_chaotic else ""
            if growing_chaotic and (best is None or s["survivors"] > best[1]["survivors"]):
                best = ((mu, sig), s, (t, W, lnM))
            print(f"  mu={mu:+.1f}: g_eff={s['g_eff']:+.3f} (drift {s['g_drift']:.3f})  "
                  f"chaos={s['chaos']:.4f}  surv={s['survivors']}/{S} (end {s['n_surv_end']}) ok={ok}{flag}")

    if best is not None:
        (mu, sig), s, (t, W, lnM) = best
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        ax[0].plot(t, lnM / np.log(10)); ax[0].set(xlabel="t", ylabel="log10 M", title=f"Aggregate (mu={mu}, sig={sig}, g_eff={s['g_eff']:.3f})")
        idx = np.argsort(W[:, -1])[-12:]
        for i in idx:
            ax[1].plot(t, np.log10(np.maximum(S * W[i], 1e-8)), lw=0.8)
        ax[1].set(xlabel="t", ylabel="log10 S_i  (S_i = N w_i)", title="Relative sizes (top 12)")
        lo = t.size // 2
        g = np.diff(np.log(np.maximum(W[:, lo:], 1e-12)), axis=1).ravel()
        g = g[np.isfinite(g)]; g = (g - g.mean()) / (g.std() + 1e-12)
        ax[2].hist(g, bins=120, density=True); ax[2].set_yscale("log")
        ax[2].set(xlabel="standardized Δln S", ylabel="PDF", title="Growth distribution (late window)")
        plt.tight_layout(); plt.savefig("/Users/ludovicofurlanetto/Code/glv/growing_glv/explore_best.png", dpi=110)
        print(f"\nbest: mu={mu}, sigma={sig}  -> growing_glv/explore_best.png")
    else:
        print("\nNo (growing + chaotic + coexisting) point found in this grid -> widen mu/sigma.")
