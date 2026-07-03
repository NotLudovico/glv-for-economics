"""Heterogeneous DMFT (degree-resolved) relaxed-phase solver -- lever 1.

Extends dmft_solver.py to a degree-indexed single-site law, following
Patil, Altieri & Aguirre-Lopez, "Dynamical systems on ultra small-world networks"
(arXiv:2605.21364), eqs (10)-(14), and Section 7 of
docs/superpowers/specs/2026-06-21-relative-glv-dmft-derivation.md.

Why bother: the noise amplitude a firm feels, sigma*sqrt(q_k), now depends on its
connectivity k. That degree-resolved noise is the ONLY object in the mean field that
can carry a (size, volatility) correlation, i.e. a nonzero MSB beta. The
fully-connected solver in dmft_solver.py is the C->inf homogeneous limit where every
firm sees the same noise -> beta = 0 by construction. So that solver cannot even ask
the MSB question; this one can.

Single-site (relaxed phase, gamma=0, the build_alpha case):

    y*_k(z) = max(0, K - mu B_k + sigma sqrt(q_k) z),    z ~ N(0,1)

    c B_k = int dm rho(m) P_km <y_m>            (eq 11, mean interaction)
    c q_k = int dm rho(m) P_km <y_m^2>          (eq 11, noise variance)
    c = C/N

K = 1 for the plain LV the paper solves; K = 1 - g* for the relative GLV, with g*
fixed by the normalisation <y> = M_1 = 1 (the mean firm). gamma=0 drops the Onsager
response term, so this is a pure static-Gaussian fixed point.

Two questions this answers (lever 1):
  (1) Well-posedness at the exponent the repo actually uses, P(k) ~ k^-2.5 (alpha=1.5),
      where the bilinear Chung-Lu closure P_lin = km/CN over-counts hub pairs (P>1) --
      the "ill-posed" blocker in Section 7. Structural cut-offs (P_cap, P_exp) cap it.
  (2) The size<->degree map and per-degree noise sqrt(q_k): does a non-collapsing
      (size, volatility) relation with a definite sign exist at all.

Relaxed phase only (sigma < sigma_c). The MSB tent lives in the fluctuating phase and
needs the two-time HDMFT (lever 3). P_SCM (hidden-degree inversion, the paper's gold
standard) is deferred: P_cap/P_exp are the closed-form well-posed proxies that bracket
it. ponytail: add P_SCM if the qualitative sign survives and the hub correction matters.
"""
import numpy as np
from scipy.optimize import brentq

from dmft_solver import w0, w1, w2, solve_fixed_point


# ---------------------------------------------------------------- degree grid
def pareto_grid(alpha, C, N, n=120, kmax=None):
    """Truncated-Pareto degree grid P(k) ~ k^{-1-alpha}, mean ~ C.

    Returns (k, w) with w_j ~ rho(k_j) dk_j the integration weights, sum(w)=1.
    kmin set from the mean; kmax is the structural cut-off (natural max of N draws,
    capped at N-1). kmax is the "C depends on N" knob for later levers.
    """
    kmin = C * (alpha - 1.0) / alpha                  # continuous-Pareto mean = alpha/(alpha-1) * kmin
    if kmax is None:
        kmax = min(kmin * N ** (1.0 / alpha), N - 1.0)  # extreme of N Pareto draws
    k = np.geomspace(kmin, kmax, n)
    rho = alpha * kmin ** alpha * k ** (-1.0 - alpha)   # Pareto pdf
    dk = np.gradient(k)
    w = rho * dk
    w /= w.sum()                                        # renormalise the truncated tail
    return k, w


def kernel(name, ki, kj, C, N):
    """Connection-probability closures P(k_i, k_j). x = ki*kj/(C N)."""
    x = np.outer(ki, kj) / (C * N)
    if name == "lin":
        return x                                        # Chung-Lu bilinear; P>1 for hub pairs
    if name == "cap":
        return np.minimum(x, 1.0)                        # P_cap, the trivial structural cap
    if name == "exp":
        return 1.0 - np.exp(-x)                          # P_exp, static-model creation
    raise ValueError(name)


# ------------------------------------------------------------ truncated moments
def _moments(a, b):
    """E[max(0,a+bz)] and E[max(0,a+bz)^2] and survival, per degree class."""
    b = np.maximum(b, 1e-300)
    d = a / b
    return b * w1(d), b * b * w2(d), w0(d)              # mean, mean-square, phi


# -------------------------------------------------------------------- solver
def solve_hdmft(mu, sigma, alpha=1.5, C=26.0, N=1162.0, kern="cap",
                normalize=True, grid=None, n=120, kmax=None,
                damp=0.5, tol=1e-10, itmax=8000):
    """Degree-resolved relaxed-phase fixed point.

    normalize=True -> relative GLV (K=1-g*, <y>=1 fixes g*). False -> plain LV (K=1).
    grid=(k,w) overrides the Pareto build (used for the fully-connected check).
    """
    if grid is None:
        k, w = pareto_grid(alpha, C, N, n=n, kmax=kmax)
    else:
        k, w = grid
    M = (N / C) * (kernel(kern, k, k, C, N) * w[None, :])   # B = M@mean, q = M@msq  (eq 11)

    mean = np.ones_like(k)
    msq = np.ones_like(k)
    gstar = 0.0
    for it in range(itmax):
        B = M @ mean
        q = np.maximum(M @ msq, 1e-300)
        b = sigma * np.sqrt(q)
        if normalize:
            def hroot(g):
                a = (1.0 - g) - mu * B
                return float(w @ (b * w1(a / b))) - 1.0
            gstar = brentq(hroot, -200.0, 200.0, xtol=1e-12)
            K = 1.0 - gstar
        else:
            K = 1.0
        a = K - mu * B
        m_new, s_new, phi = _moments(a, b)
        # converge on BOTH moments: <y> is pinned to 1 by the normalisation, so a
        # mean-only test fires before q (hence the noise) has settled.
        err = max(np.max(np.abs(m_new - mean)), np.max(np.abs(s_new - msq)))
        mean = (1 - damp) * mean + damp * m_new
        msq = (1 - damp) * msq + damp * s_new
        if err < tol:
            break

    phi_tot = float(w @ phi)
    return dict(k=k, w=w, mean=mean, msq=msq, phi=phi, B=B, q=q,
                vol=sigma * np.sqrt(q), a=a, K=K, gstar=gstar,
                mu=mu, sigma=sigma, alpha=alpha, C=C, N=N, kern=kern,
                phi_tot=phi_tot, iters=it, conv=err)


def loglog_slope(x, y, mask):
    """(slope, r2) of d log y / d log x over mask; beta_proxy = -slope for y=vol, x=size."""
    x, y = x[mask], y[mask]
    if x.size < 3:
        return np.nan, np.nan
    lx, ly = np.log(x), np.log(y)
    c = np.polyfit(lx, ly, 1)
    r2 = 1.0 - np.sum((ly - np.polyval(c, lx)) ** 2) / np.sum((ly - ly.mean()) ** 2)
    return float(c[0]), float(r2)


# ===================================================================== checks
def _check_fully_connected():
    """Degenerate single degree class (C=N, P=1) must reduce to dmft_solver."""
    for mu, sig in [(0.5, 0.5), (1.0, 1.0), (2.0, 1.2)]:
        ref = solve_fixed_point(mu=mu, sigma=sig)              # scalar fully-connected
        N = 1000.0
        r = solve_hdmft(mu, sig, C=N, N=N, kern="lin", normalize=True,
                        grid=(np.array([N]), np.array([1.0])))
        assert abs(r["gstar"] - ref["gstar"]) < 1e-6, (mu, sig, r["gstar"], ref["gstar"])
        assert abs(r["phi"][0] - ref["phi"]) < 1e-6, (mu, sig, r["phi"][0], ref["phi"])
        assert abs(r["mean"][0] - 1.0) < 1e-6                  # normalisation <y>=1
    print("[check] degenerate degree class reproduces dmft_solver (g*, phi, <y>=1)  OK")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    _check_fully_connected()

    # ---- (A) validate machinery vs the paper's Fig 2: plain LV, alpha=1.5, mutualistic
    print("\n[A] paper Fig 2 regime  alpha=1.5, mu=-0.1, sigma=0.3, C=26, N=1162 (plain LV)")
    print(f"    {'kernel':6} {'phi_tot':>8} {'frac P_lin>1':>13} {'phi(low k)':>11} {'phi(hub)':>9}")
    for kern in ("lin", "cap", "exp"):
        r = solve_hdmft(-0.1, 0.3, alpha=1.5, C=26, N=1162, kern=kern, normalize=False)
        Pl = kernel("lin", r["k"], r["k"], 26, 1162)
        frac = float((Pl > 1).mean())
        print(f"    {kern:6} {r['phi_tot']:8.4f} {frac:13.3f} "
              f"{r['phi'][:r['k'].size//5].mean():11.4f} {r['phi'][-r['k'].size//5:].mean():9.4f}")

    # ---- (B) relative GLV, competitive (mu>0), the repo regime. Size<->degree map.
    print("\n[B] relative GLV  alpha=1.5, competitive, C=26, N=4000, P_cap, normalised <y>=1")
    print(f"    {'mu':>4} {'sigma':>5} {'g*':>7} {'phi_tot':>7} {'k_surv_max':>10} "
          f"{'slope vol~S':>11} {'beta':>6} {'r2':>5} {'beta(/k_i)':>10}")
    for mu in (0.5, 1.0, 2.0):
        for sigma in (0.5, 0.8):
            r = solve_hdmft(mu, sigma, alpha=1.5, C=26, N=4000, kern="cap", normalize=True)
            surv = r["phi"] > 0.02
            S, vol, k = r["mean"], r["vol"], r["k"]
            slope, r2 = loglog_slope(S, vol, surv)                   # vol ~ S^slope
            kappa = k / r["C"]                                       # lever-2 preview (heuristic)
            slope_n, _ = loglog_slope(S, vol / kappa, surv)
            ksurv = k[surv].max() if surv.any() else np.nan
            print(f"    {mu:4.1f} {sigma:5.2f} {r['gstar']:7.3f} {r['phi_tot']:7.3f} "
                  f"{ksurv:10.1f} {slope:11.3f} {-slope:6.3f} {r2:5.2f} {-slope_n:10.3f}")
    print("\n    slope = d log(vol)/d log(size) across surviving degree classes.")
    print("    beta>0 wants vol falling with size (slope<0). '/k_i' previews lever 2.")

    # ---- (C) THE WASHOUT TEST: does beta survive N->inf?  fixed C vs C~sqrt(N).
    print("\n[C] washout test  mu=2.0, sigma=0.8 (the ~0.2 point), P_cap.  Is beta N-stable?")
    print(f"    {'scaling':12} {'N':>8} {'C':>6} {'frac P>1':>8} {'phi_tot':>7} {'beta':>6} {'r2':>5}")
    Ns = [1162, 4000, 14000, 50000, 180000]
    for label, C_of_N in [("C=26 fixed", lambda N: 26.0),
                          ("C~sqrt(N)", lambda N: 26.0 * np.sqrt(N / 4000.0))]:
        for N in Ns:
            C = C_of_N(N)
            r = solve_hdmft(2.0, 0.8, alpha=1.5, C=C, N=float(N), kern="cap", normalize=True)
            surv = r["phi"] > 0.02
            slope, r2 = loglog_slope(r["mean"], r["vol"], surv)
            Pl = kernel("lin", r["k"], r["k"], C, N)
            print(f"    {label:12} {N:8d} {C:6.1f} {float((Pl>1).mean()):8.3f} "
                  f"{r['phi_tot']:7.3f} {-slope:6.3f} {r2:5.2f}")
    print("\n    Fully-connected limit C=N collapses the degree range -> beta->0 (dmft_solver).")
    print("    A flat beta down the N column = volatility does NOT wash out in the mean field.")
