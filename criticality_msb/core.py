"""Disordered-GLV machinery (raw/original dynamics). A realization's disorder is frozen and split as
alpha(sigma) = mean_a + sigma * dis_a so sigma can be swept on a fixed graph+disorder. Integration is
LSODA (stiff-safe near criticality) with tight tolerances and an optional terminal divergence event.
"""
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.integrate import solve_ivp


def _degree_sequence(rng, S, topology, mean_degree, alpha_exp):
    C = min(mean_degree, S - 1)
    if topology == "regular":
        deg = np.full(S, C, dtype=int)
    elif topology == "exponential":
        deg = np.maximum(rng.exponential(C, S).astype(int), 1)
    elif topology == "powerlaw":
        kmin = C * (alpha_exp - 2) / (alpha_exp - 1)
        deg = np.maximum((kmin * (1 - rng.uniform(size=S)) ** (-1 / (alpha_exp - 1))).round().astype(int), 1)
    else:
        raise ValueError(f"unknown topology {topology!r}")
    if deg.sum() % 2:                                   # configuration model needs an even stub count
        deg[deg.argmin()] += 1
    return deg


def build_realization(seed, S, *, topology, mean_degree, alpha_exp, mu, gamma):
    """Graph + frozen disorder split into mean_a, dis_a (alpha = mean_a + sigma*dis_a) + a fixed IC.
    Per-edge moments follow the 1/C scaling: mean mu/C, var sigma^2/C."""
    rng = np.random.default_rng(seed)
    deg = _degree_sequence(rng, S, topology, mean_degree, alpha_exp)
    G = nx.Graph(nx.configuration_model(deg.tolist(), seed=int(seed)))
    G.remove_edges_from(nx.selfloop_edges(G))
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float); A.data[:] = 1.0
    C_eff = float(np.asarray(A.sum(1)).mean())
    Au = sparse.triu(A, k=1).tocoo(); ei, ej = Au.row, Au.col; nE = ei.size
    a = rng.normal(0, 1, nE); b = rng.normal(0, 1, nE)
    sym = (a + b) / np.sqrt(2); anti = (a - b) / np.sqrt(2)
    ds = 1.0 / np.sqrt(2 * C_eff)
    d_ij = ds * (np.sqrt(1 + gamma) * sym + np.sqrt(1 - gamma) * anti)
    d_ji = ds * (np.sqrt(1 + gamma) * sym - np.sqrt(1 - gamma) * anti)
    rows = np.concatenate([ei, ej]); cols = np.concatenate([ej, ei])
    mean_a = sparse.csr_array((np.full(rows.size, mu / C_eff), (rows, cols)), shape=(S, S))
    dis_a = sparse.csr_array((np.concatenate([d_ij, d_ji]), (rows, cols)), shape=(S, S))
    x0 = rng.uniform(0.5, 1.5, S)
    return mean_a, dis_a, C_eff, x0


def integrate(mean_a, dis_a, x0, sigma, T, *, lam, rtol, atol, method="RK45", t_eval=None, percap_thresh=None):
    """Integrate x' = x(1 - x - alpha x) + lam. If percap_thresh is set, a terminal event halts the run
    when the mean abundance exceeds it (divergence onset, S-independent). RK45 by default — LSODA's
    stiffness-switcher is ~7-30x slower here (and non-monotonic in tolerance) near criticality."""
    alpha = mean_a + sigma * dis_a
    S = x0.size

    def rhs(t, x):
        x = np.maximum(x, 0.0)
        return x * (1.0 - x - alpha @ x) + lam

    events = None
    if percap_thresh is not None:
        def runaway(t, x):
            return percap_thresh - float(np.maximum(x, 0.0).sum()) / S
        runaway.terminal = True
        runaway.direction = -1
        events = runaway
    return solve_ivp(rhs, (0.0, T), x0, method=method, t_eval=t_eval,
                     rtol=rtol, atol=atol, events=events)


def _diverged(mean_a, dis_a, x0, sigma, T, **kw):
    r = integrate(mean_a, dis_a, x0, sigma, T, percap_thresh=kw["percap_thresh"], method=kw["method"],
                  lam=kw["lam"], rtol=kw["rtol"], atol=kw["atol"])
    return (r.t[-1] < T - 1e-6) or (float(np.maximum(r.y, 0.0).sum(0).max()) / x0.size >= kw["percap_thresh"])


def locate_sigma_c(mean_a, dis_a, x0, T, *, lam, rtol, atol, percap_thresh, bracket, iters, method="RK45"):
    """Bisection divergence onset on THIS graph: smallest sigma whose run runs away within horizon T."""
    kw = dict(lam=lam, rtol=rtol, atol=atol, percap_thresh=percap_thresh, method=method)
    lo, hi = bracket
    if _diverged(mean_a, dis_a, x0, lo, T, **kw):
        return float(lo)
    if not _diverged(mean_a, dis_a, x0, hi, T, **kw):
        return float(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _diverged(mean_a, dis_a, x0, mid, T, **kw):
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))
