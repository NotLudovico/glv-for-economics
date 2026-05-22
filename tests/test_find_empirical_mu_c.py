import numpy as np
import networkx as nx

import glv.analysis


def _binary_adjacency(n=60, seed=0):
    """A small binary configuration-model adjacency (sparse csr_array)."""
    rng = np.random.default_rng(seed)
    ds = np.maximum(rng.exponential(scale=6.0, size=n).astype(int), 1)
    if ds.sum() % 2 != 0:
        ds[0] += 1
    G = nx.Graph(nx.configuration_model(list(ds)))
    G.remove_edges_from(nx.selfloop_edges(G))
    return nx.to_scipy_sparse_array(G, format="csr", dtype=float)


def test_make_W_scales_with_adjacency_values(monkeypatch):
    """find_empirical_mu_c must build W_ij = A_ij * alpha_ij, not alpha_ij alone.

    A weighted adjacency (here 0.5 * binary) must produce W matrices that are
    exactly 0.5 * the W matrices built from the binary adjacency, given the
    same RNG state. This fails if _make_W replaces W.data instead of scaling it.
    """
    A_bin = _binary_adjacency()
    A_weighted = A_bin * 0.5  # same sparsity pattern, every value halved

    captured = []

    def fake_sweep(Ws, initial_states, **kwargs):
        captured.append(Ws)
        # A sharp tanh-shaped mean-final-time so the real tanh fit converges.
        idx = np.arange(len(Ws))
        col = 5.0 + 4.0 * np.tanh(-(idx - len(Ws) / 2.0))
        return np.tile(col[:, None], (1, len(initial_states)))

    monkeypatch.setattr(glv.analysis, "sweep_final_time", fake_sweep)

    N = A_bin.shape[0]
    ics = [np.concatenate((np.full(N, 1.0 / N), [1.0], [0.0]))]
    common = dict(
        mu_c_theoretical=0.5, C=6.0, sigma=0.2,
        initial_conditions=ics, n_mu=8,
    )

    np.random.seed(123)
    glv.analysis.find_empirical_mu_c(A=A_bin, **common)
    Ws_bin = captured[0]

    np.random.seed(123)
    glv.analysis.find_empirical_mu_c(A=A_weighted, **common)
    Ws_weighted = captured[1]

    for W_b, W_w in zip(Ws_bin, Ws_weighted):
        np.testing.assert_allclose(W_w.data, 0.5 * W_b.data, rtol=1e-12)
