import numpy as np
import pytest
import scipy.sparse as sp

import glv.analysis as analysis


def _make_mock_sweep(mu_true: float, slope: float = 200.0, t_high: float = 40.0,
                     t_low: float = 2.5, noise: float = 0.5, seed: int = 0):
    """Return a function with the same signature as sweep_final_time.

    Mean final time follows a logistic drop from t_high to t_low centred at
    mu_true. mu is recovered from W (the only per-call info that varies):
    W.data = mu/C + sigma/sqrt(C) * z, so mean(W.data) ≈ mu/C, hence
    mu ≈ C * mean(W.data). The mock uses this to read mu out of each Ws.
    """
    rng = np.random.default_rng(seed)

    def mock_sweep(*, Ws, initial_states, N, **_kwargs):
        n_mu = len(Ws)
        n_ic = len(initial_states)
        out = np.empty((n_mu, n_ic))
        for i, W in enumerate(Ws):
            # Recover mu from the weight column (assumes C=N for the mock callers).
            mu_hat = N * float(W.data.mean()) if W.nnz > 0 else 0.0
            base = t_low + (t_high - t_low) / (1.0 + np.exp(slope * (mu_hat - mu_true)))
            out[i, :] = base + rng.normal(0.0, noise, size=n_ic)
        return out

    return mock_sweep


def _make_inputs(N: int = 8, n_ic: int = 3):
    """Build a fully connected adjacency (mod diagonal) so the mock's
    mu = C * mean(W.data) recovery is well-conditioned. Tests pass sigma=0
    so W.data = mu/C exactly, making mu recovery exact."""
    A = np.ones((N, N), dtype=float)
    np.fill_diagonal(A, 0.0)
    ics = [np.ones(N + 2) for _ in range(n_ic)]
    return A, ics


def test_forward_localization(monkeypatch):
    mu_true = 0.05
    A, ics = _make_inputs(N=8)

    monkeypatch.setattr(analysis, "sweep_final_time", _make_mock_sweep(mu_true))

    result = analysis.find_empirical_mu_c(
        mu_c_theoretical=0.0,
        A=A,
        C=8,  # matches N so the mock can recover mu from W.data
        sigma=0.0,
        initial_conditions=ics,
        drop_ratio=0.15,
        expand_step0=0.01,
        expand_max=0.5,
        n_bisect=6,
        n_refine=10,
        n_workers=1,
    )

    assert abs(result["mu_c"] - mu_true) < 0.01
    assert result["direction"] == "forward"
    assert result["bracket"][0] < mu_true < result["bracket"][1]
