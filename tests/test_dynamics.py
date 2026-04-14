import numpy as np
import scipy.sparse as sp
import pytest
from glv.dynamics import simulate_glv


def _zero_matrix(N):
    """Sparse NxN zero matrix — no interactions, pure logistic growth."""
    return sp.csr_array((N, N), dtype=float)


def test_output_shapes_default():
    N = 5
    x0 = np.full(N, 0.5)
    sol, t = simulate_glv(_zero_matrix(N), x0, tmax=1.0)
    assert sol.shape == (500, N)
    assert t.shape == (500,)


def test_output_shapes_custom_n_eval():
    N = 5
    x0 = np.full(N, 0.5)
    sol, t = simulate_glv(_zero_matrix(N), x0, tmax=2.0, n_eval=100)
    assert sol.shape == (100, N)
    assert t.shape == (100,)


def test_non_negative():
    N = 10
    np.random.seed(0)
    x0 = np.random.uniform(0, 1, N)
    sol, _ = simulate_glv(_zero_matrix(N), x0, tmax=5.0)
    assert np.all(sol >= 0)


def test_time_bounds():
    N = 3
    x0 = np.ones(N) * 0.5
    tmax = 3.0
    _, t = simulate_glv(_zero_matrix(N), x0, tmax=tmax)
    assert t[0] == pytest.approx(0.0)
    assert t[-1] == pytest.approx(tmax)


def test_logistic_convergence():
    # No interactions: ẋ = x(1-x), converges to 1 from x0=0.5
    N = 1
    x0 = np.array([0.5])
    sol, _ = simulate_glv(_zero_matrix(N), x0, tmax=20.0)
    assert sol[-1, 0] == pytest.approx(1.0, abs=1e-3)
