import pytest
import numpy as np
from glv.analysis import calculate_mu_c


def test_mu_c_sigma_zero_exponential_agrees_with_formula():
    # Exponential: nu(g) = exp(-g), <g^2> = integral g^2 exp(-g) dg = 2, mu_c = 0.5
    result = calculate_mu_c(sigma=0, gamma=0.0, nu_pdf=lambda g: np.exp(-g))["mu_c"]
    assert result == pytest.approx(0.5, rel=1e-3)


def test_mu_c_sigma_zero_exponential():
    # Exponential: nu(g) = exp(-g), <g^2> = 2, mu_c = 0.5
    result = calculate_mu_c(sigma=0, gamma=0.0, nu_pdf=lambda g: np.exp(-g))["mu_c"]
    assert result == pytest.approx(0.5, rel=1e-3)


def test_mu_c_sigma_nonzero_returns_float():
    result = calculate_mu_c(sigma=0.5, gamma=0.0, nu_pdf=lambda g: np.exp(-g))["mu_c"]
    assert isinstance(result, float)
    assert result > 0


# ── generate_matrix ───────────────────────────────────────────────────────────

import scipy.sparse as sp
from glv.graph import generate_matrix


def test_generate_matrix_shape():
    np.random.seed(0)
    degrees = [2, 2, 2, 2]  # 4-node 2-regular graph
    A = generate_matrix(degrees, C=2, mu=1.0, sigma=0.0)
    assert A.shape == (4, 4)


def test_generate_matrix_is_sparse():
    np.random.seed(0)
    degrees = [2, 2, 2, 2]
    A = generate_matrix(degrees, C=2, mu=1.0, sigma=0.0)
    assert sp.issparse(A)


def test_generate_matrix_diagonal_zero():
    np.random.seed(0)
    degrees = [2, 2, 2, 2]
    A = generate_matrix(degrees, C=2, mu=1.0, sigma=0.0)
    assert A.diagonal().sum() == pytest.approx(0.0)


def test_generate_matrix_odd_sum_raises():
    with pytest.raises(ValueError, match="even"):
        generate_matrix([1, 2], C=1, mu=1.0, sigma=0.0)


def test_generate_matrix_sigma_zero_weight():
    # sigma=0 means all weights = mu/C
    np.random.seed(0)
    degrees = [2, 2, 2, 2]
    mu, C = 1.0, 2.0
    A = generate_matrix(degrees, C=C, mu=mu, sigma=0.0)
    data = A.data
    assert np.allclose(data, mu / C)
