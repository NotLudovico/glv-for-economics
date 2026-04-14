import pytest
import numpy as np
from glv.graph import get_degree_sequence


def test_regular_length():
    degrees = get_degree_sequence(N=10, C=4)
    assert len(degrees) == 10


def test_regular_values():
    degrees = get_degree_sequence(N=10, C=4, topology="regular")
    assert all(d == 4 for d in degrees)


def test_regular_sum_even():
    # N=5, C=3: sum=15, odd — library must fix it
    degrees = get_degree_sequence(N=5, C=3, topology="regular")
    assert sum(degrees) % 2 == 0


def test_exponential_length():
    np.random.seed(0)
    degrees = get_degree_sequence(N=100, C=5, topology="exponential")
    assert len(degrees) == 100


def test_exponential_non_negative():
    np.random.seed(0)
    degrees = get_degree_sequence(N=100, C=5, topology="exponential")
    assert all(d >= 0 for d in degrees)


def test_exponential_sum_even():
    np.random.seed(0)
    degrees = get_degree_sequence(N=100, C=5, topology="exponential")
    assert sum(degrees) % 2 == 0


def test_unknown_topology_raises():
    with pytest.raises(ValueError, match="topology"):
        get_degree_sequence(N=10, C=4, topology="ring")


# ── compute_mu_c ──────────────────────────────────────────────────────────────

from glv.graph import compute_mu_c


def test_mu_c_regular():
    # Regular topology: all k_i = C, so g_i = 1, <g^2> = 1, mu_c = 1.0
    degrees = [4, 4, 4, 4]
    assert compute_mu_c(degrees, C=4) == pytest.approx(1.0)


def test_mu_c_formula():
    # degrees = [2, 4], C = 2
    # g = [2/2, 4/2] = [1, 2]
    # <g^2> = (1 + 4) / 2 = 2.5
    # mu_c = 1 / 2.5 = 0.4
    degrees = [2, 4]
    assert compute_mu_c(degrees, C=2) == pytest.approx(0.4)
