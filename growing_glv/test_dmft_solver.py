"""Analytic-limit tests for the relative-GLV DMFT fixed-point solver.

Each test pins a limit I can derive by hand from
docs/superpowers/specs/2026-06-21-relative-glv-dmft-derivation.md, eqs (5)-(6).
Run: uv run python -m pytest growing_glv/test_dmft_solver.py -v
"""
import numpy as np
import pytest

from dmft_solver import solve_fixed_point, sigma_c, w0, w1, w2


# ---- Gaussian moments w_n(Delta) = int_{-Delta}^inf Dz (Delta+z)^n ----

def test_w0_at_zero_is_half():
    assert w0(0.0) == pytest.approx(0.5, abs=1e-9)


def test_w1_at_zero_is_gaussian_peak():
    assert w1(0.0) == pytest.approx(1.0 / np.sqrt(2 * np.pi), abs=1e-9)


def test_w2_at_zero_is_half():
    # int_0^inf z^2 Dz = 1/2, so w2(0) = 1*0.5 + 0 = 0.5
    assert w2(0.0) == pytest.approx(0.5, abs=1e-9)


# ---- chaos-onset disorder (relaxed <-> fluctuating boundary), eq (6) ----

def test_sigma_c_is_sqrt2_at_gamma_zero():
    assert sigma_c(gamma=0.0) == pytest.approx(np.sqrt(2.0), rel=1e-4)


def test_threshold_has_half_survivors_and_zero_delta():
    # at sigma = sigma_c, eq (6) forces w0(Delta)=w2(Delta) -> Delta=0 -> phi=1/2
    sol = solve_fixed_point(mu=0.0, sigma=np.sqrt(2.0), gamma=0.0)
    assert sol["delta"] == pytest.approx(0.0, abs=1e-3)
    assert sol["phi"] == pytest.approx(0.5, abs=2e-3)


# ---- noiseless limit sigma -> 0: all survive, g* = -mu ----

def test_small_sigma_all_survive():
    sol = solve_fixed_point(mu=0.3, sigma=0.05, gamma=0.0)
    assert sol["phi"] == pytest.approx(1.0, abs=1e-3)


def test_very_small_sigma_solves():
    # root Delta ~ 1/sigma, so the bracket must widen as sigma shrinks
    sol = solve_fixed_point(mu=0.0, sigma=0.02, gamma=0.0)
    assert sol["phi"] == pytest.approx(1.0, abs=1e-3)
    assert sol["gstar"] == pytest.approx(0.0, abs=0.02)


def test_small_sigma_growth_rate_is_minus_mu():
    for mu in (-0.5, 0.0, 0.3, 1.0):
        sol = solve_fixed_point(mu=mu, sigma=0.05, gamma=0.0)
        assert sol["gstar"] == pytest.approx(-mu, abs=0.02)


# ---- mu only shifts the growth rate (regular-graph mean field is a uniform shift) ----

def test_shape_observables_are_mu_independent():
    a = solve_fixed_point(mu=0.5, sigma=1.0, gamma=0.0)
    b = solve_fixed_point(mu=2.0, sigma=1.0, gamma=0.0)
    for key in ("delta", "q", "chi", "phi"):
        assert a[key] == pytest.approx(b[key], rel=1e-6), key


def test_growth_rate_shifts_by_minus_delta_mu():
    a = solve_fixed_point(mu=0.5, sigma=1.0, gamma=0.0)
    b = solve_fixed_point(mu=2.0, sigma=1.0, gamma=0.0)
    assert (b["gstar"] - a["gstar"]) == pytest.approx(-(2.0 - 0.5), abs=1e-6)


# ---- internal consistency of any returned solution ----

def test_normalisation_M1_equals_one():
    sol = solve_fixed_point(mu=1.0, sigma=1.0, gamma=0.0)
    M1 = (sol["sqrtq"] / sol["v"]) * w1(sol["delta"]) * sol["sigma"]
    # sqrtq carries sqrt(q); the normalisation eq is (sigma*sqrtq/v) w1 = 1
    assert M1 == pytest.approx(1.0, rel=1e-5)


def test_meansquare_equation_residual_is_zero():
    sol = solve_fixed_point(mu=0.0, sigma=1.0, gamma=0.0)
    # eq (5, mean square): v^2 = sigma^2 w2(Delta)
    assert sol["v"] ** 2 == pytest.approx(sol["sigma"] ** 2 * w2(sol["delta"]), rel=1e-5)


# ---- monotonicity: more disorder -> fewer survivors (relaxed phase) ----

def test_survival_fraction_decreases_with_disorder():
    lo = solve_fixed_point(mu=0.0, sigma=0.5, gamma=0.0)["phi"]
    hi = solve_fixed_point(mu=0.0, sigma=1.3, gamma=0.0)["phi"]
    assert lo > hi
