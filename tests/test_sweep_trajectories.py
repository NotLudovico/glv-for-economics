import types

import numpy as np

import glv.sweep


def _fake_solution(y, M, n_t=5):
    """A solve_ivp-like result on the rescaled state [y(N), M, t_phys].

    y is held fixed on the simplex across n_t steps; the scale M is constant
    and large enough that x = y * M overflows float32 but not float64.
    """
    N = y.size
    Y = np.tile(y[:, None], (1, n_t))
    M_row = np.full(n_t, M)
    t_row = np.linspace(0.0, 1.0, n_t)
    state = np.vstack([Y, M_row, t_row])          # (N + 2, n_t)
    return types.SimpleNamespace(y=state, t=np.linspace(0.0, 10.0, n_t), status=0)


def test_trajectory_preserves_relative_size_when_scale_is_large(monkeypatch):
    """Regression: a diverging scale M (super-critical runs at mu_c reach
    M ~ 1e100+) must not corrupt the stored trajectory. The cross-section
    relative size depends only on the shape y, so reducing x = y*M to float32
    (overflow -> inf -> spurious extinctions) is a bug."""
    N = 4
    y = np.array([0.1, 0.2, 0.3, 0.4])            # simplex shape (sum = 1)
    M = 1e40                                       # y*M up to 4e39 > float32 max (3.4e38)

    monkeypatch.setattr(glv.sweep, "solve_ivp", lambda *a, **k: _fake_solution(y, M))
    args = (0, None, None, N, 10.0, "RK45", None, None)
    _, tr = glv.sweep._integrate_trajectory(args)

    x = tr["x"]
    assert np.isfinite(x).all(), "x overflowed (float32 cast of a large scale M)"
    assert x.dtype == np.float64

    # relative size built from the stored x must equal the intended shape (M cancels)
    S_got = N * x[:, -1] / x[:, -1].sum()
    S_exp = N * y / y.sum()
    np.testing.assert_allclose(S_got, S_exp, rtol=1e-6)
