import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from glv.analysis import _inner_gaussian_moments


def _quad_moment(delta_g, k):
    """Ground-truth inner integral int_{-delta}^inf Dz (delta + z)^k.

    Dz = standard normal measure. This is the literal integrand of the
    Aguirre-Lopez self-consistency equations (56), evaluated by direct
    quadrature so the closed forms in _inner_gaussian_moments can be pinned
    against it.
    """
    return quad(lambda z: norm.pdf(z) * (delta_g + z) ** k, -delta_g, 40.0)[0]


def test_inner_moments_match_direct_quadrature():
    """The closed forms must equal the Gaussian moments they stand in for.

    int_1, int_2, int_3 are the analytic values of the k = 1, 2, 0 moments
    int_{-delta}^inf Dz (delta + z)^k. They must agree with direct quadrature
    across the whole range of delta_g, especially small delta_g (large sigma),
    where the phi-weighted term dominates and a wrong coefficient bites hardest.
    """
    for delta_g in [0.2, 0.5, 1.0, 2.0, 3.0]:
        int_1, int_2, int_3 = _inner_gaussian_moments(delta_g)
        np.testing.assert_allclose(int_1, _quad_moment(delta_g, 1), rtol=1e-10,
                                   err_msg=f"int_1 wrong at delta_g={delta_g}")
        np.testing.assert_allclose(int_2, _quad_moment(delta_g, 2), rtol=1e-10,
                                   err_msg=f"int_2 wrong at delta_g={delta_g}")
        np.testing.assert_allclose(int_3, _quad_moment(delta_g, 0), rtol=1e-10,
                                   err_msg=f"int_3 wrong at delta_g={delta_g}")
