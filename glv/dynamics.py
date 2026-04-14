import numpy as np
from scipy.integrate import solve_ivp


def simulate_glv(
    A,
    x0: np.ndarray,
    tmax: float,
    n_eval: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the GLV system ẋ_i = x_i (1 - x_i + (A·x)_i).

    Args:
        A: Interaction matrix, shape (N, N). Sparse or dense.
        x0: Initial abundances, shape (N,).
        tmax: End time of integration.
        n_eval: Number of evenly-spaced time points to evaluate (including t=0).

    Returns:
        sol: Abundance trajectories, shape (n_eval, N). Non-negative.
        t:   Time points, shape (n_eval,).
    """
    t_eval = np.linspace(0.0, tmax, n_eval)

    def rhs(t, x, A):
        x = np.maximum(x, 0.0)
        return x * (1.0 - x + A @ x)

    result = solve_ivp(
        rhs,
        [0.0, tmax],
        x0,
        method="RK45",
        t_eval=t_eval,
        args=(A,),
        dense_output=False,
    )

    sol = np.maximum(result.y.T, 0.0)  # (n_eval, N)
    return sol, result.t
