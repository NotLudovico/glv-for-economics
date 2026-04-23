import numpy as np
import scipy.sparse as sp


def fixed_point(W, tol: float = 1e-5, max_iter: int = 3000) -> np.ndarray:
    """Solve x* = 1 + W @ x* s.t. x* >= 0 using Projected Damped Jacobi (LCP).

    Args:
        W: Interaction matrix (sparse or dense), shape (N, N).
        tol: Convergence tolerance on max absolute update.
        max_iter: Maximum number of iterations.

    Returns:
        x*: Fixed-point abundances, shape (N,). Non-negative.
    """
    N = W.shape[0]
    x = np.ones(N)
    damping = 0.1

    for _ in range(max_iter):
        x_new = np.maximum(1.0 + W.dot(x), 0.0)
        if np.max(np.abs(x_new - x)) < tol:
            return x_new
        x = x + damping * (x_new - x)

    return x


def stability_matrix(x_star: np.ndarray, W):
    """Compute Jacobian J = diag(x*) @ (W - I) at the GLV fixed point.

    Args:
        x_star: Fixed-point abundances, shape (N,).
        W: Interaction matrix (sparse or dense), shape (N, N).

    Returns:
        Jacobian matrix in the same format as W (sparse if W is sparse).
    """
    Dx = sp.diags(x_star)
    return Dx @ W - Dx
