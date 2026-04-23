import numpy as np
import scipy.sparse as sp
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import root


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


def calculate_mu_c(sigma, gamma, nu_pdf, max_g_approx=100.0):
    """
    Calculates the critical interaction strength mu_c for the generalized
    Lotka-Volterra model on a network.
    """
    if sigma == 0:
        g2_mean = quad(lambda g: g**2 * nu_pdf(g), 0, max_g_approx)[0]
        return {"mu_c": 1.0 / g2_mean, "q_star": 0.0, "chi_star": 0.0}

    def equations(vars):
        mu_c, q_star, chi_star = vars

        if gamma > 0 and chi_star > 0:
            g_star = 1.0 / (gamma * sigma**2 * chi_star)
            upper_limit = min(g_star, max_g_approx)
        else:
            upper_limit = max_g_approx

        def inner_integrals(g):
            q_safe = max(q_star, 1e-10)
            delta_g = np.sqrt(g) / (np.sqrt(q_safe) * sigma)

            phi = norm.pdf(delta_g)
            Phi = norm.cdf(delta_g)

            int_1 = delta_g * Phi + phi
            int_2 = (delta_g**2 + 1) * Phi + 3 * delta_g * phi
            int_3 = Phi

            return int_1, int_2, int_3

        def integrand_1(g):
            int_1, _, _ = inner_integrals(g)
            denominator = 1 - g * gamma * sigma**2 * chi_star
            return nu_pdf(g) * g * (np.sqrt(g * max(q_star, 1e-10)) * sigma / denominator) * int_1

        def integrand_2(g):
            _, int_2, _ = inner_integrals(g)
            denominator = 1 - g * gamma * sigma**2 * chi_star
            return nu_pdf(g) * g * ((g * sigma**2) / (denominator**2)) * int_2

        def integrand_3(g):
            _, _, int_3 = inner_integrals(g)
            denominator = 1 - g * gamma * sigma**2 * chi_star
            return nu_pdf(g) * g * (1 / denominator) * int_3

        eq1_int = quad(integrand_1, 0, upper_limit, limit=100)[0]
        eq2_int = quad(integrand_2, 0, upper_limit, limit=100)[0]
        eq3_int = quad(integrand_3, 0, upper_limit, limit=100)[0]

        res1 = mu_c * eq1_int - 1.0
        res2 = eq2_int - 1.0
        res3 = eq3_int - chi_star

        return [res1, res2, res3]

    initial_guess = [1.0, 1.0, 1.0]
    solution = root(equations, initial_guess, method='hybr')

    if solution.success:
        return {
            "mu_c": solution.x[0],
            "q_star": solution.x[1],
            "chi_star": solution.x[2]
        }
    else:
        raise ValueError(f"Solver failed to converge: {solution.message}")


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
