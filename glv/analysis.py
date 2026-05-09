import numpy as np
import scipy.sparse as sp
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import root, curve_fit
from glv.sweep import sweep_final_time


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


def find_empirical_mu_c(
    mu_c_theoretical: float,
    A,
    C: float,
    sigma: float,
    initial_conditions,
    n_mu: int = 40,
    mu_lo: float = -0.2,
    mu_hi: float = 0.2,
    tau_max: float = 1e6,
    method: str = "RK45",
    max_step: float | None = 1e2,
    n_workers: int | None = None,
) -> dict:
    """Find empirical mu_c for a fixed graph by sweeping mu and fitting tanh to mean final time.

    For each mu in the sweep, regenerates fresh weights
    alpha_ij = mu/C + sigma/sqrt(C)*z_ij on the edges of A (z_ij ~ N(0,1)),
    runs rescaled-GLV integrations from all initial_conditions, then fits a
    tanh model to mean final-time vs mu and returns the midpoint as mu_c.

    Args:
        mu_c_theoretical: Theoretical critical value — centres the mu sweep.
        A: Binary adjacency matrix (sparse or dense).
        C: Mean degree used in the weight formula.
        sigma: Std of interaction strength fluctuations.
        initial_conditions: Sequence of initial state vectors (length N+2).
        n_mu: Number of mu grid points.
        mu_lo: Lower offset from mu_c_theoretical for the sweep.
        mu_hi: Upper offset from mu_c_theoretical for the sweep.
        tau_max: End of rescaled-time integration.
        method: scipy solve_ivp method.
        max_step: Cap on solver step size (None to disable).
        n_workers: Worker processes (None → all CPUs).

    Returns:
        dict with keys:
            mu_c     – empirical critical mu (tanh midpoint).
            mu_values – full mu grid (length n_mu).
            mean_t   – mean final time per mu point (NaN where no run converged).
            popt     – tanh fit parameters (amp, a, mu0, b).

    Raises:
        RuntimeError: If too few valid runs or the tanh fit fails.
    """
    A_sp = sp.csr_array(A, dtype=float)

    mu_values = np.linspace(mu_c_theoretical + mu_lo, mu_c_theoretical + mu_hi, n_mu)

    def _make_W(mu):
        W = A_sp.copy()
        W.data = mu / C + (sigma / np.sqrt(C)) * np.random.normal(0.0, 1.0, len(W.data))
        return W

    Ws = [_make_W(mu) for mu in mu_values]
    N = len(initial_conditions[0]) - 2

    t_mat = sweep_final_time(
        Ws=Ws,
        initial_states=list(initial_conditions),
        N=N,
        tau_max=tau_max,
        method=method,
        max_step=max_step,
        n_workers=n_workers,
    )

    n_ok = np.sum(~np.isnan(t_mat), axis=1)
    mean_t = np.nanmean(t_mat, axis=1)
    valid = n_ok > 0

    if valid.sum() < 4:
        raise RuntimeError(f"Too few valid mu points for tanh fit ({valid.sum()} < 4).")

    x_fit = mu_values[valid]
    y_fit = mean_t[valid]

    def _tanh(mu, amp, a, mu0, b):
        return amp * np.tanh(-a * (mu - mu0)) + b

    p0 = [
        (y_fit.max() - y_fit.min()) / 2,
        100.0,
        x_fit[len(x_fit) // 2],
        (y_fit.max() + y_fit.min()) / 2,
    ]
    try:
        popt, _ = curve_fit(_tanh, x_fit, y_fit, p0=p0, maxfev=10000)
    except RuntimeError as exc:
        raise RuntimeError(f"Tanh fit failed: {exc}") from exc

    return {
        "mu_c": float(popt[2]),
        "mu_values": mu_values,
        "mean_t": mean_t,
        "popt": popt,
    }


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
