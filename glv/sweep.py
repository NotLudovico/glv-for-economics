import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.integrate import solve_ivp

from glv.dynamics import rescaled_glv_sparse


def _integrate_mu_chunk(args):
    """Run all initial conditions for a single W. Returns (i, t_finals[n_reps])."""
    i, W_sparse, initial_states, N, tau_max, method, max_step = args
    kwargs = {"method": method}
    if max_step is not None:
        kwargs["max_step"] = max_step

    out = np.full(len(initial_states), np.nan)
    for j, state0 in enumerate(initial_states):
        sol = solve_ivp(
            fun=rescaled_glv_sparse,
            t_span=(0.0, tau_max),
            y0=state0,
            args=(N, W_sparse),
            **kwargs,
        )
        if sol.status == 0:
            out[j] = sol.y[N + 1, -1]
    return i, out


def sweep_final_time(
    Ws,
    initial_states,
    N: int,
    tau_max: float,
    method: str = "LSODA",
    max_step: float | None = 1e2,
    n_workers: int | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Run rescaled-GLV integrations in parallel: one task per W.

    Each task integrates all initial_states for its W, so the sparse
    matrix is pickled once per W (not once per IC) and worker startup
    amortizes over n_reps integrations.

    Args:
        Ws: Sequence of interaction matrices (one per mu).
        initial_states: Sequence of state0 vectors (length N+2).
        N: Number of species.
        tau_max: End of rescaled-time integration.
        method: scipy solve_ivp method.
        max_step: Cap on solver step (None to disable).
        n_workers: Number of processes (None → all CPUs).
        verbose: Print per-mu completion.

    Returns:
        Array (len(Ws), len(initial_states)). NaN entries indicate
        failed integrations (sol.status != 0).
    """
    n_mu = len(Ws)
    n_reps = len(initial_states)
    out = np.full((n_mu, n_reps), np.nan)

    jobs = [
        (i, Ws[i], initial_states, N, tau_max, method, max_step)
        for i in range(n_mu)
    ]

    if n_workers == 1:
        for done, args in enumerate(jobs, 1):
            i, t_finals = _integrate_mu_chunk(args)
            out[i, :] = t_finals
            if verbose:
                ok = int(np.sum(~np.isnan(t_finals)))
                print(f"  [{done}/{n_mu}]  i={i}  ok={ok}/{n_reps}")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_integrate_mu_chunk, args) for args in jobs]
            done = 0
            for fut in as_completed(futures):
                i, t_finals = fut.result()
                out[i, :] = t_finals
                done += 1
                if verbose:
                    ok = int(np.sum(~np.isnan(t_finals)))
                    print(f"  [{done}/{n_mu}]  i={i}  ok={ok}/{n_reps}")

    return out
