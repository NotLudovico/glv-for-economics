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


def _integrate_observables(args):
    """Run one (W, initial_state) integration and compute observables."""
    i, W_sparse, initial_state, N, tau_max, method, max_step, n_years, t_eval = args
    kwargs = {"method": method}
    if max_step is not None:
        kwargs["max_step"] = max_step
    if t_eval is not None:
        kwargs["t_eval"] = t_eval

    sol = solve_ivp(
        fun=rescaled_glv_sparse,
        t_span=(0.0, tau_max),
        y0=initial_state,
        args=(N, W_sparse),
        **kwargs,
    )

    if sol.t.size < 3:
        return i, None

    y_traj = sol.y[:N, :]
    M_traj = sol.y[N, :]
    t_traj = sol.y[N + 1, :]
    x_traj = y_traj * M_traj

    t_final = t_traj[-1]
    if not np.isfinite(t_final) or t_final <= t_traj[0]:
        return i, None

    dt_year = (t_final - t_traj[0]) / n_years
    t_years = np.arange(t_traj[0], t_final, dt_year)
    x_yearly = np.array([np.interp(t_years, t_traj, x_traj[k]) for k in range(N)])

    totals = x_yearly.sum(axis=0, keepdims=True)
    totals = np.where(totals > 0, totals, np.nan)
    S = N * x_yearly / totals

    log_S = np.log(np.maximum(S, 1e-15))
    g = np.diff(log_S, axis=1)
    g_bar = g.mean(axis=1, keepdims=True)
    volatility = np.sqrt(np.pi / 2.0) * np.mean(np.abs(g - g_bar), axis=1)
    avg_size = S.mean(axis=1)

    return i, {
        "avg_size": avg_size,
        "volatility": volatility,
        "g_flat": g.ravel(),
        "t_max": float(t_final),
        "n_years_actual": int(len(t_years)),
    }


def sweep_observables(
    Ws,
    initial_states,
    N: int,
    tau_max: float,
    n_years: int = 100,
    method: str = "RK45",
    max_step: float | None = 1e2,
    n_workers: int | None = None,
    t_eval_n: int | None = 1000,
    verbose: bool = False,
):
    """Run rescaled-GLV integrations in parallel and compute per-run observables.

    Each task integrates one (W, initial_state) pair, then converts the
    rescaled trajectory to physical time, samples it on a uniform yearly
    grid, and computes per-species time-averaged normalized size, growth-rate
    volatility (sqrt(pi/2) * MAD), and the flattened growth-rate vector.

    Args:
        Ws: Sequence of interaction matrices (one per realization).
        initial_states: Sequence of state0 vectors (length N+2).
        N: Number of species.
        tau_max: End of rescaled-time integration.
        n_years: Target number of "years" sampled from physical time.
        method: scipy solve_ivp method.
        max_step: Cap on solver step (None to disable).
        n_workers: Number of processes (None → all CPUs, 1 → serial).
        t_eval_n: Number of evenly-spaced tau points to evaluate; None
            lets the solver choose adaptive points.
        verbose: Print per-run completion.

    Returns:
        List of length len(Ws). Each entry is either a dict with keys
        ``avg_size``, ``volatility``, ``g_flat``, ``t_max``,
        ``n_years_actual`` — or None if the integration failed.
    """
    n = len(Ws)
    if len(initial_states) != n:
        raise ValueError("Ws and initial_states must have the same length.")

    t_eval = np.linspace(0.0, tau_max, t_eval_n) if t_eval_n else None
    jobs = [
        (i, Ws[i], initial_states[i], N, tau_max, method, max_step, n_years, t_eval)
        for i in range(n)
    ]

    out = [None] * n

    if n_workers == 1:
        for done, args in enumerate(jobs, 1):
            i, res = _integrate_observables(args)
            out[i] = res
            if verbose:
                status = "ok" if res is not None else "failed"
                print(f"  [{done}/{n}] run {i}: {status}")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_integrate_observables, args) for args in jobs]
            done = 0
            for fut in as_completed(futures):
                i, res = fut.result()
                out[i] = res
                done += 1
                if verbose:
                    status = "ok" if res is not None else "failed"
                    print(f"  [{done}/{n}] run {i}: {status}")

    return out
