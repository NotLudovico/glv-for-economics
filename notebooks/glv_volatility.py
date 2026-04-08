import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import networkx as nx

    plt.style.use("default")
    return mo, np, nx, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GLV — Growth Volatility Analysis
    The model:
    $$
    \dot{x}_i = x_i \left(1 - x_i + \sum_j A_{ij} x_j\right)
    $$
    The interaction strength is set at the critical value $\mu = \mu_c = 1/\langle g^2 \rangle$,
    where $g$ is the degree sequence.

    For each realization we compute:
    - **Average size** of node $i$: $\bar{s}_i = \frac{1}{T}\sum_t x_i(t)$
    - **Growth volatility**: $\sigma_i = \frac{1}{T_i}\sum_t |g_{it} - \bar{g}_i|$,
      where $g_{it} = \log\!\left(x_i(t{+}1)/x_i(t)\right)$ and $\bar{g}_i$ is the mean growth.

    Nodes are binned into **25 equal-frequency bins** by average size (bins defined once across all realizations).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    form = (
        mo.md(
            r"""
            - $N$: {N}
            - $C$: {C}
            - Topology: {topology}
            - $\sigma$: {sigma}
            - $T$: {tmax}
            - $x_0$ distribution: {x0_dist}
            - $x_0$ mean (truncated Gaussian only): {x0_mean}
            - Number of realizations: {n_runs}
            - Log x-axis (trajectory plot): {log_x}
            """
        )
        .batch(
            N=mo.ui.number(value=1000),
            C=mo.ui.number(value=5),
            topology=mo.ui.dropdown(
                options=["regular", "exponential"],
                value="exponential",
            ),
            sigma=mo.ui.number(value=0.1),
            tmax=mo.ui.number(value=10),
            x0_dist=mo.ui.dropdown(
                options=["uniform", "truncated gaussian"],
                value="truncated gaussian",
            ),
            x0_mean=mo.ui.number(value=1),
            n_runs=mo.ui.number(value=10, start=1, stop=500),
            log_x=mo.ui.checkbox(value=True),
        )
        .form(submit_button_label="Run")
    )

    form
    return (form,)


@app.cell(hide_code=True)
def _(form, np):
    N = form.value["N"]
    C = form.value["C"]
    topology = form.value["topology"]
    sigma = form.value["sigma"]
    tmax = form.value["tmax"]
    x0_dist = form.value["x0_dist"]
    x0_mean = form.value["x0_mean"]
    n_runs = form.value["n_runs"]
    log_x = form.value["log_x"]
    np.random.seed(42)
    return C, N, log_x, n_runs, sigma, tmax, topology, x0_dist, x0_mean


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Helper functions
    """)
    return


@app.cell(hide_code=True)
def _(np):
    def get_degree_sequence(N, C, topology="regular"):
        if topology == "regular":
            degrees = np.full(N, int(C))
        elif topology == "exponential":
            degrees = np.round(
                np.random.exponential(scale=C, size=N)
            ).astype(int)
        else:
            raise ValueError("Topology must be 'regular' or 'exponential'")

        if np.sum(degrees) % 2 != 0:
            degrees[np.random.randint(0, N)] += 1

        return degrees.tolist()

    return (get_degree_sequence,)


@app.cell(hide_code=True)
def _(np, nx):
    def generate_interaction_matrix(degree_sequence, C, mu_c, sigma):
        """
        Configuration model with mu always set to mu_c.
        alpha_ij = mu_c/C + (sigma/sqrt(C)) * z_ij,  z_ij ~ N(0,1)
        """
        if sum(degree_sequence) % 2 != 0:
            raise ValueError("Sum of degree sequence must be even.")

        G_multi = nx.configuration_model(degree_sequence)
        G = nx.Graph(G_multi)
        G.remove_edges_from(nx.selfloop_edges(G))

        A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)

        rows, cols = A.nonzero()
        z = np.random.normal(0, 1.0, len(rows))
        A.data = (mu_c / C) + (sigma / np.sqrt(C)) * z
        A.setdiag(0.0)
        A.eliminate_zeros()

        return A, G

    return (generate_interaction_matrix,)


@app.cell(hide_code=True)
def _(np):
    def simulate_glv(A, tmax, x0_dist, x0_mean, log_x):
        from scipy.integrate import solve_ivp
        from scipy.stats import truncnorm

        N_sim = A.shape[0]

        if x0_dist == "uniform":
            x0 = np.random.uniform(0, 1, N_sim)
        else:
            lo, hi = x0_mean - 0.5, x0_mean + 0.5
            x0 = truncnorm.rvs(lo, hi, loc=x0_mean, scale=0.5, size=N_sim)

        if log_x:
            t_eval = np.geomspace(1e-1, tmax, 500)
        else:
            t_eval = np.linspace(0, tmax, 500)

        def glv(t, x, A):
            x = np.maximum(x, 0)
            return x * (1 - x + A @ x)

        result = solve_ivp(
            glv,
            [0, tmax],
            x0,
            method="RK45",
            t_eval=t_eval,
            args=(A,),
            dense_output=False,
        )
        sol = np.maximum(result.y.T, 0)  # (n_timepoints, N)
        return sol, result.t

    return (simulate_glv,)


@app.cell(hide_code=True)
def _(np):
    def compute_avg_size(sol):
        """Average abundance of each node over time: shape (N,)"""
        T = sol.shape[0]
        return sol.sum(axis=0) / T

    def compute_growth_volatility(sol, t_span, year_length):
        """
        g_it = log(x_i(t + year_length) / x_i(t)) for all t where
        t + year_length <= t_span[-1].

        sigma_i = (1/T_i) * sum_t |g_it - g_bar_i|
        Returns shape (N,).
        """
        eps = 1e-12
        x = np.maximum(sol, eps)  # (T, N)

        # For each start index find the closest future index at distance year_length
        end_times = t_span + year_length
        # keep only pairs that stay within the simulated window
        valid = end_times <= t_span[-1]
        if not np.any(valid):
            return np.zeros(sol.shape[1])

        idx_start = np.where(valid)[0]
        idx_end = np.searchsorted(t_span, end_times[idx_start])
        # clip to valid range
        idx_end = np.clip(idx_end, 0, len(t_span) - 1)

        log_growth = np.log(x[idx_end] / x[idx_start])  # (n_pairs, N)
        g_bar = log_growth.mean(axis=0)                  # (N,)
        volatility = np.abs(log_growth - g_bar).mean(axis=0)  # (N,)
        return volatility

    return compute_avg_size, compute_growth_volatility


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running realizations
    """)
    return


@app.cell(hide_code=True)
def _(
    C,
    N,
    generate_interaction_matrix,
    get_degree_sequence,
    log_x,
    mo,
    n_runs,
    np,
    sigma,
    simulate_glv,
    tmax,
    topology,
    x0_dist,
    x0_mean,
):
    all_sols = []      # list of (n_timepoints, N) arrays, one per run
    all_t_spans = []   # list of (n_timepoints,) arrays, one per run
    mu_c_values = []

    with mo.status.progress_bar(total=n_runs, title="Running realizations") as bar:
        for _run in range(n_runs):
            deg_seq = get_degree_sequence(N, C, topology=topology)
            g_arr = np.array(deg_seq, dtype=float) / C
            mu_c = 1.0 / np.mean(g_arr ** 2)
            mu_c_values.append(mu_c)

            A_run, _ = generate_interaction_matrix(
                deg_seq, C=C, mu_c=mu_c, sigma=sigma
            )

            sol, t_span = simulate_glv(
                A_run, tmax, x0_dist, x0_mean, log_x
            )

            all_sols.append(sol)
            all_t_spans.append(t_span)

            bar.update()

    mean_mu_c = float(np.mean(mu_c_values))
    mo.md(rf"Completed **{n_runs}** realizations. Mean $\mu_c = {mean_mu_c:.4f}$.")
    return all_sols, all_t_spans, mean_mu_c


@app.cell(hide_code=True)
def _(mo, tmax):
    year_length_slider = mo.ui.slider(
        start=0.01,
        stop=float(tmax),
        step=0.01,
        value=min(1.0, float(tmax)),
        label=r"Year length $\Delta t$ (simulation time units)",
        show_value=True,
    )
    mo.vstack([
        mo.md(r"### Year length for growth rate $g_{it}$"),
        mo.md(
            r"$g_{it} = \log\!\left(x_i(t+\Delta t)\,/\,x_i(t)\right)$ — "
            r"adjust $\Delta t$ without re-running simulations."
        ),
        year_length_slider,
    ])
    return (year_length_slider,)


@app.cell(hide_code=True)
def _(
    all_sols,
    all_t_spans,
    compute_avg_size,
    compute_growth_volatility,
    year_length_slider,
):
    _year_length = year_length_slider.value
    all_avg_sizes = [compute_avg_size(sol) for sol in all_sols]
    all_volatilities = [
        compute_growth_volatility(sol, t_span, _year_length)
        for sol, t_span in zip(all_sols, all_t_spans)
    ]
    last_sol = all_sols[0]
    last_t_span = all_t_spans[0]
    return all_avg_sizes, all_volatilities, last_sol, last_t_span


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sample trajectory plot (first realization)
    """)
    return


@app.cell(hide_code=True)
def _(
    C,
    N,
    last_sol,
    last_t_span,
    log_x,
    mean_mu_c,
    mo,
    np,
    plt,
    sigma,
    tmax,
    topology,
):
    _fig, _ax = plt.subplots(figsize=(10, 5))

    n_show = min(200, last_sol.shape[1])
    idx_show = np.random.choice(last_sol.shape[1], n_show, replace=False)
    for _i in idx_show:
        _ax.plot(
            last_t_span,
            last_sol[:, _i],
            alpha=0.3,
            linewidth=0.6,
            color="black",
        )

    _ax.set_xlabel("Time")
    _ax.set_ylabel("Abundance $x_i$")
    if log_x:
        _ax.set_xscale("log")
        _ax.set_xlim(left=1e-1)

    _ax.set_title(
        rf"$N={N}$, $C={C}$, {topology} | $\mu=\mu_c\approx{mean_mu_c:.4f}$, $\sigma={sigma}$ | $T={tmax}$",
        fontsize=9,
    )
    plt.grid(alpha=0.5)
    plt.tight_layout()
    mo.center(plt.gca())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Binning & volatility analysis

    Nodes are sorted into **25 equal-frequency bins** by their average size $\bar{s}_i$,
    with bin edges defined from the pooled distribution across all realizations.
    """)
    return


@app.cell(hide_code=True)
def _(all_avg_sizes, all_volatilities, np):
    N_BINS = 25

    # Pool all average sizes across runs to define bin edges once
    pooled_sizes = np.concatenate(all_avg_sizes)  # (n_runs * N,)
    bin_edges = np.quantile(
        pooled_sizes, np.linspace(0, 1, N_BINS + 1)
    )
    # Ensure edges are strictly increasing
    bin_edges = np.unique(bin_edges)
    actual_n_bins = len(bin_edges) - 1

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # For each run and each node, assign a bin and accumulate volatility
    bin_vol_sum = np.zeros(actual_n_bins)
    bin_counts = np.zeros(actual_n_bins, dtype=int)

    for _sizes, _vols in zip(all_avg_sizes, all_volatilities):
        # np.digitize returns 1-indexed bins; clip to [1, actual_n_bins]
        bin_idx = np.digitize(
            _sizes, bin_edges[1:-1]
        )  # 0-indexed result
        for _b in range(actual_n_bins):
            mask = bin_idx == _b
            bin_vol_sum[_b] += _vols[mask].sum()
            bin_counts[_b] += mask.sum()

    # Average volatility per bin (guard against empty bins)
    with np.errstate(invalid="ignore"):
        avg_vol_per_bin = np.where(
            bin_counts > 0, bin_vol_sum / bin_counts, np.nan
        )
    return actual_n_bins, avg_vol_per_bin, bin_centers, bin_counts, bin_edges


@app.cell(hide_code=True)
def _(avg_vol_per_bin, bin_centers, mo, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: volatility vs average size (bin centers)
    _ax = _axes[0]
    _ax.plot(bin_centers, avg_vol_per_bin, "o-", color="steelblue", linewidth=1.5)
    _ax.set_xlabel(r"Average size $\bar{s}_i$ (bin center)")
    _ax.set_ylabel(r"Mean growth volatility $\sigma_i$")
    _ax.set_title(r"Growth volatility vs average size")
    _ax.grid(alpha=0.4)

    # Right: same on log-log scale
    _ax2 = _axes[1]
    valid = (avg_vol_per_bin > 0) & (bin_centers > 0)
    _ax2.loglog(
        bin_centers[valid],
        avg_vol_per_bin[valid],
        "o-",
        color="steelblue",
        linewidth=1.5,
    )
    _ax2.set_xlabel(r"Average size $\bar{s}_i$ (bin center, log scale)")
    _ax2.set_ylabel(r"Mean growth volatility $\sigma_i$ (log scale)")
    _ax2.set_title(r"Growth volatility vs average size (log-log)")
    _ax2.grid(alpha=0.4, which="both")

    plt.tight_layout()
    mo.center(plt.gcf())
    return


@app.cell(hide_code=True)
def _(avg_vol_per_bin, bin_centers, bin_counts, mo, np):
    # Summary table
    _rows = [
        {"Bin": int(b + 1), "Center": f"{bin_centers[b]:.4f}", "Count": int(bin_counts[b]), "Avg σ": f"{avg_vol_per_bin[b]:.5f}"}
        for b in range(len(bin_centers))
        if not np.isnan(avg_vol_per_bin[b])
    ]
    mo.ui.table(_rows)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PDF of volatility $\sigma$ per bin

    Each line is the KDE-estimated PDF of individual node volatilities $\sigma_i$ within one size bin.
    The second plot shows the same PDFs after normalising each node's volatility by its bin mean
    $\hat{\sigma}(S)$, so different bins can be compared on a common scale.
    """)
    return


@app.cell(hide_code=True)
def _(
    actual_n_bins,
    all_avg_sizes,
    all_volatilities,
    avg_vol_per_bin,
    bin_centers,
    bin_edges,
    mo,
    np,
    plt,
):
    from scipy.stats import gaussian_kde as _gaussian_kde

    # Collect individual volatility values per bin
    _bin_vol_lists = [[] for _ in range(actual_n_bins)]
    for _sizes, _vols in zip(all_avg_sizes, all_volatilities):
        _bin_idx = np.digitize(_sizes, bin_edges[1:-1])  # 0-indexed
        for _b in range(actual_n_bins):
            _mask = _bin_idx == _b
            _bin_vol_lists[_b].extend(_vols[_mask].tolist())

    _cmap = plt.get_cmap("plasma", actual_n_bins)
    _norm = plt.Normalize(vmin=bin_centers[0], vmax=bin_centers[-1])

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: PDF of σ per bin ---
    for _b in range(actual_n_bins):
        _vals = np.array(_bin_vol_lists[_b])
        _vals = _vals[_vals > 0]
        if len(_vals) < 5:
            continue
        _kde = _gaussian_kde(_vals)
        _x = np.linspace(_vals.min(), _vals.max(), 300)
        _ax1.plot(
            _x, _kde(_x), color=_cmap(_b), alpha=0.85, linewidth=1.2
        )

    _ax1.set_xlabel(r"Growth volatility $\sigma$")
    _ax1.set_xlim(0.01, 10)
    _ax1.set_xscale("log")
    _ax1.set_ylim(0.001, 10)
    _ax1.set_yscale("log")
    _ax1.set_ylabel("PDF")
    _ax1.set_title(r"PDF of $\sigma$ per size bin")
    _ax1.grid(alpha=0.4)
    _sm1 = plt.cm.ScalarMappable(cmap=_cmap, norm=_norm)
    _sm1.set_array([])
    _fig.colorbar(_sm1, ax=_ax1, label=r"Bin centre $\bar{s}$")

    # --- Right: PDF of σ(S) / σ̂(S) per bin ---
    for _b in range(actual_n_bins):
        _sigma_hat = avg_vol_per_bin[_b]
        if np.isnan(_sigma_hat) or _sigma_hat == 0:
            continue
        _vals = np.array(_bin_vol_lists[_b])
        _vals = _vals[_vals > 0]
        if len(_vals) < 5:
            continue
        _ratio = _vals / _sigma_hat
        _kde = _gaussian_kde(_ratio)
        _x = np.linspace(_ratio.min(), _ratio.max(), 300)
        _ax2.plot(
            _x, _kde(_x), color=_cmap(_b), alpha=0.85, linewidth=1.2
        )

    _ax2.set_xlabel(r"$\sigma(S)\,/\,\bar{\sigma}(S)$")
    _ax2.set_ylabel("PDF")
    _ax2.set_title(
        r"PDF of $\sigma(S)\,/\,\bar{\sigma}(S)$ per size bin"
    )
    _ax2.grid(alpha=0.4)
    _ax2.set_xlim(0.01, 10)
    _ax2.set_xscale("log")
    _ax2.set_ylim(0.0001, 10)
    _ax2.set_yscale("log")
    _sm2 = plt.cm.ScalarMappable(cmap=_cmap, norm=_norm)
    _sm2.set_array([])
    _fig.colorbar(_sm2, ax=_ax2, label=r"Bin centre $\bar{s}$")

    plt.tight_layout()
    mo.center(plt.gcf())
    return


if __name__ == "__main__":
    app.run()
