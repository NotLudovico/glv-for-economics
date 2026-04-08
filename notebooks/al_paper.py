import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import scipy.sparse as sp
    import numpy as np
    import networkx as nx

    plt.style.use("default")
    return mo, np, nx, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The Model
    Definition of the differential equation
    $$
    \dot{x}_i = x_i \left(1-x_i + \sum_j A_{ij}x_j\right)
    $$
    """)
    return


@app.cell(hide_code=True)
def _(np):
    def glv_system(t, x, A):
        # Ensure abundances don't drop below zero due to numerical step errors
        x = np.maximum(x, 0)

        dxdt = x * (1 - x + A @ x)
        return dxdt

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Inputs
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
            - $\mu$: {mu} (ignored if $\mu_c$ is used)
            - Use $\mu_c = 1/\langle g^2 \rangle$ from graph: {use_mu_c}
            - $\sigma$: {sigma}
            - $T$: {tmax}
            - $x_0$ distribution: {x0_dist}
            - $x_0$ mean (truncated Gaussian only): {x0_mean}
            - Log x-axis: {log_x}
            """
        )
        .batch(
            N=mo.ui.number(value=5000),
            C=mo.ui.number(value=5),
            topology=mo.ui.dropdown(
                options=["regular", "exponential"],
                value="exponential",
            ),
            mu=mo.ui.number(value=1),
            use_mu_c=mo.ui.checkbox(value=False),
            sigma=mo.ui.number(value=0),
            tmax=mo.ui.number(value=1),
            x0_dist=mo.ui.dropdown(
                options=["uniform", "truncated gaussian"],
                value="uniform",
            ),
            x0_mean=mo.ui.number(value=1),
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
    mu = form.value["mu"]
    use_mu_c = form.value["use_mu_c"]
    sigma = form.value["sigma"]
    tmax = form.value["tmax"]
    x0_dist = form.value["x0_dist"]
    x0_mean = form.value["x0_mean"]
    log_x = form.value["log_x"]
    np.random.seed(69)
    return C, N, log_x, mu, sigma, tmax, topology, use_mu_c, x0_dist, x0_mean


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generating the graph
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Producing Degree Sequence
    """)
    return


@app.cell(hide_code=True)
def _(np):
    def get_degree_sequence(N, C, topology="regular"):
        """
        Generates a degree sequence for N nodes with average degree C.
        If regular is selected, C is casted to an integer

        Parameters:
        N : int (Number of species/nodes)
        C : int or float (Average degree)
        topology : str ('regular' or 'exponential')
        """
        if topology == "regular":
            # Every node gets exactly C connections
            # Note: N * C must be an even number. If it isn't, we add 1 to a random node.
            degrees = np.full(N, int(C))
        elif topology == "exponential":
            # Sample from an exponential distribution where the mean (scale) is C
            degrees = np.random.exponential(scale=C, size=N)

            # Degrees must be integers, so we round them
            degrees = np.round(degrees).astype(int)
        else:
            raise ValueError(
                "Topology must be 'regular' or 'exponential'"
            )

        #  The sum of all degrees must be even
        if np.sum(degrees) % 2 != 0:
            # Pick a random node and add 1 to its degree to make the total sum even
            random_index = np.random.randint(0, N)
            degrees[random_index] += 1

        return degrees.tolist()

    return (get_degree_sequence,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generate configuration model
    The interaction strength is drawn from $\alpha_{ij} = \frac{\mu}{C} + \frac{\sigma}{\sqrt{C}}z_{ij}$ where $z_{ij} \sim \mathcal{N}(0,1)$

    Adjacency matrix is sparse (**important**)
    """)
    return


@app.cell(hide_code=True)
def _(np, nx):
    def generate_configuration_model_matrix(
        degree_sequence, C, mu, sigma
    ):
        """
        Builds an interaction matrix A using the Configuration Model.

        Parameters:
        degree_sequence : list of int
            The desired degrees for each node. Sum must be even.
        interaction_strength : float
            The standard deviation or scaling factor for the interaction weights.

        Returns:
        A : ndarray
            The adjacency matrix formatted for the GLV integrator.
        """
        # 1. Ensure the sum of degrees is even (required for the configuration model)
        if sum(degree_sequence) % 2 != 0:
            raise ValueError(
                "The sum of the degree sequence must be even."
            )

        # Note: The standard configuration model can create self-loops and parallel edges (MultiGraph).
        G_multi = nx.configuration_model(degree_sequence)
        G = nx.Graph(G_multi)
        G.remove_edges_from(nx.selfloop_edges(G))

        # csr = compressed sparse row, optimized for row column multiplication
        A_sparse = nx.to_scipy_sparse_array(
            G, format="csr", dtype=float
        )

        # 2. Apply random interaction weights efficiently
        # Instead of making an N x N grid of random numbers, we only generate
        # random numbers for the specific edges that actually exist.
        rows, cols = A_sparse.nonzero()
        num_edges = len(rows)

        # Generate z_ij ~ N(0, 1)
        z = np.random.normal(0, 1.0, num_edges)

        # Calculate alpha_ij = (mu / C) + (sigma / sqrt(C)) * z_ij
        alpha = (mu / C) + (sigma / np.sqrt(C)) * z

        # Overwrite the unweighted 1s with the random weights
        A_sparse.data = alpha

        # 3. Ensure the diagonal is strictly 0.0 (no self-interactions)
        A_sparse.setdiag(0.0)

        # Clean up the matrix structure (removes any explicit zeros we just created)
        A_sparse.eliminate_zeros()

        return A_sparse, G

    return (generate_configuration_model_matrix,)


@app.cell(hide_code=True)
def _(
    C,
    N,
    generate_configuration_model_matrix,
    get_degree_sequence,
    mo,
    mu,
    np,
    sigma,
    topology,
    use_mu_c,
):
    deg_seq = get_degree_sequence(N, C, topology=topology)

    # Critical mu from the actual degree sequence: mu_c = 1 / <g^2>
    g = np.array(deg_seq, dtype=float) / C
    mu_c = 1.0 / np.mean(g**2)

    mu_eff = mu_c if use_mu_c else mu


    A, G = generate_configuration_model_matrix(
        deg_seq, C=C, mu=mu_eff, sigma=sigma
    )

    mu_label = (
        rf"\mu_c = {mu_c:.4f}" if use_mu_c else rf"\mu = {mu_eff}"
    )
    mo.md(rf"""
           - Generated **{topology}** graph with {N} nodes
           - Target Average Degree: **{C}**
           - Actual Average Degree: **{np.mean([d for n, d in G.degree()]):.2f}**
           - Maximum Degree in Network: **{max(dict(G.degree()).values())}**
           - $\mu_c = 1/\langle g^2 \rangle = {mu_c:.4f}$ &nbsp;|&nbsp; using ${mu_label}$
        """)
    return A, G, mu_c, mu_eff


@app.cell(hide_code=True)
def _(C, G, mo, np, plt):
    _degrees = np.array([d for _, d in G.degree()])
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.hist(
        _degrees,
        bins=range(0, _degrees.max() + 2),
        align="left",
        edgecolor="white",
        color="steelblue",
        density=True,
    )
    _ax.set_xlabel("Degree $k$")
    _ax.set_ylabel("PDF")
    _ax.set_title(
        rf"Degree Distribution  |  $\langle k \rangle = {_degrees.mean():.2f}$,  target $C = {C}$"
    )
    plt.grid(alpha=0.4, axis="y")
    plt.tight_layout()
    mo.center(plt.gca())
    return


@app.cell(hide_code=True)
def _(G, mo, nx):
    _components = sorted(
        nx.connected_components(G), key=len, reverse=True
    )
    _sizes = [len(c) for c in _components]
    mo.vstack(
        [
            mo.md(rf"""
        ### Connected Components
        - **Number of components:** {len(_components)}
        - **Largest connected component:** {_sizes[0]} nodes ({_sizes[0] / G.number_of_nodes() * 100:.1f}% of graph)
        - **Isolated nodes (size 1):** {sum(1 for s in _sizes if s == 1)}
        """),
            mo.ui.table(
                data=[
                    {
                        "#": i + 1,
                        "Nodes": s,
                        "% of graph": f"{s / G.number_of_nodes() * 100:.1f}%",
                    }
                    for i, s in enumerate(_sizes)
                ],
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Simulation
    """)
    return


@app.cell(hide_code=True)
def _(A, log_x, np, tmax, x0_dist, x0_mean):
    from scipy.integrate import solve_ivp
    from scipy.stats import truncnorm

    N_sim = A.shape[0]

    if x0_dist == "uniform":
        x0 = np.random.uniform(0, 1, N_sim)
    else:  # truncated gaussian
        a_clip, b_clip = 0,3000
        x0 = truncnorm.rvs(
            a_clip, b_clip, loc=x0_mean, scale=0.1, size=N_sim
        )

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
    sol = np.maximum(result.y.T, 0)  # shape: (n_timepoints, N)
    t_span = result.t
    return sol, t_span


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There's a dependence on initial condition for the stationary state at long time for a competitive environment. This is due to the fact that the initial conditions lead to an effective topology change, for example a node could survive and all of its neighbours will be dead, hence the surviving node will evolve naturally towards 1. There are also interemediate cases that make a node apporach a non 0 and non 1 stationary state.
    """)
    return


@app.cell(hide_code=True)
def _(
    C,
    N,
    log_x,
    mo,
    mu,
    mu_c,
    mu_eff,
    np,
    plt,
    sigma,
    sol,
    t_span,
    tmax,
    topology,
    use_mu_c,
    x0_dist,
):
    _fig, _ax = plt.subplots(figsize=(10, 5))

    n_show = min(200, sol.shape[1])
    idx = np.random.choice(sol.shape[1], n_show, replace=False)
    for i in idx:
        _ax.plot(
            t_span,
            sol[:, i],
            alpha=0.3,
            linewidth=0.6,
            color="black",
        )

    _ax.set_xlabel("Time")
    _ax.set_ylabel("Abundance $x_i$")
    if log_x:
        _ax.set_xscale("log")
        _ax.set_xlim(left=1e-1)

    mu_str = (
        rf"$\mu_c={mu_c:.4f}$  (using $\mu_c$)"
        if use_mu_c
        else rf"$\mu={mu}$,  $\mu_c={mu_c:.3f}$"
    )
    _ax.set_title(
        rf"$N={N}$,  $C={C}$,  {topology}  |  {mu_str},  $\sigma={sigma}$  |  $T={tmax}$,  $x_0$: {x0_dist}",
        fontsize=9,
    )
    mf_prediction = 1 / (1 - mu_eff)

    _ax.axhline(
        y=mf_prediction,
        color="red",
        linestyle="--",
        linewidth=1,
        label=r"$\frac{1}{(1-\mu)}\approx$" + f"{mf_prediction:.2f}",
    )

    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.legend()
    mo.center(plt.gca())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the effective stochastic process ($\gamma = 0$)
    $$
    \dot{x} = x(1-x+g\mu M(t) + \sqrt{g}\sigma \eta(t))
    $$
    means that the growth rates is
    $$
    r(t) = 1- x + g\mu M(t) + \sqrt{g}\sigma \eta(t)
    $$
    """)
    return


@app.cell(hide_code=True)
def _(G, mo, np, plt, sol, tmax):
    _degrees = np.array([d for _, d in G.degree()])
    _growth_rate = (sol[-1, :] - sol[0, :]) / tmax

    _unique_degrees = np.unique(_degrees)
    _mean_growth = np.array([_growth_rate[_degrees == k].mean() for k in _unique_degrees])

    _fig, _ax = plt.subplots(figsize=(10, 5))
    _ax.scatter(_degrees, _growth_rate, alpha=0.15, s=5, color="steelblue", label="species")
    _ax.plot(_unique_degrees, _mean_growth, color="red", linewidth=2, marker="o", markersize=4, label="mean per degree")

    _ax.set_xlabel("Node degree $k$")
    _ax.set_ylabel(r"Growth rate $\frac{x_i(T)-x_i(0)}{T}$")
    _ax.set_title("Degree vs Growth Rate")
    _ax.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    mo.center(plt.gca())
    return


if __name__ == "__main__":
    app.run()
