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
    # GLV — Initial Condition Dependence on Ring Structures

    We study whether the **stationary distribution** of
    $$
    \dot{x}_i = x_i \left(1 - x_i + \sum_j A_{ij} x_j\right)
    $$
    depends on the **initial condition** $x_0$, using ring-like graph topologies.

    Several initial conditions are evolved on the **same graph** and their
    distributions are compared at multiple time snapshots.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    form = (
        mo.md(r"""
        - $N$: {N}
        - Ring topology: {ring_type}
        - $k$ (neighbors per side): {k}
        - Rewiring probability $p$ (small-world only): {p_rewire}
        - $\mu$: {mu}
        - $\sigma$: {sigma}
        - $T$: {tmax}
        - Number of time snapshots: {n_snapshots}
        - Seed: {seed}
        """)
        .batch(
            N=mo.ui.number(value=300, start=10, stop=5000),
            ring_type=mo.ui.dropdown(
                options=["simple_ring", "k_nearest", "small_world"],
                value="k_nearest",
            ),
            k=mo.ui.number(value=3, start=1, stop=20),
            p_rewire=mo.ui.number(value=0.1, step=0.01),
            mu=mo.ui.number(value=0.5, step=0.05),
            sigma=mo.ui.number(value=0.1, step=0.01),
            tmax=mo.ui.number(value=20, start=1, stop=500),
            n_snapshots=mo.ui.number(value=5, start=2, stop=10),
            seed=mo.ui.number(value=42),
        )
        .form(submit_button_label="Run")
    )
    form
    return (form,)


@app.cell(hide_code=True)
def _(form, np):
    N = form.value["N"]
    ring_type = form.value["ring_type"]
    k = form.value["k"]
    p_rewire = form.value["p_rewire"]
    mu = form.value["mu"]
    sigma = form.value["sigma"]
    tmax = form.value["tmax"]
    n_snapshots = form.value["n_snapshots"]
    seed = int(form.value["seed"])
    np.random.seed(seed)
    return N, k, mu, n_snapshots, p_rewire, ring_type, sigma, tmax


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Graph generation
    """)
    return


@app.cell(hide_code=True)
def _(N, k, mo, mu, np, nx, p_rewire, ring_type, sigma):
    if ring_type == "simple_ring":
        _G = nx.cycle_graph(N)
    elif ring_type == "k_nearest":
        # Regular ring lattice: each node connects to k left and k right neighbors
        _G = nx.watts_strogatz_graph(N, 2 * k, 0)
    elif ring_type == "small_world":
        _G = nx.watts_strogatz_graph(N, 2 * k, p_rewire)
    else:
        raise ValueError(f"Unknown ring type: {ring_type}")

    G = _G

    # Weighted adjacency matrix
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    _rows, _cols = A.nonzero()
    _C_eff = np.mean([d for _, d in G.degree()])
    _z = np.random.normal(0, 1.0, len(_rows))
    A.data = (mu / _C_eff) + (sigma / np.sqrt(_C_eff)) * _z
    A.setdiag(0.0)
    A.eliminate_zeros()

    _degrees = np.array([d for _, d in G.degree()])

    mo.md(rf"""
    - Topology: **{ring_type}**, $N={N}$, $k={k}$
    - Average degree: **{_C_eff:.2f}** &nbsp;|&nbsp; Min/Max: **{_degrees.min()}** / **{_degrees.max()}**
    - Edges: **{G.number_of_edges()}** &nbsp;|&nbsp; $\mu={mu}$, $\sigma={sigma}$
    """)
    return A, G


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initial conditions

    Four qualitatively different starting states are evolved on the **same graph**.
    """)
    return


@app.cell(hide_code=True)
def _(A, N, mo, n_snapshots, np, tmax):
    from scipy.integrate import solve_ivp

    # ── Define initial conditions ──────────────────────────────────────────────
    _rng = np.random.default_rng(0)   # fixed sub-seed so ICs are reproducible

    initial_conditions = {
        r"Low uniform $\mathcal{U}(0,\,0.5)$":      _rng.uniform(0.0, 0.5, N),
        r"High uniform $\mathcal{U}(1.5,\,2.5)$":   _rng.uniform(1.5, 2.5, N),
        r"Peaked at 1":                              np.ones(N),
        r"Bimodal":                                  np.where(
                                                         _rng.random(N) < 0.5,
                                                         _rng.uniform(0.0, 0.3, N),
                                                         _rng.uniform(1.5, 2.5, N),
                                                     ),
    }

    IC_COLORS = ["steelblue", "tomato", "seagreen", "darkorange"]

    # ── Time grid ──────────────────────────────────────────────────────────────
    _t_snaps = np.linspace(0, tmax, n_snapshots + 1)[1:]
    _t_dense = np.linspace(0, tmax, 800)
    t_eval_all = np.sort(np.unique(np.concatenate([_t_dense, _t_snaps])))
    t_snapshots = _t_snaps

    # ── ODE ───────────────────────────────────────────────────────────────────
    def _glv(t, x, A):
        x = np.maximum(x, 0)
        return x * (1 - x + A @ x)

    # ── Simulate ──────────────────────────────────────────────────────────────
    results = {}
    with mo.status.progress_bar(
        total=len(initial_conditions), title="Running simulations"
    ) as _bar:
        for _name, _x0 in initial_conditions.items():
            _res = solve_ivp(
                _glv,
                [0, tmax],
                _x0,
                method="RK45",
                t_eval=t_eval_all,
                args=(A,),
                dense_output=False,
            )
            results[_name] = (np.maximum(_res.y.T, 0), _res.t)
            _bar.update()

    mo.md(
        f"Ran **{len(initial_conditions)}** initial conditions, "
        f"**{len(t_eval_all)}** time points, $T={tmax}$."
    )
    return IC_COLORS, results, t_snapshots


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Trajectories

    A random sample of 100 node trajectories per initial condition.
    The black line shows the instantaneous mean abundance $\langle x \rangle(t)$.
    """)
    return


@app.cell(hide_code=True)
def _(IC_COLORS, mo, np, plt, results):
    _n_ic = len(results)
    _fig, _axes = plt.subplots(1, _n_ic, figsize=(5 * _n_ic, 4), sharey=True)
    if _n_ic == 1:
        _axes = [_axes]

    for _ax, ((ic_name, (sol, t_out)), color) in zip(
        _axes, zip(results.items(), IC_COLORS)
    ):
        _n_show = min(100, sol.shape[1])
        _idx = np.random.default_rng(1).choice(sol.shape[1], _n_show, replace=False)
        for _i in _idx:
            _ax.plot(t_out, sol[:, _i], alpha=0.15, linewidth=0.5, color=color)
        _ax.plot(
            t_out, sol[:, _idx].mean(axis=1),
            color="black", linewidth=1.5, label=r"$\langle x \rangle$",
        )
        _ax.set_title(ic_name, fontsize=9)
        _ax.set_xlabel("Time")
        _ax.legend(fontsize=8)
        _ax.grid(alpha=0.3)

    _axes[0].set_ylabel(r"Abundance $x_i$")
    _fig.suptitle("Trajectories per initial condition", fontsize=11)
    _fig.tight_layout()
    mo.center(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Distribution snapshots

    KDE of $P(x, t)$ at each snapshot time, coloured light→dark as time increases.
    All panels share the same $x$-axis so distributions can be directly compared.
    """)
    return


@app.cell(hide_code=True)
def _(IC_COLORS, mo, np, plt, results, t_snapshots, tmax):
    from scipy.stats import gaussian_kde as _kde_fn

    _n_ic = len(results)
    _fig, _axes = plt.subplots(1, _n_ic, figsize=(5 * _n_ic, 4), sharex=True)
    if _n_ic == 1:
        _axes = [_axes]

    _x_global_max = max(
        sol[:, :].max() for sol, _ in results.values()
    )

    for _ax, ((ic_name, (sol, t_out)), color) in zip(
        _axes, zip(results.items(), IC_COLORS)
    ):
        for _snap_i, _t_snap in enumerate(t_snapshots):
            _t_idx = np.argmin(np.abs(t_out - _t_snap))
            _x = sol[_t_idx, :]
            _x = _x[_x > 1e-8]
            if len(_x) < 5:
                continue
            _alpha = 0.25 + 0.75 * (_snap_i / max(1, len(t_snapshots) - 1))
            _kde = _kde_fn(_x)
            _xr = np.linspace(0, min(_x_global_max * 1.05, _x.max() * 1.2), 300)
            _ax.plot(
                _xr, _kde(_xr),
                color=color, alpha=_alpha, linewidth=1.5,
                label=f"t={_t_snap:.1f}",
            )

        _ax.set_title(ic_name, fontsize=9)
        _ax.set_xlabel(r"Abundance $x_i$")
        _ax.legend(fontsize=7, title="Time")
        _ax.grid(alpha=0.3)

    _axes[0].set_ylabel("PDF")
    _fig.suptitle(
        rf"Distribution snapshots  (light=early → dark=late,  $T={tmax}$)",
        fontsize=11,
    )
    _fig.tight_layout()
    mo.center(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Final distribution comparison

    Overlay of the **final** ($t = T$) distributions across all initial conditions.
    Overlapping curves indicate that the stationary state is **independent of the initial condition**.
    Dashed vertical lines mark the respective means.
    """)
    return


@app.cell(hide_code=True)
def _(IC_COLORS, mo, np, plt, results, tmax):
    from scipy.stats import gaussian_kde as _kde_fn2

    _fig, _ax = plt.subplots(figsize=(9, 5))

    for (ic_name, (sol, _t)), color in zip(results.items(), IC_COLORS):
        _x_final = sol[-1, :]
        _x_final = _x_final[_x_final > 1e-8]
        if len(_x_final) < 5:
            continue
        _kde = _kde_fn2(_x_final)
        _xr = np.linspace(0, _x_final.max() * 1.1, 400)
        _ax.plot(_xr, _kde(_xr), color=color, linewidth=2, label=ic_name)
        _ax.axvline(
            _x_final.mean(), color=color,
            linestyle="--", linewidth=1, alpha=0.7,
        )

    _ax.set_xlabel(r"Abundance $x_i$")
    _ax.set_ylabel("PDF")
    _ax.set_title(
        rf"Final distribution ($t = {tmax}$) — do all ICs converge to the same state?"
    )
    _ax.legend(fontsize=9)
    _ax.grid(alpha=0.4)
    _fig.tight_layout()
    mo.center(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mean and variance over time

    Track $\langle x \rangle(t)$ and $\sigma_x(t)$ for each IC.
    Convergence of these curves signals approach to the **same stationary state**.
    """)
    return


@app.cell(hide_code=True)
def _(IC_COLORS, mo, plt, results):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for (ic_name, (sol, t_out)), color in zip(results.items(), IC_COLORS):
        _ax1.plot(t_out, sol.mean(axis=1), color=color, linewidth=1.5, label=ic_name)
        _ax2.plot(t_out, sol.std(axis=1),  color=color, linewidth=1.5, label=ic_name)

    _ax1.set_xlabel("Time")
    _ax1.set_ylabel(r"$\langle x \rangle(t)$")
    _ax1.set_title("Mean abundance over time")
    _ax1.legend(fontsize=8)
    _ax1.grid(alpha=0.4)

    _ax2.set_xlabel("Time")
    _ax2.set_ylabel(r"$\sigma_x(t)$")
    _ax2.set_title("Std of abundance over time")
    _ax2.legend(fontsize=8)
    _ax2.grid(alpha=0.4)

    _fig.tight_layout()
    mo.center(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pairwise Wasserstein-1 distance (final distributions)

    The 1-Wasserstein (earth-mover's) distance between every pair of final distributions.
    Small values → the two ICs converged to the **same** stationary state.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, plt, results):
    from scipy.stats import wasserstein_distance as _wdist

    _ic_names = list(results.keys())
    _n = len(_ic_names)
    _W = np.zeros((_n, _n))
    _finals = {name: sol[-1, :] for name, (sol, _) in results.items()}

    for _i, _ni in enumerate(_ic_names):
        for _j, _nj in enumerate(_ic_names):
            if _i != _j:
                _W[_i, _j] = _wdist(_finals[_ni], _finals[_nj])

    _fig, _ax = plt.subplots(figsize=(6, 5))
    _im = _ax.imshow(_W, cmap="YlOrRd")
    _ax.set_xticks(range(_n))
    _ax.set_yticks(range(_n))
    _labels = [
        n.replace("$", "").replace("\\", "").replace("{", "").replace("}", "")[:22]
        for n in _ic_names
    ]
    _ax.set_xticklabels(_labels, rotation=30, ha="right", fontsize=8)
    _ax.set_yticklabels(_labels, fontsize=8)
    plt.colorbar(_im, ax=_ax, label="Wasserstein-1 distance")
    _vmax = _W.max()
    for _i in range(_n):
        for _j in range(_n):
            _ax.text(
                _j, _i, f"{_W[_i,_j]:.3f}",
                ha="center", va="center", fontsize=9,
                color="white" if _W[_i, _j] > 0.6 * _vmax else "black",
            )
    _ax.set_title("Pairwise Wasserstein-1 distance (final distributions)")
    _fig.tight_layout()
    mo.center(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Network visualization

    Ring graph with nodes coloured by their **final abundance** (first IC shown).
    Only rendered for $N \leq 600$.
    """)
    return


@app.cell(hide_code=True)
def _(G, mo, nx, plt, results):
    _ic_name_show = list(results.keys())[0]
    _sol_show, _ = results[_ic_name_show]
    _x_final = _sol_show[-1, :]

    _N = G.number_of_nodes()
    if _N <= 600:
        _fig, _ax = plt.subplots(figsize=(7, 7))
        _pos = nx.circular_layout(G)
        _vmin, _vmax = _x_final.min(), _x_final.max()
        nx.draw_networkx_nodes(
            G, _pos,
            node_color=_x_final,
            cmap="viridis",
            node_size=max(5, int(1200 / _N)),
            vmin=_vmin, vmax=_vmax,
            ax=_ax,
        )
        nx.draw_networkx_edges(G, _pos, alpha=0.15, ax=_ax, width=0.4)
        _sm = plt.cm.ScalarMappable(
            cmap="viridis",
            norm=plt.Normalize(vmin=_vmin, vmax=_vmax),
        )
        _sm.set_array([])
        plt.colorbar(_sm, ax=_ax, label=r"Final abundance $x_i(T)$", shrink=0.7)
        _clean_name = _ic_name_show.replace("$", "").replace("\\", "")[:40]
        _ax.set_title(f"Ring network — final abundance\n({_clean_name})")
        _ax.axis("off")
        _fig.tight_layout()
        mo.center(_fig)
    else:
        mo.callout(
            mo.md(f"Network too large to visualise ($N={_N} > 600$)."),
            kind="warn",
        )
    return


if __name__ == "__main__":
    app.run()
