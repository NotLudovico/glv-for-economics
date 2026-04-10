# GLV Notebooks

Interactive [marimo](https://marimo.io) notebooks for simulating and analysing the **Generalised Lotka–Volterra (GLV)** model on sparse random networks.

The GLV dynamics are:

$$\dot{x}_i = x_i \left(1 - x_i + \sum_j A_{ij} x_j\right)$$

where $x_i$ is the abundance of species $i$ and $A$ is a weighted sparse interaction matrix built via the **configuration model**.

---

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Activate the virtual environment (optional – uv run handles it automatically)
source .venv/bin/activate
```

---

## Running the notebooks

Each notebook lives in `notebooks/` and is a plain Python file runnable with marimo.

### Interactive (edit) mode

Opens the notebook in a browser with live reactive cells and UI controls:

```bash
uv run marimo edit notebooks/al_paper.py
uv run marimo edit notebooks/glv_volatility.py
uv run marimo edit notebooks/ic_dependence_rings.py
```

### Read-only (run) mode

Runs the notebook as an app — UI controls are active but code is hidden:

```bash
uv run marimo run notebooks/al_paper.py
```

### As a script

Each file also runs directly as a Python script (no browser, no UI):

```bash
uv run python notebooks/al_paper.py
```

---

## Notebooks

### `al_paper.py` — GLV on Configuration Model Graphs

Reproduces the mean-field analysis from Aguirre-Lopez et al. Generates a random network via the configuration model (regular or exponential degree distribution), simulates GLV with RK45, and plots:

- Degree distribution of the generated graph
- Connected component breakdown
- Species trajectories coloured by degree
- Mean-field prediction $x^* \approx 1/(1-\mu)$ overlaid on trajectories
- Growth rate vs node degree scatter

**Key parameters (set via the form before running):**

| Parameter | Description |
|-----------|-------------|
| $N$ | Number of species |
| $C$ | Average degree |
| Topology | `regular` or `exponential` |
| $\mu$ | Mean interaction strength |
| Use $\mu_c$ | Replace $\mu$ with the critical value $\mu_c = 1/\langle g^2 \rangle$ |
| $\sigma$ | Interaction disorder |
| $T$ | Simulation time |
| $x_0$ distribution | `uniform` or `truncated gaussian` |

---

### `glv_volatility.py` — Growth Volatility vs Average Size

Studies whether the GLV model reproduces the empirical scaling $\sigma_i \sim \bar{s}_i^\beta$ (analogous to Taylor's law in ecology/economics). Runs multiple independent realisations and computes:

- Average abundance $\bar{s}_i = \frac{1}{T}\sum_t x_i(t)$
- Growth volatility $\sigma_i = \frac{1}{T}\sum_t |g_{it} - \bar{g}_i|$ where $g_{it} = \log(x_i(t+\Delta t)/x_i(t))$
- Bins nodes into 25 equal-frequency size bins and fits a power law in log-log space

**Outputs:** trajectory plot (first realisation), $\sigma$ vs $\bar{s}$ log-log plot with power-law fit, PDF of $\sigma$ per size bin (raw and rescaled).

The year-length slider $\Delta t$ can be adjusted interactively **without re-running** the simulations.

**Key parameters:** same as above plus `n_runs` (number of independent realisations) and $\sigma > 0$ (required for non-trivial volatility).

---

### `ic_dependence_rings.py` — Initial Condition Dependence on Ring Graphs

Tests whether the stationary distribution of GLV depends on the initial condition when the interaction graph is a ring-like structure. Four qualitatively different initial conditions are evolved on the **same graph**:

- Low uniform $\mathcal{U}(0, 0.5)$
- High uniform $\mathcal{U}(1.5, 2.5)$
- Peaked at 1
- Bimodal

**Outputs:**
- Trajectory panels per initial condition
- KDE snapshots of $P(x, t)$ at multiple times
- Overlay of final distributions (convergence check)
- Mean and variance over time
- Pairwise Wasserstein-1 distance matrix between final distributions
- Network visualisation coloured by final abundance (for $N \leq 600$)

**Key parameters:**

| Parameter | Description |
|-----------|-------------|
| $N$ | Number of nodes |
| Ring topology | `simple_ring`, `k_nearest`, or `small_world` |
| $k$ | Neighbours per side |
| $p$ | Rewiring probability (small-world only) |
| $\mu$, $\sigma$ | Interaction strength and disorder |
| $T$ | Simulation time |
| Snapshots | Number of time snapshots for KDE plots |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `marimo` | Reactive notebook runtime |
| `numpy` | Numerical arrays |
| `scipy` | ODE solver (`solve_ivp`), stats |
| `networkx` | Graph generation (configuration model, ring graphs) |
| `matplotlib` | Plotting |
| `altair` | (available, not yet used) |
| `polars` | (available, not yet used) |
