# GLV

Simulation and analysis library for the **Generalised Lotka–Volterra (GLV)** model on sparse random networks, with Jupyter notebooks for exploring the model numerically and comparing with mean-field theory.

## Model

The GLV dynamics are:

$$\dot{x}_i = x_i \left(1 - x_i + \sum_j W_{ij} x_j\right)$$

where $x_i$ is the abundance of species $i$ and $W_{ij} = A_{ij}\,\alpha_{ij}$ is a weighted sparse interaction matrix. The adjacency matrix $A_{ij}$ is built via the **configuration model**, and the interaction strengths are drawn as:

$$\alpha_{ij} = \frac{\mu}{C} + \frac{\sigma}{\sqrt{C}}\, z_{ij}, \quad z_{ij} \sim \mathcal{N}(0,1)$$

where $C$ is the average degree, $\mu$ controls the mean interaction (cooperative for $\mu > 0$, competitive for $\mu < 0$), and $\sigma$ controls the disorder.

---

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

---

## Library

The `glv` package exposes the following modules.

### `glv.graph`

**`generate_matrix(degree_sequence, C, mu, sigma)`**

Builds the sparse interaction matrix $W$ from a degree sequence using the configuration model. Returns a `scipy.sparse.csr_array`.

```python
import numpy as np
import glv

N, C = 1000, 50
degrees = np.round(np.random.exponential(scale=C, size=N)).astype(int)
if np.sum(degrees) % 2 != 0:
    degrees[0] += 1

W = glv.generate_matrix(degrees, C, mu=-0.5, sigma=0.3)
```

**`generate_network(degree_sequence, sigma, gamma, mu)`**

Builds a dense interaction matrix using a correlated weight model, where the symmetric and antisymmetric parts of the weight matrix are controlled separately by $\gamma$:

$$\alpha_{ij} = \sqrt{1+\gamma}\, S_{ij} + \sqrt{1-\gamma}\, V_{ij} + \mu$$

with $S = (M + M^T)/\sqrt{2}$, $V = (M - M^T)/\sqrt{2}$, $M_{ij} \sim \mathcal{N}(0, \sigma)$. Returns `(W, G)`.

**`compute_mu_c(sigma, gamma, nu_pdf, max_g_approx=100.0)`**

Computes the critical interaction strength $\mu_c$ via the full HDMFT solver (see `calculate_mu_c` below). Returns a float.

```python
nu_pdf = lambda g: np.exp(-g)  # exponential degree distribution: g = k/C ~ Exp(1)
mu_c = glv.compute_mu_c(sigma=0.5, gamma=0.0, nu_pdf=nu_pdf)
```

For $\sigma = 0$ (homogeneous interactions), the solver reduces to the closed-form limit:

$$\mu_c = \frac{1}{\langle g^2 \rangle} = \frac{1}{\int g^2\, \nu(g)\, dg}$$

---

### `glv.dynamics`

**`simulate_glv(A, x0, tmax, n_eval=500)`**

Integrates the GLV ODE with RK45. Returns `(sol, t)` where `sol` has shape `(n_eval, N)`.

```python
x0 = np.random.uniform(0.5, 1.5, N)
sol, t = glv.simulate_glv(W, x0, tmax=500)
```

**`rescaled_glv_sparse(tau, state, N, W_sparse)`**

ODE right-hand side for the **rescaled GLV system**, suitable for studying finite-time blowup near the cooperative critical point. The state vector encodes the relative fractions $y_i = x_i / M$, the total mass $M = \sum_i x_i$, and the physical time $t$:

$$\frac{dy_i}{d\tau} = y_i\left[(Wy)_i - \phi - (y_i - \|y\|^2)\right], \quad \frac{dM}{d\tau} = 1 + M(\phi - \|y\|^2), \quad \frac{dt}{d\tau} = \frac{1}{M}$$

where $\phi = y^T W y$. Pass directly to `scipy.integrate.solve_ivp`.

```python
from scipy.integrate import solve_ivp

state0 = np.concatenate([y0, [M0], [0.0]])
sol = solve_ivp(glv.rescaled_glv_sparse, [0, 5e4], state0, args=(N, W_sparse), method='RK45')
```

---

### `glv.analysis`

**`fixed_point(W, tol=1e-5, max_iter=3000)`**

Solves the GLV fixed-point equation $x^{*} = (I - W)^{-1}\mathbf{1}$ subject to $x^{*} \geq 0$ via **Projected Damped Jacobi** (a linear complementarity solver):

$$x^{(k+1)} = x^{(k)} + \eta\,\bigl[\max(1 + Wx^{(k)},\, 0) - x^{(k)}\bigr]$$

with damping $\eta = 0.1$. Returns `x*` as a dense array.

```python
x_star = glv.fixed_point(W)
survivors = x_star[x_star > 0]
```

**`stability_matrix(x_star, W)`**

Returns the Jacobian $J = \mathrm{diag}(x^{*})(W - I)$ evaluated at the fixed point.

**`calculate_mu_c(sigma, gamma, nu_pdf, max_g_approx=100.0)`**

Computes $\mu_c$ by solving the three self-consistent HDMFT equations simultaneously. Returns a dict with keys `mu_c`, `q_star`, `chi_star`.

#### The numerical trick

The HDMFT equations involve a double integral — an outer integral over the degree distribution $\nu(g)$ and an inner integral over the Gaussian measure $Dz$:

$$1 = \mu_c \int_0^{g^{*}} \nu(g)\, g\, \frac{\sqrt{g q^{*}}\,\sigma}{1 - g\gamma\sigma^2\chi^{*}} \left[\int_{-\Delta_g}^\infty Dz\,(\Delta_g + z)\right] dg$$

$$1 = \int_0^{g^{*}} \nu(g)\, g\, \frac{g\sigma^2}{(1 - g\gamma\sigma^2\chi^{*})^2} \left[\int_{-\Delta_g}^\infty Dz\,(\Delta_g + z)^2\right] dg$$

$$\chi^{*} = \int_0^{g^{*}} \nu(g)\, g\, \frac{1}{1 - g\gamma\sigma^2\chi^{*}} \left[\int_{-\Delta_g}^\infty Dz\right] dg$$

where $\Delta_g = \sqrt{g}/(\sqrt{q^{*}}\,\sigma)$ and $Dz = e^{-z^2/2}/\sqrt{2\pi}$.

The inner Gaussian integrals are solved **analytically** in terms of the standard normal CDF $\Phi$ and PDF $\phi$:

| Moment | Closed form |
|--------|-------------|
| $\int_{-\Delta_g}^\infty Dz$ | $\Phi(\Delta_g)$ |
| $\int_{-\Delta_g}^\infty Dz\,(\Delta_g + z)$ | $\Delta_g\,\Phi(\Delta_g) + \phi(\Delta_g)$ |
| $\int_{-\Delta_g}^\infty Dz\,(\Delta_g + z)^2$ | $(\Delta_g^2+1)\,\Phi(\Delta_g) + \Delta_g\,\phi(\Delta_g)$ |

Substituting these reduces each equation to a **single numerical integral** over $\nu(g)$, which is then handled by `scipy.integrate.quad`. The three residuals are passed to `scipy.optimize.root` (hybr method) to find $(\mu_c, q^{*}, \chi^{*})$ simultaneously.

For $\sigma = 0$ the inner integrals degenerate and the system reduces to the closed-form limit $\mu_c = 1/\langle g^2 \rangle$, which is computed directly.

---

### `glv.visualization`

**`plot_shortest_path_distribution(degrees, title, relative=True)`**

Builds a graph from a degree sequence, extracts the largest connected component, and plots the distribution of shortest path lengths.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `aguirre-lopez.ipynb` | GLV trajectories on an exponential configuration model graph |
| `fixed_point.ipynb` | Fixed-point abundance distribution via LCP solver |
| `rescaled.ipynb` | Rescaled dynamics and finite-time singularity at $\mu_c$ |
| `mu_critical.ipynb` | $\mu_c$ vs $\sigma$ curve from the HDMFT solver |
| `spectrum.ipynb` | Jacobian eigenvalue spectrum at the survivor fixed point |
| `volatility.ipynb` | Growth volatility scaling $\sigma_i \sim \bar{S}_i^\beta$ |
| `paths.ipynb` | Shortest path distributions for exponential and power-law graphs |
| `power_graph.ipynb` | Power-law exponent shift after configuration model cleaning |

Run any notebook with:

```bash
uv run jupyter notebook
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical arrays |
| `scipy` | ODE solver, root finding, numerical integration |
| `networkx` | Graph generation (configuration model) |
| `matplotlib` | Plotting |
| `powerlaw` | Power-law fitting (`power_graph.ipynb` only) |
