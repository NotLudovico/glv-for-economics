# GLV Library Design

**Date:** 2026-04-14  
**Status:** Approved

## Goal

Extract duplicated GLV simulation code from the marimo notebooks into a reusable, testable Python library. The library lives at `glv/` (root-level package), importable as `import glv`.

## Scope

Three submodules:

- `glv.graph` — graph generation and interaction matrix construction
- `glv.dynamics` — GLV ODE integrator
- `glv.observables` — placeholder, empty for now

Ring topologies are out of scope. Configuration model only.

## Layout

```
glv/
  __init__.py       # re-exports key public symbols
  graph.py
  dynamics.py
  observables.py    # empty placeholder
tests/
  test_graph.py
  test_dynamics.py
```

## `glv/graph.py`

### `get_degree_sequence(N, C, topology="regular") -> list[int]`

Generates a degree sequence for N nodes with mean degree C.

- `topology="regular"`: all degrees equal `int(C)`
- `topology="exponential"`: samples from `Exponential(scale=C)`, rounds to int
- Ensures the total degree sum is even (required for the configuration model); if odd, increments one randomly chosen node by 1
- Raises `ValueError` for unknown topology

### `compute_mu_c(degree_sequence, C) -> float`

Returns the critical interaction strength `μ_c = 1 / ⟨g²⟩` where `g_i = k_i / C`.

### `generate_matrix(degree_sequence, C, mu, sigma) -> scipy.sparse.csr_array`

Builds a sparse interaction matrix using the configuration model.

1. Calls `nx.configuration_model(degree_sequence)`, converts to simple graph (removes multi-edges and self-loops)
2. Assigns edge weights `α_ij = μ/C + (σ/√C) · z_ij` where `z_ij ~ N(0,1)`
3. Sets diagonal to zero, eliminates explicit zeros
4. Returns a `csr_array` (sparse, optimised for matrix-vector products)
5. Raises `ValueError` if degree sum is odd

## `glv/dynamics.py`

### `simulate_glv(A, x0, tmax, n_eval=500) -> tuple[np.ndarray, np.ndarray]`

Integrates the GLV system:

```
ẋ_i = x_i (1 - x_i + (A·x)_i)
```

- Uses `scipy.integrate.solve_ivp` with `method="RK45"`
- `t_eval = np.linspace(0, tmax, n_eval)`
- Clamps `x` to 0 at each RHS evaluation (`np.maximum(x, 0)`)
- Clamps output to 0 after integration
- Returns `(sol, t)` where `sol` is `(n_eval, N)` and `t` is `(n_eval,)`

## `glv/__init__.py`

Re-exports:

```python
from glv.graph import get_degree_sequence, compute_mu_c, generate_matrix
from glv.dynamics import simulate_glv
```

## Tests

### `tests/test_graph.py`

- `get_degree_sequence` returns a list of length N
- Sum of returned degrees is always even
- Regular topology: all degrees equal `int(C)`
- Exponential topology: all degrees are non-negative integers
- `compute_mu_c` returns the correct value for a known input
- `generate_matrix` returns a square sparse matrix of shape `(N, N)`
- Diagonal of returned matrix is zero
- `generate_matrix` raises `ValueError` on odd degree sum

### `tests/test_dynamics.py`

- `simulate_glv` returns `sol` with shape `(n_eval, N)` and `t` with shape `(n_eval,)`
- All values in `sol` are non-negative
- `t[0] == 0` and `t[-1] == tmax`
