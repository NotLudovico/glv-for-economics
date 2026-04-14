# GLV Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract duplicated GLV simulation code from marimo notebooks into a reusable, tested Python library at `glv/`.

**Architecture:** Three submodules — `glv.graph` (degree sequences + interaction matrix), `glv.dynamics` (ODE integrator), `glv.observables` (empty placeholder). Public symbols re-exported from `glv/__init__.py`. Tests live in `tests/` using pytest.

**Tech Stack:** Python 3.14, numpy, scipy, networkx, pytest, uv

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `glv/__init__.py` | Create | Re-export public API |
| `glv/graph.py` | Create | `get_degree_sequence`, `compute_mu_c`, `generate_matrix` |
| `glv/dynamics.py` | Create | `simulate_glv` |
| `glv/observables.py` | Create | Empty placeholder |
| `tests/__init__.py` | Create | Makes tests a package |
| `tests/test_graph.py` | Create | Tests for glv.graph |
| `tests/test_dynamics.py` | Create | Tests for glv.dynamics |
| `pyproject.toml` | Modify | Add pytest dev dependency |

---

### Task 1: Add pytest and scaffold package

**Files:**
- Modify: `pyproject.toml`
- Create: `glv/__init__.py`
- Create: `glv/graph.py`
- Create: `glv/dynamics.py`
- Create: `glv/observables.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Add pytest as a dev dependency**

```bash
uv add --dev pytest
```

Expected: `pyproject.toml` gains a `[tool.uv.dev-dependencies]` section with `pytest`.

- [ ] **Step 2: Create the package files**

`glv/__init__.py`:
```python
from glv.graph import get_degree_sequence, compute_mu_c, generate_matrix
from glv.dynamics import simulate_glv

__all__ = [
    "get_degree_sequence",
    "compute_mu_c",
    "generate_matrix",
    "simulate_glv",
]
```

`glv/graph.py`:
```python
# graph generation and interaction matrix construction
```

`glv/dynamics.py`:
```python
# GLV ODE integrator
```

`glv/observables.py`:
```python
# placeholder — observables to be added later
```

`tests/__init__.py`:
```python
```

- [ ] **Step 3: Verify the package is importable (will fail on missing functions — that's expected)**

```bash
uv run python -c "import glv"
```

Expected: `ImportError: cannot import name 'get_degree_sequence' from 'glv.graph'`

- [ ] **Step 4: Commit scaffold**

```bash
git add glv/ tests/__init__.py pyproject.toml
git commit -m "chore: scaffold glv package and add pytest"
```

---

### Task 2: `get_degree_sequence`

**Files:**
- Modify: `glv/graph.py`
- Create: `tests/test_graph.py`

- [ ] **Step 1: Write failing tests**

`tests/test_graph.py`:
```python
import pytest
import numpy as np
from glv.graph import get_degree_sequence


def test_regular_length():
    degrees = get_degree_sequence(N=10, C=4)
    assert len(degrees) == 10


def test_regular_values():
    degrees = get_degree_sequence(N=10, C=4, topology="regular")
    assert all(d == 4 for d in degrees)


def test_regular_sum_even():
    # N=10, C=4: sum=40, even; also works when N*C is odd
    degrees = get_degree_sequence(N=5, C=3, topology="regular")
    assert sum(degrees) % 2 == 0


def test_exponential_length():
    np.random.seed(0)
    degrees = get_degree_sequence(N=100, C=5, topology="exponential")
    assert len(degrees) == 100


def test_exponential_non_negative():
    np.random.seed(0)
    degrees = get_degree_sequence(N=100, C=5, topology="exponential")
    assert all(d >= 0 for d in degrees)


def test_exponential_sum_even():
    np.random.seed(0)
    degrees = get_degree_sequence(N=100, C=5, topology="exponential")
    assert sum(degrees) % 2 == 0


def test_unknown_topology_raises():
    with pytest.raises(ValueError, match="topology"):
        get_degree_sequence(N=10, C=4, topology="ring")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_graph.py -v
```

Expected: all 7 tests FAIL with `ImportError` or `AttributeError`

- [ ] **Step 3: Implement `get_degree_sequence` in `glv/graph.py`**

```python
import numpy as np


def get_degree_sequence(N: int, C: float, topology: str = "regular") -> list[int]:
    """Return a degree sequence of length N with mean degree C.

    Args:
        N: Number of nodes.
        C: Mean degree (exact for regular, expected value for exponential).
        topology: "regular" or "exponential".

    Returns:
        List of integer degrees whose sum is even.

    Raises:
        ValueError: If topology is not recognised.
    """
    if topology == "regular":
        degrees = np.full(N, int(C))
    elif topology == "exponential":
        degrees = np.round(np.random.exponential(scale=C, size=N)).astype(int)
    else:
        raise ValueError(f"Unknown topology '{topology}'. Use 'regular' or 'exponential'.")

    if degrees.sum() % 2 != 0:
        degrees[np.random.randint(0, N)] += 1

    return degrees.tolist()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_graph.py -v
```

Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add glv/graph.py tests/test_graph.py
git commit -m "feat: implement get_degree_sequence"
```

---

### Task 3: `compute_mu_c`

**Files:**
- Modify: `glv/graph.py`
- Modify: `tests/test_graph.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_graph.py`:
```python
from glv.graph import compute_mu_c


def test_mu_c_regular():
    # Regular topology: all k_i = C, so g_i = 1, <g^2> = 1, mu_c = 1.0
    degrees = [4, 4, 4, 4]
    assert compute_mu_c(degrees, C=4) == pytest.approx(1.0)


def test_mu_c_formula():
    # degrees = [2, 4], C = 2
    # g = [2/2, 4/2] = [1, 2]
    # <g^2> = (1 + 4) / 2 = 2.5
    # mu_c = 1 / 2.5 = 0.4
    degrees = [2, 4]
    assert compute_mu_c(degrees, C=2) == pytest.approx(0.4)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_graph.py::test_mu_c_regular tests/test_graph.py::test_mu_c_formula -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Add `compute_mu_c` to `glv/graph.py`**

Add after `get_degree_sequence`:
```python
def compute_mu_c(degree_sequence: list[int], C: float) -> float:
    """Return the critical interaction strength mu_c = 1 / <g^2>.

    g_i = k_i / C is the normalised degree of node i.

    Args:
        degree_sequence: Integer degree of each node.
        C: Mean degree used to normalise.

    Returns:
        Critical mu value.
    """
    g = np.array(degree_sequence, dtype=float) / C
    return float(1.0 / np.mean(g ** 2))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_graph.py -v
```

Expected: all 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add glv/graph.py tests/test_graph.py
git commit -m "feat: implement compute_mu_c"
```

---

### Task 4: `generate_matrix`

**Files:**
- Modify: `glv/graph.py`
- Modify: `tests/test_graph.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_graph.py`:
```python
import scipy.sparse as sp
from glv.graph import generate_matrix


def test_generate_matrix_shape():
    np.random.seed(0)
    degrees = [2, 2, 2, 2]  # 4-node 2-regular graph
    A = generate_matrix(degrees, C=2, mu=1.0, sigma=0.0)
    assert A.shape == (4, 4)


def test_generate_matrix_is_sparse():
    np.random.seed(0)
    degrees = [2, 2, 2, 2]
    A = generate_matrix(degrees, C=2, mu=1.0, sigma=0.0)
    assert sp.issparse(A)


def test_generate_matrix_diagonal_zero():
    np.random.seed(0)
    degrees = [2, 2, 2, 2]
    A = generate_matrix(degrees, C=2, mu=1.0, sigma=0.0)
    assert A.diagonal().sum() == pytest.approx(0.0)


def test_generate_matrix_odd_sum_raises():
    with pytest.raises(ValueError, match="even"):
        generate_matrix([1, 2], C=1, mu=1.0, sigma=0.0)


def test_generate_matrix_sigma_zero_weight():
    # sigma=0 means all weights = mu/C
    np.random.seed(0)
    degrees = [2, 2, 2, 2]
    mu, C = 1.0, 2.0
    A = generate_matrix(degrees, C=C, mu=mu, sigma=0.0)
    data = A.data
    assert np.allclose(data, mu / C)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_graph.py::test_generate_matrix_shape tests/test_graph.py::test_generate_matrix_is_sparse tests/test_graph.py::test_generate_matrix_diagonal_zero tests/test_graph.py::test_generate_matrix_odd_sum_raises tests/test_graph.py::test_generate_matrix_sigma_zero_weight -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Add `generate_matrix` to `glv/graph.py`**

Add `import networkx as nx` to the existing imports at the top of `glv/graph.py`, then add after `compute_mu_c`:
```python
def generate_matrix(
    degree_sequence: list[int],
    C: float,
    mu: float,
    sigma: float,
):
    """Build a sparse interaction matrix using the configuration model.

    Edge weights: alpha_ij = mu/C + (sigma/sqrt(C)) * z_ij, z_ij ~ N(0,1).
    Self-loops and multi-edges are removed. Diagonal is zero.

    Args:
        degree_sequence: Integer degree of each node. Sum must be even.
        C: Mean degree used in weight formula.
        mu: Mean interaction strength parameter.
        sigma: Std of interaction strength fluctuations.

    Returns:
        scipy.sparse.csr_array of shape (N, N).

    Raises:
        ValueError: If the sum of degree_sequence is odd.
    """
    if sum(degree_sequence) % 2 != 0:
        raise ValueError("Sum of degree_sequence must be even.")

    G_multi = nx.configuration_model(degree_sequence)
    G = nx.Graph(G_multi)
    G.remove_edges_from(nx.selfloop_edges(G))

    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)

    rows, cols = A.nonzero()
    z = np.random.normal(0.0, 1.0, len(rows))
    A.data = (mu / C) + (sigma / np.sqrt(C)) * z
    A.setdiag(0.0)
    A.eliminate_zeros()

    return A
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_graph.py -v
```

Expected: all 14 tests PASS

- [ ] **Step 5: Commit**

```bash
git add glv/graph.py tests/test_graph.py
git commit -m "feat: implement generate_matrix"
```

---

### Task 5: `simulate_glv`

**Files:**
- Modify: `glv/dynamics.py`
- Create: `tests/test_dynamics.py`

- [ ] **Step 1: Write failing tests**

`tests/test_dynamics.py`:
```python
import numpy as np
import scipy.sparse as sp
import pytest
from glv.dynamics import simulate_glv


def _zero_matrix(N):
    """Sparse NxN zero matrix — no interactions, pure logistic growth."""
    return sp.csr_array((N, N), dtype=float)


def test_output_shapes_default():
    N = 5
    x0 = np.full(N, 0.5)
    sol, t = simulate_glv(_zero_matrix(N), x0, tmax=1.0)
    assert sol.shape == (500, N)
    assert t.shape == (500,)


def test_output_shapes_custom_n_eval():
    N = 5
    x0 = np.full(N, 0.5)
    sol, t = simulate_glv(_zero_matrix(N), x0, tmax=2.0, n_eval=100)
    assert sol.shape == (100, N)
    assert t.shape == (100,)


def test_non_negative():
    N = 10
    np.random.seed(0)
    x0 = np.random.uniform(0, 1, N)
    sol, _ = simulate_glv(_zero_matrix(N), x0, tmax=5.0)
    assert np.all(sol >= 0)


def test_time_bounds():
    N = 3
    x0 = np.ones(N) * 0.5
    tmax = 3.0
    _, t = simulate_glv(_zero_matrix(N), x0, tmax=tmax)
    assert t[0] == pytest.approx(0.0)
    assert t[-1] == pytest.approx(tmax)


def test_logistic_convergence():
    # No interactions: ẋ = x(1-x), converges to 1 from x0=0.5
    N = 1
    x0 = np.array([0.5])
    sol, _ = simulate_glv(_zero_matrix(N), x0, tmax=20.0)
    assert sol[-1, 0] == pytest.approx(1.0, abs=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_dynamics.py -v
```

Expected: all 5 tests FAIL with `ImportError`

- [ ] **Step 3: Implement `simulate_glv` in `glv/dynamics.py`**

```python
import numpy as np
from scipy.integrate import solve_ivp


def simulate_glv(
    A,
    x0: np.ndarray,
    tmax: float,
    n_eval: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the GLV system ẋ_i = x_i (1 - x_i + (A·x)_i).

    Args:
        A: Interaction matrix, shape (N, N). Sparse or dense.
        x0: Initial abundances, shape (N,).
        tmax: End time of integration.
        n_eval: Number of evenly-spaced time points to evaluate (including t=0).

    Returns:
        sol: Abundance trajectories, shape (n_eval, N). Non-negative.
        t:   Time points, shape (n_eval,).
    """
    t_eval = np.linspace(0.0, tmax, n_eval)

    def rhs(t, x, A):
        x = np.maximum(x, 0.0)
        return x * (1.0 - x + A @ x)

    result = solve_ivp(
        rhs,
        [0.0, tmax],
        x0,
        method="RK45",
        t_eval=t_eval,
        args=(A,),
        dense_output=False,
    )

    sol = np.maximum(result.y.T, 0.0)  # (n_eval, N)
    return sol, result.t
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_dynamics.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add glv/dynamics.py tests/test_dynamics.py
git commit -m "feat: implement simulate_glv"
```

---

### Task 6: Wire up `__init__.py` and run full suite

**Files:**
- Modify: `glv/__init__.py` (already written in Task 1 — verify it still works)

- [ ] **Step 1: Run the full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all 19 tests PASS

- [ ] **Step 2: Verify top-level imports work**

```bash
uv run python -c "
from glv import get_degree_sequence, compute_mu_c, generate_matrix, simulate_glv
import numpy as np
degrees = get_degree_sequence(100, 5, topology='exponential')
mu_c = compute_mu_c(degrees, C=5)
A = generate_matrix(degrees, C=5, mu=mu_c, sigma=0.0)
x0 = np.random.uniform(0, 1, 100)
sol, t = simulate_glv(A, x0, tmax=1.0)
print(f'OK: sol.shape={sol.shape}, mu_c={mu_c:.4f}')
"
```

Expected output like: `OK: sol.shape=(500, 100), mu_c=0.XXXX`

- [ ] **Step 3: Commit**

```bash
git add glv/__init__.py
git commit -m "feat: wire up glv public API"
```
