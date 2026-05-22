# True-graph vs Annealed Simulation Switch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `final_time.ipynb` run its empirical-`mu_c` simulations against either the true configuration-model graph or the annealed approximation, switched by a single notebook-level flag.

**Architecture:** Make `find_empirical_mu_c` adjacency-agnostic by having its internal `_make_W` *multiply* the adjacency's stored values by the disordered weight `alpha_ij`, instead of *replacing* them. Binary adjacency → unchanged behavior. Weighted annealed adjacency `A_ij = k_i k_j/(NC)` → correct annealed matrix. The notebook then builds whichever adjacency the flag selects.

**Tech Stack:** Python 3.14, NumPy, SciPy sparse, NetworkX, pytest, Jupyter. Run everything with `uv run`.

**Spec:** [docs/superpowers/specs/2026-05-22-annealed-true-graph-switch-design.md](../specs/2026-05-22-annealed-true-graph-switch-design.md)

---

## File Structure

- `glv/analysis.py` — modify `find_empirical_mu_c`: change the `_make_W` closure and the docstring. No signature change.
- `tests/test_find_empirical_mu_c.py` — **new** file. First test in the repo; pytest is already a dev dependency.
- `notebooks/final_time.ipynb` — add a `USE_ANNEALED` flag and branch adjacency construction in two cells.

---

## Task 1: Make `_make_W` multiply by the adjacency values

**Files:**
- Create: `tests/test_find_empirical_mu_c.py`
- Modify: `glv/analysis.py:117-160` (`find_empirical_mu_c` docstring and the `_make_W` closure)

- [ ] **Step 1: Write the failing test**

Create `tests/test_find_empirical_mu_c.py` with exactly this content:

```python
import numpy as np
import networkx as nx

import glv.analysis


def _binary_adjacency(n=60, seed=0):
    """A small binary configuration-model adjacency (sparse csr_array)."""
    rng = np.random.default_rng(seed)
    ds = np.maximum(rng.exponential(scale=6.0, size=n).astype(int), 1)
    if ds.sum() % 2 != 0:
        ds[0] += 1
    G = nx.Graph(nx.configuration_model(list(ds)))
    G.remove_edges_from(nx.selfloop_edges(G))
    return nx.to_scipy_sparse_array(G, format="csr", dtype=float)


def test_make_W_scales_with_adjacency_values(monkeypatch):
    """find_empirical_mu_c must build W_ij = A_ij * alpha_ij, not alpha_ij alone.

    A weighted adjacency (here 0.5 * binary) must produce W matrices that are
    exactly 0.5 * the W matrices built from the binary adjacency, given the
    same RNG state. This fails if _make_W replaces W.data instead of scaling it.
    """
    A_bin = _binary_adjacency()
    A_weighted = A_bin * 0.5  # same sparsity pattern, every value halved

    captured = []

    def fake_sweep(Ws, initial_states, **kwargs):
        captured.append(Ws)
        # A sharp tanh-shaped mean-final-time so the real tanh fit converges.
        idx = np.arange(len(Ws))
        col = 5.0 + 4.0 * np.tanh(-(idx - len(Ws) / 2.0))
        return np.tile(col[:, None], (1, len(initial_states)))

    monkeypatch.setattr(glv.analysis, "sweep_final_time", fake_sweep)

    N = A_bin.shape[0]
    ics = [np.concatenate((np.full(N, 1.0 / N), [1.0], [0.0]))]
    common = dict(
        mu_c_theoretical=0.5, C=6.0, sigma=0.2,
        initial_conditions=ics, n_mu=8,
    )

    np.random.seed(123)
    glv.analysis.find_empirical_mu_c(A=A_bin, **common)
    Ws_bin = captured[0]

    np.random.seed(123)
    glv.analysis.find_empirical_mu_c(A=A_weighted, **common)
    Ws_weighted = captured[1]

    for W_b, W_w in zip(Ws_bin, Ws_weighted):
        np.testing.assert_allclose(W_w.data, 0.5 * W_b.data, rtol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_find_empirical_mu_c.py -v`
Expected: FAIL. The current `_make_W` does `W.data = alpha` (ignores `A`'s values), so `Ws_weighted` equals `Ws_bin` instead of half of it — `assert_allclose` reports a mismatch (ratio ~1.0 vs expected 0.5).

- [ ] **Step 3: Change `_make_W` to multiply**

In `glv/analysis.py`, the `_make_W` closure inside `find_empirical_mu_c` currently reads:

```python
    def _make_W(mu):
        W = A_sp.copy()
        W.data = mu / C + (sigma / np.sqrt(C)) * np.random.normal(0.0, 1.0, len(W.data))
        return W
```

Replace it with:

```python
    def _make_W(mu):
        W = A_sp.copy()
        z = np.random.normal(0.0, 1.0, len(W.data))
        W.data = W.data * (mu / C + (sigma / np.sqrt(C)) * z)
        return W
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_find_empirical_mu_c.py -v`
Expected: PASS.

- [ ] **Step 5: Update the `find_empirical_mu_c` docstring**

In `glv/analysis.py`, the docstring of `find_empirical_mu_c` currently opens with:

```python
    """Find empirical mu_c for a fixed graph by sweeping mu and fitting tanh to mean final time.

    For each mu in the sweep, regenerates fresh weights
    alpha_ij = mu/C + sigma/sqrt(C)*z_ij on the edges of A (z_ij ~ N(0,1)),
    runs rescaled-GLV integrations from all initial_conditions, then fits a
    tanh model to mean final-time vs mu and returns the midpoint as mu_c.
```

Replace those lines with:

```python
    """Find empirical mu_c for a fixed graph by sweeping mu and fitting tanh to mean final time.

    For each mu in the sweep, builds weights W_ij = A_ij * (mu/C +
    sigma/sqrt(C)*z_ij), z_ij ~ N(0,1), runs rescaled-GLV integrations from
    all initial_conditions, then fits a tanh model to mean final-time vs mu
    and returns the midpoint as mu_c. With a binary adjacency this places
    fresh disordered weights on the true graph's edges; with the annealed
    adjacency A_ij = k_i k_j/(NC) it builds the annealed interaction matrix.
```

Then, in the same docstring's Args section, the line:

```python
        A: Binary adjacency matrix (sparse or dense).
```

becomes:

```python
        A: Adjacency matrix (sparse or dense). Binary for the true graph,
            or the weighted annealed adjacency k_i k_j/(NC).
```

- [ ] **Step 6: Re-run the test (docstring change must not break it)**

Run: `uv run pytest tests/test_find_empirical_mu_c.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add glv/analysis.py tests/test_find_empirical_mu_c.py
git commit -m "feat(analysis): make find_empirical_mu_c adjacency-agnostic

_make_W now scales the adjacency's stored values by the disordered
weight instead of replacing them. Binary adjacency is unchanged;
a weighted annealed adjacency k_i k_j/(NC) now yields the correct
annealed interaction matrix.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: Add the `USE_ANNEALED` switch to `final_time.ipynb`

**Files:**
- Modify: `notebooks/final_time.ipynb` — cell `8ef9309d` (parameters + adjacency build) and cell `e90c4f7b` (`_dist_worker`).

Edit notebook cells with the `NotebookEdit` tool (do not hand-edit the JSON). Cell indices below are 0-based: cell 0 = imports, cell 1 = `8ef9309d`, cell 2 = `e90c4f7b`, cell 3 = `dist-plot-cell`.

- [ ] **Step 1: Add the flag and branch the adjacency build in cell 1**

Cell 1 currently is:

```python
# --- Parameters ---
N = 1000
C =25
sigma = 0.2
tau_max = 1e6
n_mu = 40
n_reps = 10

# --- Exponential degree sequence ---
degree_sequence = np.maximum(np.random.exponential(scale=C, size=N).astype(int), 1)
if np.sum(degree_sequence) % 2 != 0:
    degree_sequence[0] += 1

# --- Theoretical mu_c ---
nu_pdf = lambda g: np.exp(-g)
mu_c = glv.calculate_mu_c(sigma=sigma, gamma=0.0, nu_pdf=nu_pdf)["mu_c"]
print(f"mu_c (theoretical) = {mu_c:.4f}")

# --- Build adjacency matrix ---
G = nx.Graph(nx.configuration_model(list(degree_sequence)))
G.remove_edges_from(nx.selfloop_edges(G))
A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)

# --- Initial conditions ---
initial_conditions = []
for _ in range(n_reps):
    x_initial = np.random.uniform(0.1, 1.0, N)
    M_0 = np.sum(x_initial)
    y_0 = x_initial / M_0
    initial_conditions.append(np.concatenate((y_0, [M_0], [0.0])))

# --- Find empirical mu_c ---
emp_result = glv.find_empirical_mu_c(
    mu_c_theoretical=mu_c,
    A=A,
    C=C,
    sigma=sigma,
    initial_conditions=initial_conditions,
    n_mu=n_mu,
    tau_max=tau_max,
    n_workers=4,
)
mu_c_emp = emp_result["mu_c"]
print(f"mu_c (empirical)    = {mu_c_emp:.4f}")
```

Use `NotebookEdit` with `cell_id` `8ef9309d` and `edit_mode` `replace` to set its source to:

```python
# --- Parameters ---
N = 1000
C =25
sigma = 0.2
tau_max = 1e6
n_mu = 40
n_reps = 10
USE_ANNEALED = False  # True → annealed approximation, False → true graph

# --- Exponential degree sequence ---
degree_sequence = np.maximum(np.random.exponential(scale=C, size=N).astype(int), 1)
if np.sum(degree_sequence) % 2 != 0:
    degree_sequence[0] += 1

# --- Theoretical mu_c ---
nu_pdf = lambda g: np.exp(-g)
mu_c = glv.calculate_mu_c(sigma=sigma, gamma=0.0, nu_pdf=nu_pdf)["mu_c"]
print(f"mu_c (theoretical) = {mu_c:.4f}")

# --- Build adjacency matrix ---
if USE_ANNEALED:
    # Annealed approximation: A_ij = k_i k_j / (N C). Dense; diagonal kept.
    A = np.outer(degree_sequence, degree_sequence) / (N * C)
else:
    # True configuration-model graph: binary adjacency, self-loops removed.
    G = nx.Graph(nx.configuration_model(list(degree_sequence)))
    G.remove_edges_from(nx.selfloop_edges(G))
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)

# --- Initial conditions ---
initial_conditions = []
for _ in range(n_reps):
    x_initial = np.random.uniform(0.1, 1.0, N)
    M_0 = np.sum(x_initial)
    y_0 = x_initial / M_0
    initial_conditions.append(np.concatenate((y_0, [M_0], [0.0])))

# --- Find empirical mu_c ---
emp_result = glv.find_empirical_mu_c(
    mu_c_theoretical=mu_c,
    A=A,
    C=C,
    sigma=sigma,
    initial_conditions=initial_conditions,
    n_mu=n_mu,
    tau_max=tau_max,
    n_workers=4,
)
mu_c_emp = emp_result["mu_c"]
print(f"mu_c (empirical)    = {mu_c_emp:.4f}")
```

- [ ] **Step 2: Branch the adjacency build inside `_dist_worker` (cell 2)**

Cell 2 (`e90c4f7b`) contains the `_dist_worker` function. Its adjacency-building block currently reads:

```python
    ds = np.maximum(rng.exponential(scale=C, size=N).astype(int), 1)
    if np.sum(ds) % 2 != 0:
        ds[0] += 1

    G_i = nx.Graph(nx.configuration_model(list(ds)))
    G_i.remove_edges_from(nx.selfloop_edges(G_i))
    A_i = nx.to_scipy_sparse_array(G_i, format="csr", dtype=float)
```

Use `NotebookEdit` with `cell_id` `e90c4f7b` and `edit_mode` `replace`, changing **only** that block to:

```python
    ds = np.maximum(rng.exponential(scale=C, size=N).astype(int), 1)
    if np.sum(ds) % 2 != 0:
        ds[0] += 1

    if USE_ANNEALED:
        # Annealed approximation: A_ij = k_i k_j / (N C). Dense; diagonal kept.
        A_i = np.outer(ds, ds) / (N * C)
    else:
        # True configuration-model graph: binary adjacency, self-loops removed.
        G_i = nx.Graph(nx.configuration_model(list(ds)))
        G_i.remove_edges_from(nx.selfloop_edges(G_i))
        A_i = nx.to_scipy_sparse_array(G_i, format="csr", dtype=float)
```

Leave every other line of cell 2 (parameters, the `glv.find_empirical_mu_c` call, the multiprocessing block) exactly as it is. The cell uses `mp.get_context('fork')`, so `USE_ANNEALED` is inherited by worker processes as a module global — no extra wiring needed.

- [ ] **Step 3: Verify both branches end-to-end with a quick script**

This confirms the annealed dense adjacency flows through `find_empirical_mu_c` without executing the full (slow) notebook. Run:

```bash
uv run python -c "
import numpy as np, networkx as nx, glv

N, C, sigma = 80, 6.0, 0.2
ds = np.maximum(np.random.default_rng(0).exponential(scale=C, size=N).astype(int), 1)
if ds.sum() % 2: ds[0] += 1

# annealed branch
A_ann = np.outer(ds, ds) / (N * C)
assert A_ann.shape == (N, N) and A_ann.ndim == 2, 'annealed A must be dense NxN'

# true-graph branch
G = nx.Graph(nx.configuration_model(list(ds)))
G.remove_edges_from(nx.selfloop_edges(G))
A_true = nx.to_scipy_sparse_array(G, format='csr', dtype=float)
assert set(np.unique(A_true.data)) <= {1.0}, 'true-graph A must be binary'

ics = [np.concatenate((np.full(N, 1.0/N), [1.0], [0.0]))]
for label, A in [('true', A_true), ('annealed', A_ann)]:
    res = glv.find_empirical_mu_c(
        mu_c_theoretical=0.5, A=A, C=C, sigma=sigma,
        initial_conditions=ics, n_mu=8, tau_max=1e3, n_workers=1,
    )
    assert np.isfinite(res['mu_c']), f'{label}: mu_c not finite'
    print(f'{label}: mu_c = {res[\"mu_c\"]:.4f}  OK')
print('both branches OK')
"
```

Expected: prints `true: mu_c = ... OK`, `annealed: mu_c = ... OK`, then `both branches OK`, exit code 0. (The `mu_c` values are from a tiny throwaway system and are not meaningful — only that both branches run and return a finite number.)

- [ ] **Step 4: Commit**

```bash
git add notebooks/final_time.ipynb
git commit -m "feat(final_time): add USE_ANNEALED switch for true vs annealed graph

A single flag selects the true configuration-model adjacency or the
annealed adjacency A_ij = k_i k_j/(NC). Applied to both the main fit
and the distribution sweep.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Verification Checklist

- [ ] `uv run pytest tests/ -v` passes.
- [ ] `notebooks/final_time.ipynb` cell 1 has `USE_ANNEALED = False` and an `if USE_ANNEALED:` branch building `A`.
- [ ] `notebooks/final_time.ipynb` cell 2 (`_dist_worker`) has the matching `if USE_ANNEALED:` branch building `A_i`.
- [ ] The Task 2 Step 3 script prints `both branches OK`.
- [ ] `find_empirical_mu_c` with a binary adjacency behaves exactly as before (covered by the Task 1 test: binary `Ws` are the 1.0-scaled reference).
