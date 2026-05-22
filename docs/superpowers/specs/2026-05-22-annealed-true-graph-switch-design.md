# True-graph vs annealed simulation switch — design

## Goal

Let `final_time.ipynb` run its empirical-`mu_c` simulations against either the
true configuration-model graph or the annealed approximation, switched by a
single notebook-level flag.

## Background

`find_empirical_mu_c` ([glv/analysis.py](../../../glv/analysis.py)) is the
workhorse: it sweeps `mu`, builds a weighted matrix `W` per `mu`, integrates
the rescaled GLV system, and fits a tanh to mean final time.

Its internal `_make_W` currently *replaces* the stored values of the adjacency
matrix:

```python
W.data = mu / C + (sigma / np.sqrt(C)) * z
```

This silently assumes `A` is binary — the stored values are treated as a 0/1
mask and discarded.

The annealed approximation ([generate_annealed_matrix](../../../glv/graph.py))
replaces the realized adjacency by its configuration-model expectation
`A_ij = k_i k_j / (N C)`, then applies the same disordered weights
`alpha_ij = mu/C + sigma/sqrt(C) * z_ij`.

## Key insight

If `_make_W` *multiplies* by the stored values instead of replacing them, the
function becomes adjacency-agnostic:

```python
W.data = W.data * (mu / C + (sigma / np.sqrt(C)) * z)
```

- Binary `A` (true graph): `data` is all `1.0` → identical behavior, fully
  backward-compatible.
- Weighted `A` (annealed, `A_ij = k_i k_j/(NC)`): produces the correct annealed
  `W = A · alpha`.

The switch then lives entirely in the notebook: it builds either a binary
sparse adjacency or the dense annealed adjacency and hands it to the
unchanged-signature function.

## Changes

### 1. Library — `glv/analysis.py`

- `_make_W` inside `find_empirical_mu_c`: multiply by `W.data` instead of
  replacing it.
- Update the `find_empirical_mu_c` docstring: `A` is "binary adjacency for the
  true graph, or the weighted annealed adjacency `k_i k_j/(NC)`"; weights are
  now `A_ij * alpha_ij` rather than `alpha_ij` on edges.

No signature change. No new function.

### 2. Notebook — `notebooks/final_time.ipynb`

- Add a `USE_ANNEALED = False` flag in the parameters cell.
- Branch the construction of `A`:
  - **true graph**: existing configuration-model code →
    `nx.to_scipy_sparse_array(...)` (binary, sparse).
  - **annealed**: `A = np.outer(degree_sequence, degree_sequence) / (N * C)`
    (dense — one inline line, no library helper).
- Apply the same branch inside `_dist_worker` (the distribution-sweep cell),
  governed by the same `USE_ANNEALED` flag, so the main fit and the
  distribution sweep switch together.

## Conventions and trade-offs

- **Diagonal**: the annealed adjacency `np.outer(k, k)/(NC)` keeps its diagonal
  (`k_i^2/(NC)` self-term), matching the existing convention in
  `generate_annealed_matrix`. The true-graph adjacency has no diagonal
  (self-loops removed). Intentional, consistent with current code.
- **Shared degree sequence**: both modes use the same `degree_sequence`, giving
  an apples-to-apples comparison. The annealed mode uses `k_i` directly; the
  true-graph mode realizes it (approximately) through the configuration model.
- **Density**: the annealed `A` is dense; `find_empirical_mu_c` converts it to
  CSR internally. For `N = 1000` that is ~1M nonzeros per `W` — workable.
  Larger `N` gets memory-heavy; this is inherent to the annealed approximation,
  not a defect.

## Alternative considered

A `mode="true"|"annealed"` parameter on `find_empirical_mu_c`. Rejected: the
multiply change is strictly more general (no enum, no branching inside the
function) and the user chose a notebook-level switch.

## Out of scope

Other notebooks (`c_sweep_mu_c.ipynb`, `annealed_volatility.ipynb`) — they call
`find_empirical_mu_c` too and benefit automatically from the backward-compatible
`_make_W` change, but their own toggles are not part of this work.
