# Relative-GLV Curated Repo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone, shareable git repo at `~/Code/relative-glv` that explains the relative GLV model via one results-first narrative notebook backed by a small clean Python module.

**Architecture:** A `relative_glv/` package with one clean implementation each of the model (`model.py`), the MSB empirics (`msb.py`), and the DMFT theory (`dmft.py`). A `story.ipynb` renders the explanation from cheap live simulations plus committed heavy datasets in `data/*.npz`; `scripts/compute.py` regenerates those datasets. The model, estimators, and theory are validated by `tests/`. Science-bearing code carries its mathematics in docstrings; plumbing stays comment-light.

**Tech Stack:** Python ≥3.11, numpy, scipy, networkx, matplotlib, joblib; pytest + jupyter (dev). Managed with `uv`. Source material lifted/cleaned from `~/Code/glv/growing_glv/`.

## Global Constraints

- **Target dir:** everything is created under `~/Code/relative-glv/` (its own git repo). The existing `~/Code/glv/` repo is the *source* and is never modified.
- **Run Python via `uv run python`**, never bare `python`/`python3`.
- **Sign convention:** competition-positive — `mu > 0` means mean competition (lowers growth: `g* = g0 - mu`). Same convention in `coupling`, `integrate`, `dmft`. The thesis "chapter mu" flips the sign for presentation; this repo does not.
- **Disorder scaling:** fully-connected `alpha_ij = mu/N + (sigma/sqrt(N)) z`; power-law per-edge mean `mu/C_eff`, std `sigma/sqrt(C_eff)`. Never use raw/unscaled couplings.
- **MSB rescaling:** growth distributions are centred and divided by `sqrt(pi/2)*MAD`, never std (std is inflated by the fat tails). This lives in exactly one place: `msb.rescale`.
- **Math-behind-code:** every science-bearing function states its governing equation / estimator / rationale in its docstring; pure plumbing (plotting, npz I/O, CLI) stays comment-light.
- **License:** MIT.
- **Commit after each task.**

---

### Task 1: Scaffold the repository

**Files:**
- Create: `~/Code/relative-glv/pyproject.toml`
- Create: `~/Code/relative-glv/.gitignore`
- Create: `~/Code/relative-glv/LICENSE`
- Create: `~/Code/relative-glv/relative_glv/__init__.py`
- Create: `~/Code/relative-glv/tests/__init__.py`
- Create: `~/Code/relative-glv/data/.gitkeep`, `~/Code/relative-glv/scripts/.gitkeep`

**Interfaces:**
- Produces: an installable `relative_glv` package importable as `import relative_glv`.

- [ ] **Step 1: Create the directory tree and init git**

```bash
mkdir -p ~/Code/relative-glv/{relative_glv,tests,data,scripts}
cd ~/Code/relative-glv && git init -q
touch data/.gitkeep scripts/.gitkeep
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "relative-glv"
version = "0.1.0"
description = "The relative (scale-invariant) generalised Lotka-Volterra model: a growing economy with MSB firm-growth statistics"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
dependencies = [
    "numpy",
    "scipy",
    "networkx",
    "matplotlib",
    "joblib",
]

[project.optional-dependencies]
dev = ["pytest", "jupyter", "ipykernel"]

[tool.hatch.build.targets.wheel]
packages = ["relative_glv"]
```

- [ ] **Step 3: Write `.gitignore`**

```
__pycache__/
*.pyc
.venv/
.ipynb_checkpoints/
.pytest_cache/
.ruff_cache/
*.egg-info/
.DS_Store
```

Note: `data/*.npz` and `story.ipynb` outputs are intentionally **committed** (they are the heavy figures), so they are not ignored.

- [ ] **Step 4: Write `LICENSE`** (MIT, copyright "2026 Ludovico Furlanetto")

Use the standard MIT license text.

- [ ] **Step 5: Write `relative_glv/__init__.py`**

```python
"""The relative (scale-invariant) generalised Lotka-Volterra model.

A disordered replicator in which interactions act on the mean firm, giving a
self-consistently growing economy whose firm-size dynamics reproduce the
Moran-Santos-Bouchaud (MSB) stylized facts: a symmetric fat-tailed growth-rate
distribution and a size-variance exponent beta ~ 0.15-0.20.
"""
from relative_glv.model import coupling, integrate, growth_rate, survivors
from relative_glv.msb import rescale, size_volatility, tent_stats
from relative_glv.dmft import solve_fixed_point, sigma_c

__all__ = [
    "coupling", "integrate", "growth_rate", "survivors",
    "rescale", "size_volatility", "tent_stats",
    "solve_fixed_point", "sigma_c",
]
```

(`tests/__init__.py` is empty.)

- [ ] **Step 6: Sync and verify the env builds**

Run: `cd ~/Code/relative-glv && uv sync --extra dev`
Expected: resolves and creates `.venv` with no error. (`import relative_glv` will fail until the modules exist — that is fine; later tasks add them. Do NOT import yet.)

- [ ] **Step 7: Commit**

```bash
cd ~/Code/relative-glv
git add -A
git commit -m "chore: scaffold relative-glv package"
```

---

### Task 2: DMFT solver (`dmft.py`)

Lift the already-validated solver verbatim, then enrich the docstrings with the mathematics. This is the easiest module to test (deterministic numeric outputs) so it comes first.

**Files:**
- Create: `~/Code/relative-glv/relative_glv/dmft.py`
- Test: `~/Code/relative-glv/tests/test_dmft_solver.py`

**Interfaces:**
- Produces:
  - `sigma_c(gamma=0.0) -> float` (returns `sqrt(2)` at `gamma=0`)
  - `solve_fixed_point(mu, sigma, gamma=0.0) -> dict` with keys
    `mu, sigma, gamma, delta, q, sqrtq, v, phi, chi, g0, gstar, stable`.

- [ ] **Step 1: Write the failing test**

`tests/test_dmft_solver.py`:

```python
import numpy as np
from relative_glv.dmft import solve_fixed_point, sigma_c


def test_sigma_c_is_sqrt2_at_gamma0():
    assert abs(sigma_c(0.0) - np.sqrt(2.0)) < 1e-9


def test_mu_is_a_uniform_growth_shift():
    # mu cancels in the replicator's relative dynamics: shape (phi, q, delta) is
    # mu-independent and g* = g0 - mu.
    a = solve_fixed_point(mu=0.0, sigma=1.0)
    b = solve_fixed_point(mu=1.0, sigma=1.0)
    assert abs(a["phi"] - b["phi"]) < 1e-9
    assert abs(a["q"] - b["q"]) < 1e-9
    assert abs((b["gstar"] - a["gstar"]) - (-1.0)) < 1e-9


def test_survival_decreases_with_disorder():
    # more disorder -> fewer survivors (phi falls), in the relaxed phase
    phis = [solve_fixed_point(0.5, s)["phi"] for s in (0.2, 0.6, 1.0, 1.3)]
    assert all(x > y for x, y in zip(phis, phis[1:]))
    assert solve_fixed_point(0.5, 1.0)["stable"] is True
    assert solve_fixed_point(0.5, 2.0)["stable"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Code/relative-glv && uv run pytest tests/test_dmft_solver.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'relative_glv.dmft'`.

- [ ] **Step 3: Create `dmft.py` by copying the source, then editing docstrings**

Copy `~/Code/glv/growing_glv/dmft_solver.py` verbatim to `~/Code/relative-glv/relative_glv/dmft.py`. Then replace the module docstring (lines 1-11 of the source) with:

```python
"""Relative-GLV DMFT fixed-point solver (relaxed / single-fixed-point phase).

The relative GLV is a disordered replicator. In the high-connectivity limit its
dynamics reduce to a single-site stochastic process whose stationary statistics
obey self-consistency equations for the survival fraction phi, the overlap q,
the response chi, and the growth rate g*. With Gaussian couplings of mean mu and
variance sigma^2 (gamma the symmetric/antisymmetric correlation), the survivors'
rescaled abundances are a clipped Gaussian, x* = (sqrt(q) sigma / v) max(Delta+z, 0),
with Delta the clipping threshold and v = 1 - gamma sigma^2 chi the renormalised
self-interaction. The Gaussian moments

    w_n(Delta) = int_{-Delta}^{inf} Dz (Delta+z)^n,    Dz = e^{-z^2/2} dz / sqrt(2 pi)

are known in closed form (w0=Phi, w1=Delta Phi + phi_pdf, w2=(1+Delta^2)Phi + Delta phi_pdf),
which collapses the self-consistency to a single scalar root for Delta:

    v^2 = sigma^2 w2(Delta),   chi = w0(Delta)/v,   normalisation M_1 = (sigma sqrt(q)/v) w1 = 1.

The mean competition mu is a *uniform* shift of every fitness, so it cancels from
the replicator's relative dynamics: the shape observables (Delta, q, chi, phi) are
mu-independent and mu only lowers the growth rate, g* = g0 - mu. At gamma=0 the
relaxed<->fluctuating boundary is mu-independent at sigma_c = sqrt(2) (set by
w0(Delta)=w2(Delta) -> Delta=0, v(0)^2 = sigma_c^2 / 2).

Full derivation: glv/docs/superpowers/specs/2026-06-21-relative-glv-dmft-derivation.md
"""
```

Keep every function (`w0, w1, w2, _v_of_delta, sigma_c, solve_fixed_point`) and the `__main__` block exactly as in the source. Add a one-line math note to `sigma_c`'s docstring and `solve_fixed_point`'s docstring only if not already present (the source already documents them adequately).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Code/relative-glv && uv run pytest tests/test_dmft_solver.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Verify the module self-check**

Run: `cd ~/Code/relative-glv && uv run python -m relative_glv.dmft`
Expected: prints `sigma_c(gamma=0) = 1.414214  (= sqrt(2) = 1.414214)` and a small table.

- [ ] **Step 6: Commit**

```bash
cd ~/Code/relative-glv
git add relative_glv/dmft.py tests/test_dmft_solver.py
git commit -m "feat: DMFT fixed-point solver (sigma_c=sqrt2, mu-shift)"
```

---

### Task 3: Coupling matrices (`model.py` part 1)

**Files:**
- Create: `~/Code/relative-glv/relative_glv/model.py`
- Test: `~/Code/relative-glv/tests/test_model.py`

**Interfaces:**
- Produces: `coupling(N, mu, sigma, *, kind="fc", gamma=0.0, seed=0)` returning a dense
  `ndarray` (kind="fc") or a `scipy.sparse.csr_array` (kind="powerlaw"); both support `alpha @ w`.

- [ ] **Step 1: Write the failing test**

`tests/test_model.py`:

```python
import numpy as np
from scipy import sparse
from relative_glv.model import coupling


def test_fc_shape_diag_and_scaling():
    N, mu, sigma = 400, 1.0, 0.8
    a = coupling(N, mu, sigma, kind="fc", seed=0)
    assert a.shape == (N, N)
    assert np.allclose(np.diag(a), 0.0)            # zero self-interaction
    off = a[~np.eye(N, dtype=bool)]
    assert abs(off.mean() - mu / N) < 5e-4          # mean mu/N
    assert abs(off.std() - sigma / np.sqrt(N)) < 5e-3   # std sigma/sqrt(N)


def test_fc_is_deterministic_in_seed():
    a = coupling(50, 0.5, 1.0, kind="fc", seed=7)
    b = coupling(50, 0.5, 1.0, kind="fc", seed=7)
    assert np.array_equal(a, b)


def test_powerlaw_is_sparse_square_zero_diagonal():
    N = 600
    a = coupling(N, 1.0, 1.5, kind="powerlaw", seed=1)
    assert sparse.issparse(a)
    assert a.shape == (N, N)
    assert np.allclose(a.diagonal(), 0.0)           # config model removes self-loops
    assert a.nnz > 0


def test_unknown_kind_raises():
    import pytest
    with pytest.raises(ValueError):
        coupling(10, 0.0, 1.0, kind="banana")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Code/relative-glv && uv run pytest tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'relative_glv.model'`.

- [ ] **Step 3: Write `model.py` (coupling only)**

```python
"""The relative (scale-invariant) generalised Lotka-Volterra model.

    x_i' = x_i [ 1 - x_i/m - (alpha x)_i / m ],   m = <x> = M/N.

Interactions act on the MEAN firm m, so the competition term is O(M) not O(M^2):
the aggregate M grows exponentially in physical time with no finite-time blow-up.
We integrate in a share + log-scale split (shares w_i = x_i/M on the simplex,
ln M separate), which is numerically unconditionally stable -- see `integrate`.

Sign convention: mu > 0 is mean COMPETITION (it lowers the growth rate).
"""
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.integrate import solve_ivp

_ALPHA_PL, _MEAN_DEGREE = 2.5, 100   # power-law degree exponent, target mean degree


def coupling(N, mu, sigma, *, kind="fc", gamma=0.0, seed=0):
    """Interaction matrix alpha for the relative GLV.

    Couplings are scaled so the mean field stays finite as connectivity grows:
    per-interaction mean mu/C and std sigma/sqrt(C), with C the connectivity
    (C = N fully connected, C = mean degree on the graph). gamma sets the
    symmetric/antisymmetric correlation of the reciprocal pair via
    alpha ~ sqrt(1+gamma) S +/- sqrt(1-gamma) V, S=(M+M^T)/sqrt2, V=(M-M^T)/sqrt2.

    kind="fc":       fully-connected Gaussian, alpha_ij = mu/N + (sigma/sqrt N) z_ij,
                     zero diagonal. The clean ensemble the DMFT is derived for.
    kind="powerlaw": power-law configuration-model graph (exponent 2.5, mean
                     degree ~100), Roy disorder per edge. The realistic ensemble.

    Returns a dense ndarray (fc) or a scipy.sparse csr_array (powerlaw); both
    support `alpha @ w`.
    """
    rng = np.random.default_rng(seed)
    if kind == "fc":
        z = rng.standard_normal((N, N))
        a = mu / N + (sigma / np.sqrt(N)) * z
        np.fill_diagonal(a, 0.0)
        return a
    if kind == "powerlaw":
        C = min(_MEAN_DEGREE, N - 1)
        kmin = C * (_ALPHA_PL - 2) / (_ALPHA_PL - 1)
        deg = np.maximum(
            (kmin * (1 - rng.uniform(size=N)) ** (-1 / (_ALPHA_PL - 1))).round().astype(int), 1)
        if deg.sum() % 2:
            deg[deg.argmin()] += 1
        G = nx.Graph(nx.configuration_model(deg.tolist(), seed=int(seed)))
        G.remove_edges_from(nx.selfloop_edges(G))
        A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
        A.data[:] = 1.0
        C_eff = float(np.asarray(A.sum(axis=1)).mean())
        Au = sparse.triu(A, k=1).tocoo()
        ei, ej, nE = Au.row, Au.col, Au.row.size
        a, b = rng.normal(0, 1, nE), rng.normal(0, 1, nE)
        sym, anti = (a + b) / np.sqrt(2), (a - b) / np.sqrt(2)
        scale = sigma / np.sqrt(2 * C_eff)
        w_ij = mu / C_eff + scale * (np.sqrt(1 + gamma) * sym + np.sqrt(1 - gamma) * anti)
        w_ji = mu / C_eff + scale * (np.sqrt(1 + gamma) * sym - np.sqrt(1 - gamma) * anti)
        rows = np.concatenate([ei, ej])
        cols = np.concatenate([ej, ei])
        return sparse.csr_array((np.concatenate([w_ij, w_ji]), (rows, cols)), shape=(N, N))
    raise ValueError(f"unknown kind {kind!r} (expected 'fc' or 'powerlaw')")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Code/relative-glv && uv run pytest tests/test_model.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
cd ~/Code/relative-glv
git add relative_glv/model.py tests/test_model.py
git commit -m "feat: coupling() for fc and power-law ensembles"
```

---

### Task 4: Integrator and helpers (`model.py` part 2)

**Files:**
- Modify: `~/Code/relative-glv/relative_glv/model.py` (append `integrate`, `growth_rate`, `survivors`)
- Test: `~/Code/relative-glv/tests/test_model.py` (append)

**Interfaces:**
- Consumes: `coupling` (Task 3).
- Produces:
  - `integrate(alpha, *, tmax, n_eval=1500, lam=0.0, seed=0, method="LSODA", rtol=1e-7, atol=1e-9) -> dict(t, W, lnM, success)` where `W` has shape `(N, n_eval)` and each column sums to 1.
  - `growth_rate(t, lnM, frac=0.5) -> float`
  - `survivors(W, floor=1e-6) -> np.ndarray[bool]` (length N)

- [ ] **Step 1: Write the failing test (append to `tests/test_model.py`)**

```python
from relative_glv.model import integrate, growth_rate, survivors


def test_integrate_keeps_shares_on_the_simplex():
    a = coupling(200, 1.0, 0.8, kind="fc", seed=0)
    r = integrate(a, tmax=40.0, n_eval=200, seed=0)
    assert r["success"]
    assert r["W"].shape == (200, 200)
    assert np.allclose(r["W"].sum(axis=0), 1.0, atol=1e-8)   # simplex preserved
    assert np.isfinite(r["lnM"]).all()                        # ln M never overflows


def test_relaxed_economy_grows_at_negative_mu():
    # weak competition (mu small) in the relaxed phase -> aggregate grows: g_eff > 0
    a = coupling(300, 0.0, 0.8, kind="fc", seed=1)
    r = integrate(a, tmax=60.0, n_eval=300, seed=1)
    assert growth_rate(r["t"], r["lnM"]) > 0.0


def test_survivors_mask_length_and_dtype():
    a = coupling(150, 1.0, 0.8, kind="fc", seed=2)
    r = integrate(a, tmax=30.0, n_eval=150, seed=2)
    m = survivors(r["W"])
    assert m.shape == (150,) and m.dtype == bool
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Code/relative-glv && uv run pytest tests/test_model.py -k "integrate or relaxed or survivors_mask" -v`
Expected: FAIL — `ImportError: cannot import name 'integrate'`.

- [ ] **Step 3: Append `integrate` and helpers to `model.py`**

```python
def integrate(alpha, *, tmax, n_eval=1500, lam=0.0, seed=0,
              method="LSODA", rtol=1e-7, atol=1e-9):
    """Integrate the relative GLV in the share + log-scale split.

    Writing x_i = M w_i with w on the simplex (sum_i w_i = 1) and splitting off
    the scale gives a replicator for the shares plus a linear equation for ln M:

        f_i      = 1 - N w_i - N (alpha w)_i           (per-firm fitness, N = number of firms)
        <f>      = sum_i w_i f_i = g_eff               (aggregate growth rate)
        dw_i/dt  = w_i (f_i - <f>) + lam (1/N - w_i)   (replicator + optional share floor)
        d lnM/dt = g_eff

    The absolute abundance x_i = M w_i is NEVER formed, so nothing overflows even
    though M grows exponentially: w stays on the simplex and ln M grows linearly.
    The optional lam term is a persistent, mass-conserving immigration floor that
    fights condensation onto a single firm.

    Returns dict(t, W, lnM, success): W shape (N, n_eval), columns renormalised
    to the simplex; lnM the log aggregate (lnM(0)=0).
    """
    N = alpha.shape[0]
    rng = np.random.default_rng(seed)
    w0 = rng.uniform(0.5, 1.5, N)
    w0 /= w0.sum()

    def rhs(t, state):
        w = np.clip(state[:N], 0.0, None)
        s = w.sum()
        if s > 0:
            w = w / s
        f = 1.0 - N * w - N * (alpha @ w)
        fbar = float(w @ f)
        dw = w * (f - fbar) + lam * (1.0 / N - w)
        return np.concatenate([dw, [fbar]])

    r = solve_ivp(rhs, (0.0, tmax), np.concatenate([w0, [0.0]]), method=method,
                  t_eval=np.linspace(0.0, tmax, n_eval), rtol=rtol, atol=atol)
    W = np.clip(r.y[:N], 0.0, None)
    W = W / W.sum(0, keepdims=True)
    return dict(t=r.t, W=W, lnM=r.y[N], success=bool(r.success))


def growth_rate(t, lnM, frac=0.5):
    """Aggregate growth rate g_eff = d lnM/dt, as the slope of lnM over the late
    fraction `frac` of the trajectory (after the initial-condition transient)."""
    late = t > (1.0 - frac) * t[-1]
    return float(np.polyfit(t[late], lnM[late], 1)[0])


def survivors(W, floor=1e-6):
    """Boolean mask of firms whose share stays above `floor` across all of W
    (relative size S_i = N w_i, so floor is a fraction of the average firm)."""
    return W.min(axis=1) > floor
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Code/relative-glv && uv run pytest tests/test_model.py -v`
Expected: PASS (7 passed total).

- [ ] **Step 5: Commit**

```bash
cd ~/Code/relative-glv
git add relative_glv/model.py tests/test_model.py
git commit -m "feat: integrate() share+lnM split, growth_rate, survivors"
```

---

### Task 5: MSB estimators (`msb.py`)

**Files:**
- Create: `~/Code/relative-glv/relative_glv/msb.py`
- Test: `~/Code/relative-glv/tests/test_msb.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (operates on arrays).
- Produces:
  - `rescale(g) -> np.ndarray` — centred, divided by `sqrt(pi/2)*MAD`.
  - `size_volatility(W, t, *, window, dt, n_bins=20) -> dict(Sbar, vol, beta, r2, bin_S, bin_vol, pk, growth)`.
  - `tent_stats(g) -> dict(bowley, exkurt)`.

- [ ] **Step 1: Write the failing test**

`tests/test_msb.py`:

```python
import numpy as np
from relative_glv.msb import rescale, size_volatility, tent_stats


def test_rescale_unit_scale_on_gaussian():
    # for a Gaussian, sqrt(pi/2)*MAD == std, so rescaled data has std ~ 1, mean ~ 0
    g = np.random.default_rng(0).standard_normal(200_000)
    z = rescale(g)
    assert abs(z.mean()) < 0.02
    assert abs(z.std() - 1.0) < 0.02


def test_rescale_drops_nonfinite():
    z = rescale(np.array([1.0, 2.0, np.nan, np.inf, -1.0]))
    assert np.isfinite(z).all()


def test_tent_stats_laplace_is_symmetric_and_fat():
    g = np.random.default_rng(1).laplace(size=200_000)
    s = tent_stats(g)
    assert abs(s["bowley"]) < 0.03        # symmetric
    assert s["exkurt"] > 2.0              # fat (Laplace excess kurtosis = 3)


def test_size_volatility_recovers_a_planted_exponent():
    # plant sigma(S) ~ S^-beta with beta=0.3 and check the estimator recovers it.
    # Each firm fluctuates (stationary) around a fixed base size with log-noise
    # amplitude base^-beta; converting to shares preserves relative sizes, so
    # S_i = N w_i restores the planted sizes and the decline slope is beta.
    rng = np.random.default_rng(2)
    N, T, beta_true = 4000, 80, 0.3
    base = np.logspace(0.5, 3, N)                         # firm sizes over ~2.5 decades
    amp = 0.5 * base ** (-beta_true)                      # planted per-firm log-volatility
    lnS = np.log(base)[:, None] + amp[:, None] * rng.standard_normal((N, T))
    W = np.exp(lnS); W /= W.sum(0, keepdims=True)         # -> shares; S_i = N w_i restores size
    t = np.linspace(0, T - 1, T)
    out = size_volatility(W, t, window=(0, T - 1), dt=1.0)
    assert abs(out["beta"] - beta_true) < 0.08
    assert out["r2"] > 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Code/relative-glv && uv run pytest tests/test_msb.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'relative_glv.msb'`.

- [ ] **Step 3: Write `msb.py`**

`_decline_beta` is lifted from `~/Code/glv/growing_glv/ch4_data.py` (the `_decline_beta` function, lines 42-59), parameterised by `n_bins`.

```python
"""Moran-Santos-Bouchaud (MSB) firm-growth statistics of the relative GLV.

The MSB stylized facts this model reproduces:
  (1) growth rates g_i = Delta ln S_i (S_i = N w_i the relative firm size) follow
      a symmetric, fat-tailed, tent-shaped distribution -- closer to a Laplace
      than a Gaussian;
  (2) the volatility of growth falls with firm size as a power law,
      sigma(S) ~ S^-beta, with the empirical exponent beta ~ 0.15-0.20.

Because the aggregate M divides out of the shares, g_i = Delta ln S_i is purely
idiosyncratic (the common growth g_eff cancels), which is exactly the quantity
the firm-growth literature measures.
"""
import numpy as np
from scipy.stats import kurtosis


def rescale(g):
    """Centre and rescale growth rates by sqrt(pi/2)*MAD, NOT the std.

    The std is inflated by the fat tails and would mis-normalise the tent;
    sqrt(pi/2)*MAD equals the std for a Gaussian but is robust to heavy tails, so
    a Gaussian maps to unit variance while the model's excess kurtosis stays
    visible against the N(0,1) / Laplace reference curves.
    """
    g = np.asarray(g, float).ravel()
    g = g[np.isfinite(g)]
    scale = np.sqrt(np.pi / 2) * np.abs(g - g.mean()).mean()
    return (g - g.mean()) / (scale if scale > 0 else 1.0)


def _decline_beta(Sbar, vol, n_bins=20):
    """sigma(S) ~ S^-beta on the large-S DECLINE branch.

    Bin firms by mean size, take the median volatility per bin, find the peak,
    and least-squares fit the log-log slope from the peak rightward (the small-S
    side is a floor/plateau artefact and is excluded). Returns
    (beta, r2, bin_S, bin_vol, peak_index).
    """
    live = (vol > 0) & np.isfinite(vol) & (Sbar > 0)
    Sbar, vol = Sbar[live], vol[live]
    if Sbar.size < 50:
        return np.nan, np.nan, np.array([]), np.array([]), 0
    o = np.argsort(Sbar)
    P = np.array_split(np.arange(o.size), n_bins)
    bx = np.array([Sbar[o][p].mean() for p in P])
    by = np.array([np.median(vol[o][p]) for p in P])
    m = (bx > 0) & (by > 0)
    bx, by = bx[m], by[m]
    if bx.size <= 5:
        return np.nan, np.nan, bx, by, 0
    pk = int(np.argmax(by))
    if by.size - pk < 5:
        return np.nan, np.nan, bx, by, pk
    c = np.polyfit(np.log10(bx[pk:]), np.log10(by[pk:]), 1)
    yh = np.polyval(c, np.log10(bx[pk:]))
    lo = np.log10(by[pk:])
    ss = np.sum((lo - lo.mean()) ** 2)
    r2 = float(1 - np.sum((lo - yh) ** 2) / ss) if ss > 0 else np.nan
    return float(-c[0]), r2, bx, by, pk


def size_volatility(W, t, *, window, dt, n_bins=20):
    """Size-volatility relation sigma(S) ~ S^-beta from a shares trajectory.

    Relative sizes S_i = N w_i are regridded onto a fixed Delta-t = dt log grid
    over `window` (a fixed grid is required so volatility is not aliased by the
    integrator's adaptive steps). Per firm: mean size Sbar_i and MAD-volatility
    sigma_i = sqrt(pi/2) * mean|g - <g>|, g = Delta ln S_i. beta is the
    decline-branch slope (see _decline_beta). Only firms above the survival floor
    across the whole window contribute. Returns dict(Sbar, vol, beta, r2, bin_S,
    bin_vol, pk, growth).
    """
    N = W.shape[0]
    win = (t >= window[0]) & (t <= window[1])
    live = W[:, win].min(1) > 1e-6
    tg = np.arange(window[0], window[1] + 1e-9, dt)
    lnS = np.array([np.interp(tg, t, np.log(np.maximum(N * W[i], 1e-12)))
                    for i in np.where(live)[0]])
    g = np.diff(lnS, axis=1)
    Sbar = np.exp(lnS).mean(1)
    vol = np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)
    beta, r2, bx, by, pk = _decline_beta(Sbar, vol, n_bins)
    return dict(Sbar=Sbar, vol=vol, beta=beta, r2=r2,
                bin_S=bx, bin_vol=by, pk=pk, growth=g)


def tent_stats(g):
    """Shape of the (rescaled) growth-rate distribution: Bowley skewness
    (quartile-based, robust) and excess kurtosis. A symmetric fat tent has
    bowley ~ 0 and excess kurtosis well above 0 (Laplace = 3, Gaussian = 0).
    """
    z = rescale(g)
    q1, q2, q3 = np.percentile(z, [25, 50, 75])
    bowley = (q3 + q1 - 2 * q2) / (q3 - q1)
    return dict(bowley=float(bowley), exkurt=float(kurtosis(z)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Code/relative-glv && uv run pytest tests/test_msb.py -v`
Expected: PASS (4 passed). If `test_size_volatility_recovers_a_planted_exponent` is marginal, it is a stochastic test — re-run once; it should pass comfortably with the planted beta=0.3.

- [ ] **Step 5: Commit**

```bash
cd ~/Code/relative-glv
git add relative_glv/msb.py tests/test_msb.py
git commit -m "feat: MSB estimators (rescale, size_volatility, tent_stats)"
```

---

### Task 6: DMFT-vs-simulation validation test (`tests/test_validation.py`)

The scientific integration test: the model (`coupling` + `integrate`) reproduces the DMFT predictions in the relaxed phase, the chaos turns on near `sigma_c=sqrt(2)`, and `mu` is a pure growth shift. Kept small enough for CI.

**Files:**
- Create: `~/Code/relative-glv/tests/test_validation.py`

**Interfaces:**
- Consumes: `coupling`, `integrate` (Tasks 3-4), `solve_fixed_point`, `sigma_c` (Task 2).

- [ ] **Step 1: Write the test**

Logic adapted from `~/Code/glv/growing_glv/dmft_validate.py` (the `simulate` + assertions), shrunk to `N=800`, 2 seeds, fully-connected matched ensemble, `lam=0`.

```python
import numpy as np
from relative_glv.model import coupling, integrate
from relative_glv.dmft import solve_fixed_point, sigma_c

N, TMAX, LATE = 800, 140.0, (90.0, 130.0)
SEEDS = (101, 202)


def _sim(mu, sigma):
    """Matched fully-connected sim: return (g_eff, survival, fluctuation), seed-averaged."""
    out = []
    for sd in SEEDS:
        a = coupling(N, mu, sigma, kind="fc", seed=sd)
        r = integrate(a, tmax=TMAX, n_eval=700, lam=0.0, seed=sd, method="LSODA")
        t, W, lnM = r["t"], r["W"], r["lnM"]
        late = t > LATE[0]
        g_eff = float(np.polyfit(t[late], lnM[late], 1)[0])
        win = (t >= LATE[0]) & (t <= LATE[1])
        surv_mask = W[:, win].min(1) > 1e-6
        surv = float(surv_mask.mean())
        if surv_mask.sum() > 20:
            tg = np.arange(LATE[0], LATE[1] + 1e-9, 0.5)
            lnS = np.array([np.interp(tg, t, np.log(np.maximum(N * W[i], 1e-12)))
                            for i in np.where(surv_mask)[0]])
            fluct = float(np.diff(lnS, axis=1).std())
        else:
            fluct = 0.0
        out.append((g_eff, surv, fluct))
    return np.mean(out, axis=0)


def test_relaxed_phase_matches_dmft():
    sc = sigma_c(0.0)
    errs_g, errs_s = [], []
    for sigma in (0.7, 1.0):
        ge, su, _ = _sim(0.5, sigma)
        d = solve_fixed_point(0.5, sigma)
        assert sigma < sc
        errs_g.append(abs(ge - d["gstar"]))
        errs_s.append(abs(su - d["phi"]))
    assert np.mean(errs_g) < 0.12, "relaxed-phase growth rate disagrees with DMFT"
    assert np.mean(errs_s) < 0.10, "relaxed-phase survival disagrees with DMFT"


def test_chaos_turns_on_near_sigma_c():
    _, _, f_lo = _sim(0.5, 0.7)     # below sqrt(2): relaxed
    _, _, f_hi = _sim(0.5, 2.0)     # above sqrt(2): fluctuating
    assert f_lo < 0.01 and f_hi > 0.02, "chaos onset not near sigma_c=sqrt(2)"


def test_mu_is_a_pure_growth_shift():
    g0, s0, _ = _sim(0.0, 1.0)
    g1, s1, _ = _sim(1.0, 1.0)
    assert abs(s1 - s0) < 0.05, "survival should be mu-independent"
    assert abs((g1 - g0) - (-1.0)) < 0.10, "g_eff should shift by -delta_mu"
```

- [ ] **Step 2: Run the test**

Run: `cd ~/Code/relative-glv && uv run pytest tests/test_validation.py -v`
Expected: PASS (3 passed). Runtime a few minutes (it integrates ~14 economies at N=800). If a relaxed-phase tolerance is marginally exceeded, raise `N` to 1000 or add a third seed — do not loosen the asserted physics tolerances below the dmft_validate originals (0.12 / 0.10).

- [ ] **Step 3: Commit**

```bash
cd ~/Code/relative-glv
git add tests/test_validation.py
git commit -m "test: DMFT-vs-simulation validation (relaxed match, chaos onset, mu-shift)"
```

---

### Task 7: Heavy-data compute script (`scripts/compute.py`)

Generates the three committed datasets the notebook embeds. Ported from `ch4_data.py`, `phase_diagram.py`, and `dmft_validate.py`, rewired onto the `relative_glv` package and writing into `data/`. A `--smoke` flag runs a tiny version for verification.

**Files:**
- Create: `~/Code/relative-glv/scripts/compute.py`
- Produces (committed): `data/msb.npz`, `data/phase_diagram.npz`, `data/dmft_validation.npz`

**Interfaces:**
- Consumes: `relative_glv.model` (coupling/integrate), `relative_glv.msb` (size_volatility/rescale/tent_stats), `relative_glv.dmft`.

- [ ] **Step 1: Write `scripts/compute.py`**

Structure (the heavy MSB block ports `ch4_data.py`'s `simulate` + `__main__`, using `coupling(kind="powerlaw")` + `integrate` + `msb.size_volatility`; the phase block ports `phase_diagram.py`; the dmft block ports `dmft_validate.py`'s sweep). Locked regime from `ch4_data.py`: `SIGMA, MU, LAM = 1.75, 1.76, 1e-3`. Use `joblib.Parallel` for the seed/N loops.

```python
"""Regenerate the heavy datasets the story notebook embeds:

  data/msb.npz            MSB at the locked regime (sigma=1.75, mu=1.76, power-law):
                          single-run churn, size-volatility + tent, beta-vs-N + extrapolation,
                          freeze-vs-persist trajectories.
  data/phase_diagram.npz  (mu, sigma) grid of growth rate g_eff and chaos amplitude.
  data/dmft_validation.npz  matched fully-connected sim vs DMFT (growth, survival, chaos onset).

Dataset-oriented: compute once, store raw-enough results, replot freely in the notebook.

    uv run python scripts/compute.py [--smoke]
"""
import os, sys, time
import numpy as np
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from relative_glv.model import coupling, integrate
from relative_glv import msb
from relative_glv.dmft import solve_fixed_point, sigma_c

SMOKE = "--smoke" in sys.argv
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SIGMA, MU, LAM = 1.75, 1.76, 1e-3
N_JOBS = 6
# ... (full parameter block + the three compute functions, ported as described) ...
```

Implement three functions and call them under `__main__`:

1. `compute_msb()` — port `ch4_data.simulate` but call `coupling(N, MU, SIGMA, kind="powerlaw", seed=seed)` then `integrate(a, tmax=400, n_eval=1600, lam=LAM, seed=seed, method="RK45", rtol=1e-4, atol=1e-7)`, and use `msb.size_volatility(W, t, window=(320,390), dt=0.5)` for beta and `msb.rescale`/`msb.tent_stats` for the tent. Save the same arrays `ch4_data.py` saves (`sc_*`, `msb_*`, `ns_*`, `traj_*`) into `data/msb.npz`. Big sizes: `N_SHOWCASE=6400`, `MSB_N=6400, MSB_SEEDS=8`, `NSCAN=[800,1600,3200,6400,12800], NSCAN_SEEDS=8`, `TRAJ_N=600, TRAJ_SEEDS=12`.
2. `compute_phase()` — port `phase_diagram.py` (`coupling(kind="powerlaw")` + `integrate`), grid `MUS=[0,0.5,1,1.5,2,2.5,3,3.5,4]`, `SIGS=[0.5,1,1.5,2,2.5,3]`, `N=2000`, 2 seeds, save `G, F, MUS, SIGS` to `data/phase_diagram.npz`.
3. `compute_dmft_validation()` — port `dmft_validate.py` sweep: `coupling(kind="fc")` + `integrate`, `sigmas=[0.4,0.7,1,1.2,1.35,1.6,2,2.5]` at `mu=0.5`, plus mu-independence rows at sigma=1; save `sigmas, sim, mui_mus, mui` to `data/dmft_validation.npz`.

Under `--smoke`, shrink every N/seed/grid to tiny values (e.g. `MSB_N=400, MSB_SEEDS=2, NSCAN=[400,800], TMAX=120`, phase `N=300` on a 3x3 grid, dmft `N=400` on 3 sigmas) so the whole script runs in ~1-2 minutes.

- [ ] **Step 2: Smoke-run to verify the pipeline end-to-end**

Run: `cd ~/Code/relative-glv && uv run python scripts/compute.py --smoke`
Expected: prints progress for all three blocks and writes `data/msb.npz`, `data/phase_diagram.npz`, `data/dmft_validation.npz` with no error. Confirm:
Run: `uv run python -c "import numpy as np; [print(f, sorted(np.load('data/'+f).files)) for f in ('msb.npz','phase_diagram.npz','dmft_validation.npz')]"`
Expected: each npz lists its expected keys.

- [ ] **Step 3: Full run (the committed data)**

Delete the smoke npz first, then run the full compute (minutes; uses 6 cores).
Run: `cd ~/Code/relative-glv && rm -f data/*.npz && uv run python scripts/compute.py`
Expected: completes; prints the locked-regime summary (`beta_inf`, MSB pooled beta, churn g_eff). Sanity-check the printed `beta_inf` is in/near 0.2-0.25 and pooled beta is positive.

- [ ] **Step 4: Commit (script + the heavy data)**

```bash
cd ~/Code/relative-glv
git add scripts/compute.py data/msb.npz data/phase_diagram.npz data/dmft_validation.npz
git commit -m "feat: compute.py + committed heavy datasets (MSB, phase, DMFT validation)"
```

---

### Task 8: The story notebook (`story.ipynb`)

Results-first narrative, rendering from the committed `data/*.npz` plus a couple of cheap live sims. Committed **with cell outputs** so it renders on GitHub without execution (this is the repo's figure delivery — no separate `figures/` dir).

**Files:**
- Create: `~/Code/relative-glv/story.ipynb`

**Interfaces:**
- Consumes: `relative_glv` package; `data/*.npz`.

- [ ] **Step 1: Author the notebook (sections below), each plotting cell preceded by a markdown cell with the prose + the relevant equation in LaTeX**

Build it as a script-style notebook with these cells:

1. **Title + intro (markdown):** what the relative GLV is, one-paragraph claim (growing economy + MSB facts), link to the thesis/DMFT derivation doc.
2. **The phenomenon (markdown):** the two MSB stylized facts — tent-shaped growth PDF and `sigma(S) ~ S^-beta`, beta ~ 0.15-0.20 — stated as the target. (Equations in LaTeX.)
3. **The model (markdown + code):** the model equation and the share+lnM split (LaTeX); then a *live* small sim:
   ```python
   import numpy as np, matplotlib.pyplot as plt
   from relative_glv import coupling, integrate, growth_rate
   a = coupling(1200, mu=1.76, sigma=1.75, kind="powerlaw", seed=0)
   r = integrate(a, tmax=200, n_eval=1000, lam=1e-3, seed=0, method="RK45", rtol=1e-4, atol=1e-7)
   # plot log10 M(t) (rising) and a sample of relative sizes S_i = N w_i (churning)
   ```
4. **MSB in the model (markdown + code):** load `data/msb.npz`; reproduce the two ch4 figures — size-volatility (binned-median + decline fit beta) and the rescaled growth tent vs Laplace/Gaussian — using `ch4_plots.py` sections 1-2 as the plotting template; then beta-vs-N and the `beta_inf` extrapolation (ch4_plots sections 3-4). Prose explains beta shallowing toward ~0.2 as N grows.
5. **Phase diagram (markdown + code):** load `data/phase_diagram.npz`; render the classified scatter (growing/chaotic/coexisting) from `phase_diagram.py`'s plotting block; one *live* analytic companion from `dmft.solve_fixed_point`/`sigma_c` (the `dmft_phase.py` panels — cheap, no sim). Prose: where the firm-growth facts live.
6. **Appendix — DMFT in brief (markdown + code):** state the single-site result, `sigma_c=sqrt(2)` (mu-independent), then load `data/dmft_validation.npz` and render the 3-panel validation (growth rate, survival, chaos onset) from `dmft_validate.py`'s figure block. Link the full derivation doc.

Reuse the exact matplotlib styling from `ch4_plots.py` / `phase_diagram.py` / `dmft_phase.py` / `dmft_validate.py` for the loaded-data figures (they are already publication-styled).

- [ ] **Step 2: Execute the notebook top-to-bottom and embed outputs**

Run: `cd ~/Code/relative-glv && uv run jupyter nbconvert --to notebook --execute --inplace story.ipynb`
Expected: completes with no errors; every figure renders; total runtime a few minutes (the two live sims at N=1200 + plotting from npz).

- [ ] **Step 3: Eyeball the rendered figures**

Open `story.ipynb` and confirm: M(t) rises, sizes churn, the tent is symmetric and fatter than Gaussian, the size-volatility decline fit shows beta, beta-vs-N shallows toward the empirical band, the phase diagram shows the growing+chaotic region, and the DMFT validation panels agree below `sigma_c`.

- [ ] **Step 4: Commit (notebook with outputs)**

```bash
cd ~/Code/relative-glv
git add story.ipynb
git commit -m "feat: story.ipynb — results-first relative-GLV explainer"
```

---

### Task 9: README and final polish

**Files:**
- Create: `~/Code/relative-glv/README.md`

**Interfaces:**
- Consumes: the finished package, notebook, data.

- [ ] **Step 1: Write `README.md`**

Sections (with LaTeX-rendered equations where they carry the science):
- **Title + one-paragraph pitch:** the relative GLV — interactions on the mean firm → a self-consistently growing economy reproducing the MSB firm-growth facts.
- **The model:** the equation `x_i' = x_i[1 - x_i/m - (alpha x)_i/m]`, the share+lnM split, why it doesn't blow up, the sign convention (mu>0 = competition).
- **Key results:** the symmetric fat tent, `beta` in the empirical band (beta_inf toward ~0.2 as N→∞), `sigma_c=sqrt(2)` (mu-independent), validated against simulation. One sentence each; point to `story.ipynb`.
- **Repo layout:** the file tree with one line per item.
- **Install & run:**
  ```
  uv sync --extra dev
  uv run jupyter notebook story.ipynb     # read the explainer
  uv run pytest                            # validate the model vs DMFT
  uv run python scripts/compute.py         # (optional) regenerate the heavy data
  ```
- **Provenance/citation:** links to the thesis and `glv/docs/.../2026-06-21-relative-glv-dmft-derivation.md` as the authoritative long-form derivation.

- [ ] **Step 2: Verify the full test suite is green**

Run: `cd ~/Code/relative-glv && uv run pytest -q`
Expected: all tests pass (dmft solver, model, msb, validation).

- [ ] **Step 3: Verify a clean reader path**

Run: `cd ~/Code/relative-glv && uv run python -c "import relative_glv as r; print(r.sigma_c(0.0)); print([k for k in r.__all__])"`
Expected: prints `1.4142135...` and the export list.

- [ ] **Step 4: Commit**

```bash
cd ~/Code/relative-glv
git add README.md
git commit -m "docs: README for the relative-GLV explainer repo"
```

---

## Self-Review

**Spec coverage:**
- Narrative notebook + clean module → Tasks 3-5 (module), 8 (notebook). ✓
- Standalone git repo at `~/Code/relative-glv` → Task 1. ✓
- Results-first depth, DMFT in appendix → Task 8 section order. ✓
- One implementation each (no triplicated RHS / MAD) → `integrate` (Task 4), `rescale` (Task 5). ✓
- Math-behind-code in docstrings + prose → docstrings in Tasks 2-5; markdown cells in Task 8; README in Task 9. ✓
- `model.py`/`msb.py`/`dmft.py` split, sign convention, disorder scaling → Global Constraints + Tasks 2-5. ✓
- Compute split (light notebook / heavy script) → Tasks 7-8. ✓
- Committed `data/` + figures as notebook outputs (spec's `figures/` folded into notebook cell outputs — flagged in Task 8) → Tasks 7-8. ✓
- Tests = DMFT-vs-sim validation → Task 6; plus solver/model/msb unit tests. ✓
- Provenance map, MIT license, link-not-copy the derivation → Task 1 (license), Tasks 8-9 (links). ✓
- Out-of-scope (`_attic`, original GLV, criticality, exploratory nbs) → never referenced as build targets. ✓

**Deviation from spec (intentional, ponytail):** no separate `figures/` directory — the heavy figures are the committed cell outputs of `story.ipynb`, which also makes them render on GitHub without execution. Same deliverable, less to maintain. Flagged in Task 8.

**Placeholder scan:** the only non-literal block is `scripts/compute.py` (Task 7), which is described as a port of three named, fully-specified source files with exact parameters rather than pasted in full (~250 lines across three ports). Source paths, function names, parameters, output keys, and the smoke-shrink are all given. Acceptable: the port is mechanical and the sources are committed and verified.

**Type consistency:** `coupling`/`integrate`/`size_volatility`/`solve_fixed_point` signatures and the `integrate` return dict keys (`t, W, lnM, success`) are used consistently across Tasks 4, 6, 7, 8. `size_volatility` return keys (`beta, r2, bin_S, bin_vol, pk`) match `ch4_plots` field usage in Task 8. ✓
