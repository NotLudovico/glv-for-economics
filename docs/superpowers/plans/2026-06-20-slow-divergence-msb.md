# Slow-Diverging-Regime MSB Study — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the MSB firm-growth observables (size–volatility β, growth distribution) in the slowly-diverging regime just above the MA→Unbounded boundary of Roy's disordered GLV, and decide whether that regime supports a self-similar growing state or is only a condensation transient.

**Architecture:** Pure, correctness-critical analysis primitives (MSB kernel, measurement-window selection, self-similarity diagnostic, β fit) go into `glv/analysis.py` with pytest tests, following the project's established pattern (analysis tested in `glv/`, e.g. `find_mu_c_shape_scalar`). Graph + disorder + integration construction stays inline in a new self-contained notebook `notebooks/slow_divergence_msb.ipynb` (the no-degree-abstraction preference). The notebook generates a per-realization f·σ_c ensemble around the boundary, persists a raw dataset, then slices it post-hoc for the slow divergers.

**Tech Stack:** numpy, scipy (`solve_ivp`, `logsumexp`, `stats.skew`), networkx (configuration-model graphs), scipy.sparse, joblib (`Parallel`), matplotlib, pytest. Run everything with `uv run` (e.g. `uv run pytest`, `uv run jupyter nbconvert`).

## Global Constraints

- Python execution: always `uv run python` / `uv run pytest` / `uv run jupyter ...`, never bare `python`.
- Disorder scaling: per-edge mean `mu/C_eff`, variance `sigma^2/C_eff` (scale `sigma/sqrt(2*C_eff)`); built on the edge list, stored sparse. Never use unscaled `generate_network`.
- Model: Roy convention `ẋ_i = x_i(1 − x_i − (α x)_i) + λ`, positive α mean = competition. Base spec (μ, σ, γ, λ) = (4, 2, 0, 1e−10), power-law graph (degree exponent ALPHA=2.5, MEAN_DEGREE=100).
- σ_c locator: **per-capita** (mean-abundance) divergence threshold, never total biomass (total ~ S false-fires at large S).
- MSB measurement: **fixed Δt = 1** for all pooled comparisons, never the adaptive grid (horizon-mixing manufactures fat tails). Δt must stay above the trajectory floor `tmax/(n_eval−1)`.
- Size is the Moran cross-sectional share `S_i = N x_i / Σ_j x_j` (divides out aggregate growth). Volatility is MAD: `σ_i = sqrt(π/2)·mean|g_i − ⟨g_i⟩|`.
- Plot labels: proper mathtext, not raw identifiers. Use `glv.style.apply_style`.
- Big dataset artifact (`notebooks/slow_divergence_dataset.npz`) is **gitignored**, not committed. Commit the notebook + the `slow_divergence_*.png` figures.

---

### Task 1: `binned_beta` — size-volatility exponent fit

**Files:**
- Modify: `glv/analysis.py` (add function)
- Modify: `glv/__init__.py:3` (add to the `from glv.analysis import ...` line)
- Test: `tests/test_slow_divergence.py` (create)

**Interfaces:**
- Produces: `binned_beta(Sbar, vol, *, nbins=18, min_firms=50) -> float`

- [ ] **Step 1: Write the failing test**

Create `tests/test_slow_divergence.py`:

```python
import numpy as np
import glv.analysis as analysis


def test_binned_beta_recovers_known_exponent():
    Sbar = np.logspace(-1, 2, 2000)
    vol = 0.3 * Sbar ** (-0.2)            # exact power law -> beta = 0.2
    assert abs(analysis.binned_beta(Sbar, vol) - 0.2) < 1e-6


def test_binned_beta_nan_when_too_few_firms():
    assert np.isnan(analysis.binned_beta(np.array([1.0, 2.0]), np.array([1.0, 1.0])))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_slow_divergence.py -q`
Expected: FAIL with `AttributeError: module 'glv.analysis' has no attribute 'binned_beta'`

- [ ] **Step 3: Write minimal implementation**

Add to `glv/analysis.py` (the file already imports `numpy as np`):

```python
def binned_beta(Sbar, vol, *, nbins: int = 18, min_firms: int = 50) -> float:
    """Size-volatility exponent beta from a binned-median log-log fit
    (vol ~ Sbar^-beta). Firms are split into `nbins` equal-count bins by Sbar;
    the fit is over bin medians of (log10 Sbar, log10 vol). Returns nan with
    fewer than `min_firms` usable firms or fewer than 2 positive bins."""
    Sbar = np.asarray(Sbar, float)
    vol = np.asarray(vol, float)
    ok = np.isfinite(Sbar) & np.isfinite(vol) & (Sbar > 0) & (vol > 0)
    Sbar, vol = Sbar[ok], vol[ok]
    if Sbar.size < min_firms:
        return float("nan")
    o = np.argsort(Sbar)
    xs, ys = Sbar[o], vol[o]
    parts = np.array_split(np.arange(xs.size), nbins)
    bx = np.array([xs[p].mean() for p in parts])
    by = np.array([np.median(ys[p]) for p in parts])
    m = (bx > 0) & (by > 0)
    if m.sum() < 2:
        return float("nan")
    return float(-np.polyfit(np.log10(bx[m]), np.log10(by[m]), 1)[0])
```

Add `binned_beta` to the import list in `glv/__init__.py:3`:

```python
from glv.analysis import fixed_point, stability_matrix, calculate_mu_c, calculate_mu_c_regular, find_empirical_mu_c, find_mu_c_shape_scalar, binned_beta
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_slow_divergence.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add glv/analysis.py glv/__init__.py tests/test_slow_divergence.py
git commit -m "feat(analysis): binned_beta size-volatility exponent fit"
```

---

### Task 2: `msb_observables` — Moran cross-sectional MSB kernel

**Files:**
- Modify: `glv/analysis.py` (add function + `logsumexp` import)
- Modify: `glv/__init__.py:3`
- Test: `tests/test_slow_divergence.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (independent of `binned_beta`).
- Produces: `msb_observables(logx, t_grid, *, floor=1e-8) -> dict` with keys `Sbar` (S,), `vol` (S,), `g` (S, n_t−1), `live` (S,) bool, `persistent` (S,) bool, `n_inc` int. `logx` is the (S, n_t) array of log-sizes on the common grid `t_grid`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_slow_divergence.py`:

```python
from scipy.special import logsumexp


def test_msb_common_mode_invariance():
    # Adding the same time function to every firm = multiplying all x by a common
    # M(t). The cross-sectional shares must be invariant -> aggregate growth divides out.
    rng = np.random.default_rng(1)
    logx = rng.normal(0, 1, (50, 40)).cumsum(1)
    t = np.arange(40, dtype=float)
    common = np.linspace(0, 30, 40)              # GDP-like aggregate blow-up
    a = analysis.msb_observables(logx, t)
    b = analysis.msb_observables(logx + common[None, :], t)
    np.testing.assert_allclose(a["Sbar"], b["Sbar"], rtol=1e-9)
    np.testing.assert_allclose(a["vol"], b["vol"], rtol=1e-9)


def test_msb_cross_section_sums_to_S():
    rng = np.random.default_rng(2)
    logx = rng.normal(0, 1, (30, 10))
    a = analysis.msb_observables(logx, np.arange(10.0))
    # reconstruct S_i each snapshot from the returned definition; columns sum to S
    logS = logx - logsumexp(logx, axis=0, keepdims=True) + np.log(30)
    np.testing.assert_allclose(np.exp(logS).sum(0), 30.0, rtol=1e-9)
    assert a["n_inc"] == 9


def test_msb_extinct_firm_delisted():
    logx = np.zeros((5, 8))
    logx[0] = -np.inf                            # firm 0 extinct throughout
    a = analysis.msb_observables(logx, np.arange(8.0))
    assert not a["live"][0]
    assert a["live"][1:].all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_slow_divergence.py -q -k msb`
Expected: FAIL with `AttributeError: ... 'msb_observables'`

- [ ] **Step 3: Write minimal implementation**

Ensure `glv/analysis.py` imports `logsumexp` (add near the top imports if absent):

```python
from scipy.special import logsumexp
```

Add the function:

```python
def msb_observables(logx, t_grid, *, floor: float = 1e-8) -> dict:
    """MSB per-firm observables from gridded log-sizes.

    logx : (S, n_t) log-abundances on the common grid `t_grid` (length n_t);
           extinct firms may carry -inf. Returns per-firm arrays plus masks:
      Sbar       : time-mean Moran cross-sectional size S_i = N x_i / sum_j x_j
      vol        : MAD volatility sqrt(pi/2) * mean|g_i - <g_i>|
      g          : (S, n_t-1) growth increments g_i = Delta ln S_i
      live       : firms with finite positive vol (Moran delisting rule)
      persistent : firms whose abundance never drops to `floor`
      n_inc      : growth-increment count (n_t - 1)
    The cross-sectional normalization divides out any common-mode (aggregate)
    growth, so Sbar/vol/g are invariant to multiplying every firm by a common
    time function M(t)."""
    logx = np.asarray(logx, float)
    S, n_t = logx.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        logS = logx - logsumexp(logx, axis=0, keepdims=True) + np.log(S)
        g = np.diff(logS, axis=1)
        Sbar = np.exp(logS).mean(1)
        vol = np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)
    live = (Sbar > 0) & (vol > 0) & np.isfinite(vol)
    with np.errstate(over="ignore"):
        xmin = np.exp(logx).min(1)
    persistent = np.isfinite(logx).all(1) & (xmin > floor)
    return {"Sbar": Sbar, "vol": vol, "g": g, "live": live,
            "persistent": persistent, "n_inc": int(n_t - 1)}
```

Add `msb_observables` to `glv/__init__.py:3`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_slow_divergence.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add glv/analysis.py glv/__init__.py tests/test_slow_divergence.py
git commit -m "feat(analysis): msb_observables Moran cross-sectional kernel"
```

---

### Task 3: `select_msb_window` — slow-diverger measurement window

**Files:**
- Modify: `glv/analysis.py`
- Modify: `glv/__init__.py:3`
- Test: `tests/test_slow_divergence.py`

**Interfaces:**
- Produces: `select_msb_window(t_grid, t_div, status, *, t_burn_frac=0.2, rho=0.8, t_min=150.0, w_min=50) -> dict` with keys `qualifies` (bool), `i0` (int), `i1` (int), `n_snap` (int), `t0` (float), `t1` (float). `i0:i1` indexes the window in `t_grid`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_slow_divergence.py`:

```python
def test_window_qualifies_slow_diverger():
    t = np.arange(0, 301, 1.0)
    w = analysis.select_msb_window(t, t_div=300.0, status="diverged")
    assert w["qualifies"] and w["n_snap"] >= 50
    assert abs(w["t0"] - 60) < 1.0 and abs(w["t1"] - 240) < 1.0


def test_window_rejects_fast_diverger():
    t = np.arange(0, 101, 1.0)
    assert not analysis.select_msb_window(t, t_div=100.0, status="diverged")["qualifies"]


def test_window_rejects_bounded():
    t = np.arange(0, 501, 1.0)
    assert not analysis.select_msb_window(t, t_div=500.0, status="bounded")["qualifies"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_slow_divergence.py -q -k window`
Expected: FAIL with `AttributeError: ... 'select_msb_window'`

- [ ] **Step 3: Write minimal implementation**

Add to `glv/analysis.py`:

```python
def select_msb_window(t_grid, t_div, status, *, t_burn_frac: float = 0.2,
                      rho: float = 0.8, t_min: float = 150.0, w_min: int = 50) -> dict:
    """Quasi-stationary measurement window for a slowly-diverging run.

    Window = [t_burn_frac*t_div, rho*t_div]. A run qualifies as a usable slow
    diverger when status == 'diverged', t_div >= t_min, and the window holds
    >= w_min snapshots of `t_grid`. i0:i1 indexes that window."""
    t_grid = np.asarray(t_grid, float)
    t0 = t_burn_frac * t_div
    t1 = rho * t_div
    i0 = int(np.searchsorted(t_grid, t0, side="left"))
    i1 = int(np.searchsorted(t_grid, t1, side="right"))
    n_snap = max(0, i1 - i0)
    qualifies = (status == "diverged") and (t_div >= t_min) and (n_snap >= w_min)
    return {"qualifies": bool(qualifies), "i0": i0, "i1": i1,
            "n_snap": int(n_snap), "t0": float(t0), "t1": float(t1)}
```

Add `select_msb_window` to `glv/__init__.py:3`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_slow_divergence.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add glv/analysis.py glv/__init__.py tests/test_slow_divergence.py
git commit -m "feat(analysis): select_msb_window slow-diverger window picker"
```

---

### Task 4: `self_similarity_report` — the make-or-break diagnostic

**Files:**
- Modify: `glv/analysis.py` (add function + `scipy.stats.skew` import)
- Modify: `glv/__init__.py:3`
- Test: `tests/test_slow_divergence.py`

**Interfaces:**
- Consumes: `msb_observables`, `binned_beta` (Tasks 1–2).
- Produces: `self_similarity_report(logx_win, t_win, *, floor=1e-8, r2_min=0.95, beta_drift_max=0.10) -> dict` with keys `lnM_slope`, `lnM_r2`, `beta_early`, `beta_late`, `beta_drift`, `skew_early`, `skew_late`, `stationary` (bool). `logx_win` is the (S, n_w) gridded log-sizes inside the window; `t_win` its times.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_slow_divergence.py`:

```python
def test_self_similar_growing_state_is_stationary():
    rng = np.random.default_rng(3)
    S, n_w = 200, 120
    t = np.arange(n_w, dtype=float)
    base = rng.normal(0, 1, (S, 1))              # fixed per-firm levels (stationary shares)
    noise = 0.05 * rng.normal(0, 1, (S, n_w))    # small stationary fluctuation
    common = 0.5 * t                             # steady exponential aggregate growth
    logx = base + noise + common[None, :]
    rep = analysis.self_similarity_report(logx, t)
    assert rep["lnM_r2"] > 0.95
    assert abs(rep["lnM_slope"] - 0.5) < 0.05
    assert rep["stationary"]


def test_condensing_state_is_not_stationary():
    rng = np.random.default_rng(4)
    S, n_w = 200, 120
    t = np.arange(n_w, dtype=float)
    logx = rng.normal(0, 1, (S, 1)) + 0.05 * rng.normal(0, 1, (S, n_w)) + 0.5 * t[None, :]
    logx[0] += 0.30 * t                          # firm 0 escapes -> condensation completes
    rep = analysis.self_similarity_report(logx, t)
    assert not rep["stationary"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_slow_divergence.py -q -k stationary`
Expected: FAIL with `AttributeError: ... 'self_similarity_report'`

- [ ] **Step 3: Write minimal implementation**

Ensure `glv/analysis.py` imports skew (add near the top if absent):

```python
from scipy.stats import skew as _skew
```

Add the function:

```python
def self_similarity_report(logx_win, t_win, *, floor: float = 1e-8,
                           r2_min: float = 0.95, beta_drift_max: float = 0.10) -> dict:
    """Is the windowed state a self-similar growing chaotic state?

    Checks (i) ln M(t) ~ linear (steady exponential aggregate growth) via an
    R^2 of a least-squares line, and (ii) the size-volatility beta does not
    drift between the window's early and late halves. `stationary` is True when
    R^2 >= r2_min and |beta_drift| <= beta_drift_max."""
    logx_win = np.asarray(logx_win, float)
    t_win = np.asarray(t_win, float)
    n_w = logx_win.shape[1]
    lnM = logsumexp(logx_win, axis=0)                       # ln total biomass per snapshot
    A = np.vstack([t_win, np.ones_like(t_win)]).T
    slope, intercept = np.linalg.lstsq(A, lnM, rcond=None)[0]
    resid = lnM - (slope * t_win + intercept)
    ss_tot = float(np.sum((lnM - lnM.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 0.0

    def _beta_skew(sl):
        o = msb_observables(logx_win[:, sl], t_win[sl], floor=floor)
        live = o["live"]
        b = binned_beta(o["Sbar"][live], o["vol"][live])
        gl = o["g"][live].ravel()
        gl = gl[np.isfinite(gl)]
        sk = float(_skew(gl)) if gl.size > 2 else float("nan")
        return b, sk

    half = n_w // 2
    be, ske = _beta_skew(slice(0, half + 1))
    bl, skl = _beta_skew(slice(half, n_w))
    drift = abs(bl - be) if (np.isfinite(be) and np.isfinite(bl)) else float("inf")
    stationary = bool(r2 >= r2_min and drift <= beta_drift_max)
    return {"lnM_slope": float(slope), "lnM_r2": float(r2),
            "beta_early": be, "beta_late": bl, "beta_drift": float(drift),
            "skew_early": ske, "skew_late": skl, "stationary": stationary}
```

Add `self_similarity_report` to `glv/__init__.py:3`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_slow_divergence.py -q`
Expected: PASS (10 passed). If `test_condensing_state_is_not_stationary` passes by the r2 path but you want to confirm the beta-drift path also fires, print `rep` once; no code change needed if `stationary` is False.

- [ ] **Step 5: Commit**

```bash
git add glv/analysis.py glv/__init__.py tests/test_slow_divergence.py
git commit -m "feat(analysis): self_similarity_report growing-state diagnostic"
```

---

### Task 5: Notebook scaffold + inline construction + `run_and_store`

**Files:**
- Create: `notebooks/slow_divergence_msb.ipynb`
- Modify: `.gitignore` (add the dataset artifact)

**Interfaces:**
- Consumes: `glv.analysis.msb_observables` (Task 2).
- Produces (in-notebook): `build_realization(run_seed, S_run)`, `_integrate(mean_a, dis_a, x0, sig, T, t_eval=None)`, `_diverged_at(mean_a, dis_a, x0, sig, T, percap=10.0)`, `locate_sigma_c(mean_a, dis_a, x0, T, bracket=(1.0,3.5), iters=7)`, `run_and_store(run_seed, S_run, f_list, tmax, n_eval, dt_store, locate_T, percap, lam) -> list[dict]`.

- [ ] **Step 1: Create the notebook with a params + imports cell**

Create `notebooks/slow_divergence_msb.ipynb`. First a markdown cell:

```markdown
# MSB observables in the slowly-diverging regime (MA → Unbounded boundary)

Roy's disordered GLV at the chaotic Multiple-Attractors spec. We place each
realization at sigma = f * sigma_c around its own divergence onset, keep the
runs that diverge slowly, and measure the MSB firm-growth observables in the
quasi-stationary growing window. Spec:
`docs/superpowers/specs/2026-06-20-slow-divergence-msb-design.md`.
```

Then a code cell:

```python
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from glv.analysis import (msb_observables, binned_beta, select_msb_window,
                          self_similarity_report)
from glv.style import apply_style
apply_style()

# --- base spec (Roy chaotic-MA) ---
MU, GAMMA, LAM = 4.0, 0.0, 1e-10
ALPHA, MEAN_DEGREE = 2.5, 100
TMAX, N_EVAL, DT_STORE = 500.0, 3000, 1.0
LOCATE_T, PERCAP = TMAX, 10.0           # locate sigma_c at the full horizon -> f>1 diverges slowly
F_LIST = [0.90, 0.97, 1.02, 1.05, 1.10, 1.20, 1.50]
SEED, N_JOBS = 20240620, 4
```

- [ ] **Step 2: Add the inline construction helpers cell**

Copy `build_realization`, `_integrate`, `_diverged_at`, `locate_sigma_c` from `notebooks/chaotic_glv.ipynb` cell 31 (the power-law topology branch), adapting them to read the module-level constants above (`MU`, `GAMMA`, `LAM`, `ALPHA`, `MEAN_DEGREE`). They must keep: per-edge mean `MU/C_eff` and disorder scale `1/sqrt(2*C_eff)`; `_diverged_at` using the **per-capita** max (`r.y.sum(0).max()/S >= percap`) or a finite-time integration failure; `locate_sigma_c` bisecting that predicate at horizon `T`. Verify the disorder split matches:

```python
def build_realization(run_seed, S_run):
    rng = np.random.default_rng(run_seed)
    C = min(MEAN_DEGREE, S_run - 1)
    kmin = C * (ALPHA - 2) / (ALPHA - 1)
    deg = np.maximum((kmin * (1 - rng.uniform(size=S_run)) ** (-1 / (ALPHA - 1))).round().astype(int), 1)
    if deg.sum() % 2:
        deg[deg.argmin()] += 1
    G = nx.Graph(nx.configuration_model(deg.tolist(), seed=int(run_seed)))
    G.remove_edges_from(nx.selfloop_edges(G))
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float); A.data[:] = 1.0
    C_eff = float(np.asarray(A.sum(axis=1)).mean())
    Au = sparse.triu(A, k=1).tocoo(); ei, ej = Au.row, Au.col; nE = ei.size
    a = rng.normal(0, 1, nE); b = rng.normal(0, 1, nE)
    sym = (a + b) / np.sqrt(2); anti = (a - b) / np.sqrt(2)
    dscale = 1.0 / np.sqrt(2 * C_eff)
    d_ij = dscale * (np.sqrt(1 + GAMMA) * sym + np.sqrt(1 - GAMMA) * anti)
    d_ji = dscale * (np.sqrt(1 + GAMMA) * sym - np.sqrt(1 - GAMMA) * anti)
    rows = np.concatenate([ei, ej]); cols = np.concatenate([ej, ei])
    mean_a = sparse.csr_array((np.full(rows.size, MU / C_eff), (rows, cols)), shape=(S_run, S_run))
    dis_a  = sparse.csr_array((np.concatenate([d_ij, d_ji]), (rows, cols)), shape=(S_run, S_run))
    x0 = rng.uniform(0.5, 1.5, S_run)
    return mean_a, dis_a, C_eff, x0


def _integrate(mean_a, dis_a, x0, sig, T, t_eval=None):
    alpha = mean_a + sig * dis_a
    def rhs(t, x, a=alpha):
        x = np.maximum(x, 0.0); return x * (1.0 - x - a @ x) + LAM
    return solve_ivp(rhs, (0.0, T), x0, method="RK45", t_eval=t_eval, rtol=1e-8, atol=1e-10)


def _diverged_at(mean_a, dis_a, x0, sig, T, percap=PERCAP):
    r = _integrate(mean_a, dis_a, x0, sig, T)
    percap_max = float(np.maximum(r.y, 0.0).sum(0).max()) / r.y.shape[0]
    return (r.t[-1] < T - 1e-6) or (percap_max >= percap)


def locate_sigma_c(mean_a, dis_a, x0, T, bracket=(1.0, 3.5), iters=7):
    lo, hi = bracket
    if _diverged_at(mean_a, dis_a, x0, lo, T): return lo
    if not _diverged_at(mean_a, dis_a, x0, hi, T): return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _diverged_at(mean_a, dis_a, x0, mid, T): hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)
```

- [ ] **Step 3: Add the `run_and_store` cell**

```python
def run_and_store(run_seed, S_run, f_list, tmax, n_eval, dt_store, locate_T, percap, lam):
    """One realization across the f-band. For each f: integrate once, classify
    bounded/diverged, and store the raw-enough payload. Diverged runs keep the
    full per-firm log-size grid (re-sliceable window); bounded runs keep only the
    reference MSB observables over a standard window."""
    mean_a, dis_a, C_eff, x0 = build_realization(run_seed, S_run)
    sig_c = locate_sigma_c(mean_a, dis_a, x0, locate_T)
    t_loc = np.linspace(0.0, tmax, n_eval)
    out = []
    for f in f_list:
        sig = f * sig_c
        r = _integrate(mean_a, dis_a, x0, sig, tmax, t_eval=t_loc)
        x = np.maximum(r.y, 0.0)                            # (S_run, n_reached)
        t_used = r.t
        diverged = (not r.success) or (t_used[-1] < tmax - 1e-6)
        t_div = float(t_used[-1])
        status = "diverged" if diverged else "bounded"
        t_grid = np.arange(0.0, t_div + 0.5 * dt_store, dt_store)
        with np.errstate(divide="ignore"):
            lnx = np.log(np.maximum(x, lam))               # floor at lam (1e-10) -> finite, dead firms ~ log(lam)
        logx_grid = np.array([np.interp(t_grid, t_used, lnx[i]) for i in range(S_run)]).astype(np.float32)
        rec = {"seed": int(run_seed), "C_eff": C_eff, "sigma_c": float(sig_c),
               "f": float(f), "sigma": float(sig), "status": status,
               "t_div": t_div, "t_grid": t_grid.astype(np.float32)}
        if status == "diverged":
            rec["logx_grid"] = logx_grid                   # full trajectory for re-slicing
        else:
            o = msb_observables(logx_grid, t_grid, floor=1e-8)
            live = o["live"]
            rec["ref_Sbar"] = o["Sbar"][live].astype(np.float32)
            rec["ref_vol"] = o["vol"][live].astype(np.float32)
            rec["ref_g"] = o["g"][live].ravel().astype(np.float32)
        out.append(rec)
    return out
```

- [ ] **Step 4: Add `.gitignore` entry, then validate one realization**

Add to `.gitignore`:

```
notebooks/slow_divergence_dataset.npz
```

Add a temporary validation cell and execute the notebook:

```python
_chk = run_and_store(SEED, 400, F_LIST, TMAX, N_EVAL, DT_STORE, LOCATE_T, PERCAP, LAM)
print("sigma_c =", _chk[0]["sigma_c"])
print([(r["f"], r["status"], round(r["t_div"], 1)) for r in _chk])
assert any(r["status"] == "diverged" for r in _chk), "no diverged run in the f-band"
assert any(r["status"] == "bounded" for r in _chk), "no bounded run in the f-band"
assert _chk[0]["logx_grid"].dtype == np.float32 if _chk[0]["status"] == "diverged" else True
```

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/slow_divergence_msb.ipynb`
Expected: executes cleanly; printout shows a finite `sigma_c` (~2–2.5), at least one `diverged` and one `bounded` row, and t_div decreasing as f rises (small f-above-1 = slow).

- [ ] **Step 5: Commit** (remove the temporary `_chk` validation cell first, or keep it — it is cheap and self-documenting; keep it.)

```bash
git add notebooks/slow_divergence_msb.ipynb .gitignore
git commit -m "feat(notebook): slow_divergence scaffold + run_and_store"
```

---

### Task 6: Dataset-generation ensemble (prototype scale)

**Files:**
- Modify: `notebooks/slow_divergence_msb.ipynb`
- Output (gitignored): `notebooks/slow_divergence_dataset.npz`

**Interfaces:**
- Consumes: `run_and_store` (Task 5).
- Produces (in-notebook): module-level `runs` (list of dicts) and the on-disk `slow_divergence_dataset.npz` with an object array under key `runs`.

- [ ] **Step 1: Add the ensemble cell (PROTOTYPE flag on)**

```python
PROTOTYPE = True                                   # flip to False for production in Task 11
S_RUN  = 1000 if PROTOTYPE else 2000
N_REAL = 20 if PROTOTYPE else 150

seeds = np.random.default_rng(SEED).integers(0, 2**31 - 1, size=N_REAL)
batches = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(
    delayed(run_and_store)(int(s), S_RUN, F_LIST, TMAX, N_EVAL, DT_STORE, LOCATE_T, PERCAP, LAM)
    for s in seeds)
runs = [rec for b in batches for rec in b]
np.savez("slow_divergence_dataset.npz", runs=np.array(runs, dtype=object))
print(f"stored {len(runs)} runs ({N_REAL} realizations x {len(F_LIST)} f)")
```

- [ ] **Step 2: Add a dataset-summary / validation cell**

```python
import collections
by_status = collections.Counter(r["status"] for r in runs)
div = [r for r in runs if r["status"] == "diverged"]
tdivs = np.array([r["t_div"] for r in div])
print("status counts:", dict(by_status))
print(f"diverged t_div: min={tdivs.min():.0f} med={np.median(tdivs):.0f} max={tdivs.max():.0f}")
slow = [r for r in div if r["t_div"] >= 150.0]
print(f"slow divergers (t_div >= 150): {len(slow)}")
assert by_status["diverged"] > 0 and by_status["bounded"] > 0
assert tdivs.max() - tdivs.min() > 50, "no t_div spread -> f-band not bracketing the onset; adjust F_LIST/LOCATE_T"
```

- [ ] **Step 3: Execute the notebook**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/slow_divergence_msb.ipynb`
Expected: prints status counts with both `diverged` and `bounded` present, a t_div spread > 50, and a nonzero slow-diverger count. If slow count is 0, note it — the prototype may be too small; production scale (Task 11) should populate it, but if even production yields none, that is itself evidence the slow regime is vanishingly thin (record in the verdict).

- [ ] **Step 4: Commit**

```bash
git add notebooks/slow_divergence_msb.ipynb
git commit -m "feat(notebook): slow_divergence dataset ensemble (prototype scale)"
```

---

### Task 7: Slow-diverger selection + t_div / f distribution figure

**Files:**
- Modify: `notebooks/slow_divergence_msb.ipynb`
- Output: `notebooks/slow_divergence_tdiv.png`

**Interfaces:**
- Consumes: `runs` (Task 6), `select_msb_window` (Task 3).
- Produces (in-notebook): `slow` (list of qualifying diverged dicts, each annotated with its `window` dict).

- [ ] **Step 1: Add the selection cell**

```python
T_MIN, W_MIN = 150.0, 50
slow = []
for r in runs:
    if r["status"] != "diverged":
        continue
    w = select_msb_window(r["t_grid"], r["t_div"], r["status"], t_min=T_MIN, w_min=W_MIN)
    if w["qualifies"]:
        slow.append({**r, "window": w})
print(f"qualifying slow divergers: {len(slow)} / {sum(x['status']=='diverged' for x in runs)} diverged")
```

- [ ] **Step 2: Add the distribution figure cell**

```python
div = [r for r in runs if r["status"] == "diverged"]
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.5))
axL.scatter([r["f"] for r in div], [r["t_div"] for r in div], s=12, alpha=0.5, color="#9bb8d3")
axL.scatter([r["f"] for r in slow], [r["t_div"] for r in slow], s=18, color="#c1121f", label="slow (kept)")
axL.axhline(T_MIN, ls=":", color="#888", label=fr"$t_\mathrm{{min}}={T_MIN:g}$")
axL.set(xlabel=r"$f=\sigma/\sigma_c$", ylabel=r"divergence time $t_\mathrm{div}$",
        title="Divergence time vs distance through the boundary"); axL.legend()
axR.hist([r["t_div"] for r in slow], bins=15, color="#457b9d", edgecolor="white")
axR.set(xlabel=r"$t_\mathrm{div}$ (slow divergers)", ylabel="count", title="Slowness distribution")
plt.tight_layout(); plt.savefig("slow_divergence_tdiv.png", dpi=120); plt.show()
```

- [ ] **Step 3: Execute + sanity-check**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/slow_divergence_msb.ipynb`
Expected: figure saved; red points sit at small f-above-1 / large t_div (slow divergers are nearest the boundary).

- [ ] **Step 4: Commit**

```bash
git add notebooks/slow_divergence_msb.ipynb notebooks/slow_divergence_tdiv.png
git commit -m "feat(notebook): slow-diverger selection + t_div distribution"
```

---

### Task 8: Self-similarity diagnostic (make-or-break) + figure

**Files:**
- Modify: `notebooks/slow_divergence_msb.ipynb`
- Output: `notebooks/slow_divergence_selfsim.png`

**Interfaces:**
- Consumes: `slow` (Task 7), `self_similarity_report` (Task 4).
- Produces (in-notebook): `slow` items annotated with `ssr` (the self-similarity report); module-level `frac_stationary`.

- [ ] **Step 1: Add the diagnostic cell**

```python
for r in slow:
    w = r["window"]
    logx_win = r["logx_grid"][:, w["i0"]:w["i1"]]
    t_win = r["t_grid"][w["i0"]:w["i1"]].astype(float)
    r["ssr"] = self_similarity_report(logx_win, t_win)
frac_stationary = np.mean([r["ssr"]["stationary"] for r in slow]) if slow else float("nan")
print(f"stationary (self-similar growing) fraction: {frac_stationary:.2f} of {len(slow)} slow divergers")
print("median lnM_r2 =", np.median([r["ssr"]["lnM_r2"] for r in slow]) if slow else "n/a",
      " median beta_drift =", np.median([r["ssr"]["beta_drift"] for r in slow]) if slow else "n/a")
```

- [ ] **Step 2: Add the diagnostic figure cell**

```python
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.5))
# left: a representative slow diverger's ln M(t) over its window (linear -> self-similar)
if slow:
    r = max(slow, key=lambda z: z["t_div"])
    w = r["window"]
    from scipy.special import logsumexp
    t_win = r["t_grid"][w["i0"]:w["i1"]].astype(float)
    lnM = logsumexp(r["logx_grid"][:, w["i0"]:w["i1"]].astype(float), axis=0)
    axL.plot(t_win, lnM, color="#2a9d8f")
    s = r["ssr"]
    axL.plot(t_win, s["lnM_slope"] * t_win + (lnM[0] - s["lnM_slope"] * t_win[0]), "--",
             color="#264653", label=fr"linear fit ($R^2={s['lnM_r2']:.3f}$)")
    axL.set(xlabel=r"$t$", ylabel=r"$\ln M(t)$",
            title=fr"Aggregate growth, slowest run ($t_\mathrm{{div}}={r['t_div']:.0f}$)"); axL.legend()
# right: early vs late beta per slow diverger (on-diagonal -> no drift -> stationary)
be = [r["ssr"]["beta_early"] for r in slow]; bl = [r["ssr"]["beta_late"] for r in slow]
axR.scatter(be, bl, s=20, c=["#c1121f" if not r["ssr"]["stationary"] else "#457b9d" for r in slow])
lim = [0, max([x for x in be + bl if np.isfinite(x)] + [1])]
axR.plot(lim, lim, ls=":", color="#888"); axR.set(xlabel=r"$\beta$ early half", ylabel=r"$\beta$ late half",
        title="Window self-similarity (blue = stationary)")
plt.tight_layout(); plt.savefig("slow_divergence_selfsim.png", dpi=120); plt.show()
```

- [ ] **Step 2b: Add a markdown verdict-gate cell**

```markdown
**Make-or-break.** If `frac_stationary` is high (most slow divergers show linear ln M and no
beta drift), the slow-diverger MSB in the next cells is a genuine phase property. If it is low,
the regime is a condensation transient and the MSB below is reported as such (per the spec's
accepted negative finding), not as a clean boundary law.
```

- [ ] **Step 3: Execute + record the outcome**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/slow_divergence_msb.ipynb`
Expected: prints `frac_stationary` and the figure. Record the value — it drives the verdict in Task 11.

- [ ] **Step 4: Commit**

```bash
git add notebooks/slow_divergence_msb.ipynb notebooks/slow_divergence_selfsim.png
git commit -m "feat(notebook): self-similarity make-or-break diagnostic"
```

---

### Task 9: MSB measurement in the window (β all/persistent + growth distribution)

**Files:**
- Modify: `notebooks/slow_divergence_msb.ipynb`
- Output: `notebooks/slow_divergence_msb.png`

**Interfaces:**
- Consumes: `slow` (Task 7/8), `msb_observables`, `binned_beta` (Tasks 1–2).
- Produces (in-notebook): pooled `Sbar_all`, `vol_all`, `g_all`, `Sbar_p`, `vol_p`, `g_p`; scalars `beta_all`, `beta_persist`, `skew_all`, `skew_persist`, `qtail_all`.

- [ ] **Step 1: Add the pooled-measurement cell**

```python
Sbar_all, vol_all, g_all, Sbar_p, vol_p, g_p = ([] for _ in range(6))
for r in slow:
    w = r["window"]
    o = msb_observables(r["logx_grid"][:, w["i0"]:w["i1"]],
                        r["t_grid"][w["i0"]:w["i1"]].astype(float), floor=1e-8)
    live, pers = o["live"], o["persistent"]
    Sbar_all.append(o["Sbar"][live]); vol_all.append(o["vol"][live])
    g_all.append(o["g"][live].ravel())
    lp = live & pers
    Sbar_p.append(o["Sbar"][lp]); vol_p.append(o["vol"][lp]); g_p.append(o["g"][lp].ravel())
Sbar_all = np.concatenate(Sbar_all); vol_all = np.concatenate(vol_all)
g_all = np.concatenate(g_all); g_all = g_all[np.isfinite(g_all)]
Sbar_p = np.concatenate(Sbar_p); vol_p = np.concatenate(vol_p)
g_p = np.concatenate(g_p); g_p = g_p[np.isfinite(g_p)]

from scipy.stats import skew
beta_all, beta_persist = binned_beta(Sbar_all, vol_all), binned_beta(Sbar_p, vol_p)
def _qtail(g):                                   # Dt-robust tail: 99.9th pct of |g| / MAD
    g = g - np.median(g); mad = np.median(np.abs(g))
    return np.percentile(np.abs(g), 99.9) / mad if mad > 0 else np.nan
skew_all, skew_persist, qtail_all = skew(g_all), skew(g_p), _qtail(g_all)
print(f"beta_all={beta_all:.3f}  beta_persist={beta_persist:.3f}")
print(f"skew_all={skew_all:.3f}  skew_persist={skew_persist:.3f}  qtail99.9/MAD(all)={qtail_all:.2f}")
print("empirical MSB anchors: beta~0.15-0.20, symmetric tent (skew~0), Laplace qtail~7.6")
```

- [ ] **Step 2: Add the MSB figure cell (size-volatility + growth PDF)**

```python
def _binmed(x, y, nb=18):
    o = np.argsort(x); xs, ys = x[o], y[o]; P = np.array_split(np.arange(xs.size), nb)
    return np.array([xs[p].mean() for p in P]), np.array([np.median(ys[p]) for p in P])

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.8))
bx, by = _binmed(Sbar_all, vol_all)
axL.loglog(Sbar_all, vol_all, ".", ms=2, alpha=0.15, color="gray")
axL.loglog(bx, by, "o", ms=5, color="#2a9d8f", label=fr"all firms ($\beta={beta_all:.3f}$)")
bxp, byp = _binmed(Sbar_p, vol_p)
axL.loglog(bxp, byp, "s", ms=5, color="#e76f51", label=fr"persistent ($\beta={beta_persist:.3f}$)")
axL.set(xlabel=r"time-average size $\bar S_i$", ylabel=r"volatility $\sigma_i$",
        title="Size--volatility, slow-diverger window"); axL.legend()
for g, lab, c in [(g_all, "all", "#457b9d"), (g_p, "persistent", "#e76f51")]:
    gg = (g - g.mean()) / g.std()
    h, e = np.histogram(gg, bins=120, density=True); m = h > 0
    axR.semilogy(0.5 * (e[:-1] + e[1:])[m], h[m], "-", lw=1.3, color=c, label=lab)
zz = np.linspace(-8, 8, 200); axR.semilogy(zz, np.exp(-np.abs(zz) * np.sqrt(2)) / np.sqrt(2), "k--", lw=1, label="Laplace")
axR.set(xlabel=r"standardized growth $g/\mathrm{std}$", ylabel="PDF",
        title=fr"Growth distribution (skew$_\mathrm{{all}}={skew_all:.2f}$)"); axR.legend()
plt.tight_layout(); plt.savefig("slow_divergence_msb.png", dpi=120); plt.show()
```

- [ ] **Step 3: Execute + sanity-check**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/slow_divergence_msb.ipynb`
Expected: prints β_all, β_persist (compare to the bounded-side ≈0.3–0.5 and empirical 0.15–0.20) and the growth skew/tail; figure saved. If `slow` is empty, guard the cell to print "no slow divergers — see Task 11 production scale" instead of erroring (wrap the body in `if slow:`).

- [ ] **Step 4: Commit**

```bash
git add notebooks/slow_divergence_msb.ipynb notebooks/slow_divergence_msb.png
git commit -m "feat(notebook): slow-diverger MSB observables (beta + growth dist)"
```

---

### Task 10: Comparison — slow-diverger vs bounded side vs empirical

**Files:**
- Modify: `notebooks/slow_divergence_msb.ipynb`
- Output: `notebooks/slow_divergence_compare.png`

**Interfaces:**
- Consumes: `runs` (bounded `ref_*` arrays from Task 6), pooled slow-diverger arrays (Task 9), `binned_beta`.
- Produces (in-notebook): `beta_bounded`, and the comparison figure.

- [ ] **Step 1: Add the comparison cell**

```python
ref = [r for r in runs if r["status"] == "bounded"]
Sb_b = np.concatenate([r["ref_Sbar"] for r in ref]) if ref else np.array([])
vl_b = np.concatenate([r["ref_vol"] for r in ref]) if ref else np.array([])
g_b = np.concatenate([r["ref_g"] for r in ref]) if ref else np.array([])
beta_bounded = binned_beta(Sb_b, vl_b)
print(f"beta_bounded(f<1) = {beta_bounded:.3f}   beta_slow_all = {beta_all:.3f}   "
      f"beta_slow_persist = {beta_persist:.3f}   empirical ~ 0.15-0.20")

fig, ax = plt.subplots(figsize=(7, 4.8))
labels = ["bounded\n(f<1)", "slow\n(all)", "slow\n(persistent)", "empirical\nMSB"]
vals = [beta_bounded, beta_all, beta_persist, 0.175]
ax.bar(labels, vals, color=["#9bb8d3", "#457b9d", "#e76f51", "#2a9d8f"])
ax.axhspan(0.15, 0.20, color="#2a9d8f", alpha=0.15)
ax.set(ylabel=r"size-volatility exponent $\beta$", title="MSB beta across regimes")
plt.tight_layout(); plt.savefig("slow_divergence_compare.png", dpi=120); plt.show()
```

- [ ] **Step 2: Execute + sanity-check**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/slow_divergence_msb.ipynb`
Expected: bar chart contrasting the regimes; prints the β triple vs empirical band.

- [ ] **Step 3: Commit**

```bash
git add notebooks/slow_divergence_msb.ipynb notebooks/slow_divergence_compare.png
git commit -m "feat(notebook): regime comparison of MSB beta"
```

---

### Task 11: Production run + verdict + memory

**Files:**
- Modify: `notebooks/slow_divergence_msb.ipynb` (flip `PROTOTYPE`, add verdict markdown)
- Create: `/Users/ludovicofurlanetto/.claude/projects/-Users-ludovicofurlanetto-Code-glv/memory/project_slow_divergence_msb.md` + index line in `MEMORY.md`

**Interfaces:**
- Consumes: the full notebook (Tasks 5–10).

- [ ] **Step 1: Flip to production scale**

In the Task 6 ensemble cell set `PROTOTYPE = False` (→ S=2000, N_REAL=150). Optionally add a one-off S-independence spot-check cell that reruns ~5 realizations at `S_RUN=10000` and prints their slow-diverger β for comparison (no need to fold into the main pool).

- [ ] **Step 2: Execute the full notebook at production scale**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/slow_divergence_msb.ipynb`
Expected: completes (may take ~30–60 min); dataset regenerated; all figures refreshed; `frac_stationary`, the β triple, growth skew/qtail all populated.

- [ ] **Step 3: Write the verdict markdown cell**

Add a final markdown cell stating, with the actual numbers produced: (a) whether a quasi-stationary growing window exists (`frac_stationary`, median lnM_r2, beta_drift) and over what t_div/f range; (b) if yes, the slow-diverger β (all/persistent) and growth skew + qtail vs the bounded side and the empirical anchors, with a one-paragraph answer to "does the slowly-diverging boundary regime give a cleaner MSB law than the bounded side?"; (c) if the self-similarity check failed, state the accepted negative finding: "the slow-diverger MSB is a condensation transient, not a phase property." Match the plain-prose thesis style (no em-dashes, no \emph) since this may feed the thesis.

- [ ] **Step 4: Final commit**

```bash
git add notebooks/slow_divergence_msb.ipynb notebooks/slow_divergence_*.png
git commit -m "feat(notebook): slow-divergence MSB production run + verdict"
```

- [ ] **Step 5: Persist the finding to memory**

Create `memory/project_slow_divergence_msb.md` (frontmatter `type: project`) summarizing: the regime (Roy spec, per-realization f·σ_c, slow divergers t_div≥150), the self-similarity outcome, the β triple and growth shape vs the bounded side and empirical MSB, and the verdict. Link `[[project_chaotic_glv_dt_sampling]]`, `[[project_ma_phase_needs_gamma]]`, `[[reference_msb_paper]]`. Add a one-line pointer in `MEMORY.md`. Reference the spec and plan paths.

---

## Self-Review

**1. Spec coverage:**
- Dataset generation (per-realization f·σ_c band, store raw) → Tasks 5–6. ✓
- Slow-diverger selection (t_div ≥ T_min, window ≥ W) → Task 7 (`select_msb_window`, Task 3). ✓
- MSB measurement (fixed Δt=1, Moran shares, β all/persistent, growth skew + qtail) → Task 9 (`msb_observables` Task 2, `binned_beta` Task 1). ✓
- Make-or-break self-similarity diagnostic (ln M linear + early/late drift) → Task 8 (`self_similarity_report`, Task 4). ✓
- Comparisons (slow vs bounded vs empirical) → Task 10. ✓
- Failure = accepted finding → Task 8 verdict-gate markdown + Task 11 verdict. ✓
- S-independence spot-check at S=10000 → Task 11 Step 1. ✓
- Constraints (per-capita σ_c threshold, fixed Δt above floor, disorder scaling, gitignored dataset) → Global Constraints + Tasks 5/6. ✓
- Out-of-scope (λ sweep, phase diagram, Lyapunov, mechanism; no edits to chaotic_glv.ipynb / glv core dynamics) → respected; only `glv/analysis.py` (additive) and the new notebook are touched. ✓

**2. Placeholder scan:** No TBD/TODO; every code step shows complete code; every command has expected output. Notebook-cell tasks carry concrete cell bodies. ✓

**3. Type consistency:** `msb_observables` returns `Sbar/vol/g/live/persistent/n_inc` — consumed with those exact keys in Tasks 8/9 and inside `self_similarity_report`. `select_msb_window` returns `qualifies/i0/i1/n_snap/t0/t1` — consumed as `window["i0"]`/`["i1"]` in Tasks 8/9. `self_similarity_report` returns `lnM_slope/lnM_r2/beta_early/beta_late/beta_drift/skew_early/skew_late/stationary` — consumed in Task 8. `run_and_store` records use `status/t_div/t_grid/logx_grid/ref_Sbar/ref_vol/ref_g` consistently across Tasks 6–10. `binned_beta(Sbar, vol)` signature consistent throughout. ✓
