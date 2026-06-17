# Langevin GLV at the Critical Point — MSB Observables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one self-contained notebook that integrates a multiplicative-noise (Langevin) GLV sitting exactly at the cooperative critical point μ=μ_c with disorder σ=0.2, and computes the Moran–Secchi–Bouchaud size–volatility relation and growth-rate distribution.

**Architecture:** Integrate the *relative composition* (simplex-renormalized) via Euler–Maruyama so trajectories stay bounded despite the M-divergence at μ_c. Per graph realization: build a power-law graph, locate empirical μ_c with the repo's shape-scalar locator on a frozen disorder draw, rebuild the weight matrix at that μ_c with the *same* draw, integrate the SDE, accumulate relative-size and growth observables. Pool over realizations per noise strength s, cache, and plot.

**Tech Stack:** Python via `uv run`, numpy, scipy.sparse, networkx, matplotlib; `glv` package (`apply_style`, `find_mu_c_shape_scalar`, `calculate_mu_c_regular`).

## Global Constraints

- Run all Python with `uv run python` / `uv run jupyter`, never bare `python`.
- Keep the graph builder and integrator inline in the notebook — do not add a library abstraction.
- Plot labels/titles use proper LaTeX mathtext, not raw identifiers.
- float64 throughout the integrator (never cast state to float32 — the prior float32 overflow trap).
- σ here = interaction-matrix disorder σ_mat = 0.2; s = Langevin multiplicative-noise strength (independent sweep).
- New files use the `langevin_glv_critical_*` prefix.

---

### Task 1: Notebook scaffold — config + inline graph builder

**Files:**
- Create: `notebooks/langevin_glv_critical_msb.ipynb`

**Interfaces:**
- Produces: `build_adjacency(seed) -> (A_coo, C)` where `A` is a binary COO adjacency (N×N) and `C=float` mean degree; module constants `N, ALPHA, KMIN, KMAX, SIGMA_MAT, S_GRID, dt, tmax, burn, sample_dt, n_runs, FLOOR, RES`.

- [ ] **Step 1: Create the notebook with a title markdown cell**

Markdown cell content:

```markdown
# Langevin GLV at the critical point — MSB observables

A multiplicative-noise GLV sitting exactly at the cooperative critical point $\mu=\mu_c$
($\sigma_{\rm mat}=0.2$). At $\mu_c$ the total abundance $M=\sum_i x_i$ diverges, so we integrate the
**relative composition** $y$ (simplex-renormalized, bounded) and read MSB off the relative size
$S_i=N y_i$. Euler–Maruyama in rescaled time $\tau$:
$$d\ln y_i=\big(F_i-\phi-y_i+\textstyle\sum_j y_j^2\big)\,d\tau+s\,dW_i,\qquad F=Wy,\ \phi=y^\top F,$$
renormalizing $y\leftarrow y/\sum y$ each step. Growth $g_i=\Delta\ln S_i$, volatility
$\sigma_i=\sqrt{\pi/2}\,\overline{|g-\bar g|}$, swept over noise strength $s$.
```

- [ ] **Step 2: Config + imports code cell**

```python
import os
import numpy as np
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

import glv
from glv.analysis import find_mu_c_shape_scalar, calculate_mu_c_regular
glv.apply_style()

N          = 400
ALPHA      = 1.5          # power-law degree tail P(k) ~ k^-(1+alpha)
KMIN       = 5.0          # <k> ~ 11
KMAX       = 120
SIGMA_MAT  = 0.2          # interaction-matrix disorder
S_GRID     = np.array([0.05, 0.1, 0.2, 0.3])   # Langevin noise strengths
dt         = 0.02         # Euler-Maruyama step in rescaled tau
tmax       = 600.0        # total rescaled-time horizon
burn       = 0.3          # discard first fraction (reach stochastic steady state)
sample_dt  = 1.0          # record every sample_dt in tau
n_runs     = 6            # graph realizations pooled per s
FLOOR      = 1e-6         # live-size mask on mean relative size
RES        = "langevin_glv_critical_results.npz"
print(f"Langevin GLV @ mu_c | sigma_mat={SIGMA_MAT}, <k>~11 | noise s in {list(S_GRID)} "
      f"| dt={dt}, tmax={tmax:g} | N={N} x {n_runs} runs/s")
```

- [ ] **Step 3: Inline graph builder cell**

```python
# Inline power-law configuration-model graph (kept inline, not abstracted).
def build_adjacency(seed):
    rng = np.random.default_rng(seed)
    k = np.clip(np.round(KMIN * rng.uniform(0, 1, N) ** (-1.0 / ALPHA)).astype(int), 1, KMAX)
    if k.sum() % 2:
        k[0] += 1
    G = nx.Graph(nx.configuration_model(list(k), seed=int(seed)))
    G.remove_edges_from(nx.selfloop_edges(G))
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float).tocoo()
    return A, float(A.getnnz(axis=1).mean())
```

- [ ] **Step 4: Run cells 1-3; smoke-check the builder**

Add a temporary line at the end of the builder cell, run it, confirm output, then delete the line:

```python
_A, _C = build_adjacency(7000); print(f"smoke: N={_A.shape[0]}, <k>={_C:.1f}, nnz={_A.getnnz()}")
```
Expected: `N=400, <k>=` roughly 10–12, nnz a few thousand. Delete this line after confirming.

- [ ] **Step 5: Commit**

```bash
git add notebooks/langevin_glv_critical_msb.ipynb
git commit -m "feat(notebooks): Langevin critical-MSB scaffold + graph builder"
```

---

### Task 2: μ_c locator + weight matrix at the critical point

**Files:**
- Modify: `notebooks/langevin_glv_critical_msb.ipynb`

**Interfaces:**
- Consumes: `build_adjacency`, `SIGMA_MAT`, `find_mu_c_shape_scalar`, `calculate_mu_c_regular`.
- Produces: `make_W(A, C, mu, seed) -> csr_array` (weights with the SAME frozen disorder draw keyed by `seed`); `locate_mu_c(A, C, seed) -> float`.

- [ ] **Step 1: Weight-matrix builder cell (frozen draw keyed by seed)**

The locator `find_mu_c_shape_scalar(A, C, sigma, mus, seed=seed)` draws `z = default_rng(seed).standard_normal(nnz)` over the COO edges. Reuse the exact same draw so the Langevin run sits on the located point:

```python
# Weights W_ij = A_ij*(mu/C + sigma/sqrt(C) z_ij) with z frozen by `seed`
# (identical draw to find_mu_c_shape_scalar, so we sit on the located mu_c).
def make_W(A, C, mu, seed):
    z = np.random.default_rng(seed).standard_normal(A.row.size)
    data = mu / C + (SIGMA_MAT / np.sqrt(C)) * z
    return sp.csr_array((data, (A.row, A.col)), shape=A.shape)
```

- [ ] **Step 2: μ_c locator wrapper cell**

```python
# Locate empirical mu_c on the realized graph via the shape-scalar zero-crossing.
# Grid centered on the regular-graph theoretical value, widened (power-law != regular).
def locate_mu_c(A, C, seed):
    center = calculate_mu_c_regular(SIGMA_MAT)            # ~1 at sigma=0.2
    mus = np.linspace(center - 0.4, center + 0.4, 41)
    out = find_mu_c_shape_scalar(A, C, sigma=SIGMA_MAT, mus=mus, seed=seed)
    return float(out["mu_c"])
```

- [ ] **Step 3: Run and verify a single μ_c is finite and inside the grid**

Temporary cell (delete after confirming):

```python
_A, _C = build_adjacency(7000)
_mc = locate_mu_c(_A, _C, 7000)
print(f"theoretical mu_c(regular,0.2)={calculate_mu_c_regular(SIGMA_MAT):.4f}  empirical mu_c={_mc:.4f}")
assert np.isfinite(_mc), "mu_c not located — widen the grid"
```
Expected: both values finite, empirical within the ±0.4 window of the center. If `nan`, widen the grid and rerun. Delete this cell after confirming.

- [ ] **Step 4: Commit**

```bash
git add notebooks/langevin_glv_critical_msb.ipynb
git commit -m "feat(notebooks): mu_c locator + frozen-draw weight matrix"
```

---

### Task 3: Euler–Maruyama integrator for the relative composition

**Files:**
- Modify: `notebooks/langevin_glv_critical_msb.ipynb`

**Interfaces:**
- Consumes: `make_W`, `N, dt, tmax, sample_dt`.
- Produces: `integrate_relative(W, s, seed) -> (t, S)` where `t` is shape (T,) sampled rescaled-times and `S` is shape (N, T) relative sizes (S=N·y, columns sum to N). Raises `AssertionError` on non-finite / off-simplex state.

- [ ] **Step 1: Integrator cell (log-space EM, renormalize each step, float64, inline assertions)**

```python
# EM on the relative composition y (Sum y = 1), bounded by construction.
# d ln y = (F - phi - y + sum y^2) dtau + s dW ; renormalize y each step.
def integrate_relative(W, s, seed):
    rng = np.random.default_rng(seed)
    y = np.full(N, 1.0 / N, dtype=np.float64)
    nstep = int(tmax / dt)
    stride = max(int(sample_dt / dt), 1)
    sdt = s * np.sqrt(dt)
    rec, rect = [], []
    for n in range(nstep):
        F = W @ y
        phi = float(y @ F)
        sq = float(y @ y)
        u = np.log(y) + (F - phi - y + sq) * dt + sdt * rng.standard_normal(N)
        u -= u.max()                       # stabilize exp
        y = np.exp(u)
        y /= y.sum()                       # renormalize to simplex
        assert np.isfinite(y).all(), f"non-finite state at step {n}"
        if n % stride == 0:
            rec.append(N * y); rect.append(n * dt)
    S = np.array(rec, dtype=np.float64).T  # (N, T)
    assert np.allclose(S.sum(axis=0), N, rtol=1e-6), "columns of S must sum to N"
    return np.array(rect), S
```

- [ ] **Step 2: Run on one realization and assert bounded/finite behaviour**

Temporary cell (delete after confirming):

```python
_A, _C = build_adjacency(7000); _mc = locate_mu_c(_A, _C, 7000)
_W = make_W(_A, _C, _mc, 7000)
_t, _S = integrate_relative(_W, 0.1, 7000)
print(f"S shape={_S.shape}, min={_S.min():.2e}, max={_S.max():.2f}, "
      f"col-sum≈{_S.sum(axis=0).mean():.3f} (=N)")
assert np.isfinite(_S).all() and _S.max() < N, "S must be finite and < N (relative size)"
```
Expected: `S shape=(400, ~420)`, all finite, max well below N, col-sum ≈ 400. Delete after confirming.

- [ ] **Step 3: Commit**

```bash
git add notebooks/langevin_glv_critical_msb.ipynb
git commit -m "feat(notebooks): relative-composition Euler-Maruyama integrator"
```

---

### Task 4: Sweep over noise s, accumulate observables, cache

**Files:**
- Modify: `notebooks/langevin_glv_critical_msb.ipynb`

**Interfaces:**
- Consumes: `build_adjacency, locate_mu_c, make_W, integrate_relative, S_GRID, n_runs, burn, tmax, RES`.
- Produces: in-memory `by_s` dict keyed by float s → `{"avg": (M,), "vol": (M,), "g": (K,), "mu_c": [..]}` pooled over realizations; cached `.npz` at `RES`.

- [ ] **Step 1: Sweep + observables cell with caching**

```python
# Sweep noise s; per realization locate mu_c, integrate, pool steady-state observables.
def observables(S, t):
    Sl = S[:, t >= burn * tmax]
    log_S = np.log(np.maximum(Sl, 1e-15))
    g = np.diff(log_S, axis=1)
    g_bar = g.mean(axis=1, keepdims=True)
    vol = np.sqrt(np.pi / 2.0) * np.mean(np.abs(g - g_bar), axis=1)
    return Sl.mean(axis=1), vol, g.ravel()

if os.path.exists(RES):
    d = np.load(RES, allow_pickle=True)
    by_s = d["by_s"].item()
    print(f"loaded cache {RES}")
else:
    by_s = {float(s): {"avg": [], "vol": [], "g": [], "mu_c": []} for s in S_GRID}
    for i, s in enumerate(S_GRID):
        for r in range(n_runs):
            seed = 7000 + 13 * i + r
            A, C = build_adjacency(seed)
            mc = locate_mu_c(A, C, seed)
            if not np.isfinite(mc):
                print(f"  s={s:g} run {r}: mu_c not found, skipped"); continue
            W = make_W(A, C, mc, seed)
            t, S = integrate_relative(W, float(s), seed)
            a, v, g = observables(S, t)
            d = by_s[float(s)]
            d["avg"].append(a); d["vol"].append(v); d["g"].append(g); d["mu_c"].append(mc)
            print(f"  s={s:g} run {r}: mu_c={mc:.3f}, <S>max={a.max():.2f}, vol[med]={np.median(v):.3f}")
    for s in by_s:
        for key in ("avg", "vol", "g"):
            by_s[s][key] = np.concatenate(by_s[s][key]) if by_s[s][key] else np.array([])
    np.savez(RES, by_s=np.array(by_s, dtype=object))
    print(f"saved {RES}")
```

- [ ] **Step 2: Run the sweep (uncached) to completion**

Run the cell. Expected: 4 noise levels × 6 runs = 24 lines printing finite `mu_c` and `vol[med]`, then `saved ...`. No assertion errors, no `mu_c not found`. If the run is slow, that is expected (24 integrations); let it finish.

- [ ] **Step 3: Commit (code + cache)**

```bash
git add notebooks/langevin_glv_critical_msb.ipynb notebooks/langevin_glv_critical_results.npz
git commit -m "feat(notebooks): noise sweep + cached critical-MSB observables"
```

---

### Task 5: Plots — size–volatility relation and growth distribution

**Files:**
- Modify: `notebooks/langevin_glv_critical_msb.ipynb`

**Interfaces:**
- Consumes: `by_s, S_GRID, FLOOR, SIGMA_MAT`.
- Produces: PNGs `langevin_glv_critical_volatility_size.png`, `langevin_glv_critical_growth_dist.png`.

- [ ] **Step 1: Binning + two-branch slope helpers cell**

```python
def vbin(x, y, nb=22):
    o = np.argsort(x); xp, yp = x[o], y[o]
    parts = np.array_split(np.arange(xp.size), nb)
    return (np.array([xp[p].mean() for p in parts]), np.array([np.median(yp[p]) for p in parts]))

def two_slopes(x, y):
    bx, by = vbin(x, y)
    j = int(np.argmin(by))
    if j < 1 or j > len(bx) - 2:
        sl = np.polyfit(np.log10(bx), np.log10(by), 1)[0]
        return bx[j], sl, sl
    lo = np.polyfit(np.log10(bx[:j + 1]), np.log10(by[:j + 1]), 1)[0]
    hi = np.polyfit(np.log10(bx[j:]), np.log10(by[j:]), 1)[0]
    return bx[j], lo, hi
```

- [ ] **Step 2: Plot 1 — size–volatility, one V-curve per s**

```python
fig, ax = plt.subplots(figsize=(7.8, 5.8))
cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(S_GRID)))
for s, col in zip(sorted(by_s), cmap):
    b = by_s[s]
    if b["avg"].size == 0:
        continue
    live = (b["avg"] > FLOOR) & (b["vol"] > 0) & np.isfinite(b["vol"])
    if live.sum() < 50:
        continue
    bx, by = vbin(b["avg"][live], b["vol"][live])
    _, blo, bhi = two_slopes(b["avg"][live], b["vol"][live])
    ax.loglog(bx, by, "o-", ms=4, color=col,
              label=rf"$s={s:g}$  ($\beta_{{\rm lo}}={blo:+.2f},\ \beta_{{\rm hi}}={bhi:+.2f}$)")
ax.set(xlabel=r"time-average relative size $\bar S_i$",
       ylabel=r"growth-rate volatility $\sigma_i=\sqrt{\pi/2}\,\overline{|g-\bar g|}$",
       title=rf"Langevin GLV at $\mu_c$: size--volatility vs noise $s$  ($\sigma_{{\rm mat}}={SIGMA_MAT}$)")
ax.grid(True, which="both", alpha=0.15)
ax.legend(fontsize=8, title="noise strength")
plt.tight_layout()
plt.savefig("langevin_glv_critical_volatility_size.png", dpi=120)
plt.show()
```

- [ ] **Step 3: Plot 2 — pooled growth distribution at a representative s**

```python
s_show = 0.2 if 0.2 in by_s else sorted(by_s)[len(by_s) // 2]
g = by_s[s_show]["g"]; g = g[np.isfinite(g)]
mu_g, sd_g, med_g = g.mean(), g.std(), np.median(g)
b_lap = np.mean(np.abs(g - med_g))
lo, hi = np.percentile(g, [0.2, 99.8])
edges = np.linspace(lo, hi, 80); centers = 0.5 * (edges[:-1] + edges[1:])
dens, _ = np.histogram(g, bins=edges, density=True)
gauss = np.exp(-0.5 * ((centers - mu_g) / sd_g) ** 2) / (sd_g * np.sqrt(2 * np.pi))
lap = np.exp(-np.abs(centers - med_g) / b_lap) / (2 * b_lap)
fig, ax = plt.subplots(figsize=(7.5, 5.6))
ax.semilogy(centers, np.where(dens > 0, dens, np.nan), "o", ms=3, color="#1d3557",
            label=rf"pooled $g=\Delta\ln S$ ($s={s_show:g}$)")
ax.semilogy(centers, gauss, "-", color="#457b9d", lw=1.5, label=r"Gaussian (same $\sigma$)")
ax.semilogy(centers, lap, "--", color="#c1121f", lw=1.5, label="Laplace (same scale)")
ax.set(xlabel=r"growth rate $g=\Delta\ln S_i$", ylabel="density",
       title=rf"Langevin GLV at $\mu_c$: growth-rate distribution ($\sigma_{{\rm mat}}={SIGMA_MAT}$, $s={s_show:g}$)")
ax.grid(True, which="both", alpha=0.15)
ax.legend()
plt.tight_layout()
plt.savefig("langevin_glv_critical_growth_dist.png", dpi=120)
plt.show()
print(f"s={s_show}: mean={mu_g:+.4f}, median={med_g:+.4f}, std={sd_g:.4f}, "
      f"skew={skew(g):+.2f}, excess_kurtosis={kurtosis(g):+.2f}")
```

- [ ] **Step 4: Run both plot cells; confirm figures render and PNGs are written**

Expected: two figures display; `langevin_glv_critical_volatility_size.png` and `langevin_glv_critical_growth_dist.png` exist on disk; the printed line shows finite skew/kurtosis.

- [ ] **Step 5: Commit**

```bash
git add notebooks/langevin_glv_critical_msb.ipynb notebooks/langevin_glv_critical_volatility_size.png notebooks/langevin_glv_critical_growth_dist.png
git commit -m "feat(notebooks): critical-MSB size-volatility and growth-distribution plots"
```

---

### Task 6: Summary + findings

**Files:**
- Modify: `notebooks/langevin_glv_critical_msb.ipynb`

- [ ] **Step 1: Markdown summary cell**

Write a short summary cell reporting the measured numbers (read off the run): the located μ_c range across realizations, the two-branch size–volatility slopes β per s and whether the exponent is robust across s, and the growth-distribution shape (skew, excess kurtosis, Gaussian vs Laplace). State plainly what the regime shows; do not claim it "reproduces MSB" beyond what the numbers support.

- [ ] **Step 2: Commit**

```bash
git add notebooks/langevin_glv_critical_msb.ipynb
git commit -m "docs(notebooks): critical-MSB summary and findings"
```

---

## Self-Review

**Spec coverage:**
- Config (N, power-law, σ=0.2, s-sweep) → Task 1. ✓
- Locate μ_c on realized graph, reuse frozen draw → Task 2. ✓
- Approach-B relative/rescaled EM, simplex renormalize, inline assertions, float64 → Task 3. ✓
- Observables (S̄_i, σ_i, pooled g), cache .npz → Task 4. ✓
- Two PNGs (size–volatility, growth dist) → Task 5. ✓
- Deletions of prior work → done before this plan (not a task here). ✓
- Out of scope (no frozen-field null, no pytest, no absolute-x) → honored. ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code. ✓

**Type consistency:** `build_adjacency -> (A, C)`, `make_W(A, C, mu, seed)`, `locate_mu_c(A, C, seed)`, `integrate_relative(W, s, seed) -> (t, S)`, `observables(S, t) -> (avg, vol, g)`, `by_s[float(s)]` schema consistent across Tasks 1–5. ✓
