# Adaptive `find_empirical_mu_c` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the uniform-grid sweep in `glv.find_empirical_mu_c` with a three-stage adaptive search (geometric expansion → bisection → dense refine + tanh fit) so evaluation budget is concentrated near the empirical critical point.

**Architecture:** Single function rewrite in `glv/analysis.py`. Three internal helpers (`_eval_mu`, `_expand`, `_bisect`) keep the public API identical in shape but adaptive in sample placement. Tests use a monkeypatched `sweep_final_time` mock that returns a smooth sigmoidal `t̄(mu)` — no real ODE integration. Three notebook callers updated to drop removed kwargs.

**Tech Stack:** Python 3.14, NumPy, SciPy (`curve_fit`, `solve_ivp` via existing `sweep_final_time`), pytest. Run everything with `uv run`.

**Spec:** [docs/superpowers/specs/2026-05-09-adaptive-find-empirical-mu-c-design.md](../specs/2026-05-09-adaptive-find-empirical-mu-c-design.md)

---

## File Structure

- **Modify** [glv/analysis.py](../../../glv/analysis.py) — replace body of `find_empirical_mu_c` (lines 102–198). Public name and import path unchanged.
- **Create** [tests/test_find_empirical_mu_c.py](../../../tests/test_find_empirical_mu_c.py) — unit tests with mocked `sweep_final_time`.
- **Modify** [notebooks/volatility.ipynb](../../../notebooks/volatility.ipynb) — drop `n_mu` / `mu_lo` / `mu_hi` from the one call site.
- **Modify** [notebooks/final_time.ipynb](../../../notebooks/final_time.ipynb) — drop the same kwargs from two call sites.

No new module files. The helpers stay private (underscore-prefixed) inside `analysis.py`.

---

## Task 1: Test scaffolding and forward localization test

**Files:**
- Create: `tests/test_find_empirical_mu_c.py`

- [ ] **Step 1: Write the test file with a mock fixture and the forward-localization test**

```python
# tests/test_find_empirical_mu_c.py
import numpy as np
import pytest
import scipy.sparse as sp

import glv.analysis as analysis


def _make_mock_sweep(mu_true: float, slope: float = 200.0, t_high: float = 40.0,
                     t_low: float = 2.5, noise: float = 0.5, seed: int = 0):
    """Return a function with the same signature as sweep_final_time.

    Mean final time follows a logistic drop from t_high to t_low centred at
    mu_true. mu is recovered from W (the only per-call info that varies):
    W.data = mu/C + sigma/sqrt(C) * z, so mean(W.data) ≈ mu/C, hence
    mu ≈ C * mean(W.data). The mock uses this to read mu out of each Ws.
    """
    rng = np.random.default_rng(seed)

    def mock_sweep(*, Ws, initial_states, N, **_kwargs):
        n_mu = len(Ws)
        n_ic = len(initial_states)
        out = np.empty((n_mu, n_ic))
        for i, W in enumerate(Ws):
            # Recover mu from the weight column (assumes C=N for the mock callers).
            mu_hat = N * float(W.data.mean()) if W.nnz > 0 else 0.0
            base = t_low + (t_high - t_low) / (1.0 + np.exp(slope * (mu_hat - mu_true)))
            out[i, :] = base + rng.normal(0.0, noise, size=n_ic)
        return out

    return mock_sweep


def _make_inputs(N: int = 8, n_ic: int = 3):
    """Build a fully connected adjacency (mod diagonal) so the mock's
    mu = C * mean(W.data) recovery is well-conditioned. Tests pass sigma=0
    so W.data = mu/C exactly, making mu recovery exact."""
    A = np.ones((N, N), dtype=float)
    np.fill_diagonal(A, 0.0)
    ics = [np.ones(N + 2) for _ in range(n_ic)]
    return A, ics


def test_forward_localization(monkeypatch):
    mu_true = 0.05
    A, ics = _make_inputs(N=8)

    monkeypatch.setattr(analysis, "sweep_final_time", _make_mock_sweep(mu_true))

    result = analysis.find_empirical_mu_c(
        mu_c_theoretical=0.0,
        A=A,
        C=8,  # matches N so the mock can recover mu from W.data
        sigma=0.0,
        initial_conditions=ics,
        drop_ratio=0.15,
        expand_step0=0.01,
        expand_max=0.5,
        n_bisect=6,
        n_refine=10,
        n_workers=1,
    )

    assert abs(result["mu_c"] - mu_true) < 0.01
    assert result["direction"] == "forward"
    assert result["bracket"][0] < mu_true < result["bracket"][1]
```

- [ ] **Step 2: Run the test to confirm it fails on the old API**

Run: `uv run pytest tests/test_find_empirical_mu_c.py::test_forward_localization -v`
Expected: FAIL — old `find_empirical_mu_c` doesn't accept `drop_ratio` / `expand_step0` / `expand_max` / `n_bisect` / `n_refine` and returns no `direction` / `bracket` keys.

- [ ] **Step 3: Commit**

```bash
git add tests/test_find_empirical_mu_c.py
git commit -m "test: add forward-localization test for adaptive find_empirical_mu_c"
```

---

## Task 2: Implement the adaptive algorithm

**Files:**
- Modify: `glv/analysis.py:102-198`

- [ ] **Step 1: Add the `warnings` import at the top of `glv/analysis.py`**

Add `import warnings` to the import block at the top of the file (currently lines 1–6).

- [ ] **Step 2: Replace the entire `find_empirical_mu_c` function (lines 102–198) with the adaptive implementation**

```python
def find_empirical_mu_c(
    mu_c_theoretical: float,
    A,
    C: float,
    sigma: float,
    initial_conditions,
    *,
    drop_ratio: float = 0.15,
    expand_step0: float = 0.02,
    expand_max: float = 0.5,
    n_bisect: int = 6,
    n_refine: int = 10,
    tau_max: float = 1e6,
    method: str = "RK45",
    max_step: float | None = 1e2,
    n_workers: int | None = None,
) -> dict:
    """Adaptively locate empirical mu_c by expansion + bisection + tanh fit.

    Probes at mu_c_theoretical, then walks forward (increasing mu) with a
    doubling step until mean final time drops below drop_ratio * t_initial.
    Bisects the resulting bracket n_bisect times, then samples n_refine
    points inside the final bracket. Fits tanh to the union of all samples.

    If forward expansion exhausts (transition not crossed within
    expand_max), retries backward and emits a RuntimeWarning.

    Args:
        mu_c_theoretical: Starting mu for the probe.
        A: Binary adjacency matrix (sparse or dense).
        C: Mean degree used in the weight formula mu/C + sigma/sqrt(C) * z.
        sigma: Std of interaction-strength fluctuations.
        initial_conditions: Sequence of initial state vectors (length N+2).
        drop_ratio: Trigger threshold; bracket forms when t < drop_ratio * t_initial.
        expand_step0: Initial step in the geometric expansion.
        expand_max: Hard cap on expansion step size.
        n_bisect: Number of bisection iterations.
        n_refine: Uniform points to add inside the final bracket.
        tau_max: End of rescaled-time integration passed to sweep_final_time.
        method, max_step, n_workers: forwarded to sweep_final_time.

    Returns:
        dict with keys mu_c, mu_values, mean_t, popt, bracket, direction.

    Raises:
        RuntimeError: transition not found in either direction, or tanh fit fails.
    """
    A_sp = sp.csr_array(A, dtype=float)
    N = len(initial_conditions[0]) - 2
    ics = list(initial_conditions)

    samples: list[tuple[float, float]] = []  # (mu, mean_t)

    def _eval_mu(mu: float) -> float:
        W = A_sp.copy()
        W.data = mu / C + (sigma / np.sqrt(C)) * np.random.normal(0.0, 1.0, len(W.data))
        t_mat = sweep_final_time(
            Ws=[W],
            initial_states=ics,
            N=N,
            tau_max=tau_max,
            method=method,
            max_step=max_step,
            n_workers=n_workers,
        )
        row = t_mat[0]
        mean_t = float(np.nanmean(row)) if np.any(~np.isnan(row)) else float("nan")
        samples.append((mu, mean_t))
        return mean_t

    t_initial = _eval_mu(mu_c_theoretical)
    threshold = drop_ratio * t_initial

    def _expand(direction: int) -> tuple[float, float] | None:
        delta = expand_step0
        mu_prev = mu_c_theoretical
        while delta <= expand_max:
            mu_next = mu_prev + direction * delta
            t_next = _eval_mu(mu_next)
            if t_next < threshold:
                return tuple(sorted([mu_prev, mu_next]))
            mu_prev = mu_next
            delta *= 2
        return None

    bracket = _expand(+1)
    direction = "forward"
    if bracket is None:
        warnings.warn(
            "forward expansion exhausted; empirical transition may be below "
            "theoretical mu_c. Retrying backward.",
            RuntimeWarning,
            stacklevel=2,
        )
        bracket = _expand(-1)
        direction = "backward"
        if bracket is None:
            raise RuntimeError(
                "transition not found within expand_max in either direction"
            )

    mu_lo_b, mu_hi_b = bracket
    for _ in range(n_bisect):
        mu_mid = 0.5 * (mu_lo_b + mu_hi_b)
        t_mid = _eval_mu(mu_mid)
        mid_is_low = t_mid < threshold
        if direction == "forward":
            if mid_is_low:
                mu_hi_b = mu_mid
            else:
                mu_lo_b = mu_mid
        else:
            if mid_is_low:
                mu_lo_b = mu_mid
            else:
                mu_hi_b = mu_mid

    refine_pts = np.linspace(mu_lo_b, mu_hi_b, n_refine + 2)[1:-1]
    for mu in refine_pts:
        _eval_mu(float(mu))

    samples.sort(key=lambda p: p[0])
    mu_values = np.array([m for m, _ in samples])
    mean_t = np.array([t for _, t in samples])

    valid = ~np.isnan(mean_t)
    if valid.sum() < 4:
        raise RuntimeError(f"Too few valid mu points for tanh fit ({valid.sum()} < 4).")

    x_fit = mu_values[valid]
    y_fit = mean_t[valid]

    def _tanh(mu, amp, a, mu0, b):
        return amp * np.tanh(-a * (mu - mu0)) + b

    p0 = [
        (y_fit.max() - y_fit.min()) / 2,
        100.0,
        0.5 * (mu_lo_b + mu_hi_b),
        (y_fit.max() + y_fit.min()) / 2,
    ]
    try:
        popt, _ = curve_fit(_tanh, x_fit, y_fit, p0=p0, maxfev=10000)
    except RuntimeError as exc:
        raise RuntimeError(f"Tanh fit failed: {exc}") from exc

    return {
        "mu_c": float(popt[2]),
        "mu_values": mu_values,
        "mean_t": mean_t,
        "popt": popt,
        "bracket": (float(mu_lo_b), float(mu_hi_b)),
        "direction": direction,
    }
```

- [ ] **Step 3: Run the forward test to confirm it now passes**

Run: `uv run pytest tests/test_find_empirical_mu_c.py::test_forward_localization -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add glv/analysis.py
git commit -m "feat: adaptive expansion + bisection in find_empirical_mu_c"
```

---

## Task 3: Backward fallback test

**Files:**
- Modify: `tests/test_find_empirical_mu_c.py`

- [ ] **Step 1: Append the backward test to the test file**

```python
def test_backward_fallback_with_warning(monkeypatch):
    mu_true = 0.05
    A, ics = _make_inputs(N=8)

    monkeypatch.setattr(analysis, "sweep_final_time", _make_mock_sweep(mu_true))

    with pytest.warns(RuntimeWarning, match="forward expansion exhausted"):
        result = analysis.find_empirical_mu_c(
            mu_c_theoretical=0.20,  # already past the transition
            A=A,
            C=8,
            sigma=0.0,
            initial_conditions=ics,
            drop_ratio=0.15,
            expand_step0=0.01,
            expand_max=0.5,
            n_bisect=6,
            n_refine=10,
            n_workers=1,
        )

    assert result["direction"] == "backward"
    assert abs(result["mu_c"] - mu_true) < 0.01
    assert result["bracket"][0] < mu_true < result["bracket"][1]
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_find_empirical_mu_c.py::test_backward_fallback_with_warning -v`
Expected: PASS (the implementation already handles backward fallback from Task 2).

- [ ] **Step 3: Commit**

```bash
git add tests/test_find_empirical_mu_c.py
git commit -m "test: backward fallback emits warning and locates mu_c"
```

---

## Task 4: Out-of-range failure test

**Files:**
- Modify: `tests/test_find_empirical_mu_c.py`

- [ ] **Step 1: Append the failure test**

```python
def test_out_of_range_raises(monkeypatch):
    # Transition far away from mu_theoretical, expand_max too small to reach it.
    mu_true = 1.0
    A, ics = _make_inputs(N=8)

    monkeypatch.setattr(analysis, "sweep_final_time", _make_mock_sweep(mu_true))

    with pytest.raises(RuntimeError, match="transition not found"):
        analysis.find_empirical_mu_c(
            mu_c_theoretical=0.0,
            A=A,
            C=8,
            sigma=0.0,
            initial_conditions=ics,
            drop_ratio=0.15,
            expand_step0=0.01,
            expand_max=0.05,  # too small to reach mu_true=1.0
            n_bisect=6,
            n_refine=10,
            n_workers=1,
        )
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_find_empirical_mu_c.py::test_out_of_range_raises -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_find_empirical_mu_c.py
git commit -m "test: raise RuntimeError when transition outside expand_max"
```

---

## Task 5: Output invariants test

**Files:**
- Modify: `tests/test_find_empirical_mu_c.py`

- [ ] **Step 1: Append the invariants test**

```python
def test_output_invariants(monkeypatch):
    mu_true = 0.05
    A, ics = _make_inputs(N=8)

    monkeypatch.setattr(analysis, "sweep_final_time", _make_mock_sweep(mu_true))

    n_bisect = 6
    n_refine = 10
    result = analysis.find_empirical_mu_c(
        mu_c_theoretical=0.0,
        A=A,
        C=8,
        sigma=0.0,
        initial_conditions=ics,
        drop_ratio=0.15,
        expand_step0=0.01,
        expand_max=0.5,
        n_bisect=n_bisect,
        n_refine=n_refine,
        n_workers=1,
    )

    mus = result["mu_values"]
    assert mus.ndim == 1
    assert np.all(np.diff(mus) >= 0), "mu_values must be sorted ascending"
    # probe + at least one expansion step + n_bisect mids + n_refine
    assert len(mus) >= 2 + n_bisect + n_refine
    assert np.any(np.isclose(mus, 0.0)), "mu_theoretical must appear in mu_values"
    lo, hi = result["bracket"]
    assert np.any(np.isclose(mus, lo))
    assert np.any(np.isclose(mus, hi))
    assert result["mean_t"].shape == mus.shape
    assert result["popt"].shape == (4,)
```

- [ ] **Step 2: Run the full test file**

Run: `uv run pytest tests/test_find_empirical_mu_c.py -v`
Expected: 4 PASSED.

- [ ] **Step 3: Commit**

```bash
git add tests/test_find_empirical_mu_c.py
git commit -m "test: output invariants for adaptive find_empirical_mu_c"
```

---

## Task 6: Update `notebooks/volatility.ipynb` caller

**Files:**
- Modify: `notebooks/volatility.ipynb` (the cell containing the `glv.find_empirical_mu_c(` call near line 106)

- [ ] **Step 1: Inspect the current call to know what kwargs to remove**

Run: `uv run jupyter nbconvert --to script notebooks/volatility.ipynb --stdout 2>/dev/null | grep -n -A 20 "find_empirical_mu_c"`
Note which of `n_mu`, `mu_lo`, `mu_hi` are passed and what other kwargs survive.

- [ ] **Step 2: Open the notebook in the IDE and edit the call cell**

In the cell that contains `glv.find_empirical_mu_c(`:
- Remove any `n_mu=...`, `mu_lo=...`, `mu_hi=...` kwargs.
- Keep all other kwargs unchanged.
- If the old `mu_hi - mu_lo` was significantly less than the new default `expand_max=0.5`, add `expand_max=<old_mu_hi>` so the search range matches what the notebook previously used. Otherwise leave defaults.

- [ ] **Step 3: Restart kernel and run the affected cell to confirm it executes without error**

Open the notebook, restart the kernel, run the cells up to and including the `find_empirical_mu_c` call. Confirm no `TypeError: unexpected keyword argument` and that `result["mu_c"]` is finite.

- [ ] **Step 4: Commit**

```bash
git add notebooks/volatility.ipynb
git commit -m "chore: update volatility.ipynb for adaptive find_empirical_mu_c API"
```

---

## Task 7: Update `notebooks/final_time.ipynb` callers (two sites)

**Files:**
- Modify: `notebooks/final_time.ipynb` (cells with calls near lines 64 and 112)

- [ ] **Step 1: Identify both call sites**

Run: `uv run jupyter nbconvert --to script notebooks/final_time.ipynb --stdout 2>/dev/null | grep -n -B 1 -A 15 "find_empirical_mu_c"`

- [ ] **Step 2: Edit each call**

For each of the two cells containing `glv.find_empirical_mu_c(`:
- Remove any `n_mu=...`, `mu_lo=...`, `mu_hi=...` kwargs.
- If the old span was tighter than `expand_max=0.5`, set `expand_max=<old_span>`.
- Leave everything else unchanged.

- [ ] **Step 3: Restart kernel and run both affected cells**

Confirm no kwarg errors and that returned `mu_c` values are finite.

- [ ] **Step 4: Commit**

```bash
git add notebooks/final_time.ipynb
git commit -m "chore: update final_time.ipynb for adaptive find_empirical_mu_c API"
```

---

## Task 8: Final verification

- [ ] **Step 1: Run the full test suite to confirm nothing else broke**

Run: `uv run pytest -v`
Expected: all tests pass, including the four new ones in `tests/test_find_empirical_mu_c.py` and the pre-existing `tests/test_dynamics.py` and `tests/test_graph.py`.

- [ ] **Step 2: Confirm `git status` is clean**

Run: `git status`
Expected: working tree clean (all changes committed in tasks 1–7).
