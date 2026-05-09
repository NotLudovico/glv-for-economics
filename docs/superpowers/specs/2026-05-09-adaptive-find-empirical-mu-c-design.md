# Adaptive `find_empirical_mu_c` — Design

**Date:** 2026-05-09
**Module:** `glv/analysis.py`
**Function:** `find_empirical_mu_c` (in-place replacement)

## Motivation

The current implementation evaluates `n_mu` ODE sweeps on a uniform grid in
`[mu_c_theoretical + mu_lo, mu_c_theoretical + mu_hi]`, then fits a tanh to
`(mu, mean_t)` and returns the midpoint. Most of those evaluations land far
from the transition, where the mean final time is essentially constant — the
useful information is concentrated in a narrow band around the empirical
critical value. We want to spend evaluation budget on that band.

Empirical observation that drives the design: well below the transition,
mean final times are ~40; well above, they collapse to ~2–3. The drop is
sharp and monotonic in `mu`, so a binary search on `t̄(mu)` is well-posed.

## Algorithm

Three stages plus a probe and a fit, sharing one per-mu evaluation primitive
(regenerate `W` on `A`, run all `initial_conditions` via `sweep_final_time`,
return the mean final time `t̄(mu)`).

### Stage 0 — Probe

1. Evaluate `t_initial = t̄(mu_theoretical)`.

This single value cannot by itself tell us which side of the transition we
are on (the typical pre/post times are instance-dependent). We assume
forward and let stage 1 fail over to backward if needed.

### Stage 1 — Geometric expansion (forward, then fallback backward)

Walk away from `mu_theoretical` with a doubling step. The trigger uses
`t_initial` as the high-side reference, so if `t_initial` is itself already
low, no forward step will satisfy the trigger and the loop exhausts. In
that case, emit a `RuntimeWarning` and retry with `direction = "backward"`.

```
def _expand(direction):  # direction in (+1, -1)
    delta = expand_step0
    mu_prev, t_prev = mu_theoretical, t_initial
    history = []
    while delta <= expand_max:
        mu_next = mu_prev + direction * delta
        t_next = t̄(mu_next)
        history.append((mu_next, t_next))
        if t_next < drop_ratio * t_initial:
            bracket = sorted([mu_prev, mu_next])
            return bracket, history
        mu_prev, t_prev = mu_next, t_next
        delta *= 2
    return None, history  # exhausted

bracket, fwd_hist = _expand(+1)
direction = "forward"
if bracket is None:
    warnings.warn(
        "forward expansion exhausted; empirical transition may be below "
        "theoretical mu_c. Retrying backward.",
        RuntimeWarning,
    )
    bracket, bwd_hist = _expand(-1)
    direction = "backward"
    if bracket is None:
        raise RuntimeError(
            "transition not found within expand_max in either direction"
        )
```

```
delta = expand_step0
mu_prev = mu_theoretical
while delta <= expand_max:
    mu_next = mu_prev + sign(direction) * delta
    t_next = t̄(mu_next)
    if t_next < drop_ratio * t_initial:
        bracket = sorted([mu_prev, mu_next])     # high-t, low-t pair
        break
    mu_prev = mu_next
    delta *= 2
else:
    raise RuntimeError("transition not found within expand_max")
```

Concretely, for `direction = "forward"`, `sign = +1`; for `"backward"`,
`sign = -1`. The bracket invariant after stage 1:

- `t̄(mu_high_t_side) ≥ drop_ratio * t_initial`
- `t̄(mu_low_t_side)  < drop_ratio * t_initial`

For a symmetric API the bracket is stored as `(mu_lo_b, mu_hi_b)` with
`mu_lo_b < mu_hi_b`; which endpoint is the "high-t" side depends on
`direction`.

### Stage 2 — Bisection

Run `n_bisect` iterations:

```
for _ in range(n_bisect):
    mu_mid = 0.5 * (mu_lo_b + mu_hi_b)
    t_mid = t̄(mu_mid)
    if t_mid < drop_ratio * t_initial:
        # mid is on the low-t side
        if direction == "forward":
            mu_hi_b = mu_mid
        else:
            mu_lo_b = mu_mid
    else:
        if direction == "forward":
            mu_lo_b = mu_mid
        else:
            mu_hi_b = mu_mid
```

After `n_bisect` steps the bracket is the stage-1 bracket width divided by
`2^n_bisect`. With defaults that's typically < 0.01.

### Stage 3 — Dense refine

Pick `n_refine` uniformly-spaced points strictly inside the final
`(mu_lo_b, mu_hi_b)` bracket (open interval, e.g. `np.linspace(..., n_refine + 2)[1:-1]`)
and evaluate `t̄` at each.

### Stage 4 — Tanh fit on union

Combine **all** `(mu, t̄)` pairs from stages 0–3 (probe + expansion +
bisection + refine), sort by mu, drop NaN entries, and fit:

```
t̄(mu) ≈ amp * tanh(-a * (mu - mu0)) + b
```

with the same initial guess strategy as the current implementation. Return
`mu_c = popt[2]`.

## API

Replace the body of `glv.analysis.find_empirical_mu_c`. New signature
(keyword-only after `initial_conditions`):

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
) -> dict
```

**Removed args:** `n_mu`, `mu_lo`, `mu_hi`. (Bracket is now adaptive.)

**Returned dict:**

| key          | type                | meaning                                              |
|--------------|---------------------|------------------------------------------------------|
| `mu_c`       | `float`             | tanh midpoint (`popt[2]`).                           |
| `mu_values`  | `np.ndarray`        | All evaluated mu, sorted ascending.                  |
| `mean_t`     | `np.ndarray`        | Mean final time at each mu (NaN where no run converged). |
| `popt`       | `np.ndarray`        | Tanh fit params `(amp, a, mu0, b)`.                  |
| `bracket`    | `tuple[float, float]` | Final bisection bracket `(mu_lo_b, mu_hi_b)`.      |
| `direction`  | `str`               | `"forward"` or `"backward"`.                         |

## Per-mu evaluation primitive

Refactor the current per-mu logic out of the function body into a small
internal helper:

```python
def _eval_mu(mu: float) -> tuple[float, np.ndarray]:
    """Build W from A at this mu, run all ICs, return (mean_t, raw_t_row)."""
```

This keeps the algorithm body readable and ensures stages 0–3 use identical
evaluation semantics. Each call still creates fresh random weights — this
preserves the existing stochastic structure of the current sweep.

## Error handling

- **Expansion exhausted in both directions**: `RuntimeError("transition not
  found within expand_max in either direction")`. Don't silently return a
  bad estimate.
- **Backward fallback**: when forward exhausts but backward succeeds, emit
  one `RuntimeWarning` and continue. Caller can suppress via standard
  `warnings` filters. The cost of the wasted forward expansion is
  `O(log2(expand_max / expand_step0))` evals — small.
- **Tanh fit failure**: same as today — `RuntimeError("Tanh fit failed: ...")`.
  The richer sample set should make this rarer than the current
  uniform-grid case.
- **Stochastic flukes near the threshold**: not specially handled. Bisection
  remains correct as long as the *aggregate* `t̄(mu)` curve is monotonic on
  average; per-realization wobbles get smoothed by the tanh fit.

## Testing

`tests/test_find_empirical_mu_c.py` (new). Patch
`glv.analysis.sweep_final_time` with a fast mock that returns a smooth
sigmoidal `t̄(mu) = t_high / (1 + exp(slope * (mu - mu_true)))` plus small
Gaussian noise — no ODE integration.

Test cases:

1. **Forward localization.** `mu_theoretical = mu_true - 0.05`. Asserts
   `|result["mu_c"] - mu_true| < 0.01` and `result["direction"] == "forward"`.
2. **Backward fallback with warning.** `mu_theoretical = mu_true + 0.05`.
   Asserts `RuntimeWarning` is emitted, `result["direction"] == "backward"`,
   and `mu_c` is still within tolerance.
3. **Out-of-range failure.** `expand_max` set so the transition is
   unreachable in both directions. Asserts `RuntimeError`.
4. **Output invariants.** `result["mu_values"]` is sorted ascending,
   contains `mu_theoretical` and the bracket endpoints, and has length
   `≥ 2 + n_bisect + n_refine` (probe + at least one expansion eval +
   bisection mids + refine).

Run with `uv run pytest tests/test_find_empirical_mu_c.py`.

## Caller updates

Three call sites use the removed args and need touch-ups:

- `notebooks/volatility.ipynb` (1 call)
- `notebooks/final_time.ipynb` (2 calls)

Each will be updated to drop `n_mu` / `mu_lo` / `mu_hi` and pass the new
adaptive parameters where the existing values implied something specific
(e.g. tighter `expand_max` if the old `mu_hi - mu_lo` was small).

## Out of scope

- Cheap-probe variants (single-IC stages 1–2). The user opted to keep all
  ICs at every evaluation. Can be revisited later as an `probe_n_ic` arg if
  needed.
- Parallelizing stage-3 refinement points across workers. The existing
  `sweep_final_time` already parallelizes ICs per mu; cross-mu parallelism
  is a separate optimization.
- Replacing the tanh fit with another functional form (e.g. logistic). The
  current fit is good enough; the speedup comes from sample placement, not
  fit shape.
