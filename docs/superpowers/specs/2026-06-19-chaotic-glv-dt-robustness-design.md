# Δt-sampling robustness of the MSB observables in `chaotic_glv.ipynb` — design

**Date:** 2026-06-19
**Status:** Approved (proceeding to implementation)
**Deliverable:** additive cells in `notebooks/chaotic_glv.ipynb` (existing cells untouched)

## Goal

Check whether the **adaptive vs fixed Δt** choice in the ensemble MSB cells distorts the two
firm-growth observables — the size–volatility exponent β and the pooled growth distribution
(skew, excess kurtosis, tent shape). Deliver a clear verdict: are the MSB facts robust to the
sampling choice, or partly a pooling artifact?

## Motivation — the distortion mechanism

The ensemble cells (`run_once`, cells 18–23) snapshot each run on an **adaptive** yearly grid
`t_yr = linspace(0, t_reached, N_YEARS+1)`, so the per-year horizon is

```
Δt_adaptive = t_reached / N_YEARS   (run-specific)
```

In the Multiple-Attractors / Unbounded phase runs diverge at very different times: one that
blows up at t≈6 gets Δt≈0.06, one that reaches `tmax=500` gets Δt=5.0 — an ~80× spread. The
pooled growth histogram concatenates `g = Δln S` measured over these heterogeneous horizons.

Pooling growth rates measured over different horizons is a textbook way to manufacture fat
tails: a **scale mixture of Gaussians is leptokurtic** even when every component is Gaussian.
So the headline MSB "fat-tailed tent" — and the β fit — may be inflated by horizon mixing
rather than by genuine granular reshuffling. The notebook already flags this in prose
("pooled growth rates are therefore not on a common Δt"); this study quantifies it.

## Approach (paired re-gridding inside one integration)

For each realization, integrate **once**, then grid that same trajectory at:

- the **adaptive** grid (`N_YEARS` snapshots over `[0, t_reached]`, matching the existing cells), and
- a small sweep of **fixed physical Δt** grids: snapshot at `t = 0, Δt, 2Δt, … ≤ t_reached`, so
  every growth increment spans the same horizon Δt (the snapshot *count* varies per run).

Return only the small per-firm arrays per grid (`Sbar`, `vol`, `g`, per-run β). One integration
per run, cheap re-gridding, low memory. Because every grid sees identical realizations, any
difference in β / kurtosis is purely the Δt choice, not realization noise — a clean paired test.

## Headline diagnostic — mixing inflation (weighting-robust)

Naive pooling has a confound: adaptive gives every run exactly `N_YEARS` increments (equal
weight, heterogeneous Δt), while fixed Δt gives long-lived runs many more increments than
short ones (common Δt, unequal weight). To isolate the **horizon-mixing** effect from the
weighting effect, the primary metric is

```
mixing_inflation(grid) = exkurt(pooled g)  −  median_over_runs( exkurt(per-run g) )
```

Per-run kurtosis is computed within a single run (single Δt, no mixing); the pooled value adds
the cross-run contribution. Expectation: large positive inflation for **adaptive** (heterogeneous
Δt), small inflation for **fixed** Δt (common horizon). This holds regardless of count weighting.
β and the raw pooled moments are reported alongside for context.

## What changes (additive)

- New function `run_once_dt(run_seed, S_run, dt_fixed, n_years)` — same graph/disorder/integration
  as `run_once`, but returns `{grid_label: {Sbar, vol, g, beta, n_inc, dt}}` for the adaptive grid
  plus each fixed Δt, from one integration. Shares a small helper `_msb_from_grid(sol, t_used,
  t_grid, S_run)` that applies the Moran cross-sectional normalization and the MAD volatility.
- New ensemble cell: run `run_once_dt` over the same seed set (`joblib.Parallel`), collect per-grid.
- New diagnostic cell: histogram of per-run `Δt_adaptive = t_reached/N_YEARS` (shows the horizon
  spread driving the mixing) + a summary table of β, skew, exkurt, and mixing-inflation per grid.
- New plot cells: **excess kurtosis vs Δt** (smoking gun — does adaptive sit above the fixed-Δt
  curve?), **β vs Δt**, and overlaid growth PDFs (adaptive vs a representative fixed Δt).
- New markdown cell: interpretation / verdict.

## Constraints and caveats

- **Trajectory-resolution floor.** `run_once` samples on `t_eval = linspace(0, tmax=500, n_eval=3000)`
  (spacing ≈0.17); a diverged run keeps only the points before blow-up. A fixed Δt finer than that
  spacing would alias under `np.interp`. So the fixed Δt set stays above the floor (default
  `{0.25, 0.5, 1.0, 2.0}`), and each grid reports its effective Δt and increment count so the floor
  is visible, not hidden.
- **Cost.** One ensemble pass for the study (separate from the existing ensemble cell). Prototype at
  small `N_RUNS`/`S_ENS` to validate the code, then scale.
- Existing cells are not modified; the study is self-contained below them.

## Success criterion

A quantified verdict: Δ(excess kurtosis) and Δβ between the adaptive grid and a matched fixed Δt,
plus the mixing-inflation metric for each, with a one-paragraph statement of whether the MSB
fat-tail and β claims survive a common Δt or are partly a horizon-mixing artifact.
