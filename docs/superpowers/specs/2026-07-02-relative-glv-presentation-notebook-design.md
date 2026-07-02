# Relative-GLV presentation notebook — design

**Date:** 2026-07-02
**Target:** `../relative-glv/story.ipynb` (enhanced in place)

## Goal

Turn the existing `story.ipynb` into a self-contained, GitHub-facing presentation of
the relative GLV model: model → MSB stylized facts → discriminating tests → phase
diagram → wealth → DMFT. Every figure uses one consistent palette.

## Decisions

- **Base:** enhance existing `story.ipynb` in place (no new notebook).
- **Compute:** live-light — recompute cheap figures inline, load precomputed `.npz`
  for heavy ones.
- **Palette:** adopt the already-de-facto palette used across `scripts/`.
- **Scope:** the curated story set + own-degree-consistent extras. Explicitly drop the
  finite-economy / mean-degree / β-vs-N narrative (own-degree only).

## Palette (single source)

Defined once in the notebook setup cell as a `PALETTE` dict + `plt.rcParams` update,
applied by every figure. Scripts already hardcode these same hexes and stay untouched.

```
navy   #1d3557   categorical 1 / q=1 / small firms / model-primary
blue   #457b9d   categorical 2 / q=2 / mid firms
coral  #e76f51   categorical 3 / q=3 / large firms / empirical highlight
teal   #2a9d8f   categorical 4 / q=4 / model-secondary
empirical  = coral        (MSB data points)
reference  = black dashed (analytic / null lines)
neutral    = grays ("0.5", grid)
```

## Figures

| Section | Figure | Source |
|---|---|---|
| Model | intro / equations / normalisation / share-log split | unchanged (markdown) |
| MSB facts | size-volatility relation + growth-rate tent | live |
| 3 discriminating tests | D1 (volatility collapse) + D3 (un-mixed kurtosis) | `data/msb_conditional.npz` |
| Multiscaling (D2) | ζ_q vs q vs MSB + granular null | `data/msb_conditional.npz` |
| Phase diagram | own-degree fine grid, σ_c line | `data/phase_owndeg_fine*.npz` |
| Wealth | firm-size distribution vs empirical | `data/wealth.npz` |
| DMFT (main) | fixed-point observables, live | live (`solve_fixed_point`, `sigma_c`) |
| DMFT validation | DMFT vs matched FC-Gaussian simulation | `data/dmft_validation.npz` |
| DMFT two-time (appendix) | correlation C(t,t') decay | copy `.npz` from `../glv/growing_glv/` |

Two-time DMFT and its data are the FC-Gaussian analytic limit; presented in an appendix
subsection **clearly labelled as the fully-connected Gaussian limit**, separate from the
own-degree main line, so it does not muddy the sparse-network results.

## Dropped

- β-vs-N ("sustained") figure and section.
- β-crossover DMFT (β vs σ).
- finite-economy (`finite_economy.npz`, mean-degree framing).

Rationale: the finite-economy / β-scaling narrative is being retired; own-degree only.

## Two-time DMFT handling

The two-time solver is not in the `relative_glv` package. Lazy path: copy the precomputed
`.npz` into `../relative-glv/data/` and add a load+restyle plotting cell. No solver port.

**ponytail:** copy-`.npz` only. Add-when: the two-time result must be reproducible from
the package → port the solver into `relative_glv/dmft.py` later.

## Notebook structure (final cell order)

1. Setup (imports, `%matplotlib inline`, `PALETTE`, rcParams, `DATA` path)
2. Title + "what this notebook is"
3. Target: two MSB stylized facts
4. The model (equations, own-degree normalisation, share-log split, relative sizes)
5. MSB facts in the model (size-volatility + tent) — live
6. Three discriminating tests: D1, D2 (multiscaling), D3
7. Phase diagram (own-degree)
8. Wealth distribution
9. DMFT: analytic phase structure (live) + validation
10. Appendix: DMFT derivation notes + two-time correlation (FC-Gaussian limit)

## Verification

- Notebook runs top-to-bottom without error (`uv run jupyter nbconvert --execute`).
- Every axes' colors come from `PALETTE` (visual check: no stray matplotlib defaults).
- All referenced `.npz` exist in `../relative-glv/data/`.
- README figure/section list stays consistent with the notebook.
