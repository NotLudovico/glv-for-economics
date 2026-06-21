# criticality_msb — compute-once, analyze-many campaign

Numerical study of the MSB firm-growth stylized facts in Roy's disordered GLV, by approaching the
Multiple-Attractors → Unbounded boundary σ_c. The expensive simulations run **once** over a parameter
grid and are stored to disk; analyses read the dataset and never re-simulate.

## Layout
```
config.py        # grid + simulation settings (edit here)
core.py          # disordered-GLV machinery: build, integrate (LSODA + divergence event), locate σ_c
simulate.py      # dataset builder — sweep -> one npz per run, resumable
dataset.py       # loader / manifest over the run files
observables.py   # analysis toolkit (Moran size, β, growth dist, tails, extinction)
analyze_msb.py   # example analysis -> figure (β/survival/growth-PDF)
data/, figures/  # outputs (git-ignored)
```

## Run
```bash
# fast pipeline check (tiny grid)
uv run python -m criticality_msb.simulate --smoke
uv run python -m criticality_msb.analyze_msb --smoke

# the real campaign
uv run python -m criticality_msb.simulate --jobs 8        # builds data/  (resumable: re-run to continue)
uv run python -m criticality_msb.analyze_msb              # reads data/ -> figures/
uv run python -m criticality_msb.analyze_msb --dt 2 --alive 1e-3   # re-analyze, no recompute
```
Run from the repo root (the `-m` form puts it on the path). The simulator is **resumable** — a run is
done iff its `.npz` exists, so you can stop/restart or grow the grid freely.

## Dataset format (one npz per run)
| field | shape | meaning |
|---|---|---|
| `logx` | (S, n_snap) float32 | per-firm log-abundance on the fine grid (floored at log(1e-30)); logsumexp recovers the Moran cross-section |
| `minx` | (S,) float32 | per-firm minimum over the grid (extinction/persistence) |
| `t_snap` | (n_snap,) float32 | snapshot times |
| metadata | scalars | `sigma, sigma_c, f, seed, S, mu, gamma, lam, topology, tmax, C_eff, reached, diverged, dt_store, rtol, atol` |

Because only raw `log(x)` is stored, the **Moran normalization, the analysis horizon Δt, and the
extinction floor/listing rule are all chosen at analysis time** — one dataset, many analyses.

## Method
Raw (original) GLV dynamics; **LSODA + terminal divergence event** with tight tolerances
(`rtol=1e-8, atol=1e-10`) for fidelity. Per realization: bisection-locate σ_c (cached per seed), then
integrate at σ = f·σ_c for each f. f<1 ⇒ bounded ⇒ a common Δt across runs (no adaptive-grid sampling
pathology). Add parameter slices (topology, S, γ, λ) by editing `config.py` and re-running — existing
runs are skipped.
