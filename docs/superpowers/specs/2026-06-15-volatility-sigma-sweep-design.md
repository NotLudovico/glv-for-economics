# Growth-rate volatility vs size — fixed-μ, high-σ sweep — design

**Date:** 2026-06-15
**Status:** Approved (pending spec review)
**Deliverable:** `notebooks/volatility_sigma_sweep.ipynb`

## Update (2026-06-15, post smoke-test/probe)

A cheap probe (Task 1 of the plan) revealed the key physics and refined the design:

- **At μ=0.2 the supercritical onset is σ≈0.6.** σ∈{0.2, 0.5} stay subcritical (M finite,
  solver reaches τ_max=1e6, physical time ~700 yr). σ≥0.8 are **supercritical**: the scale M
  diverges (overflows ~1e308), the solver stops early, and physical time reaches only ~0.5–2 yr.
- **Consequence:** the physical-time (MSB-faithful) yearly resampling has valid data only in the
  subcritical band; above σ≈0.6 it degenerates. The **shape y stays valid** regardless (1000+ τ
  points even at σ=2.0), but per the chosen direction we **keep physical time and show the
  breakdown** rather than switch clocks.
- **Design changes adopted:**
  1. `run_one` builds the full rescaled state `[y, M, t]` internally
     (`np.concatenate([x0/x0.sum(), [x0.sum()], [0.0]])`) — the original plan passed a length-N
     vector, which is wrong (state must be length N+2).
  2. `run_one` returns a `reached` flag (`sol.status == 0` ⇔ subcritical) and `t_final`.
  3. The sweep classifies each run subcritical/supercritical and reports both counts plus median
     physical-time-reached per σ.
  4. Headline figure: **solid** V-curves for subcritical σ; **dashed/faded "invalid"** curves for
     supercritical σ (shows the V degenerating on the MSB axes).
  5. Summary figure gains a **breakdown** view: supercritical fraction and median physical-time
     reached collapsing vs σ.

The μ, σ grid, N, n_runs, n_years choices are unchanged.

## Goal

Reproduce the Moran–Secchi–Bouchaud size–volatility relation in the rescaled-GLV model,
holding the mean interaction $\mu$ **fixed** and sweeping the disorder $\sigma$ from the
established baseline ($\sigma=0.2$) up to **strong disorder** ($\sigma=4.0$). For each
$\sigma$ we plot per-firm growth-rate volatility against time-average relative size and
watch how the curve morphs as disorder grows.

This is a **descriptive** study (one of three options considered): it observes how the
size–volatility curve changes with $\sigma$. It deliberately does **not** test whether
high-$\sigma$ volatility is a genuine steady-state phenomenon or still a relaxation
transient — that mechanistic question (the start-at-fixed-point / burn-in intervention
from `volatility_v_cause.ipynb`) is out of scope here.

### Background

Every prior volatility notebook (`volatility_degree`, `volatility_finalsize`,
`volatility_v_cause`) operates at a single low disorder $\sigma=0.2$, in the
single-fixed-point phase, at $\mu=\mu_c\approx0.486$. There the size–volatility relation
is a **V** (two power-law branches meeting near the mean size), and `volatility_v_cause`
*proved* that V is a transient relaxation from the initial condition, not steady-state
noise. This notebook leaves that regime: fixing $\mu$ subcritical and cranking $\sigma$
pushes the system toward the multiple-equilibria / chaotic regime where that conclusion
need not hold — we simply look at what the MSB plot does.

## Model (unchanged)

Sparse configuration-model graph (exponential degree law), interaction weights from
`glv.generate_matrix`:
$$\alpha_{ij}=\frac{\mu}{C}+\frac{\sigma}{\sqrt C}\,z_{ij},\qquad z_{ij}\sim\mathcal N(0,1),$$
zero diagonal, multi-edges/self-loops removed. Rescaled dynamics `glv.rescaled_glv_sparse`
with state $(y_1,\dots,y_N,M,t)$, $y$ on the simplex ($x_i=y_iM$).

## Measurement (inline, float64, shape-based)

Per the project convention (degree construction stays inline; no library wrapper) the
measurement is an inline `run_one(W, x0)` lifted from `volatility_finalsize.ipynb`, **not**
`glv.sweep_observables`:

1. One float64 integration `solve_ivp(rescaled_glv_sparse, (0, tau_max), state, method="RK45", max_step=1e2)`.
2. Relative size built **from the shape $y$ directly**: $S_i = N\,y_i/\sum_j y_j$
   (the scale $M$ cancels). This never forms $x=yM$, so it is immune to the float32/M
   overflow trap — essential at high $\sigma$ where $M$ may diverge far past $10^{308}$.
3. Resample on a uniform yearly physical-time grid ($n_\text{years}=100$ between
   $t_\text{phys}[0]$ and $t_\text{final}$).
4. Growth $g_{iy}=\Delta\ln S_i$; whole-window volatility
   $\sigma_i=\sqrt{\pi/2}\,\overline{|g_i-\bar g_i|}$ (adjusted MAD); time-average size
   $\bar S_i$.
5. Return `(avg_size, sigma_i)`. Failed integrations (`sol.t.size < 3`, or
   non-finite/non-increasing $t_\text{final}$) return `None`.

The whole-window measure mixes the relaxation transient with any persistent dynamics —
acceptable for a descriptive "what does the curve look like" study, and noted as a caveat.

## Sweep

For each $\sigma$ in the grid: build `n_runs` fresh graphs at $\mu=$ `MU`, integrate each
from a random initial condition, pool all surviving firms. **Failed runs are dropped and
counted** (no silent truncation — the dropped fraction is reported per $\sigma$). Results
cached to `volatility_sigma_sweep_results.npz`; the sweep cell loads the cache if present.

## Notebook layout

| # | Section | Content |
|---|---------|---------|
| 0 | Intro (md) | Goal, background (prior V-is-transient result), caveats |
| 1 | Setup | `mp.set_start_method("fork")`, imports, `glv.apply_style()`, parameter block |
| 2 | Degree builders (inline) | `exponential_degrees`, `make_even`, `realized_degree` |
| 3 | Measurement (inline) | `run_one(W, x0)` (float64, shape-based, whole-window) |
| 4 | Sweep + cache | loop over $\sigma$ × `n_runs`; pool firms; drop+count failures; save/load `.npz` |
| 5 | Plot 1 — headline | log–log $\bar S$ vs $\sigma_i$, one binned-median V-curve per $\sigma$, viridis-by-$\sigma$ + colorbar; save `volatility_sigma_sweep.png` |
| 6 | Plot 2 — summary vs $\sigma$ | descending-branch slope (MSB exponent $\beta$), median volatility level, collapse fraction ($S<$ floor), dropped-run fraction |
| 7 | Summary (md) | Observations (filled after running) |

**Inline helpers** (lifted from sibling notebooks): `vbin` (equal-count binned median for
log–log), `two_slopes` (V-minimum split + per-branch power-law fits).

## Default parameters (top-of-notebook, all adjustable)

| Symbol | Value | Note |
|--------|-------|------|
| $N$ | 1000 | species / firms |
| $C$ | 5 | mean degree |
| $\mu$ | 0.2 | **fixed**, subcritical (clean unique fixed point at low $\sigma$) |
| $\sigma$ grid | `[0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]` | baseline → strong disorder |
| `n_runs` | 10 | graph realizations per $\sigma$ (pooled) |
| $\tau_{\max}$ | $10^6$ | rescaled-time horizon |
| $n_\text{years}$ | 100 | yearly resample grid |
| `n_workers` | `min(8, cpu_count)` | parallel pool |

Compute: ~70 integrations ($N=1000$, $\tau_{\max}=10^6$), ≈2× one sibling notebook;
cached after first run.

## Conventions

- **Plots:** `glv.apply_style()`; all labels/titles/legends in LaTeX mathtext via raw
  strings ($\mu$, $\sigma$, $\bar S$, $\sigma_i$, $\beta$). Faint log grid allowed.
- **Reproducibility:** seeded `np.random.default_rng`; per-$\sigma$, per-run seeds.
- **No new library code.** Measurement and degree construction stay inline.
- **Live-firm binning:** V-curves restrict to firms with $\bar S>$ floor ($10^{-3}$);
  collapse fraction reported separately.

## Caveats (baked into the intro markdown)

- $\mu_c$ drifts with $\sigma$; $\mu=0.2$ is a held subcritical reference, not critical at
  every $\sigma$.
- High $\sigma$: integrations may fail or $M$ may diverge — shape-based $S$ dodges
  overflow; failures dropped and **reported**.
- Whole-window volatility cannot separate transient from genuine steady-state dynamics.

## Out of scope (not selected)

The mechanistic transient-vs-genuine intervention (start-at-fixed-point, late burn-in
window); locating a precise $\sigma$-transition / order-parameter phase diagram; varying
$\mu$; degree-resolved or final-size analyses (covered by sibling notebooks).
