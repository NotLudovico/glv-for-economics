# Critical disorder $\sigma_c$ at fixed $\mu$ — locate then measure — design

**Date:** 2026-06-17
**Status:** Approved (pending spec review)
**Deliverable:** `notebooks/sigma_critical.ipynb`

## Goal

The $\sigma$-analog of the existing $\mu_c$ workflow. Holding the mean interaction
$\mu=0.2$ **fixed**, sweep the disorder $\sigma$ to **locate the critical point $\sigma_c$**
(the subcritical→supercritical boundary), then **simulate at $\sigma_c$** and measure two
observables:

1. growth-rate volatility vs time-average relative size (the Moran–Secchi–Bouchaud V), and
2. the pooled growth-rate distribution.

This mirrors how the $\mu$ workflow locates $\mu_c$ (via `glv.find_empirical_mu_c` /
`glv.find_mu_c_shape_scalar`) and then measures observables at $\mu_c$. Here the roles of
$\mu$ and $\sigma$ are swapped.

### Relationship to existing notebooks

- `notebooks/volatility_sigma_sweep.ipynb` is a **descriptive** $\sigma$-sweep of the V-curve
  at fixed $\mu=0.2$. It noticed the supercritical onset $\sigma\approx0.6$ as a byproduct but
  does **not** locate a precise $\sigma_c$ and does **not** measure the growth-rate distribution.
- This is a **new standalone notebook**. It leaves the descriptive sweep untouched.
- That sibling notebook's summary explicitly defers "measuring on the rescaled-$\tau$ clock from
  the shape $y$" as "a separate experiment." **This notebook is that experiment.**

## Model (unchanged from the sigma family)

Sparse configuration-model graph (exponential degree law). Interaction weights
$$\alpha_{ij}=A_{ij}\left(\frac{\mu}{C}+\frac{\sigma}{\sqrt C}\,z_{ij}\right),\qquad z_{ij}\sim\mathcal N(0,1),$$
on the binary adjacency $A$ (self-loops removed). Rescaled dynamics
`glv.rescaled_glv_sparse` with state $(y_1,\dots,y_N,M,t)$, $y$ on the simplex, $x_i=y_iM$.

## Method — locating $\sigma_c$ (shape-scalar zero-crossing)

Mirrors `glv.find_mu_c_shape_scalar`, but sweeps $\sigma$ at fixed $\mu$ instead of $\mu$ at
fixed $\sigma$. Inlined in the notebook per the sigma-notebook convention (no new library code).

For a fixed binary adjacency $A$ and one **frozen** disorder draw $z$ over its edges, for each
$\sigma$ on a grid:

1. Build $W_{ij}=A_{ij}(\mu/C+(\sigma/\sqrt C)z_{ij})$ with $z$ held fixed (single realization
   followed across the whole $\sigma$ sweep).
2. Integrate the rescaled dynamics to $\tau_{\max}=10^4$ (the shape relaxes long before then),
   with an event that halts when $M$ reaches `m_cap` $=10^{250}$ so supercritical runs return a
   valid relaxed shape instead of overflowing.
3. Compute the shape scalar from the relaxed shape $y^*$:
   $$c=(y^*)^\top W\,y^* - (y^*)^\top y^*.$$
   The sign of $c$ fixes the sign of $\mathrm dM/\mathrm d\tau=1+cM$: $c<0$ subcritical (M bounded),
   $c>0$ supercritical (M diverges).
4. $\sigma_c$ is the **first zero-crossing** of $c(\sigma)$, read by linear interpolation on the
   $\sigma$ grid.

Sweep several `(graph, seed)` realizations and aggregate: report $\sigma_c$ as
mean $\pm$ std across realizations. Expected near $\sigma\approx0.6$ at $\mu=0.2$ (consistent with
the sibling sweep, which is subcritical for $\sigma\lesssim0.6$).

**Caveat (baked into intro):** above $\sigma_c$ the shape no longer relaxes to a fixed point, so
$y^*$ at $\tau_{\max}$ is not a true fixed point there and $c$ is noisy. The first sign change is
read from the subcritical side where $y^*$ genuinely relaxes, so $\sigma_c$ itself is well
determined; values of $c$ deep in the supercritical band are indicative only.

## Method — measuring at $\sigma_c$ (rescaled-$\tau$ shape clock)

At the critical point the physical-time MSB clock degenerates ($M$ diverges, the integrator
reaches $\sim1$ yr), so per the sibling's deferred plan and the project memory we measure on the
**rescaled-$\tau$ shape clock**. Inline `run_one(W, x0)` (overflow-safe, lifted from
`volatility_sigma_sweep.ipynb`), one float64 integration to $\tau_{\max}=10^6$:

1. Relative size from the **shape directly**: $S_i=N\,y_i/\sum_j y_j$ ($M$ cancels — never forms
   $x=yM$, immune to overflow).
2. Resample on a uniform grid in **rescaled time $\tau$** (not physical years) between the first
   and last $\tau$ reached, $n_{\text{samples}}=100$.
3. Growth $g_i=\Delta\ln S_i$; whole-window volatility
   $\sigma_i=\sqrt{\pi/2}\,\overline{|g_i-\bar g_i|}$ (adjusted MAD); time-average size $\bar S_i$.
4. Return `(avg_size, volatility, g_flat)` — the third entry is the pooled growth array for the
   distribution plot. Failed integrations (`sol.t.size<3`, non-finite/non-increasing $\tau$
   window) return `None` and are dropped+counted.

Pool over `n_runs` fresh graphs at $\mu=0.2$, $\sigma=\sigma_c$.

**Caveat (baked in):** volatility here is per unit rescaled time, not literal MSB calendar years;
it is the well-defined critical-point analog of the physical-year measure, not identical to it.

## Notebook layout

| # | Section | Content |
|---|---------|---------|
| 0 | Intro (md) | Goal ($\sigma$-analog of $\mu_c$), shape-scalar method, $\tau$-clock choice, caveats |
| 1 | Setup | `mp.set_start_method("fork")`, imports, `glv.apply_style()`, parameter block |
| 2 | Degree builder (inline) | `exponential_degrees`, `make_even`, config-model binary adjacency $A$ |
| 3 | $\sigma_c$ locator + cache | inline `c_of_sigma`; $\sigma$-grid sweep over realizations; zero-crossing → $\sigma_c\pm$ spread; plot $c(\sigma)$ with crossing; cache `.npz` |
| 4 | Simulate at $\sigma_c$ + cache | inline $\tau$-clock `run_one`; `n_runs` graphs at $\sigma_c$; pool firms; drop+count failures; cache `.npz` |
| 5 | Plot — volatility vs size | log–log $\bar S$ vs $\sigma_i$, binned-median V-curve at $\sigma_c$ (`vbin`, `two_slopes`); save `.png` |
| 6 | Plot — growth distribution | pooled $g$ PDF on log-$y$, Gaussian + Laplace overlays, report tail shape; save `.png` |
| 7 | Summary (md) | observations (filled after running) |

**Inline helpers** lifted from sibling notebooks: `vbin` (equal-count binned median for log–log),
`two_slopes` (V-minimum split + per-branch power-law fits), `run_one`, `c_of_sigma`.

## Default parameters (top-of-notebook, all adjustable)

| Symbol | Value | Note |
|--------|-------|------|
| $N$ | 1000 | species / firms |
| $C$ | 5 | mean degree (matches sigma family) |
| $\mu$ | 0.2 | **fixed** |
| $\sigma$ locator grid | `np.linspace(0.3, 1.0, 20)` | brackets the expected $\sigma_c\approx0.6$ |
| `n_real` | 10 | `(graph, seed)` realizations for the $\sigma_c$ estimate |
| $\tau_{\max}$ (locator) | $10^4$ | shape relaxes well before then |
| `m_cap` | $10^{250}$ | halt M before overflow |
| $\tau_{\max}$ (measurement) | $10^6$ | trajectory horizon at $\sigma_c$ |
| `n_runs` | 10–20 | graphs pooled at $\sigma_c$ |
| `n_samples` | 100 | $\tau$-grid resample points |
| `n_workers` | `min(8, cpu_count)` | parallel pool |

## Conventions

- **Plots:** `glv.apply_style()`; all labels/titles/legends in LaTeX mathtext via raw strings
  ($\mu$, $\sigma$, $\sigma_c$, $\bar S$, $\sigma_i$, $\beta$, $c$). Faint log grid allowed.
- **Reproducibility:** seeded `np.random.default_rng`; per-realization, per-run seeds.
- **No new library code.** Measurement, degree construction, and the shape-scalar are inline
  (mirrors `volatility_sigma_sweep.ipynb`). Alternative considered: a library
  `find_sigma_c_shape_scalar` + test — rejected to stay consistent with the inline sigma family.
- **No silent truncation.** Dropped runs and the supercritical fraction are reported.

## Out of scope

- Forcing the physical-year clock at $\sigma_c$ (the sibling already showed it breaks down).
- A precise $\sigma$-resolved order-parameter phase diagram beyond the single $\sigma_c$ estimate.
- The transient-vs-steady-state intervention from `volatility_v_cause.ipynb`.
- Varying $\mu$, or degree-resolved / final-size analyses (covered by sibling notebooks).
